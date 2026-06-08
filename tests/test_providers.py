"""v0.2 (A2/A3): concrete CredProvider implementations.

Each provider maps `mint(scope, ttl)` onto a cloud's short-lived-credential API.
The network call is injected, so request construction and the stateless/stateful
contract are unit-tested here; the real transport is the deployment edge.

- AWS STS / GCP SA impersonation / GitHub App = stateless (handle is None, the
  cred expires on its own, revoke is a no-op).
- Cloudflare API token = stateful (mint returns a revoke handle, revoke deletes).
"""

import json

import pytest

from kagura_agent.membrane.launcher import LaunchSpec, docker_run_args
from kagura_agent.membrane.lease import Budget, CredentialBroker
from kagura_agent.membrane.providers import (
    AwsStsProvider,
    CloudflareTokenProvider,
    GcpImpersonationProvider,
    GitHubAppProvider,
    MemoryCloudProvider,
    MemoryWriteLocked,
    resolve_memory_scope,
)


def _clock() -> float:
    return 1000.0


# --- AWS STS (stateless) --------------------------------------------------

async def test_aws_sts_mints_env_creds_and_no_handle() -> None:
    seen: dict[str, object] = {}

    def fake_assume_role(req: dict[str, object]) -> dict[str, object]:
        seen.update(req)
        return {
            "Credentials": {
                "AccessKeyId": "AKIAEXAMPLE",
                "SecretAccessKey": "secret",
                "SessionToken": "session-token",
            }
        }

    provider = AwsStsProvider(assume_role=fake_assume_role)
    cred, handle = await provider.mint("arn:aws:iam::123:role/deploy", 900)

    assert handle is None
    assert provider.stateful is False
    assert seen["RoleArn"] == "arn:aws:iam::123:role/deploy"
    assert seen["DurationSeconds"] == 900
    env = json.loads(cred)
    assert env["AWS_ACCESS_KEY_ID"] == "AKIAEXAMPLE"
    assert env["AWS_SESSION_TOKEN"] == "session-token"


async def test_aws_sts_revoke_is_a_safe_noop() -> None:
    provider = AwsStsProvider(
        assume_role=lambda req: {
            "Credentials": {"AccessKeyId": "a", "SecretAccessKey": "b", "SessionToken": "c"}
        }
    )
    await provider.revoke(None)  # STS creds expire; revoke must not raise


async def test_aws_sts_drives_through_broker() -> None:
    provider = AwsStsProvider(
        assume_role=lambda req: {
            "Credentials": {"AccessKeyId": "a", "SecretAccessKey": "b", "SessionToken": "c"}
        }
    )
    broker = CredentialBroker({"aws": provider}, clock=_clock)
    lease = await broker.acquire("aws", scope="arn:role/x", ttl=900, budget=Budget(3600))

    assert lease.stateful is False
    assert json.loads(lease.cred)["AWS_ACCESS_KEY_ID"] == "a"


# --- GCP SA impersonation (stateless) -------------------------------------

async def test_gcp_impersonation_requests_token_for_target_sa() -> None:
    seen: dict[str, object] = {}

    def fake_generate(req: dict[str, object]) -> dict[str, object]:
        seen.update(req)
        return {"accessToken": "ya29.token", "expireTime": "2026-06-08T00:30:00Z"}

    provider = GcpImpersonationProvider(generate_token=fake_generate)
    cred, handle = await provider.mint("deploy@proj.iam.gserviceaccount.com", 1800)

    assert handle is None
    assert provider.stateful is False
    assert str(seen["name"]).endswith("deploy@proj.iam.gserviceaccount.com")
    assert seen["lifetime"] == "1800s"
    assert cred == "ya29.token"


# --- GitHub App installation token (stateless) ----------------------------

async def test_github_app_mints_installation_token() -> None:
    seen: dict[str, object] = {}

    def fake_create(req: dict[str, object]) -> dict[str, object]:
        seen.update(req)
        return {"token": "ghs_example", "expires_at": "2026-06-08T01:00:00Z"}

    provider = GitHubAppProvider(create_token=fake_create)
    cred, handle = await provider.mint("installation:42", 3600)

    assert handle is None
    assert provider.stateful is False
    assert seen["installation_id"] == "42"
    assert cred == "ghs_example"


# --- Cloudflare API token (stateful: mint -> use -> revoke) ----------------

async def test_cloudflare_mints_token_with_revoke_handle() -> None:
    created: dict[str, object] = {}
    revoked: list[str] = []

    def fake_create(req: dict[str, object]) -> dict[str, object]:
        created.update(req)
        return {"result": {"id": "tok-1", "value": "cf-secret"}}

    provider = CloudflareTokenProvider(create=fake_create, delete=revoked.append)
    cred, handle = await provider.mint("zone:edit", 600)

    assert provider.stateful is True
    assert cred == "cf-secret"
    assert handle == "tok-1"

    await provider.revoke(handle)
    assert revoked == ["tok-1"]


async def test_cloudflare_revoke_ignores_none_handle() -> None:
    def explode(token_id: str) -> None:
        raise AssertionError("delete must not be called with no handle")

    provider = CloudflareTokenProvider(
        create=lambda req: {"result": {"id": "x", "value": "y"}},
        delete=explode,
    )
    await provider.revoke(None)  # nothing to delete; must be a no-op


async def test_cloudflare_orphan_swept_by_broker_on_restart() -> None:
    revoked: list[str] = []
    provider = CloudflareTokenProvider(
        create=lambda req: {"result": {"id": "tok-1", "value": "v"}},
        delete=revoked.append,
    )
    broker = CredentialBroker({"cf": provider}, clock=_clock)
    await broker.acquire("cf", scope="zone:edit", ttl=600, budget=Budget(3600))

    await broker.sweep()  # crash/restart leaves the CF token open; sweeper revokes it

    assert revoked == ["tok-1"]


# --- memory-cloud (stateless, scoped, leased from the host auth-login session) --
# The trusted host holds the `kagura auth login` refresh token; the provider
# exchanges it for a short-lived, scoped access token (kagura auth refresh
# --scope ... ; kagura auth token) that the membrane leases into the container.
# The container only ever holds the access token — never the refresh token.

async def test_memory_cloud_mints_scoped_short_lived_token() -> None:
    seen: dict[str, object] = {}

    def fake_exchange(req: dict[str, object]) -> dict[str, object]:
        # The host's refresh session lives in this closure (host-side); only the
        # access token is handed back for injection into the container.
        seen.update(req)
        return {"access_token": "kmc-access-1", "expires_in": req["ttl"]}

    provider = MemoryCloudProvider(exchange=fake_exchange)
    cred, handle = await provider.mint("memory:read", 300)

    assert handle is None
    assert provider.stateful is False
    assert seen["scope"] == "memory:read"
    assert seen["ttl"] == 300
    assert cred == "kmc-access-1"


async def test_memory_cloud_default_scope_is_read_only() -> None:
    # Widening to memory:write triggers a device-flow re-approval (HITL) at the
    # CLI, so the broker-driven path defaults to read-only and never silently
    # grants write to the shared backbone.
    assert MemoryCloudProvider.READ_ONLY_SCOPE == "memory:read"


async def test_memory_cloud_drives_through_broker_as_leased_token() -> None:
    provider = MemoryCloudProvider(exchange=lambda req: {"access_token": "kmc-1"})
    broker = CredentialBroker({"memory": provider}, clock=_clock)
    lease = await broker.acquire("memory", scope="memory:read", ttl=300, budget=Budget(3600))

    assert lease.cred == "kmc-1"  # container gets only the short-lived leased token
    assert lease.stateful is False


@pytest.mark.parametrize(
    "provider",
    [
        AwsStsProvider(
            assume_role=lambda req: {
                "Credentials": {"AccessKeyId": "a", "SecretAccessKey": "b", "SessionToken": "c"}
            }
        ),
        GcpImpersonationProvider(generate_token=lambda req: {"accessToken": "t"}),
        GitHubAppProvider(create_token=lambda req: {"token": "t"}),
        MemoryCloudProvider(exchange=lambda req: {"access_token": "t"}),
    ],
)
async def test_stateless_providers_return_no_handle(provider: object) -> None:
    _, handle = await provider.mint("scope", 600)  # type: ignore[attr-defined]
    assert handle is None
    assert provider.stateful is False  # type: ignore[attr-defined]
    await provider.revoke(None)  # type: ignore[attr-defined]  # no-op, must not raise


# --- v0.2-A6 acceptance: write default-locked ----------------------------
# memory write is locked by default. Widening to memory:write requires an
# explicit approval (device-flow HITL, wired in v0.3 #14 / v0.4 #15). This
# helper is the fail-closed seam: no approval -> read-only, always.

def test_resolve_memory_scope_defaults_to_read_only() -> None:
    assert resolve_memory_scope() == "memory:read"
    assert resolve_memory_scope(write_approved=False) == "memory:read"


def test_resolve_memory_scope_grants_write_only_with_approval() -> None:
    assert resolve_memory_scope(write_approved=True) == "memory:write"


# --- v0.2-A6 (CSO hardening): the write-lock is STRUCTURAL, not advisory ---
# resolve_memory_scope is the policy default; MemoryCloudProvider.mint is the
# enforcement point. A caller cannot bypass the lock by passing "memory:write"
# straight into broker.acquire — the provider itself refuses unless it was
# constructed write_approved=True (where the #14/#15 device-flow approval lands).

async def test_memory_cloud_read_is_always_allowed() -> None:
    provider = MemoryCloudProvider(exchange=lambda req: {"access_token": "kmc-r"})
    cred, _ = await provider.mint("memory:read", 300)
    assert cred == "kmc-r"


async def test_memory_cloud_write_is_locked_by_default() -> None:
    provider = MemoryCloudProvider(exchange=lambda req: {"access_token": "kmc-w"})
    with pytest.raises(MemoryWriteLocked):
        await provider.mint("memory:write", 300)


async def test_memory_cloud_write_allowed_only_when_approved() -> None:
    provider = MemoryCloudProvider(
        exchange=lambda req: {"access_token": "kmc-w"}, write_approved=True
    )
    cred, _ = await provider.mint("memory:write", 300)
    assert cred == "kmc-w"


async def test_broker_acquire_cannot_bypass_the_write_lock() -> None:
    # The CSO bypass concern: broker.acquire(scope="memory:write") must NOT yield
    # a write token when the provider is read-locked — enforcement is structural.
    provider = MemoryCloudProvider(exchange=lambda req: {"access_token": "kmc-w"})
    broker = CredentialBroker({"memory": provider}, clock=_clock)
    with pytest.raises(MemoryWriteLocked):
        await broker.acquire("memory", scope="memory:write", ttl=300, budget=Budget(900))


# --- v0.2-A6 acceptance: MemoryCloudProvider rides the full broker lifecycle --

async def test_memory_cloud_rides_acquire_renew_release_sweep() -> None:
    minted: list[tuple[str, int]] = []

    def exchange(req: dict[str, object]) -> dict[str, object]:
        minted.append((str(req["scope"]), int(req["ttl"])))  # type: ignore[arg-type]
        return {"access_token": f"kmc-{len(minted)}"}

    provider = MemoryCloudProvider(exchange=exchange)
    broker = CredentialBroker({"memory": provider}, clock=_clock)

    lease = await broker.acquire(
        "memory", scope=resolve_memory_scope(), ttl=300, budget=Budget(900)
    )
    assert lease.cred == "kmc-1"
    assert lease.scope == "memory:read"

    renewed = await broker.renew(lease, ttl=300)
    assert renewed.cred == "kmc-2"
    assert renewed.budget.remaining() == 600  # 900 - 300 spent on renew

    # stateless: release/sweep must not attempt a revoke and the ledger stays empty
    await broker.release(renewed)
    await broker.sweep()  # no open stateful leases -> no-op, must not raise
    assert minted == [("memory:read", 300), ("memory:read", 300)]


# --- v0.2-A6 acceptance: container holds only the short-lived access token --

async def test_leased_memory_token_into_container_is_access_only() -> None:
    # The exchange returns the *raw* host-side response, which in reality carries
    # both the refresh token and the freshly-minted access token. The provider
    # must extract ONLY the access token into the lease — so what the membrane
    # injects into the container can never contain the refresh token. Returning
    # both here makes the assertion real: if mint leaked the whole response (or
    # the wrong field), the refresh token would reach the rendered env and fail.
    refresh_token = "kmc-refresh-SECRET-must-stay-host-side"
    access_token = "kmc-access-short-lived"

    def host_exchange(req: dict[str, object]) -> dict[str, object]:
        return {"access_token": access_token, "refresh_token": refresh_token}

    provider = MemoryCloudProvider(exchange=host_exchange)
    broker = CredentialBroker({"memory": provider}, clock=_clock)
    lease = await broker.acquire(
        "memory", scope=resolve_memory_scope(), ttl=300, budget=Budget(900)
    )

    assert lease.cred == access_token  # only the access token is leased
    spec = LaunchSpec(image="kagura-agent:python", env={"KAGURA_MEMORY_TOKEN": lease.cred or ""})
    rendered = " ".join(docker_run_args(spec))

    assert access_token in rendered
    assert refresh_token not in rendered  # the refresh token never crosses the membrane
