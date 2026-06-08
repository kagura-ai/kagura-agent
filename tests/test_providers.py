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

from kagura_agent.membrane.lease import Budget, CredentialBroker
from kagura_agent.membrane.providers import (
    AwsStsProvider,
    CloudflareTokenProvider,
    GcpImpersonationProvider,
    GitHubAppProvider,
    MemoryCloudProvider,
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
