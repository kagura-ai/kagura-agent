"""Concrete CredProvider implementations for the CredentialBroker.

Each maps `mint(scope, ttl)` onto a cloud's short-lived-credential API:

- **stateless** (AWS STS, GCP SA impersonation, GitHub App): mint and forget.
  The cred expires on its own, so `handle` is `None` and `revoke` is a no-op.
- **stateful** (Cloudflare API token): mint -> use -> revoke. `mint` returns a
  revoke handle and `revoke(handle)` deletes the token (swept on restart).

The actual network call is **injected** as a callable, so request construction
and the stateless/stateful contract are unit-testable and the core keeps its
zero-dependency invariant (tests run without boto3/google-auth/httpx installed).
Deployment supplies the transport, e.g.::

    import boto3
    sts = boto3.client("sts")
    provider = AwsStsProvider(assume_role=lambda req: sts.assume_role(**req))

The injected callables are intentionally synchronous: the real SDKs are blocking
and these run from the trusted cockpit, not inside the agent container.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from kagura_agent.membrane.cred_env import (
    aws_cred_env,
    cloudflare_cred_env,
    gcp_cred_env,
    github_cred_env,
)

# A provider's injected transport speaks raw JSON-ish dicts in and out; the
# cloud-specific shape is the provider's concern, not the broker's.
Json = Mapping[str, Any]


class AwsStsProvider:
    """AWS STS AssumeRole. `scope` is the role ARN; `ttl` is DurationSeconds.

    The cred is an env-JSON blob the launcher injects as the container's AWS_*
    environment — no long-lived key ever lands in the image or ambient env.
    """

    stateful = False

    def __init__(
        self,
        *,
        assume_role: Callable[[Json], Json],
        session_name: str = "kagura-agent",
    ) -> None:
        self._assume_role = assume_role
        self._session_name = session_name

    async def mint(self, scope: str, ttl: int) -> tuple[str, str | None]:
        resp = self._assume_role(
            {
                "RoleArn": scope,
                "RoleSessionName": self._session_name,
                "DurationSeconds": ttl,
            }
        )
        c = resp["Credentials"]
        cred = json.dumps(
            {
                "AWS_ACCESS_KEY_ID": c["AccessKeyId"],
                "AWS_SECRET_ACCESS_KEY": c["SecretAccessKey"],
                "AWS_SESSION_TOKEN": c["SessionToken"],
            }
        )
        return cred, None

    async def revoke(self, handle: str | None) -> None:
        # STS session credentials cannot be revoked; they expire at DurationSeconds.
        return None

    def cred_to_env(self, cred: str) -> dict[str, str]:
        return aws_cred_env(cred)


class GcpImpersonationProvider:
    """GCP service-account impersonation. `scope` is the target SA email."""

    stateful = False

    def __init__(self, *, generate_token: Callable[[Json], Json]) -> None:
        self._generate_token = generate_token

    async def mint(self, scope: str, ttl: int) -> tuple[str, str | None]:
        resp = self._generate_token(
            {
                "name": f"projects/-/serviceAccounts/{scope}",
                "scope": ["https://www.googleapis.com/auth/cloud-platform"],
                "lifetime": f"{ttl}s",
            }
        )
        return str(resp["accessToken"]), None

    async def revoke(self, handle: str | None) -> None:
        # Impersonation access tokens are not revocable; they expire at `lifetime`.
        return None

    def cred_to_env(self, cred: str) -> dict[str, str]:
        return gcp_cred_env(cred)


class GitHubAppProvider:
    """GitHub App installation token. `scope` is ``installation:<id>``.

    GitHub fixes the token lifetime at ~1h regardless of request, so `ttl` is
    advisory only (the broker's budget still bounds total renewals).
    """

    stateful = False

    def __init__(self, *, create_token: Callable[[Json], Json]) -> None:
        self._create_token = create_token

    async def mint(self, scope: str, ttl: int) -> tuple[str, str | None]:
        installation_id = scope.split(":", 1)[1] if ":" in scope else scope
        resp = self._create_token({"installation_id": installation_id})
        return str(resp["token"]), None

    async def revoke(self, handle: str | None) -> None:
        # Installation tokens expire (~1h); nothing to revoke.
        return None

    def cred_to_env(self, cred: str) -> dict[str, str]:
        return github_cred_env(cred)


class MemoryWriteLocked(RuntimeError):
    """A read-locked MemoryCloudProvider was asked to mint a ``memory:write`` token."""


class MemoryCloudProvider:
    """memory-cloud scoped access token, leased from the host's auth-login session.

    The trusted host runs ``kagura auth login`` once (OAuth2 device flow; the
    refresh token is stored host-side at ~/.kagura/credentials.json). This
    provider exchanges that session for a short-lived, **scoped** access token
    (``kagura auth refresh --scope <scope>`` then ``kagura auth token``) that the
    membrane leases into the agent container as env. The container only ever
    holds the short-lived access token — never the refresh token.

    Scope is read-only by default. Widening to ``memory:write`` triggers a device
    flow (human re-approval) at the CLI, which fails closed in an unattended
    container, so a hijacked agent cannot silently obtain write access to the
    shared memory backbone (the dominant prompt-injection persistence risk).

    The write-lock is **structural, not advisory**: ``mint`` refuses a
    ``memory:write`` scope unless this provider was constructed
    ``write_approved=True``. That closes the bypass where a caller hands
    ``memory:write`` straight to ``CredentialBroker.acquire`` (which forwards any
    scope string verbatim). The ``write_approved`` flag is where the device-flow
    HITL approval lands — wired in v0.3 (#14) / v0.4 graduation (#15); v0.2 only
    ever constructs it read-locked.
    """

    stateful = False  # access tokens expire (auto-refreshed near expiry); none to revoke

    READ_ONLY_SCOPE = "memory:read"
    WRITE_SCOPE = "memory:write"

    def __init__(
        self, *, exchange: Callable[[Json], Json], write_approved: bool = False
    ) -> None:
        self._exchange = exchange
        self._write_approved = write_approved

    @property
    def write_approved(self) -> bool:
        """Whether this provider was constructed with write approval (read-only by
        default). Public so the broker can hard-gate ``memory:write`` without
        reaching into a private attribute (see CredentialBroker.acquire)."""
        return self._write_approved

    @classmethod
    def requires_write_approval(cls, scope: str) -> bool:
        """Whether a memory scope needs write approval — **fail-closed**.

        Any ``memory:*`` scope that is not the read-only scope (``memory:write``,
        and any future ``memory:admin`` / ``memory:delete``) requires approval. A
        new scope is treated as privileged by default; only ``memory:read`` is
        exempt. This is the single source of truth the broker and ``mint`` both
        consult, so a literal ``"memory:write"`` comparison can't silently miss a
        future write-ish scope.

        The scope is normalized (strip + casefold) for the decision so a case or
        whitespace variant (``"MEMORY:WRITE"``, ``" memory:write "``) cannot slip
        past the guard — fail-closed against malformed-but-privileged scopes.
        """
        normalized = scope.strip().casefold()
        return normalized != cls.READ_ONLY_SCOPE and normalized.startswith("memory:")

    async def mint(self, scope: str, ttl: int) -> tuple[str, str | None]:
        if self.requires_write_approval(scope) and not self._write_approved:
            raise MemoryWriteLocked(
                "memory:write is locked: this provider is read-only. Widening requires "
                "device-flow re-approval (HITL), wired in #14 (v0.3) / #15 (v0.4). "
                "A read-locked provider never mints a write token, even via broker.acquire."
            )
        resp = self._exchange({"scope": scope, "ttl": ttl})
        return str(resp["access_token"]), None

    async def revoke(self, handle: str | None) -> None:
        # Access tokens expire on their own; there is nothing to revoke.
        return None


def resolve_memory_scope(*, write_approved: bool = False) -> str:
    """Resolve the memory scope to lease — write is locked by default (v0.2-A6).

    The fail-closed seam for "write は既定ロック": absent an explicit approval the
    scope is read-only, so a hijacked/unattended agent can never silently widen
    to ``memory:write`` on the shared backbone. Granting write requires a device
    flow re-approval (HITL); that approval path is wired in v0.3 (#14) and the
    quarantine→trusted graduation gate in v0.4 (#15). This function only encodes
    the default *policy* — `MemoryCloudProvider.mint` is the structural
    enforcement point (it refuses ``memory:write`` unless ``write_approved``), so
    the lock holds even if a caller bypasses this resolver.
    """
    if write_approved:
        return MemoryCloudProvider.WRITE_SCOPE
    return MemoryCloudProvider.READ_ONLY_SCOPE


class CloudflareTokenProvider:
    """Cloudflare API token: mint a scoped child token, revoke it by id on release.

    Stateful — an un-revoked token stays valid at the provider, so the broker's
    ledger sweeps any orphan on restart via `revoke`.
    """

    stateful = True

    def __init__(
        self,
        *,
        create: Callable[[Json], Json],
        delete: Callable[[str], None],
    ) -> None:
        self._create = create
        self._delete = delete

    async def mint(self, scope: str, ttl: int) -> tuple[str, str | None]:
        resp = self._create({"scope": scope, "ttl": ttl})
        result = resp["result"]
        return str(result["value"]), str(result["id"])

    async def revoke(self, handle: str | None) -> None:
        if handle is None:
            return
        self._delete(handle)

    def cred_to_env(self, cred: str) -> dict[str, str]:
        return cloudflare_cred_env(cred)


class StandingSecretRefused(RuntimeError):
    """Raised when a :class:`StaticEnvProvider` is built without explicit
    standing-secret consent (``standing_secret=True``)."""


class StaticEnvProvider:
    """A long-lived static API key (Slack / Discord / Resend) exposed as container env.

    The membrane prefers short-lived minted creds; some APIs only ever issue a
    static long-lived key. This provider carries one — but the **documented
    exception** to no-standing-secret is fail-closed: it refuses to construct
    unless the operator explicitly accepts the risk with ``standing_secret=True``,
    so a standing secret is never used by accident.

    The key is long-lived: there is nothing to mint or revoke, so ``mint`` ignores
    scope/ttl and returns the value, and ``revoke`` is a no-op. Pair it with a
    tight egress allowlist (see docs/operations.md) so a hijacked agent can only
    use the token against the intended API.
    """

    stateful = False  # the key is long-lived — nothing to revoke

    def __init__(self, *, value: str, env_var: str, standing_secret: bool = False) -> None:
        # Identity check, not truthiness: only the literal bool ``True`` opens the
        # gate. A non-bool (e.g. a quoted ``"false"`` from TOML, which is truthy)
        # must NOT bypass it — fail-closed against a malformed consent value.
        if standing_secret is not True:
            raise StandingSecretRefused(
                f"static_env carries a long-lived standing secret for {env_var!r}; set "
                "standing_secret = true (a bare TOML boolean) to explicitly accept the risk"
            )
        self._value = value
        self._env_var = env_var

    async def mint(self, scope: str, ttl: int) -> tuple[str, str | None]:
        # Static key: scope/ttl do not apply; return the value, nothing to revoke.
        return self._value, None

    async def revoke(self, handle: str | None) -> None:
        return None

    def cred_to_env(self, cred: str) -> dict[str, str]:
        return {self._env_var: cred}
