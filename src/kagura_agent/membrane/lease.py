"""CredentialBroker / Lease — the launcher's credential interface.

The core idea (CSO/CTO H2): approval is a **time-boxed renewable budget**, not a
credential. This resolves the contradiction between long-running tasks and
short-lived creds — the budget outlives any single cred and is what a checkpoint
persists, while the live cred is released at every checkpoint.

The broker absorbs two provider shapes behind one interface:
- **stateless** (AWS STS, GCP SA impersonation, GitHub App token): mint and
  forget; expiry handles cleanup.
- **stateful** (Cloudflare Tokens API): mint → use → **revoke**; orphans must be
  swept (a `LeaseLedger` tracks open leases for the restart sweeper).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Protocol

from kagura_agent.membrane.cred_env import EnvCredProvider
from kagura_agent.membrane.providers import MemoryCloudProvider, MemoryWriteLocked

log = logging.getLogger(__name__)


class BudgetExhausted(RuntimeError):
    """A renewal would exceed the approved budget."""


@dataclass(frozen=True)
class Budget:
    total_seconds: int
    spent_seconds: int = 0

    def remaining(self) -> int:
        return max(0, self.total_seconds - self.spent_seconds)

    def spend(self, seconds: int) -> Budget:
        if seconds > self.remaining():
            raise BudgetExhausted(
                f"renewal needs {seconds}s but only {self.remaining()}s of budget remain"
            )
        return replace(self, spent_seconds=self.spent_seconds + seconds)


@dataclass(frozen=True)
class Lease:
    provider: str
    scope: str
    budget: Budget
    cred: str | None
    expires_at: float
    handle: str | None  # stateful revoke handle (None for stateless)
    stateful: bool

    def for_checkpoint(self) -> Lease:
        """A checkpoint-safe copy: budget preserved, live cred dropped."""
        return replace(self, cred=None)


class CredProvider(Protocol):
    stateful: bool

    async def mint(self, scope: str, ttl: int) -> tuple[str, str | None]: ...

    async def revoke(self, handle: str | None) -> None: ...


class LeaseLedger:
    """Tracks open leases so orphaned stateful creds can be swept on restart."""

    def __init__(self) -> None:
        self._open: dict[str, Lease] = {}

    def _key(self, lease: Lease) -> str:
        return f"{lease.provider}:{lease.handle}"

    def record(self, lease: Lease) -> None:
        if lease.stateful and lease.handle is not None:
            self._open[self._key(lease)] = lease

    def forget(self, lease: Lease) -> None:
        self._open.pop(self._key(lease), None)

    def open_leases(self) -> list[Lease]:
        return list(self._open.values())


class CredentialBroker:
    def __init__(
        self,
        providers: Mapping[str, CredProvider],
        *,
        clock: Callable[[], float],
        ledger: LeaseLedger | None = None,
    ) -> None:
        self._providers = dict(providers)
        self._clock = clock
        self._ledger = ledger or LeaseLedger()

    def _assert_scope_allowed(self, p: CredProvider, scope: str) -> None:
        """Broker-level write-lock (defense-in-depth) for the highest-risk scope.

        ``memory:write`` (and any future privileged ``memory:*`` scope) is the
        dominant prompt-injection persistence risk, so the broker itself refuses
        to mint it unless the resolved provider is a recognized, write-approved
        ``MemoryCloudProvider``. The broker does NOT trust an arbitrary provider
        to self-enforce: a look-alike wired under the "memory" name, or one
        lacking the check, is rejected HERE — not just inside ``mint``. This is an
        intentional cross-layer reference (broker -> concrete provider): the
        knowledge "who may mint a write token" must live in the broker, because a
        capability check delegated to the provider can be subverted by the very
        look-alike this guards against. The broker is the single chokepoint;
        acquiring creds outside it is outside the membrane's contract.
        """
        if MemoryCloudProvider.requires_write_approval(scope) and not (
            isinstance(p, MemoryCloudProvider) and p.write_approved
        ):
            raise MemoryWriteLocked(
                f"broker refuses scope {scope!r}: it requires a write-approved "
                "MemoryCloudProvider (a read-locked or look-alike provider cannot "
                "mint a privileged memory token, even via broker.acquire/renew)."
            )

    def open_leases(self) -> list[Lease]:
        """Stateful leases the ledger is currently tracking (for the sweeper and
        for observability — exposed so callers need not reach into the ledger)."""
        return self._ledger.open_leases()

    def container_env(self, leases: Iterable[Lease]) -> dict[str, str]:
        """Translate live leases into the env the launcher injects (#39).

        Each lease's cred is mapped to env vars by its provider's
        ``cred_to_env`` (only providers implementing ``EnvCredProvider`` — memory
        is reached by the CLI, not a generic env var, so it is skipped). The
        result feeds ``LaunchSpec.env`` -> ``docker_run_args`` -> ``-e KEY=VALUE``.

        Fail-closed on ambiguity, not silently:
        - an unknown provider name raises ``KeyError`` (the lease was minted
          outside this broker's registry — never wave it through);
        - two leases setting the *same* env var to *different* values raise
          ``ValueError`` (a silent last-writer-wins would hand the container a
          half-overwritten cred);
        - a checkpoint-safe lease (``cred is None``) contributes nothing.
        """
        env: dict[str, str] = {}
        for lease in leases:
            if lease.cred is None:
                continue
            provider = self._providers[lease.provider]  # KeyError if unknown — fail-closed
            if not isinstance(provider, EnvCredProvider):
                continue  # e.g. memory: no generic container env mapping
            for key, value in provider.cred_to_env(lease.cred).items():
                if key in env and env[key] != value:
                    raise ValueError(
                        f"container env conflict on {key!r}: two leases set it to "
                        "different values (refusing a half-overwritten cred)"
                    )
                env[key] = value
        return env

    async def acquire(self, provider: str, *, scope: str, ttl: int, budget: Budget) -> Lease:
        p = self._providers[provider]
        self._assert_scope_allowed(p, scope)
        cred, handle = await p.mint(scope, ttl)
        lease = Lease(
            provider=provider,
            scope=scope,
            budget=budget,
            cred=cred,
            expires_at=self._clock() + ttl,
            handle=handle,
            stateful=p.stateful,
        )
        self._ledger.record(lease)
        return lease

    async def renew(self, lease: Lease, *, ttl: int) -> Lease:
        budget = lease.budget.spend(ttl)  # raises BudgetExhausted
        p = self._providers[lease.provider]
        self._assert_scope_allowed(p, lease.scope)  # same broker-level write-lock as acquire
        # Mint the replacement FIRST. If mint raises (network/rate-limit), the
        # current lease is left fully intact — still valid and still tracked in
        # the ledger (it is never forgotten below until a successful revoke), so
        # the caller keeps a working cred and nothing is orphaned. Never
        # revoke-before-mint: that would destroy a working cred on mint failure.
        cred, handle = await p.mint(lease.scope, ttl)
        renewed = replace(
            lease,
            budget=budget,
            cred=cred,
            expires_at=self._clock() + ttl,
            handle=handle,
        )
        # Record the new token BEFORE touching the old one: a revoke failure
        # below must never leave the freshly-minted (live) token untracked.
        self._ledger.record(renewed)
        # If the provider reissued the SAME handle, old and new are the same
        # token at the provider — revoking "the old" would kill the live one,
        # and forgetting it would drop the renewed entry (same ledger key). Skip.
        if lease.handle == handle:
            return renewed
        # Symmetric with release(): a stateful old token must be revoked, or it
        # leaks (still valid at the provider). If revoke fails, keep the old
        # lease tracked so the restart sweeper retries it — do not orphan it.
        if lease.stateful and lease.handle is not None:
            try:
                await p.revoke(lease.handle)
            except Exception:
                log.exception(
                    "renew: revoke of old lease %s:%s failed; left tracked for sweep",
                    lease.provider,
                    lease.handle,
                )
                return renewed
        self._ledger.forget(lease)
        return renewed

    async def release(self, lease: Lease) -> None:
        if lease.stateful:
            await self._providers[lease.provider].revoke(lease.handle)
        self._ledger.forget(lease)

    async def sweep(self) -> None:
        """Revoke every still-open stateful lease (orphan cleanup on restart).

        Resilient per lease: a single failing revoke (e.g. a handle revoked
        out-of-band -> 404) must not abort the sweep or wedge the ledger on the
        same poison handle. Log, forget, and continue.
        """
        for lease in self._ledger.open_leases():
            try:
                await self.release(lease)
            except Exception:
                log.exception("sweep failed to revoke lease %s:%s", lease.provider, lease.handle)
                self._ledger.forget(lease)  # unwedge: do not re-hit this handle
