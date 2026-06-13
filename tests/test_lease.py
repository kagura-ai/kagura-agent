"""v0.2: CredentialBroker / Lease.

Approval grants a time-boxed renewable *budget*, not a credential. acquire /
renew / release absorb both stateless STS (AWS/GCP/GitHub: mint, no revoke) and
stateful Cloudflare (mint → use → revoke). Checkpoints persist the budget and
release the live cred. An orphaned stateful cred is swept on restart.
"""

import pytest

from kagura_agent.membrane.lease import (
    Budget,
    BudgetExhausted,
    CredentialBroker,
    LeaseLedger,
)


class FakeStatelessProvider:
    """STS-style: mint short-lived creds, nothing to revoke."""

    stateful = False

    def __init__(self) -> None:
        self.minted = 0

    async def mint(self, scope: str, ttl: int) -> tuple[str, str | None]:
        self.minted += 1
        return f"sts-token-{self.minted}", None

    async def revoke(self, handle: str | None) -> None:  # pragma: no cover
        raise AssertionError("stateless provider must never be revoked")


class FakeStatefulProvider:
    """Cloudflare-style: mint a child token, must revoke it later."""

    stateful = True

    def __init__(self) -> None:
        self.minted = 0
        self.revoked: list[str] = []

    async def mint(self, scope: str, ttl: int) -> tuple[str, str | None]:
        self.minted += 1
        handle = f"cf-handle-{self.minted}"
        return f"cf-token-{self.minted}", handle

    async def revoke(self, handle: str | None) -> None:
        assert handle is not None
        self.revoked.append(handle)


def _clock() -> float:
    return 1000.0


async def test_acquire_grants_live_cred_and_budget() -> None:
    broker = CredentialBroker({"aws": FakeStatelessProvider()}, clock=_clock)
    lease = await broker.acquire("aws", scope="s3:read", ttl=300, budget=Budget(3600))

    assert lease.cred == "sts-token-1"
    assert lease.budget.remaining() == 3600
    assert lease.expires_at == 1300.0


async def test_release_revokes_stateful_but_not_stateless() -> None:
    stateless = FakeStatelessProvider()
    stateful = FakeStatefulProvider()
    broker = CredentialBroker({"aws": stateless, "cf": stateful}, clock=_clock)

    cf_lease = await broker.acquire("cf", scope="zone:edit", ttl=300, budget=Budget(3600))
    aws_lease = await broker.acquire("aws", scope="s3:read", ttl=300, budget=Budget(3600))

    await broker.release(cf_lease)
    await broker.release(aws_lease)

    assert stateful.revoked == ["cf-handle-1"]  # stateful revoked
    # stateless's revoke would have raised; reaching here proves it wasn't called


async def test_checkpoint_preserves_budget_and_drops_cred() -> None:
    broker = CredentialBroker({"aws": FakeStatelessProvider()}, clock=_clock)
    lease = await broker.acquire("aws", scope="s3:read", ttl=300, budget=Budget(3600))

    frozen = lease.for_checkpoint()

    assert frozen.cred is None          # no live cred in a checkpoint
    assert frozen.budget.remaining() == 3600  # budget survives


async def test_renew_spends_budget_and_remints() -> None:
    provider = FakeStatelessProvider()
    broker = CredentialBroker({"aws": provider}, clock=_clock)
    lease = await broker.acquire("aws", scope="s3:read", ttl=300, budget=Budget(500))

    renewed = await broker.renew(lease, ttl=300)

    assert renewed.cred == "sts-token-2"           # re-minted
    assert renewed.budget.remaining() == 200       # 500 - 300 spent
    assert provider.minted == 2


async def test_renew_past_budget_raises() -> None:
    broker = CredentialBroker({"aws": FakeStatelessProvider()}, clock=_clock)
    lease = await broker.acquire("aws", scope="s3:read", ttl=300, budget=Budget(200))

    with pytest.raises(BudgetExhausted):
        await broker.renew(lease, ttl=300)


async def test_ledger_sweeps_orphaned_stateful_creds() -> None:
    stateful = FakeStatefulProvider()
    broker = CredentialBroker({"cf": stateful}, clock=_clock, ledger=LeaseLedger())
    await broker.acquire("cf", scope="zone:edit", ttl=300, budget=Budget(3600))
    await broker.acquire("cf", scope="zone:edit", ttl=300, budget=Budget(3600))

    # crash/restart leaves two open CF leases; sweeper revokes them all
    await broker.sweep()

    assert sorted(stateful.revoked) == ["cf-handle-1", "cf-handle-2"]


# --- renew never loses track of a live token (#21) ---------------------------
# Invariant: at no point may a live stateful token exist that the ledger does
# not track. renew mints the replacement first (a mint failure must leave the
# current lease intact and usable — never revoke-before-mint), records the new
# token BEFORE revoking the old one (so a revoke failure can't orphan the fresh
# token), and keeps the old lease tracked if its revoke fails (sweep retries).


class FlakyMintProvider:
    """Stateful provider whose Nth mint raises (simulates rate-limit/network)."""

    stateful = True

    def __init__(self, fail_on_mint: int) -> None:
        self.minted = 0
        self.revoked: list[str] = []
        self._fail_on = fail_on_mint

    async def mint(self, scope: str, ttl: int) -> tuple[str, str | None]:
        self.minted += 1
        if self.minted == self._fail_on:
            raise RuntimeError("mint failed (rate limited)")
        return f"cf-token-{self.minted}", f"cf-handle-{self.minted}"

    async def revoke(self, handle: str | None) -> None:
        assert handle is not None
        self.revoked.append(handle)


class RevokeFailsProvider:
    """Stateful provider whose revoke always raises."""

    stateful = True

    def __init__(self) -> None:
        self.minted = 0
        self.revoke_attempts: list[str] = []

    async def mint(self, scope: str, ttl: int) -> tuple[str, str | None]:
        self.minted += 1
        return f"cf-token-{self.minted}", f"cf-handle-{self.minted}"

    async def revoke(self, handle: str | None) -> None:
        assert handle is not None
        self.revoke_attempts.append(handle)
        raise RuntimeError("revoke failed")


class SameHandleProvider:
    """Pathological provider that reissues the SAME handle every mint."""

    stateful = True

    def __init__(self) -> None:
        self.minted = 0
        self.revoked: list[str] = []

    async def mint(self, scope: str, ttl: int) -> tuple[str, str | None]:
        self.minted += 1
        return f"cf-token-{self.minted}", "cf-handle-fixed"

    async def revoke(self, handle: str | None) -> None:
        assert handle is not None
        self.revoked.append(handle)


async def test_renew_stateful_revokes_old_and_tracks_only_new() -> None:
    provider = FakeStatefulProvider()
    broker = CredentialBroker({"cf": provider}, clock=_clock, ledger=LeaseLedger())
    lease = await broker.acquire("cf", scope="zone:edit", ttl=300, budget=Budget(3600))

    renewed = await broker.renew(lease, ttl=300)

    assert renewed.cred == "cf-token-2"
    assert provider.revoked == ["cf-handle-1"]  # old revoked
    handles = [ln.handle for ln in broker.open_leases()]
    assert handles == ["cf-handle-2"]  # only the new token tracked


async def test_renew_mint_failure_keeps_old_lease_tracked_and_unrevoked() -> None:
    provider = FlakyMintProvider(fail_on_mint=2)  # acquire ok, renew's mint fails
    broker = CredentialBroker({"cf": provider}, clock=_clock, ledger=LeaseLedger())
    lease = await broker.acquire("cf", scope="zone:edit", ttl=300, budget=Budget(3600))

    with pytest.raises(RuntimeError, match="mint failed"):
        await broker.renew(lease, ttl=300)

    # Old token must remain live AND tracked; it was never revoked.
    handles = [ln.handle for ln in broker.open_leases()]
    assert handles == ["cf-handle-1"]
    assert provider.revoked == []
    # A failed renew must not consume budget: the caller's lease is unchanged
    # (Budget is immutable; spend() returns a new value that was discarded), so
    # a retry spends from the original budget — no double-spend.
    assert lease.budget.remaining() == 3600


async def test_renew_revoke_failure_tracks_new_and_keeps_old_for_sweep() -> None:
    provider = RevokeFailsProvider()
    broker = CredentialBroker({"cf": provider}, clock=_clock, ledger=LeaseLedger())
    lease = await broker.acquire("cf", scope="zone:edit", ttl=300, budget=Budget(3600))

    # renew succeeds (new cred is valid); the old token's revoke fails but is
    # swallowed — the renewal must not fail just because cleanup of the old
    # token did.
    renewed = await broker.renew(lease, ttl=300)

    assert renewed.cred == "cf-token-2"
    assert provider.revoke_attempts == ["cf-handle-1"]  # revoke was attempted
    # BOTH tokens tracked: the new one (never orphaned) and the old one (left
    # for the restart sweeper to retry).
    handles = sorted(ln.handle for ln in broker.open_leases())
    assert handles == ["cf-handle-1", "cf-handle-2"]


async def test_renew_same_handle_does_not_drop_or_revoke_the_live_token() -> None:
    # If the provider reissues the same handle, old and new are the same token
    # at the provider: revoking "the old" would kill the live one, and
    # forgetting it would drop the renewed ledger entry (same key). Do neither.
    provider = SameHandleProvider()
    broker = CredentialBroker({"cf": provider}, clock=_clock, ledger=LeaseLedger())
    lease = await broker.acquire("cf", scope="zone:edit", ttl=300, budget=Budget(3600))

    renewed = await broker.renew(lease, ttl=300)

    assert renewed.cred == "cf-token-2"
    assert provider.revoked == []  # must NOT revoke the shared handle
    handles = [ln.handle for ln in broker.open_leases()]
    assert handles == ["cf-handle-fixed"]  # exactly one tracked entry, not dropped
