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
