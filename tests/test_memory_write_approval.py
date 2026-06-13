"""v0.3 #14 slice: gate memory:write behind operator HITL approval.

`MemoryWriteApprover` consumes the cockpit approval loop (#32): it asks the
operator (via `request_capability`), awaits the decision with a hard timeout
(`asyncio.wait_for` — the lazy-expiry fail-closed bound), and only on approval
runs `grant`, which the caller wires to a broker.acquire using a write_approved
`MemoryCloudProvider` (the write_approved flip happens *post-approval*). A
denied or timed-out request grants nothing — fail-closed.
"""

import asyncio

from kagura_agent.cockpit.approval import PendingApprovalRegistry
from kagura_agent.cockpit.core import Cockpit
from kagura_agent.cockpit.memory_write import MemoryWriteApprover
from kagura_agent.cockpit.transports.cli import CliTransport
from kagura_agent.membrane.lease import Budget, CredentialBroker
from kagura_agent.membrane.providers import MemoryCloudProvider
from kagura_agent.patterns.checkpoint import InMemoryCheckpointStore


class _FakeBrain:
    caps = None

    def run(self, task, *, resume=None):  # type: ignore[no-untyped-def]  # pragma: no cover
        raise NotImplementedError


def _cockpit(transport, reg):  # type: ignore[no-untyped-def]
    return Cockpit(transport, _FakeBrain(), InMemoryCheckpointStore(), approvals=reg)


async def _drive_until_pending(reg, thread_id):  # type: ignore[no-untyped-def]
    for _ in range(100):
        await asyncio.sleep(0)
        if reg.pending(thread_id):
            return
    raise AssertionError("request never became pending")


async def test_approved_runs_grant_and_returns_its_result() -> None:
    reg = PendingApprovalRegistry()
    cockpit = _cockpit(CliTransport(inbox=[]), reg)
    calls = []
    sentinel = object()

    async def grant():  # type: ignore[no-untyped-def]
        calls.append(1)
        return sentinel

    approver = MemoryWriteApprover(cockpit, grant, timeout=5)
    task = asyncio.create_task(approver.request("t1", "persist note"))
    await _drive_until_pending(reg, "t1")
    reg.resolve("t1", approved=True)  # operator approves

    assert await task is sentinel
    assert calls == [1]  # grant ran exactly once, only after approval


async def test_denied_does_not_grant() -> None:
    reg = PendingApprovalRegistry()
    cockpit = _cockpit(CliTransport(inbox=[]), reg)
    calls = []

    async def grant():  # type: ignore[no-untyped-def]
        calls.append(1)  # pragma: no cover - must NOT run
        return object()

    approver = MemoryWriteApprover(cockpit, grant, timeout=5)
    task = asyncio.create_task(approver.request("t1", "persist note"))
    await _drive_until_pending(reg, "t1")
    reg.resolve("t1", approved=False)  # operator denies

    assert await task is None  # fail-closed
    assert calls == []         # grant never ran


async def test_timeout_does_not_grant() -> None:
    reg = PendingApprovalRegistry()
    cockpit = _cockpit(CliTransport(inbox=[]), reg)
    calls = []

    async def grant():  # type: ignore[no-untyped-def]
        calls.append(1)  # pragma: no cover - must NOT run
        return object()

    approver = MemoryWriteApprover(cockpit, grant, timeout=0.01)
    result = await approver.request("t1", "persist note")  # nobody resolves → times out

    assert result is None  # fail-closed on no operator decision
    assert calls == []


async def test_approved_acquires_memory_write_lease_through_the_broker() -> None:
    # End-to-end: on approval, grant mints a memory:write lease THROUGH the broker
    # using a write_approved MemoryCloudProvider (the broker's _assert_scope_allowed
    # gate passes only because the provider is write_approved — the post-approval flip).
    reg = PendingApprovalRegistry()
    cockpit = _cockpit(CliTransport(inbox=[]), reg)
    provider = MemoryCloudProvider(
        exchange=lambda req: {"access_token": "kmc-write-1"}, write_approved=True
    )
    broker = CredentialBroker({"memory": provider}, clock=lambda: 1000.0)

    async def grant():  # type: ignore[no-untyped-def]
        return await broker.acquire("memory", scope="memory:write", ttl=300, budget=Budget(3600))

    approver = MemoryWriteApprover(cockpit, grant, timeout=5)
    task = asyncio.create_task(approver.request("t1", "persist note"))
    await _drive_until_pending(reg, "t1")
    reg.resolve("t1", approved=True)

    lease = await task
    assert lease is not None
    assert lease.scope == "memory:write"
    assert lease.cred == "kmc-write-1"
