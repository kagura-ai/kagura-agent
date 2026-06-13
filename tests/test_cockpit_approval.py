"""v0.3 cockpit foundation: the async HITL approval loop (#32).

`PendingApprovalRegistry` holds a capability request that is awaiting an
operator decision, so a *later* `intent=approve`/`deny` event can resolve it.
This is the producer-seam primitive #14 (memory:write device-flow) and #15
(graduation) consume. It is fail-closed: an expired pending denies.

Pure (injected clock, no transport) so it is unit-testable in isolation —
mirroring the `lease.py` injected-clock pattern.
"""

import pytest

from kagura_agent.cockpit.approval import PendingApprovalExists, PendingApprovalRegistry
from kagura_agent.cockpit.core import Cockpit
from kagura_agent.cockpit.hitl import CapabilityRequest
from kagura_agent.cockpit.intent import Intent, classify
from kagura_agent.cockpit.transports.base import Event
from kagura_agent.cockpit.transports.cli import CliTransport
from kagura_agent.mcp.memory_cloud import LocalMemoryClient
from kagura_agent.patterns.checkpoint import InMemoryCheckpointStore

_REQ = CapabilityRequest(thread_id="t1", capability="memory:write", reason="persist note")


async def test_register_creates_a_pending_request() -> None:
    reg = PendingApprovalRegistry()
    fut = reg.register(_REQ)

    assert reg.pending("t1") is True
    assert not fut.done()  # awaits the operator's later decision


async def test_resolve_approve_sets_future_and_clears_pending() -> None:
    reg = PendingApprovalRegistry()
    fut = reg.register(_REQ)

    returned = reg.resolve("t1", approved=True)

    assert returned == _REQ  # the resolved request is handed back to the caller
    assert (await fut).approved is True
    assert reg.pending("t1") is False  # one-shot: cleared after resolution


async def test_resolve_deny_sets_a_denied_decision() -> None:
    reg = PendingApprovalRegistry()
    fut = reg.register(_REQ)

    reg.resolve("t1", approved=False)

    assert (await fut).approved is False


async def test_resolve_with_no_pending_returns_none() -> None:
    reg = PendingApprovalRegistry()
    assert reg.resolve("ghost", approved=True) is None


async def test_second_register_on_same_thread_is_rejected() -> None:
    # v0.3: one pending per thread. A second request must NOT silently supersede
    # a request the operator may be about to approve.
    reg = PendingApprovalRegistry()
    reg.register(_REQ)

    with pytest.raises(PendingApprovalExists):
        reg.register(CapabilityRequest(thread_id="t1", capability="other", reason="y"))


async def test_expired_pending_fails_closed() -> None:
    # An approval that arrives after the timeout is too late — fail closed (deny),
    # and grant nothing. Lazy expiry: observed on the next resolve/pending check.
    now = [100.0]
    reg = PendingApprovalRegistry(clock=lambda: now[0], ttl_seconds=300)
    fut = reg.register(_REQ)

    now[0] += 301  # past the ttl

    assert reg.resolve("t1", approved=True) is None  # expired → nothing to grant
    assert (await fut).approved is False              # fail-closed: expired = denied
    assert reg.pending("t1") is False                 # purged


# --- cockpit wiring: producer seam + intent=approve/deny resolution -----------


class _FakeBrain:
    caps = None

    def run(self, task, *, resume=None):  # type: ignore[no-untyped-def]  # pragma: no cover
        raise NotImplementedError("approval tests never drive the brain")


def _cockpit(transport, *, memory=None):  # type: ignore[no-untyped-def]
    return Cockpit(transport, _FakeBrain(), InMemoryCheckpointStore(), memory=memory)


async def test_slash_deny_is_deny_intent() -> None:
    event = Event(thread_id="t1", text="/deny", is_thread_reply=True)
    assert classify(event, known_sessions=set()) is Intent.DENY


async def test_request_capability_posts_request_and_registers_pending() -> None:
    transport = CliTransport(inbox=[])
    cockpit = _cockpit(transport)

    fut = await cockpit.request_capability(_REQ)

    assert not fut.done()  # non-blocking: resolves on a later approve event
    assert transport.sent and transport.sent[-1][0] == "t1"
    assert "memory:write" in transport.sent[-1][1]


async def test_approve_event_resolves_pending_grants_and_records() -> None:
    transport = CliTransport(inbox=[])
    memory = LocalMemoryClient()
    cockpit = _cockpit(transport, memory=memory)

    fut = await cockpit.request_capability(_REQ)
    await cockpit.handle(Event(thread_id="t1", text="/approve", is_thread_reply=True))

    assert (await fut).approved is True
    trail = await memory.recall("memory:write", tags=("graduation-trail",))
    assert trail and "approved" in trail[0].text
    assert any("approved" in text for _, text in transport.sent)


async def test_deny_event_resolves_denied_and_records() -> None:
    transport = CliTransport(inbox=[])
    memory = LocalMemoryClient()
    cockpit = _cockpit(transport, memory=memory)

    fut = await cockpit.request_capability(_REQ)
    await cockpit.handle(Event(thread_id="t1", text="/deny", is_thread_reply=True))

    assert (await fut).approved is False
    trail = await memory.recall("memory:write", tags=("graduation-trail",))
    assert trail and "denied" in trail[0].text


async def test_approve_with_no_pending_preserves_legacy_reply() -> None:
    transport = CliTransport(inbox=[])
    cockpit = _cockpit(transport)

    await cockpit.handle(Event(thread_id="t1", text="/approve", is_thread_reply=True))

    assert transport.sent[-1] == ("t1", "no pending approval")
