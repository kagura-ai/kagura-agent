"""v0.3 #14 slice: only the trusted operator may resolve a pending approval.

#32 wired the async approval loop, but `_resolve_pending` resolved on ANY
`/approve` for the thread with no sender check — in a multi-party transport a
hijacked agent could post `/approve` and self-approve its own capability
request. This slice closes that: the cockpit can be configured with an
`operator_id`, and a pending request is resolved ONLY by an event whose
`sender` matches. A non-operator approve/deny is rejected (fail-closed): the
pending stays open for the real operator. When no `operator_id` is configured
(the single-user CLI default) behavior is unchanged (#32-compatible).
"""

from kagura_agent.cockpit.approval import PendingApprovalRegistry
from kagura_agent.cockpit.core import Cockpit
from kagura_agent.cockpit.hitl import CapabilityRequest
from kagura_agent.cockpit.transports.base import Event
from kagura_agent.cockpit.transports.cli import CliTransport
from kagura_agent.patterns.checkpoint import InMemoryCheckpointStore

_REQ = CapabilityRequest(thread_id="t1", capability="memory:write", reason="persist note")


class _FakeBrain:
    caps = None

    def run(self, task, *, resume=None):  # type: ignore[no-untyped-def]  # pragma: no cover
        raise NotImplementedError


def test_event_sender_defaults_to_none() -> None:
    # Additive field — existing constructions stay valid, sender is optional.
    assert Event(thread_id="t1", text="hi", is_thread_reply=False).sender is None
    assert Event(thread_id="t1", text="hi", is_thread_reply=False, sender="op1").sender == "op1"


async def test_non_operator_approve_does_not_resolve() -> None:
    transport = CliTransport(inbox=[])
    reg = PendingApprovalRegistry()
    cockpit = Cockpit(
        transport, _FakeBrain(), InMemoryCheckpointStore(), approvals=reg, operator_id="op1"
    )
    fut = await cockpit.request_capability(_REQ)

    # a hijacked agent (or anyone who is not the operator) posts /approve
    await cockpit.handle(
        Event(thread_id="t1", text="/approve", is_thread_reply=True, sender="agent")
    )

    assert not fut.done()           # NOT granted — fail-closed
    assert reg.pending("t1") is True  # stays open for the real operator
    assert any("operator" in text.lower() for _, text in transport.sent)  # rejection surfaced


async def test_operator_approve_resolves() -> None:
    transport = CliTransport(inbox=[])
    reg = PendingApprovalRegistry()
    cockpit = Cockpit(
        transport, _FakeBrain(), InMemoryCheckpointStore(), approvals=reg, operator_id="op1"
    )
    fut = await cockpit.request_capability(_REQ)

    await cockpit.handle(Event(thread_id="t1", text="/approve", is_thread_reply=True, sender="op1"))

    assert (await fut).approved is True
    assert reg.pending("t1") is False


async def test_no_operator_id_resolves_on_any_sender_backward_compat() -> None:
    # The single-user CLI default (operator_id=None) keeps #32 behavior: any
    # /approve resolves (no sender to check against).
    transport = CliTransport(inbox=[])
    reg = PendingApprovalRegistry()
    cockpit = Cockpit(transport, _FakeBrain(), InMemoryCheckpointStore(), approvals=reg)
    fut = await cockpit.request_capability(_REQ)

    await cockpit.handle(Event(thread_id="t1", text="/approve", is_thread_reply=True))

    assert (await fut).approved is True


async def test_non_operator_deny_is_also_ignored() -> None:
    transport = CliTransport(inbox=[])
    reg = PendingApprovalRegistry()
    cockpit = Cockpit(
        transport, _FakeBrain(), InMemoryCheckpointStore(), approvals=reg, operator_id="op1"
    )
    fut = await cockpit.request_capability(_REQ)

    await cockpit.handle(Event(thread_id="t1", text="/deny", is_thread_reply=True, sender="agent"))

    assert not fut.done()             # a non-operator cannot deny either
    assert reg.pending("t1") is True


async def test_operator_deny_resolves_denied() -> None:
    # The operator may also deny (not only approve) — completes the gate matrix.
    transport = CliTransport(inbox=[])
    reg = PendingApprovalRegistry()
    cockpit = Cockpit(
        transport, _FakeBrain(), InMemoryCheckpointStore(), approvals=reg, operator_id="op1"
    )
    fut = await cockpit.request_capability(_REQ)

    await cockpit.handle(Event(thread_id="t1", text="/deny", is_thread_reply=True, sender="op1"))

    assert (await fut).approved is False
    assert reg.pending("t1") is False


async def test_operator_gated_but_sender_none_is_fail_closed() -> None:
    # A CLI-style event (sender=None) against an operator-gated cockpit must NOT
    # resolve: None != operator_id → rejected, pending stays open. (The single-user
    # CLI sender-less path is only permissive when operator_id is itself None.)
    transport = CliTransport(inbox=[])
    reg = PendingApprovalRegistry()
    cockpit = Cockpit(
        transport, _FakeBrain(), InMemoryCheckpointStore(), approvals=reg, operator_id="op1"
    )
    fut = await cockpit.request_capability(_REQ)

    # a sender-less /approve (CLI-style) against an operator-gated cockpit
    await cockpit.handle(Event(thread_id="t1", text="/approve", is_thread_reply=True))

    assert not fut.done()             # fail-closed
    assert reg.pending("t1") is True  # stays open for the real operator
