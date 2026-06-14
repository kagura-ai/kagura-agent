"""Regression tests for two cockpit control-surface bugs.

A) `/kill` was not covered by the operator-identity gate (#14 only gated
   approve/deny), so a non-operator could close a session / kill its container.
B) `_resolve_pending` resolved the approval future (which lets the consumer mint
   the capability) BEFORE writing the graduation-trail audit, so a failing audit
   write left a grant with no recorded evidence (fail-open on audit).
"""

from __future__ import annotations

import asyncio

import pytest

from kagura_agent.cockpit.approval import PendingApprovalRegistry
from kagura_agent.cockpit.core import Cockpit
from kagura_agent.cockpit.hitl import CapabilityRequest
from kagura_agent.cockpit.registry import SessionRegistry
from kagura_agent.cockpit.transports.base import Event


class _NoBrain:
    async def run(self, *_a, **_k):  # type: ignore[no-untyped-def]
        if False:
            yield None


class _RecordingTransport:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def listen(self):  # type: ignore[no-untyped-def]
        if False:
            yield None

    async def send(self, thread_id: str, text: str) -> None:
        self.sent.append((thread_id, text))

    async def ask(self, *_a: object) -> str:
        return ""


class _Killer:
    def __init__(self) -> None:
        self.killed: list[str] = []

    async def kill(self, container_id: str) -> None:
        self.killed.append(container_id)


class _Checkpoints:
    async def load(self, _sid: str):  # type: ignore[no-untyped-def]
        return None

    async def save(self, _sid: str, _cp: object) -> None:
        pass


def _cockpit(transport, *, registry=None, launcher=None, operator_id=None, memory=None):  # type: ignore[no-untyped-def]
    return Cockpit(
        transport,
        _NoBrain(),
        _Checkpoints(),
        registry=registry,
        launcher=launcher,
        operator_id=operator_id,
        memory=memory,
    )


# --- A: /kill operator-identity gate ----------------------------------------

async def test_kill_rejected_for_non_operator() -> None:
    reg = SessionRegistry()
    reg.add("t1", container_id="c1")
    killer = _Killer()
    t = _RecordingTransport()
    cockpit = _cockpit(t, registry=reg, launcher=killer, operator_id="op1")

    await cockpit.handle(
        Event(thread_id="t1", text="/kill", is_thread_reply=True, sender="attacker")
    )

    assert killer.killed == []  # container NOT killed
    assert reg.get("t1").status == "running"  # session NOT closed
    assert any("only the operator" in msg for _, msg in t.sent)


async def test_kill_allowed_for_operator() -> None:
    reg = SessionRegistry()
    reg.add("t1", container_id="c1")
    killer = _Killer()
    t = _RecordingTransport()
    cockpit = _cockpit(t, registry=reg, launcher=killer, operator_id="op1")

    await cockpit.handle(Event(thread_id="t1", text="/kill", is_thread_reply=True, sender="op1"))

    assert killer.killed == ["c1"]
    assert reg.get("t1").status == "closed"


async def test_kill_ungated_when_no_operator_configured() -> None:
    # Single-user CLI default (operator_id=None): no gate, backward-compatible.
    reg = SessionRegistry()
    reg.add("t1", container_id="c1")
    killer = _Killer()
    cockpit = _cockpit(_RecordingTransport(), registry=reg, launcher=killer, operator_id=None)

    await cockpit.handle(Event(thread_id="t1", text="/kill", is_thread_reply=True, sender=None))

    assert killer.killed == ["c1"]
    assert reg.get("t1").status == "closed"


# --- B: audit recorded before the grant becomes observable ------------------

class _FailingMemory:
    """A MemoryClient whose write fails — to prove the grant is gated on audit."""

    async def remember(self, *_a: object, **_k: object) -> str:
        raise RuntimeError("memory backend down")

    async def recall(self, *_a: object, **_k: object) -> list:  # type: ignore[type-arg]
        return []

    async def create_edge(self, *_a: object, **_k: object) -> None:
        pass


class _OkMemory:
    def __init__(self) -> None:
        self.written: list[str] = []

    async def remember(self, text: str, **_k: object) -> str:
        self.written.append(text)
        return "m1"

    async def recall(self, *_a: object, **_k: object) -> list:  # type: ignore[type-arg]
        return []

    async def create_edge(self, *_a: object, **_k: object) -> None:
        pass


async def test_failed_audit_does_not_resolve_the_grant() -> None:
    # The consumer awaits the future request_capability returns; if the audit
    # write fails, the future must NOT resolve approved (no grant without audit).
    t = _RecordingTransport()
    memory = _FailingMemory()
    cockpit = _cockpit(t, operator_id="op1", memory=memory)

    req = CapabilityRequest(thread_id="t1", capability="memory:write", reason="r")
    future = await cockpit.request_capability(req)

    with pytest.raises(RuntimeError):
        await cockpit.handle(
            Event(thread_id="t1", text="/approve", is_thread_reply=True, sender="op1")
        )

    # Grant did not happen: the consumer's future is still unresolved.
    assert not future.done()
    future.cancel()


async def test_successful_approve_records_then_grants() -> None:
    t = _RecordingTransport()
    memory = _OkMemory()
    cockpit = _cockpit(t, operator_id="op1", memory=memory)

    req = CapabilityRequest(thread_id="t1", capability="memory:write", reason="r")
    future = await cockpit.request_capability(req)
    await cockpit.handle(
        Event(thread_id="t1", text="/approve", is_thread_reply=True, sender="op1")
    )

    decision = await asyncio.wait_for(future, timeout=1)
    assert decision.approved is True
    assert any("memory:write" in w for w in memory.written)  # audit written


# --- claim(): pop without resolving (no false-audit on TTL expiry race) ------

async def test_claim_pops_entry_without_resolving_future() -> None:
    reg = PendingApprovalRegistry(clock=lambda: 0.0, ttl_seconds=300.0)
    req = CapabilityRequest(thread_id="t", capability="c", reason="r")
    future = reg.register(req)

    claimed = reg.claim("t")
    assert claimed is not None
    request, fut = claimed
    assert request is req
    assert fut is future
    assert not fut.done()  # claim must NOT resolve the future (caller does, post-audit)
    assert reg.pending("t") is False  # entry removed
    fut.cancel()


async def test_claim_on_expired_returns_none_and_denies() -> None:
    now = {"t": 0.0}
    reg = PendingApprovalRegistry(clock=lambda: now["t"], ttl_seconds=10.0)
    future = reg.register(CapabilityRequest(thread_id="t", capability="c", reason="r"))

    now["t"] = 100.0  # past TTL
    # Expired → no claim (so the caller writes NO audit), and the consumer's
    # future is failed-closed (denied) by the purge.
    assert reg.claim("t") is None
    assert future.done() and future.result().approved is False


async def test_claim_absent_returns_none() -> None:
    assert PendingApprovalRegistry().claim("nope") is None


# --- /kill: a failed container kill must not leave the session "running" -----

class _RaisingKiller:
    async def kill(self, _container_id: str) -> None:
        raise RuntimeError("docker kill failed")


async def test_kill_failure_closes_session_and_reports() -> None:
    reg = SessionRegistry()
    reg.add("t1", container_id="c1")
    t = _RecordingTransport()
    cockpit = _cockpit(t, registry=reg, launcher=_RaisingKiller(), operator_id="op1")

    # No exception escapes; session is closed (not left running) and the failure
    # is surfaced — regression guard for runtime.kill now raising on non-zero.
    await cockpit.handle(Event(thread_id="t1", text="/kill", is_thread_reply=True, sender="op1"))

    assert reg.get("t1").status == "closed"
    assert any("FAILED" in msg for _, msg in t.sent)
