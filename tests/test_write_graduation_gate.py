"""v0.4 #15 Slice 2: connect graduation → HITL proposal → host promote.

`WriteGraduationGate` ties the existing `GraduationEngine` (capability graduation
by category — verified successes across distinct tasks, input-trust gate,
fail-closed demotion) to the quarantine→trusted write promotion built in Slice 1.
It is fail-closed at every step and there is NO auto-promotion:

- not-yet-graduated (too few verified successes / untrusted input / cooldown) →
  no HITL proposal is even posted, nothing is promoted.
- graduated → an HITL proposal is surfaced via the #32/#14 approval loop; only an
  operator approve promotes the quarantined memories host-side. Deny/timeout →
  nothing promoted (timeout also withdraws the pending, mirroring #14's fix).
"""

import asyncio

from kagura_agent.cockpit.approval import PendingApprovalRegistry
from kagura_agent.cockpit.core import Cockpit
from kagura_agent.cockpit.memory_write import WriteGraduationGate
from kagura_agent.cockpit.transports.cli import CliTransport
from kagura_agent.mcp.memory_cloud import LocalMemoryClient, QuarantinedMemoryClient
from kagura_agent.membrane.graduation import GraduationEngine, GraduationPolicy
from kagura_agent.patterns.checkpoint import InMemoryCheckpointStore

_CATEGORY = "memctx"


class _FakeBrain:
    caps = None

    def run(self, task, *, resume=None):  # type: ignore[no-untyped-def]  # pragma: no cover
        raise NotImplementedError


def _cockpit(reg):  # type: ignore[no-untyped-def]
    return Cockpit(CliTransport(inbox=[]), _FakeBrain(), InMemoryCheckpointStore(), approvals=reg)


def _eligible_engine():  # type: ignore[no-untyped-def]
    # 5 verified successes across 3 distinct tasks → meets the default policy.
    engine = GraduationEngine(GraduationPolicy(), clock=lambda: 1000.0)
    for i in range(5):
        engine.record_success(_CATEGORY, task_id=f"t{i % 3}", verified=True)
    return engine


async def _drive_until_pending(reg, thread_id):  # type: ignore[no-untyped-def]
    for _ in range(100):
        await asyncio.sleep(0)
        if reg.pending(thread_id):
            return
    raise AssertionError("request never became pending")


async def test_not_graduated_makes_no_proposal_and_promotes_nothing() -> None:
    engine = GraduationEngine(GraduationPolicy(), clock=lambda: 1000.0)  # zero successes
    reg = PendingApprovalRegistry()
    promoted: list[str] = []
    gate = WriteGraduationGate(engine, _cockpit(reg), promote=lambda mid: promoted.append(mid))

    result = await gate.propose_promotion(
        _CATEGORY, ["m1"], thread_id="t1", input_trust="trusted", reason="r"
    )

    assert result is None
    assert promoted == []
    assert reg.pending("t1") is False  # no HITL proposal was ever posted


async def test_untrusted_input_blocks_proposal_even_when_thresholds_met() -> None:
    reg = PendingApprovalRegistry()
    promoted: list[str] = []
    gate = WriteGraduationGate(
        _eligible_engine(), _cockpit(reg), promote=lambda mid: promoted.append(mid)
    )

    result = await gate.propose_promotion(
        _CATEGORY, ["m1"], thread_id="t1", input_trust="untrusted", reason="r"
    )

    assert result is None
    assert promoted == []
    assert reg.pending("t1") is False


async def test_failure_demotion_blocks_proposal() -> None:
    engine = _eligible_engine()
    engine.record_failure(_CATEGORY)  # wipes progress + demotes for the cooldown window
    reg = PendingApprovalRegistry()
    promoted: list[str] = []
    gate = WriteGraduationGate(engine, _cockpit(reg), promote=lambda mid: promoted.append(mid))

    result = await gate.propose_promotion(
        _CATEGORY, ["m1"], thread_id="t1", input_trust="trusted", reason="r"
    )

    assert result is None
    assert promoted == []


async def test_graduated_and_approved_promotes_quarantined_into_trusted() -> None:
    backend = LocalMemoryClient()
    agent = QuarantinedMemoryClient(backend)
    mid = await agent.remember("learned fact")  # lands quarantined (Slice 1)
    reg = PendingApprovalRegistry()
    gate = WriteGraduationGate(_eligible_engine(), _cockpit(reg), promote=backend.promote)

    task = asyncio.create_task(
        gate.propose_promotion(
            _CATEGORY, [mid], thread_id="t1", input_trust="trusted", reason="5 verified successes"
        )
    )
    await _drive_until_pending(reg, "t1")
    reg.resolve("t1", approved=True)  # operator grants the proposal

    assert await task == [mid]
    trusted = await agent.recall("learned", trusted_only=True)
    assert [m.id for m in trusted] == [mid]  # promoted into the trusted backbone


async def test_graduated_but_denied_promotes_nothing() -> None:
    backend = LocalMemoryClient()
    agent = QuarantinedMemoryClient(backend)
    mid = await agent.remember("note")
    reg = PendingApprovalRegistry()
    gate = WriteGraduationGate(_eligible_engine(), _cockpit(reg), promote=backend.promote)

    task = asyncio.create_task(
        gate.propose_promotion(_CATEGORY, [mid], thread_id="t1", input_trust="trusted", reason="r")
    )
    await _drive_until_pending(reg, "t1")
    reg.resolve("t1", approved=False)  # operator denies

    assert await task is None
    assert await agent.recall("note", trusted_only=True) == []  # stays quarantined


async def test_graduated_timeout_withdraws_and_promotes_nothing() -> None:
    backend = LocalMemoryClient()
    agent = QuarantinedMemoryClient(backend)
    mid = await agent.remember("note")
    reg = PendingApprovalRegistry()
    gate = WriteGraduationGate(
        _eligible_engine(), _cockpit(reg), promote=backend.promote, timeout=0.01
    )

    result = await gate.propose_promotion(
        _CATEGORY, [mid], thread_id="t1", input_trust="trusted", reason="r"
    )

    assert result is None
    assert reg.pending("t1") is False  # withdrawn — no orphan
    assert await agent.recall("note", trusted_only=True) == []  # stays quarantined
