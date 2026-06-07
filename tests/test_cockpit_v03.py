"""v0.3: HITL approval, session registry + reconcile, status/kill intents.

The cockpit is the human's control surface: it asks before granting powers
beyond baseline (timeout = deny), records every decision to memory as a
graduation trail, and reconciles its in-memory session table against the live
containers on restart.
"""

from kagura_agent.cockpit.hitl import CapabilityRequest, HitlGate
from kagura_agent.cockpit.intent import Intent, classify
from kagura_agent.cockpit.registry import SessionRegistry
from kagura_agent.cockpit.transports.base import Event
from kagura_agent.cockpit.transports.cli import CliTransport
from kagura_agent.mcp.memory_cloud import LocalMemoryClient

# --- HITL approval --------------------------------------------------------

async def test_hitl_approve_records_decision_and_grants() -> None:
    transport = CliTransport(inbox=[], answers=["approve"])
    memory = LocalMemoryClient()
    gate = HitlGate(transport=transport, memory=memory)

    decision = await gate.review(
        CapabilityRequest(thread_id="t1", capability="aws:s3:write", reason="upload build")
    )

    assert decision.approved is True
    trail = await memory.recall("aws:s3:write", tags=("graduation-trail",))
    assert trail and "approved" in trail[0].text


async def test_hitl_deny_blocks_and_records() -> None:
    transport = CliTransport(inbox=[], answers=["deny"])
    memory = LocalMemoryClient()
    gate = HitlGate(transport=transport, memory=memory)

    decision = await gate.review(
        CapabilityRequest(thread_id="t1", capability="cf:zone:edit", reason="dns change")
    )

    assert decision.approved is False
    trail = await memory.recall("cf:zone:edit", tags=("graduation-trail",))
    assert trail and "denied" in trail[0].text


async def test_hitl_unknown_answer_is_treated_as_deny() -> None:
    # timeout / garbage answer must fail closed
    transport = CliTransport(inbox=[], answers=["maybe later"])
    gate = HitlGate(transport=transport, memory=LocalMemoryClient())

    decision = await gate.review(
        CapabilityRequest(thread_id="t1", capability="aws:s3:write", reason="x")
    )

    assert decision.approved is False


# --- registry + reconcile -------------------------------------------------

def test_registry_tracks_container_and_status() -> None:
    reg = SessionRegistry()
    reg.add("t1", container_id="c1", image="kagura-agent:python")

    rec = reg.get("t1")
    assert rec is not None
    assert rec.container_id == "c1"
    assert rec.status == "running"


def test_reconcile_marks_missing_containers_dead() -> None:
    reg = SessionRegistry()
    reg.add("t1", container_id="c1")
    reg.add("t2", container_id="c2")

    reg.reconcile(live_container_ids={"c1"})  # c2 vanished while cockpit was down

    assert reg.get("t1").status == "running"
    assert reg.get("t2").status == "dead"


# --- status / kill intents ------------------------------------------------

def test_slash_status_is_status_intent() -> None:
    event = Event(thread_id="t1", text="/status", is_thread_reply=True)
    assert classify(event, known_sessions={"t1"}) is Intent.STATUS


def test_slash_kill_is_kill_intent() -> None:
    event = Event(thread_id="t1", text="/kill", is_thread_reply=True)
    assert classify(event, known_sessions={"t1"}) is Intent.KILL


def test_plain_reply_still_continues() -> None:
    event = Event(thread_id="t1", text="keep going", is_thread_reply=True)
    assert classify(event, known_sessions={"t1"}) is Intent.CONTINUE
