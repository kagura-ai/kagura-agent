"""v0.1 walking skeleton: transport -> intent -> Session, end to end.

This is the milestone's Definition of Done: a CLI thread can launch a brain,
get a result, and continue the same session with state carried across.
"""

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest

from kagura_agent.cockpit.core import Cockpit
from kagura_agent.cockpit.intent import Intent
from kagura_agent.cockpit.transports.base import Event
from kagura_agent.cockpit.transports.cli import CliTransport
from kagura_agent.core.brain.base import BrainCaps, BrainEvent, Checkpoint, DoneEvent, Task
from kagura_agent.mcp.memory_cloud import ALWAYS_DELIVERY, LocalMemoryClient
from kagura_agent.patterns.checkpoint import InMemoryCheckpointStore


class CountingBrain:
    """A brain whose loop increments a turn counter, honoring resume state."""

    caps = BrainCaps(name="counting")

    async def run(
        self, task: Task, *, resume: Checkpoint | None = None
    ) -> AsyncIterator[BrainEvent]:
        prior = resume.state.get("turn", 0) if resume else 0
        turn = prior + 1
        yield DoneEvent(result=f"turn {turn}: {task.prompt}", state={"turn": turn})


async def test_walking_skeleton_launch_then_continue() -> None:
    transport = CliTransport(
        inbox=[
            Event(thread_id="t1", text="start task", is_thread_reply=False),
            Event(thread_id="t1", text="keep going", is_thread_reply=True),
        ]
    )
    checkpoints = InMemoryCheckpointStore()
    cockpit = Cockpit(transport=transport, brain=CountingBrain(), checkpoints=checkpoints)

    await cockpit.serve()

    # both turns answered, on the same thread, with state advancing
    assert transport.sent == [
        ("t1", "turn 1: start task"),
        ("t1", "turn 2: keep going"),
    ]
    cp = await checkpoints.load("t1")
    assert cp is not None and cp.state == {"turn": 2}


async def test_reply_without_prior_session_launches_fresh() -> None:
    transport = CliTransport(inbox=[Event(thread_id="t5", text="huh?", is_thread_reply=True)])
    cockpit = Cockpit(
        transport=transport,
        brain=CountingBrain(),
        checkpoints=InMemoryCheckpointStore(),
    )

    await cockpit.serve()

    assert transport.sent == [("t5", "turn 1: huh?")]


async def test_cockpit_task_uses_bootstrap_grounding_and_remembers_outcome() -> None:
    transport = CliTransport(inbox=[Event(thread_id="t1", text="deploy", is_thread_reply=False)])
    checkpoints = InMemoryCheckpointStore()
    memory = LocalMemoryClient()
    await memory.remember(
        "Require operator approval",
        trust_tier="trusted",
        delivery_mode=ALWAYS_DELIVERY,
    )
    cockpit = Cockpit(
        transport=transport,
        brain=CountingBrain(),
        checkpoints=checkpoints,
        memory=memory,
    )

    await cockpit.serve()

    assert (
        "Standing guardrails (always apply):\n- Require operator approval" in (transport.sent[0][1])
    )
    summaries = await memory.recall("deploy", tags=("task-summary",))
    assert len(summaries) == 1 and summaries[0].trust_tier == "quarantine"
    checkpoint = await checkpoints.load("t1")
    assert checkpoint is not None and checkpoint.state == {"turn": 1}


async def test_cockpit_outcome_write_propagates_cancellation() -> None:
    cockpit = Cockpit(
        transport=CliTransport(),
        brain=CountingBrain(),
        checkpoints=InMemoryCheckpointStore(),
        memory=LocalMemoryClient(),
    )
    event = Event(thread_id="t1", text="deploy", is_thread_reply=False)

    with patch(
        "kagura_agent.cockpit.core.remember_outcome",
        new=AsyncMock(side_effect=asyncio.CancelledError()),
    ):
        with pytest.raises(asyncio.CancelledError):
            await cockpit._handle_task(event, Intent.LAUNCH)
