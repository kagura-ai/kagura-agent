"""v0.1: Session orchestrates a BrainProvider through tasks + checkpoints.

The Iron Law of the seam: Session depends only on the BrainProvider protocol;
the provider *owns its agentic loop*. Session orchestrates tasks and
checkpoints, never individual tool calls.
"""

from collections.abc import AsyncIterator

from kagura_agent.core.brain.base import (
    BrainCaps,
    BrainEvent,
    Checkpoint,
    DoneEvent,
    MessageEvent,
    Task,
)
from kagura_agent.core.session import Session
from kagura_agent.patterns.checkpoint import InMemoryCheckpointStore


class FakeBrain:
    """A brain that owns a trivial loop: one message, then done."""

    caps = BrainCaps(name="fake")

    def __init__(self) -> None:
        self.resumed_from: Checkpoint | None = None

    async def run(
        self, task: Task, *, resume: Checkpoint | None = None
    ) -> AsyncIterator[BrainEvent]:
        self.resumed_from = resume
        prior = resume.state.get("turn", 0) if resume else 0
        yield MessageEvent(text="thinking")
        yield DoneEvent(result=f"done: {task.prompt}", state={"turn": prior + 1})


async def test_session_run_returns_result_and_persists_checkpoint() -> None:
    store = InMemoryCheckpointStore()
    session = Session(brain=FakeBrain(), checkpoints=store)

    result = await session.run(Task(prompt="hello", session_id="s1"))

    assert result.text == "done: hello"
    cp = await store.load("s1")
    assert cp is not None
    assert cp.session_id == "s1"
    assert cp.state == {"turn": 1}


class _MisbehavingBrain:
    """Yields events AFTER its terminal DoneEvent — must be ignored."""

    caps = BrainCaps(name="bad")

    async def run(
        self, task: Task, *, resume: Checkpoint | None = None
    ) -> AsyncIterator[BrainEvent]:
        yield MessageEvent(text="working")
        yield DoneEvent(result="first", state={"turn": 1})
        yield MessageEvent(text="LATE")  # post-terminal narration
        yield DoneEvent(result="second", state={"turn": 99})  # would corrupt result


async def test_session_treats_first_done_as_terminal() -> None:
    store = InMemoryCheckpointStore()
    session = Session(brain=_MisbehavingBrain(), checkpoints=store)

    result = await session.run(Task(prompt="x", session_id="s1"))

    assert result.text == "first"  # not "second"
    assert result.messages == ["working"]  # "LATE" not recorded
    cp = await store.load("s1")
    assert cp is not None
    assert cp.state == {"turn": 1}  # not 99


async def test_session_resume_feeds_prior_checkpoint_to_brain() -> None:
    store = InMemoryCheckpointStore()
    brain = FakeBrain()
    session = Session(brain=brain, checkpoints=store)

    await session.run(Task(prompt="first", session_id="s1"))
    result = await session.resume("s1", prompt="second")

    assert result.text == "done: second"
    # the brain saw the checkpoint from the first run...
    assert brain.resumed_from is not None
    assert brain.resumed_from.state == {"turn": 1}
    # ...and the new checkpoint advanced the turn counter
    cp = await store.load("s1")
    assert cp is not None
    assert cp.state == {"turn": 2}
