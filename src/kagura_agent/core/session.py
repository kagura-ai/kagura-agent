"""The orchestration loop — depends on `BrainProvider`, never the SDK directly.

`Session` drives a brain through a task, collects narration, and persists a
checkpoint when the brain reports `DoneEvent`. It never inspects provider state
or drives individual tool calls — the brain owns its loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kagura_agent.core.brain.base import (
    BrainProvider,
    Checkpoint,
    DoneEvent,
    MessageEvent,
    Task,
)
from kagura_agent.patterns.checkpoint import CheckpointStore


@dataclass(frozen=True)
class SessionResult:
    session_id: str
    text: str
    checkpoint: Checkpoint
    messages: list[str] = field(default_factory=list)


class SessionError(RuntimeError):
    """Raised when a brain ends without a terminal DoneEvent."""


class Session:
    def __init__(self, brain: BrainProvider, checkpoints: CheckpointStore) -> None:
        self._brain = brain
        self._checkpoints = checkpoints

    async def run(self, task: Task) -> SessionResult:
        return await self._drive(task, resume=None)

    async def drive(self, task: Task, *, resume: Checkpoint | None) -> SessionResult:
        """Run ``task``, resuming from an already-loaded ``resume`` checkpoint when
        given (else a fresh launch). The public seam for callers that have already
        loaded the checkpoint (e.g. ``drive_task``) — it avoids the second store
        read ``resume(session_id, ...)`` would do, and the race that re-load opens.
        """
        return await self._drive(task, resume=resume)

    async def resume(self, session_id: str, prompt: str) -> SessionResult:
        prior = await self._checkpoints.load(session_id)
        if prior is None:
            raise SessionError(f"no checkpoint to resume for session {session_id!r}")
        return await self._drive(Task(prompt=prompt, session_id=session_id), resume=prior)

    async def _drive(self, task: Task, *, resume: Checkpoint | None) -> SessionResult:
        messages: list[str] = []
        done: DoneEvent | None = None
        async for event in self._brain.run(task, resume=resume):
            # DoneEvent is terminal: once seen, DRAIN the rest of the stream but
            # ignore it, so post-terminal narration can't be recorded nor the
            # result overwritten by a second Done. We drain rather than `break`
            # because breaking mid-stream leaves the brain's underlying async
            # generator (e.g. the Claude SDK's) to be aclose()'d while still
            # running — "aclose(): asynchronous generator is already running" on
            # every run. Letting it run to exhaustion closes it cleanly.
            if done is not None:
                continue
            if isinstance(event, MessageEvent):
                messages.append(event.text)
            elif isinstance(event, DoneEvent):
                done = event
        if done is None:
            raise SessionError(f"brain ended without DoneEvent for session {task.session_id!r}")

        checkpoint = Checkpoint(
            session_id=task.session_id,
            turn=done.state.get("turn", 0),
            state=done.state,
        )
        await self._checkpoints.save(checkpoint)
        return SessionResult(
            session_id=task.session_id,
            text=done.result,
            checkpoint=checkpoint,
            messages=messages,
        )
