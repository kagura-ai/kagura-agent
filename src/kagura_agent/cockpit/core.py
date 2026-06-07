"""The transport-agnostic cockpit core (v0.1 slice).

Wires a `Transport` to the intent router and a `Session`. v0.1 handles
launch/continue only; v0.3 adds status/approve/kill and HITL escalation. The
cockpit is the *trusted host process* — it is the only side that will hold the
bot token and (later) speak to Docker. Agent work happens behind the brain.
"""

from __future__ import annotations

from typing import Protocol

from kagura_agent.cockpit.intent import Intent, classify
from kagura_agent.cockpit.registry import SessionRegistry
from kagura_agent.cockpit.transports.base import Event, Transport
from kagura_agent.core.brain.base import BrainProvider, Task
from kagura_agent.core.session import Session
from kagura_agent.patterns.checkpoint import CheckpointStore


class _Killer(Protocol):
    async def kill(self, container_id: str) -> None: ...


class Cockpit:
    def __init__(
        self,
        transport: Transport,
        brain: BrainProvider,
        checkpoints: CheckpointStore,
        registry: SessionRegistry | None = None,
        launcher: _Killer | None = None,
    ) -> None:
        self._transport = transport
        self._brain = brain
        self._checkpoints = checkpoints
        self._registry = registry or SessionRegistry()
        self._launcher = launcher

    async def serve(self) -> None:
        async for event in self._transport.listen():
            await self.handle(event)

    async def handle(self, event: Event) -> None:
        intent = classify(event, known_sessions=self._registry.sessions())

        if intent is Intent.STATUS:
            await self._handle_status(event)
        elif intent is Intent.KILL:
            await self._handle_kill(event)
        else:  # LAUNCH or CONTINUE — drive the brain
            await self._handle_task(event, intent)

    async def _handle_task(self, event: Event, intent: Intent) -> None:
        session = Session(self._brain, self._checkpoints)
        if intent is Intent.CONTINUE:
            result = await session.resume(event.thread_id, prompt=event.text)
        else:  # LAUNCH
            result = await session.run(Task(prompt=event.text, session_id=event.thread_id))
            self._registry.add(event.thread_id)
        await self._transport.send(event.thread_id, result.text)

    async def _handle_status(self, event: Event) -> None:
        rec = self._registry.get(event.thread_id)
        status = rec.status if rec is not None else "unknown"
        await self._transport.send(event.thread_id, f"session {event.thread_id}: {status}")

    async def _handle_kill(self, event: Event) -> None:
        rec = self._registry.get(event.thread_id)
        if rec is not None and rec.container_id and self._launcher is not None:
            await self._launcher.kill(rec.container_id)
        self._registry.close(event.thread_id)
        await self._transport.send(event.thread_id, f"session {event.thread_id} killed")
