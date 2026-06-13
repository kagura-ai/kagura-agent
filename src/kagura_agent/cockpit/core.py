"""The transport-agnostic cockpit core (v0.1 slice).

Wires a `Transport` to the intent router and a `Session`. v0.1 handles
launch/continue only; v0.3 adds status/approve/kill and HITL escalation. The
cockpit is the *trusted host process* — it is the only side that will hold the
bot token and (later) speak to Docker. Agent work happens behind the brain.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Protocol

from kagura_agent.cockpit.approval import PendingApprovalRegistry
from kagura_agent.cockpit.hitl import CapabilityRequest, Decision, record_decision
from kagura_agent.cockpit.intent import Intent, classify
from kagura_agent.cockpit.registry import SessionRegistry
from kagura_agent.cockpit.transports.base import Event, Transport
from kagura_agent.core.brain.base import BrainProvider, Task
from kagura_agent.core.session import Session
from kagura_agent.mcp.memory_cloud import MemoryClient
from kagura_agent.patterns.checkpoint import CheckpointStore

log = logging.getLogger(__name__)


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
        approvals: PendingApprovalRegistry | None = None,
        memory: MemoryClient | None = None,
        operator_id: str | None = None,
    ) -> None:
        self._transport = transport
        self._brain = brain
        self._checkpoints = checkpoints
        self._registry = registry or SessionRegistry()
        self._launcher = launcher
        self._approvals = approvals or PendingApprovalRegistry()
        self._memory = memory
        # When set, only an event whose `sender` matches may resolve a pending
        # HITL approval (#14: prevents a hijacked agent self-approving). None =
        # single-user CLI default: no operator gate (#32-compatible).
        self._operator_id = operator_id

    async def request_capability(self, request: CapabilityRequest) -> asyncio.Future[Decision]:
        """Producer seam (#32): register a pending approval, surface it to the
        operator, and return the future its decision resolves.

        NON-blocking: the caller awaits the future OUTSIDE `serve()`, so the loop
        stays free to process the later `/approve`|`/deny` event that resolves it
        (awaiting here would deadlock the single consumer loop). Raises
        `PendingApprovalExists` if the thread already has a live pending request.
        """
        future = self._approvals.register(request)
        try:
            await self._transport.send(
                request.thread_id,
                f"approval requested: {request.capability} ({request.reason}) "
                "— reply /approve or /deny",
            )
        except BaseException:
            # A failed surface must not strand the thread with an orphan pending
            # (which would reject every later request until it expires). Roll back
            # (deny the discarded future + remove the entry — fail-closed), re-raise.
            self._approvals.resolve(request.thread_id, approved=False)
            raise
        return future

    async def withdraw_pending(self, thread_id: str) -> None:
        """Fail-closed teardown of a pending approval the *producer* gave up on
        (e.g. a consumer's `asyncio.wait_for` timed out). Resolves it denied and
        clears the registry entry, so (a) a late `/approve` cannot record a
        misleading "approved" with nothing actually granted, and (b) the next
        `request_capability` for the thread is not wedged by
        `PendingApprovalExists` until the registry TTL elapses. Records the
        timeout as a denial on the graduation-trail for audit symmetry."""
        request = self._approvals.resolve(thread_id, approved=False)
        if request is not None and self._memory is not None:
            await record_decision(self._memory, request, approved=False)

    async def serve(self) -> None:
        # Per-event isolation: the cockpit is the sole message consumer and the
        # sole HITL surface, so one bad event must never silently kill the loop
        # (which would strand every pending approval and drop all later messages).
        async for event in self._transport.listen():
            try:
                await self.handle(event)
            except Exception:
                log.exception("cockpit failed to handle event on thread %s", event.thread_id)
                with contextlib.suppress(Exception):
                    await self._transport.send(event.thread_id, "internal error — see logs")

    async def handle(self, event: Event) -> None:
        intent = classify(event, known_sessions=self._registry.sessions())

        if intent is Intent.STATUS:
            await self._handle_status(event)
        elif intent is Intent.KILL:
            await self._handle_kill(event)
        elif intent is Intent.APPROVE:
            await self._handle_approve(event)
        elif intent is Intent.DENY:
            await self._handle_deny(event)
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

    async def _handle_approve(self, event: Event) -> None:
        await self._resolve_pending(event, approved=True)

    async def _handle_deny(self, event: Event) -> None:
        await self._resolve_pending(event, approved=False)

    async def _resolve_pending(self, event: Event, *, approved: bool) -> None:
        # A typed /approve|/deny resolves the thread's *pending* capability
        # request (registered by request_capability). No live pending — including
        # one that already expired — keeps the legacy reply and grants nothing
        # (fail-closed); never fall through to LAUNCH (which would spin a brain
        # run on the literal "/approve" and clobber the session record).
        if not self._approvals.pending(event.thread_id):
            await self._transport.send(event.thread_id, "no pending approval")
            return
        # Operator-identity gate (#14): when an operator is configured, only that
        # identity may resolve. A non-operator approve/deny is rejected and the
        # request is LEFT PENDING for the real operator — fail-closed against a
        # hijacked agent self-approving its own capability request.
        if self._operator_id is not None and event.sender != self._operator_id:
            await self._transport.send(
                event.thread_id, "approval ignored: only the operator may approve/deny"
            )
            return
        request = self._approvals.resolve(event.thread_id, approved=approved)
        if request is None:  # raced/expired between the pending() check and resolve
            await self._transport.send(event.thread_id, "no pending approval")
            return
        if self._memory is not None:
            await record_decision(self._memory, request, approved=approved)
        verb = "approved" if approved else "denied"
        await self._transport.send(event.thread_id, f"{verb} {request.capability}")

    async def _handle_kill(self, event: Event) -> None:
        rec = self._registry.get(event.thread_id)
        if rec is not None and rec.container_id and self._launcher is not None:
            await self._launcher.kill(rec.container_id)
        self._registry.close(event.thread_id)
        await self._transport.send(event.thread_id, f"session {event.thread_id} killed")
