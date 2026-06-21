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
from collections.abc import Set
from typing import Protocol

from kagura_agent.cockpit.approval import PendingApprovalRegistry
from kagura_agent.cockpit.hitl import CapabilityRequest, Decision, record_decision
from kagura_agent.cockpit.intent import Intent, classify
from kagura_agent.cockpit.registry import SessionRegistry
from kagura_agent.cockpit.transports.base import Event, Transport, click_authorized
from kagura_agent.core.brain.base import BrainProvider, Task
from kagura_agent.core.brain.container import BrainContainerSession, ContainerBrainProvider
from kagura_agent.core.session import Session
from kagura_agent.mcp.memory_cloud import MemoryClient
from kagura_agent.membrane.launcher import LaunchSpec, validate_spec
from kagura_agent.patterns.checkpoint import CheckpointStore

log = logging.getLogger(__name__)


class _Killer(Protocol):
    async def kill(self, container_id: str) -> None: ...


class _ContainerBackend(Protocol):
    """Launches the brain in a hardened container and streams it back (#102 PR2).

    Deliberately separate from ``_launcher`` (which only *kills*): a deployment
    that runs the brain in-container wires BOTH — this backend to launch + stream
    + enumerate, and the launcher to kill — so the kill-only launchers that
    existing callers pass keep working unchanged. ``spec_for`` builds the per-run
    :class:`LaunchSpec` (image, project-root RO mount, leased creds env, per-run
    egress); ``start`` validates-then-runs the container ATTACHED and streams its
    stdout event lines; ``live_container_ids`` feeds restart reconciliation. The
    real implementation (the streaming ``docker run``) is the PR3 deployment edge.
    """

    project_root: str

    def spec_for(self, session_id: str) -> LaunchSpec: ...

    async def start(self, spec: LaunchSpec, stdin: bytes) -> BrainContainerSession: ...

    async def live_container_ids(self) -> Set[str]: ...


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
        container: _ContainerBackend | None = None,
        require_operator: bool = False,
    ) -> None:
        self._transport = transport
        self._brain = brain
        self._checkpoints = checkpoints
        self._registry = registry or SessionRegistry()
        self._launcher = launcher
        self._approvals = approvals or PendingApprovalRegistry()
        self._memory = memory
        # When set, a LAUNCH/CONTINUE runs the brain INSIDE a hardened container
        # (the membrane gate validates the spec first) instead of in-process. None
        # = the in-process default (every existing caller), unchanged.
        self._container = container
        # When set, only an event whose `sender` matches may resolve a pending
        # HITL approval (#14: prevents a hijacked agent self-approving). None =
        # single-user CLI default: no operator gate (#32-compatible).
        self._operator_id = operator_id
        # Opt-in fail-closed gate (#165 S1 part 4): when set, an UNSET operator DENIES
        # privileged actions (approve/kill) instead of the permissive single-user
        # fallback — so a hijacked agent's sender-less /approve can never self-approve.
        # A non-trivial deployment (brain-in-container / egress / multi-party) should
        # enable it; the CLI/serve wiring that turns it on — paired with a STARTUP
        # operator requirement so it can't silently lock the operator out — is a
        # follow-up, not an auto-default here.
        self._require_operator = require_operator

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
            # The future is ALREADY denied (fail-closed) — a failed audit write on the
            # producer's timeout path must not surface as a raised exception there (it
            # would crash the consumer's wait_for/timeout handling). Suppress + log; the
            # deny stands regardless. (Contrast _resolve_pending's GRANT path, which is
            # audit-GATED and must NOT grant on a failed audit.)
            try:
                await record_decision(self._memory, request, approved=False)
            except Exception:
                log.exception(
                    "withdraw_pending: failed to record the timeout denial on the audit "
                    "trail for thread %s (the request is still denied)",
                    thread_id,
                )

    async def serve(self) -> None:
        # Reconcile the session table against the live containers BEFORE the loop:
        # a session whose container vanished while the cockpit was down is marked
        # dead so a follow-up isn't routed to a stale CONTINUE (replaying a
        # checkpoint whose container is gone).
        await self._reconcile_on_start()
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

    async def _reconcile_on_start(self) -> None:
        """Mark sessions whose container is no longer live as dead. Best-effort:
        a failed enumeration (``DockerRuntime.list`` fails closed on a docker
        error) leaves records running rather than risk marking a live container
        dead — the conservative direction."""
        if self._container is None:
            return
        try:
            live = await self._container.live_container_ids()
        except Exception:
            log.exception("session reconcile skipped: could not enumerate live containers")
            return
        self._registry.reconcile(live)

    def _container_brain(self, thread_id: str) -> ContainerBrainProvider:
        """Build the container-backed brain for one thread, validating the spec at
        the membrane gate FIRST so a ``MembraneViolation`` refuses the run before
        any container is created (the #102 acceptance). ``on_start`` records the
        container id (in place, preserving ``granted_caps``) the instant the
        container starts, so ``/kill`` can tear down the real container even mid-run.

        Failure-path lifecycle: if the in-container run then fails (a hijacked or
        crashed container that ends without a terminal DoneEvent), ``Session``
        raises; the container is already reaped (the provider's finally →
        ``session.aclose()``), and ``_handle_task`` marks the record ``closed``
        (non-resumable, ``container_id`` retained so ``/kill`` still works) so a
        follow-up message cannot replay a stale checkpoint against the dead
        container (#124 5a). Restart reconciliation independently marks a record
        ``dead`` if its cid has vanished."""
        assert self._container is not None  # guarded by the caller
        spec = validate_spec(
            self._container.spec_for(thread_id), project_root=self._container.project_root
        )
        backend = self._container
        return ContainerBrainProvider(
            lambda stdin: backend.start(spec, stdin),
            caps=self._brain.caps,
            on_start=lambda cid: self._registry.set_container(
                thread_id, container_id=cid, image=spec.image
            ),
        )

    async def _handle_task(self, event: Event, intent: Intent) -> None:
        # In-container when a backend is wired (membrane-validated), else the
        # in-process brain. The container path registers the session via on_start
        # the moment the container starts (so /kill works mid-run); the in-process
        # LAUNCH registers after the run (no container id) — today's behaviour.
        brain = self._brain if self._container is None else self._container_brain(event.thread_id)
        session = Session(brain, self._checkpoints)
        try:
            if intent is Intent.CONTINUE:
                result = await session.resume(event.thread_id, prompt=event.text)
            else:  # LAUNCH
                result = await session.run(Task(prompt=event.text, session_id=event.thread_id))
                if self._container is None:
                    self._registry.add(event.thread_id)
        except Exception:
            # A failed in-container run leaves the on_start-armed record "running"
            # pointing at a now-gone container (the provider's finally already reaped
            # it via session.aclose()). Mark it "closed" — non-resumable, but the
            # container_id is retained so /kill still works — BEFORE the error
            # propagates to serve(). Otherwise sessions() keeps treating it as
            # resumable and the next message replays a stale checkpoint against the
            # dead container (#124 item 5a). The in-process path registers only on
            # success (below), so there is nothing to close there.
            if self._container is not None:
                self._registry.close(event.thread_id)
            raise
        await self._transport.send(event.thread_id, result.text)

    async def _handle_status(self, event: Event) -> None:
        rec = self._registry.get(event.thread_id)
        status = rec.status if rec is not None else "unknown"
        await self._transport.send(event.thread_id, f"session {event.thread_id}: {status}")

    async def _handle_approve(self, event: Event) -> None:
        await self._resolve_pending(event, approved=True)

    async def _handle_deny(self, event: Event) -> None:
        await self._resolve_pending(event, approved=False)

    def _sender_is_operator(self, event: Event) -> bool:
        """Whether this event may drive a privileged control action.

        When an operator is configured, only that identity qualifies; with no
        operator (single-user CLI default) every event qualifies — unless this is a
        fail-closed deployment (`require_operator`, e.g. brain-in-container), where an
        unset operator denies every event (#165 S1 part 4). Shared by the approve/deny
        and /kill gates so destructive control surfaces enforce the same
        operator-identity boundary (#14). Single source of truth for the rule is
        `click_authorized` (also used by the transport button path)."""
        return click_authorized(
            event.sender, self._operator_id, require_operator=self._require_operator
        )

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
        if not self._sender_is_operator(event):
            await self._transport.send(
                event.thread_id, "approval ignored: only the operator may approve/deny"
            )
            return
        # Claim the entry WITHOUT resolving its future: liveness is decided and the
        # entry removed atomically here, THEN we write the audit, THEN fulfil the
        # future (the grant the consumer observes). This guarantees (a) a grant is
        # never observable before its audit, and (b) an audit is never written for
        # a request that wasn't live (e.g. expired during the audit await).
        claimed = self._approvals.claim(event.thread_id)
        if claimed is None:  # expired/raced between the gate and the claim
            await self._transport.send(event.thread_id, "no pending approval")
            return
        request, future = claimed
        if self._memory is not None:
            await record_decision(self._memory, request, approved=approved)
        if not future.done():
            future.set_result(Decision(approved=approved))
        verb = "approved" if approved else "denied"
        await self._transport.send(event.thread_id, f"{verb} {request.capability}")

    async def _handle_kill(self, event: Event) -> None:
        # /kill is destructive (tears down the container + closes the session), so
        # it enforces the same operator-identity gate as approve/deny (#14): a
        # non-operator — e.g. a hijacked agent posting /kill — must not be able to
        # drive it. Fail-closed: reject and leave the session untouched.
        if not self._sender_is_operator(event):
            await self._transport.send(
                event.thread_id, "kill ignored: only the operator may kill a session"
            )
            return
        rec = self._registry.get(event.thread_id)
        if rec is not None and rec.container_id and self._launcher is not None:
            try:
                await self._launcher.kill(rec.container_id)
            except Exception:
                # The container kill failed (e.g. docker error — runtime.kill now
                # raises on non-zero). Don't leave the session falsely "running"
                # (it would keep routing follow-ups as CONTINUE): close it so it
                # isn't treated as resumable, and surface the failure so the
                # operator can clean up a possible orphan.
                log.exception("kill of container %s failed", rec.container_id)
                self._registry.close(event.thread_id)
                await self._transport.send(
                    event.thread_id,
                    f"session {event.thread_id}: container kill FAILED (see logs); "
                    "session closed",
                )
                return
        self._registry.close(event.thread_id)
        await self._transport.send(event.thread_id, f"session {event.thread_id} killed")
