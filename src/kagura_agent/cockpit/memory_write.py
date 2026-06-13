"""Gate privileged memory operations behind operator HITL approval (#14, #15).

Two consumers of the cockpit approval loop (#32) live here, both fail-closed:

- `MemoryWriteApprover` (#14): when a container agent wants to widen its memory
  scope to `memory:write`, the host asks the operator via
  `Cockpit.request_capability`, awaits with a hard `asyncio.wait_for` timeout, and
  only on approval runs `grant` (wired to a broker.acquire using a
  **write_approved** `MemoryCloudProvider`; the broker's `_assert_scope_allowed`
  is the structural backstop). Deny/timeout grants nothing.

- `WriteGraduationGate` (#15, v0.4): connects the `GraduationEngine` (capability
  graduation by category) to the quarantine→trusted write promotion. There is NO
  auto-promotion — the engine only makes a category *eligible*; the operator's
  HITL grant is the final gate, and only then are the quarantined memories
  promoted host-side. Not-graduated / untrusted-input / cooldown → no proposal at
  all; deny/timeout → nothing promoted (timeout withdraws the pending, like #14).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from kagura_agent.cockpit.core import Cockpit
from kagura_agent.cockpit.hitl import CapabilityRequest
from kagura_agent.membrane.graduation import GraduationEngine
from kagura_agent.membrane.lease import Lease

log = logging.getLogger(__name__)

_MEMORY_WRITE = "memory:write"


class MemoryWriteApprover:
    def __init__(
        self,
        cockpit: Cockpit,
        grant: Callable[[], Awaitable[Lease]],
        *,
        timeout: float = 300.0,
    ) -> None:
        self._cockpit = cockpit
        self._grant = grant
        self._timeout = timeout

    async def request(self, thread_id: str, reason: str) -> Lease | None:
        """Ask the operator to approve memory:write for `thread_id`. Returns the
        minted write lease on approval, or None (fail-closed) on deny/timeout."""
        future = await self._cockpit.request_capability(
            CapabilityRequest(thread_id=thread_id, capability=_MEMORY_WRITE, reason=reason)
        )
        try:
            decision = await asyncio.wait_for(future, self._timeout)
        except TimeoutError:
            # No operator decision within the window — fail-closed. Withdraw the
            # pending so the timed-out request leaves no orphan in the registry (a
            # late /approve can't record a false "approved", and a re-request isn't
            # wedged by PendingApprovalExists until the registry TTL elapses).
            await self._cockpit.withdraw_pending(thread_id)
            return None
        if not decision.approved:
            return None  # operator denied — fail-closed
        return await self._grant()  # approved → mint the write-approved lease


class WriteGraduationGate:
    """Connect capability graduation to quarantine→trusted write promotion (#15).

    The `GraduationEngine` decides only *eligibility* (verified successes across
    distinct tasks, input-trust gate, fail-closed demotion/cooldown). This gate
    turns an eligible category into an operator HITL *proposal* via the #32/#14
    approval loop, and promotes the quarantined memories host-side **only** on an
    approve. No auto-promotion at any step; not-eligible never even surfaces a
    proposal.
    """

    def __init__(
        self,
        engine: GraduationEngine,
        cockpit: Cockpit,
        promote: Callable[[str], None],
        *,
        timeout: float = 300.0,
    ) -> None:
        self._engine = engine
        self._cockpit = cockpit
        self._promote = promote
        self._timeout = timeout

    async def propose_promotion(
        self,
        category: str,
        memory_ids: list[str],
        *,
        thread_id: str,
        input_trust: str,
        reason: str,
    ) -> list[str] | None:
        """Promote `memory_ids` from quarantine to trusted iff `category` is
        graduation-eligible AND the operator approves. Returns the promoted ids on
        approval, or None (fail-closed) when not eligible / denied / timed out."""
        if not self._engine.should_propose(category, input_trust=input_trust):
            return None  # not graduated (or untrusted input / cooldown) — no proposal
        future = await self._cockpit.request_capability(
            CapabilityRequest(
                thread_id=thread_id, capability=f"memory:promote:{category}", reason=reason
            )
        )
        try:
            decision = await asyncio.wait_for(future, self._timeout)
        except TimeoutError:
            # The operator never decided — fail-closed, and do NOT burn the
            # cooldown: a proposal nobody saw must stay re-surfaceable, so
            # mark_proposed is deliberately skipped on this path.
            await self._cockpit.withdraw_pending(thread_id)
            return None
        # A real decision (approve OR deny) was seen → start the cooldown so the
        # operator is not re-nagged with the same proposal during the window.
        self._engine.mark_proposed(category)
        if not decision.approved:
            return None  # operator denied — nothing promoted
        # Promote per-id and resiliently: a single failing promote (stale id, or
        # an MCP-backed backend rejecting one) must not strand the rest of the
        # batch. A failed promote leaves that memory quarantined — fail-safe (no
        # over-grant). Return the ids that actually landed in the trusted backbone.
        promoted: list[str] = []
        for memory_id in memory_ids:
            try:
                self._promote(memory_id)
            except Exception:
                log.exception(
                    "write-graduation: promote of %s failed; left quarantined", memory_id
                )
                continue
            promoted.append(memory_id)
        return promoted
