"""Gate memory:write behind operator HITL approval (#14).

`MemoryWriteApprover` is the first real consumer of the cockpit approval loop
(#32). When a container agent wants to widen its memory scope to `memory:write`,
the host asks the operator via `Cockpit.request_capability`, then awaits the
decision with a hard timeout (`asyncio.wait_for` — the bound that turns the
registry's lazy fail-closed expiry into an actual deny if the operator never
answers). Only on an approved decision does it run `grant`, which the caller
wires to a broker.acquire using a **write_approved** `MemoryCloudProvider` — the
write_approved flip happens *after* approval, and the broker's
`_assert_scope_allowed` gate is the structural backstop (a read-locked or
look-alike provider still cannot mint a write token). A denied or timed-out
request grants nothing: fail-closed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from kagura_agent.cockpit.core import Cockpit
from kagura_agent.cockpit.hitl import CapabilityRequest
from kagura_agent.membrane.lease import Lease

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
