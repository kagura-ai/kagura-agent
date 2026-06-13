"""HITL approval — the cockpit's reason to exist.

When the launcher needs powers beyond baseline, the cockpit asks the human in
the thread (✅/❌). The decision is **fail-closed** (anything that is not an
explicit approval denies) and is recorded to memory as a graduation trail — the
same evidence the graduation curve later reads.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from kagura_agent.cockpit.transports.base import Transport
from kagura_agent.mcp.memory_cloud import MemoryClient

_APPROVE = "approve"
_DENY = "deny"


@dataclass(frozen=True)
class CapabilityRequest:
    thread_id: str
    capability: str
    reason: str


@dataclass(frozen=True)
class Decision:
    approved: bool


async def record_decision(
    memory: MemoryClient,
    request: CapabilityRequest,
    *,
    approved: bool,
    failure: str | None = None,
) -> None:
    """Record a HITL decision to memory as a graduation trail.

    Shared by the synchronous `HitlGate.review` and the cockpit's async approval
    loop (#32) so both write the same evidence the graduation curve later reads.
    """
    if approved:
        verb = "approved"
    elif failure:
        verb = f"denied (transport_error: {failure})"
    else:
        verb = "denied"
    await memory.remember(
        f"HITL {verb} {request.capability}: {request.reason}",
        tags=("hitl", "graduation-trail"),
    )


class HitlGate:
    def __init__(self, transport: Transport, memory: MemoryClient) -> None:
        self._transport = transport
        self._memory = memory

    async def review(self, request: CapabilityRequest) -> Decision:
        failure: str | None = None
        try:
            answer = await self._transport.ask(
                request.thread_id,
                f"grant {request.capability}? ({request.reason})",
                options=[_APPROVE, _DENY],
            )
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise  # never swallow cancellation
        except Exception as exc:
            # Fail-closed for ANY transport failure (timeout, NotImplementedError
            # from an unwired transport, API/network errors) — not just timeout.
            answer = _DENY
            failure = type(exc).__name__

        approved = answer.strip().lower() == _APPROVE
        await record_decision(self._memory, request, approved=approved, failure=failure)
        return Decision(approved=approved)
