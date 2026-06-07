"""HITL approval — the cockpit's reason to exist.

When the launcher needs powers beyond baseline, the cockpit asks the human in
the thread (✅/❌). The decision is **fail-closed** (anything that is not an
explicit approval denies) and is recorded to memory as a graduation trail — the
same evidence the graduation curve later reads.
"""

from __future__ import annotations

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


class HitlGate:
    def __init__(self, transport: Transport, memory: MemoryClient) -> None:
        self._transport = transport
        self._memory = memory

    async def review(self, request: CapabilityRequest) -> Decision:
        try:
            answer = await self._transport.ask(
                request.thread_id,
                f"grant {request.capability}? ({request.reason})",
                options=[_APPROVE, _DENY],
            )
        except TimeoutError:
            answer = _DENY  # timeout = deny

        approved = answer.strip().lower() == _APPROVE
        verb = "approved" if approved else "denied"
        await self._memory.remember(
            f"HITL {verb} {request.capability}: {request.reason}",
            tags=("hitl", "graduation-trail"),
        )
        return Decision(approved=approved)
