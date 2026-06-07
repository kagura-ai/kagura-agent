"""Failure learning: turn outcomes into memory edges and graduation signal.

A failure writes a `prevents` edge (failure → the action it should prevent) so a
future run recalls the hazard, and demotes the category in the graduation engine.
A verified success advances the curve. The accumulation of `prevents` edges is
the trust score the graduation curve reads.
"""

from __future__ import annotations

from kagura_agent.mcp.memory_cloud import MemoryClient
from kagura_agent.membrane.graduation import GraduationEngine


class FailureLearner:
    def __init__(self, memory: MemoryClient, graduation: GraduationEngine) -> None:
        self._memory = memory
        self._graduation = graduation

    async def failed(self, category: str, *, action_mid: str, description: str) -> None:
        fail_mid = await self._memory.remember(description, tags=("failure", category))
        await self._memory.create_edge(fail_mid, action_mid, type="prevents")
        self._graduation.record_failure(category)

    async def succeeded(self, category: str, *, task_id: str, verified: bool) -> None:
        self._graduation.record_success(category, task_id=task_id, verified=verified)
