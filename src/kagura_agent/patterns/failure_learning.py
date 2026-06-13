"""Failure learning: turn outcomes into memory edges and graduation signal.

A failure writes a `prevents` edge (failure → the action it should prevent) so a
future run recalls the hazard, and demotes the category in the graduation engine.
A verified success advances the curve. The accumulation of `prevents` edges is
the trust score the graduation curve reads.
"""

from __future__ import annotations

from kagura_agent.mcp.memory_cloud import QUARANTINE_TIER, MemoryClient
from kagura_agent.membrane.graduation import GraduationEngine


class FailureLearner:
    def __init__(self, memory: MemoryClient, graduation: GraduationEngine) -> None:
        self._memory = memory
        self._graduation = graduation

    async def failed(self, category: str, *, action_mid: str, description: str) -> None:
        # The failure `description` can be agent/tool-derived (e.g. a tool's error
        # text), so it is untrusted provenance — write it to the quarantine tier,
        # not the trusted backbone. This keeps a poisoned failure record out of
        # trusted-only recalls (which steer graduation/behaviour) until a host
        # promotes it, mirroring QuarantinedMemoryClient's write confinement.
        fail_mid = await self._memory.remember(
            description, tags=("failure", category), trust_tier=QUARANTINE_TIER
        )
        await self._memory.create_edge(fail_mid, action_mid, type="prevents")
        self._graduation.record_failure(category)

    async def succeeded(self, category: str, *, task_id: str, verified: bool) -> None:
        self._graduation.record_success(category, task_id=task_id, verified=verified)
