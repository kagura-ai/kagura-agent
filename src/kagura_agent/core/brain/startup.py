"""Startup gating for brains.

MCP is a *gate*, not a feature flag: memory-cloud is an MCP server, and the
agent's whole thesis is memory-grounded. A brain that cannot speak MCP is
rejected at startup rather than silently degraded to a memory-less actor.
"""

from __future__ import annotations

from kagura_agent.core.brain.base import BrainCaps


class BrainStartupError(RuntimeError):
    """A brain cannot be started in the current environment."""


def ensure_startable(caps: BrainCaps, *, mcp_available: bool) -> None:
    if caps.requires_mcp and not mcp_available:
        raise BrainStartupError(
            f"brain {caps.name!r} requires MCP (memory-cloud) but it is unavailable; "
            "refusing to start (no silent degrade)"
        )
