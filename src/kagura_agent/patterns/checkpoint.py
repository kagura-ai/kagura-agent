"""Long-task resume: persist a provider's opaque state between turns.

v1 ships an in-memory store plus a memory-cloud-backed store (see
`MemoryCloudCheckpointStore`). The store is deliberately tiny — the session is
the only writer, and `Checkpoint.state` is opaque to it.
"""

from __future__ import annotations

from typing import Protocol

from kagura_agent.core.brain.base import Checkpoint


class CheckpointStore(Protocol):
    async def save(self, checkpoint: Checkpoint) -> None: ...

    async def load(self, session_id: str) -> Checkpoint | None: ...


class InMemoryCheckpointStore:
    """Process-local checkpoint store (tests + the cockpit's hot path)."""

    def __init__(self) -> None:
        self._by_session: dict[str, Checkpoint] = {}

    async def save(self, checkpoint: Checkpoint) -> None:
        self._by_session[checkpoint.session_id] = checkpoint

    async def load(self, session_id: str) -> Checkpoint | None:
        return self._by_session.get(session_id)
