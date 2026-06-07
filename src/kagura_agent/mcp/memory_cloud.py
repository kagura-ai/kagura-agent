"""The agent runtime's view of memory-cloud — deliberately narrow.

The runtime gets append (`remember`), scoped read (`recall`, with a trust-tier
filter), and `create_edge` (to record `prevents` relationships for failure
learning). It gets **no admin** (delete/forget/merge/rollback/schema): a
prompt-injected agent must not be able to amplify a hijack into destructive
writes. `LocalMemoryClient` is the self-host backend (here in-memory; SQLite in
deployment); a real deployment swaps in an MCP-backed client with the same
surface.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Memory:
    id: str
    text: str
    tags: tuple[str, ...] = ()
    trust_tier: str = "trusted"


@runtime_checkable
class MemoryClient(Protocol):
    async def remember(
        self, text: str, *, tags: tuple[str, ...] = (), trust_tier: str = "trusted"
    ) -> str: ...

    async def recall(
        self,
        query: str,
        *,
        trusted_only: bool = False,
        tags: tuple[str, ...] = (),
    ) -> list[Memory]: ...

    async def create_edge(self, src_id: str, dst_id: str, *, type: str) -> None: ...


class LocalMemoryClient:
    """Self-host backend. No admin verbs by construction."""

    def __init__(self) -> None:
        self._memories: dict[str, Memory] = {}
        self._edges: dict[str, list[tuple[str, str]]] = {}
        self._ids = itertools.count(1)

    async def remember(
        self, text: str, *, tags: tuple[str, ...] = (), trust_tier: str = "trusted"
    ) -> str:
        mid = f"m{next(self._ids)}"
        self._memories[mid] = Memory(id=mid, text=text, tags=tuple(tags), trust_tier=trust_tier)
        return mid

    async def recall(
        self,
        query: str,
        *,
        trusted_only: bool = False,
        tags: tuple[str, ...] = (),
    ) -> list[Memory]:
        terms = [t.lower() for t in query.split()]
        results: list[Memory] = []
        for mem in self._memories.values():
            if trusted_only and mem.trust_tier != "trusted":
                continue
            if tags and not set(tags) & set(mem.tags):
                continue
            haystack = mem.text.lower()
            if any(term in haystack for term in terms):
                results.append(mem)
        return results

    async def create_edge(self, src_id: str, dst_id: str, *, type: str) -> None:
        self._edges.setdefault(src_id, []).append((dst_id, type))

    def edges_of(self, src_id: str) -> list[tuple[str, str]]:
        return list(self._edges.get(src_id, []))


def mcp_available() -> bool:  # pragma: no cover - environment probe
    """Whether the memory-cloud MCP server is reachable (startup gate input)."""

    import os

    return bool(os.environ.get("KAGURA_MEMORY_MCP_URL"))
