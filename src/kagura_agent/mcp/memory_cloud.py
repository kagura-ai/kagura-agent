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


class MemoryUnreachableError(RuntimeError):
    """Memory-cloud cannot be reached/authenticated; refuse to start."""


def ensure_memory_reachable(*, reachable: bool) -> None:
    """The redefined startup gate (v0.2-A6): memory must be reachable.

    The old gate was "the brain requires MCP" — coupling memory to the brain.
    Memory is now CLI-primary and brain-independent, so the gate asserts only
    that memory is reachable + authenticated (via the CLI). It is fail-closed:
    if memory cannot be reached, we refuse to start rather than silently run a
    memory-less agent. The reachability *decision* is injected (so the gate is
    unit-tested); the live probe is `memory_reachable()`.
    """
    if not reachable:
        raise MemoryUnreachableError(
            "memory-cloud is not reachable/authenticated via the kagura CLI; "
            "refusing to start (no silent degrade). Run `kagura auth login` on the host."
        )


def memory_reachable() -> bool:  # pragma: no cover - shells out to the kagura CLI
    """Whether memory is reachable: can the host mint a token via the CLI?

    Asks the CLI for a short-lived access token (`kagura auth token`). Memory is
    reachable only when the CLI exits zero AND actually prints a token — a zero
    exit with empty stdout (e.g. a CLI that no-ops) is treated as unreachable, so
    the gate stays fail-closed. This is the CLI-primary replacement for the old
    `KAGURA_MEMORY_MCP_URL` env probe.
    """
    import subprocess

    try:
        proc = subprocess.run(
            ["kagura", "auth", "token"],
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and bool(proc.stdout.strip())
