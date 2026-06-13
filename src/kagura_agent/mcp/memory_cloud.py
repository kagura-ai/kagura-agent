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
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

#: The tier an agent's writes land in by default (#15). Read-side recall filters
#: it out of the trusted backbone (``trusted_only=True``), so a quarantined write
#: cannot pollute trusted memory until a host-side promote graduates it.
QUARANTINE_TIER = "quarantine"
TRUSTED_TIER = "trusted"


@dataclass(frozen=True)
class Memory:
    id: str
    text: str
    tags: tuple[str, ...] = ()
    trust_tier: str = TRUSTED_TIER


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
            if trusted_only and mem.trust_tier != TRUSTED_TIER:
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

    def promote(self, memory_id: str) -> None:
        """Host-side ONLY: graduate a quarantined memory into the trusted backbone.

        Deliberately NOT on the ``MemoryClient`` protocol (the agent surface): the
        agent can never promote its own writes. Promotion is the effect of a
        post-graduation HITL grant (#15), applied host-side. Unknown id raises
        ``KeyError`` — fail-closed, no silent no-op that could mask a bad id.
        """
        mem = self._memories[memory_id]  # KeyError if unknown — fail-closed
        self._memories[memory_id] = replace(mem, trust_tier=TRUSTED_TIER)


class QuarantinedMemoryClient:
    """The confined ``MemoryClient`` the membrane leases into the agent container.

    Every write is forced into the quarantine tier — the caller-supplied
    ``trust_tier`` is intentionally ignored, so a hijacked agent cannot mint a
    trusted memory by simply asking for one. There is no promote path here;
    graduating a quarantined write into the trusted backbone is host-side only
    (``LocalMemoryClient.promote``), gated by graduation HITL (#15). This mirrors
    the ``write_approved`` / broker write-lock posture (#12/#20): the agent's
    self-asserted trust tier is never trusted. ``recall``/``create_edge`` delegate
    unchanged — confinement is on the write path only.
    """

    def __init__(self, inner: MemoryClient) -> None:
        self._inner = inner

    async def remember(
        self, text: str, *, tags: tuple[str, ...] = (), trust_tier: str = TRUSTED_TIER
    ) -> str:
        # trust_tier is accepted (protocol parity) but IGNORED — fail-closed: the
        # agent's writes always land quarantined, regardless of what it requests.
        return await self._inner.remember(text, tags=tags, trust_tier=QUARANTINE_TIER)

    async def recall(
        self,
        query: str,
        *,
        trusted_only: bool = False,
        tags: tuple[str, ...] = (),
    ) -> list[Memory]:
        return await self._inner.recall(query, trusted_only=trusted_only, tags=tags)

    async def create_edge(self, src_id: str, dst_id: str, *, type: str) -> None:
        await self._inner.create_edge(src_id, dst_id, type=type)


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
