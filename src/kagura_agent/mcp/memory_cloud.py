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

#: Delivery mode (#88) — orthogonal to trust. ``always`` = pinned: deterministically
#: surfaced every turn via ``load_pinned()`` (Goal/Guardrail), never left to
#: probabilistic ``recall``. ``on_recall`` = the default; only via ``recall``.
ALWAYS_DELIVERY = "always"
ON_RECALL_DELIVERY = "on_recall"
_VALID_DELIVERY = (ALWAYS_DELIVERY, ON_RECALL_DELIVERY)


@dataclass(frozen=True)
class Memory:
    id: str
    text: str
    tags: tuple[str, ...] = ()
    trust_tier: str = TRUSTED_TIER
    delivery_mode: str = ON_RECALL_DELIVERY


@dataclass(frozen=True)
class FeedbackRecord:
    """A retrieval-quality datum (#90): was ``memory_id`` useful for ``query``?

    ``helpful`` is an **independent** verdict (HITL approval / task outcome), never
    the agent's self-report. Lives in a side lane, never embedded, never surfaced by
    ``recall`` — so it cannot pollute the recall space it measures.
    """

    memory_id: str
    query: str
    helpful: bool


@runtime_checkable
class MemoryClient(Protocol):
    async def remember(
        self,
        text: str,
        *,
        tags: tuple[str, ...] = (),
        trust_tier: str = "trusted",
        delivery_mode: str = ON_RECALL_DELIVERY,
    ) -> str: ...

    async def recall(
        self,
        query: str,
        *,
        trusted_only: bool = False,
        tags: tuple[str, ...] = (),
    ) -> list[Memory]: ...

    async def load_pinned(self) -> list[Memory]:
        """The deterministic, unranked counterpart to ``recall`` (#88): the COMPLETE
        set of ``delivery_mode="always"`` (pinned) memories, every call. Goal /
        Guardrail / critical-policy memories load this way so they are never missed
        by probabilistic ``recall`` (the structural flaw deterministic delivery
        closes). Mirrors the SDK's ``load_pinned`` (kagura-memory-python-sdk#172)."""
        ...

    async def create_edge(self, src_id: str, dst_id: str, *, type: str) -> None: ...


class LocalMemoryClient:
    """Self-host backend. No admin verbs by construction."""

    def __init__(self) -> None:
        self._memories: dict[str, Memory] = {}
        self._edges: dict[str, list[tuple[str, str]]] = {}
        self._feedback: list[FeedbackRecord] = []  # #90: retrieval-quality side lane
        self._ids = itertools.count(1)

    async def remember(
        self,
        text: str,
        *,
        tags: tuple[str, ...] = (),
        trust_tier: str = "trusted",
        delivery_mode: str = ON_RECALL_DELIVERY,
    ) -> str:
        # Validate the delivery mode at the write boundary — fail-CLOSED for the
        # standing-guardrail lane (#88). Without this, a host typo ("Always") would
        # be stored verbatim and silently never pin (load_pinned matches on exact
        # equality), dropping a guardrail the operator believes is always-on.
        if delivery_mode not in _VALID_DELIVERY:
            raise ValueError(
                f"unknown delivery_mode {delivery_mode!r} (expected one of {_VALID_DELIVERY})"
            )
        mid = f"m{next(self._ids)}"
        self._memories[mid] = Memory(
            id=mid,
            text=text,
            tags=tuple(tags),
            trust_tier=trust_tier,
            delivery_mode=delivery_mode,
        )
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

    async def load_pinned(self) -> list[Memory]:
        # Deterministic + unranked: the COMPLETE pinned set, insertion order, every
        # call. No query, no ranking, no trust filter — pinned guardrails are
        # host-curated and load whole (#88).
        return [m for m in self._memories.values() if m.delivery_mode == ALWAYS_DELIVERY]

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

    def record_feedback(self, memory_id: str, query: str, *, helpful: bool) -> None:
        """Host-side ONLY: record whether a recalled memory was useful (#90).

        Deliberately **NOT** on the ``MemoryClient`` protocol (the agent surface) —
        like ``promote``. A ranking-affecting signal a confined/hijacked agent could
        emit would let it up/down-rank a memory it recalled; the "helpful" verdict
        must come from an INDEPENDENT source (HITL approval / task outcome), never
        the agent's self-report (the capability-graduation independent-signal rule).
        Stored in a side lane separate from ``_memories``, so feedback never surfaces
        via ``recall``. Unknown ``memory_id`` raises ``KeyError`` — fail-closed, no
        silent record against a bad id (mirrors ``promote``).
        """
        if memory_id not in self._memories:
            raise KeyError(memory_id)
        self._feedback.append(
            FeedbackRecord(memory_id=memory_id, query=query, helpful=helpful)
        )

    def feedback_for(self, memory_id: str) -> list[FeedbackRecord]:
        """Host-side inspection of the feedback lane for one memory (like ``edges_of``)."""
        return [f for f in self._feedback if f.memory_id == memory_id]


class QuarantinedMemoryClient:
    """The confined ``MemoryClient`` the membrane leases into the agent container.

    Every write is forced into the quarantine tier AND to ``on_recall`` delivery —
    the caller-supplied ``trust_tier`` and ``delivery_mode`` are intentionally
    ignored, so a hijacked agent can neither mint a trusted memory nor **pin its
    own write** as an always-loaded standing instruction (#88). There is no promote
    path here; graduating a quarantined write into the trusted backbone is host-side
    only (``LocalMemoryClient.promote``), gated by graduation HITL (#15). This
    mirrors the ``write_approved`` / broker write-lock posture (#12/#20): the
    agent's self-asserted trust/delivery is never honoured. ``recall`` /
    ``load_pinned`` / ``create_edge`` delegate unchanged — confinement is on the
    write path only (reading the host-curated pinned set is safe).
    """

    def __init__(self, inner: MemoryClient) -> None:
        self._inner = inner

    async def remember(
        self,
        text: str,
        *,
        tags: tuple[str, ...] = (),
        trust_tier: str = TRUSTED_TIER,
        delivery_mode: str = ON_RECALL_DELIVERY,
    ) -> str:
        # trust_tier AND delivery_mode are accepted (protocol parity) but IGNORED —
        # fail-closed: the agent's writes always land quarantined and never pinned,
        # regardless of what it requests. Pinning a guardrail is host-side only.
        return await self._inner.remember(
            text, tags=tags, trust_tier=QUARANTINE_TIER, delivery_mode=ON_RECALL_DELIVERY
        )

    async def recall(
        self,
        query: str,
        *,
        trusted_only: bool = False,
        tags: tuple[str, ...] = (),
    ) -> list[Memory]:
        return await self._inner.recall(query, trusted_only=trusted_only, tags=tags)

    async def load_pinned(self) -> list[Memory]:
        # Read path: the pinned set is host-curated (the agent cannot pin), so
        # exposing it to the confined agent is safe and is how guardrails reach it.
        return await self._inner.load_pinned()

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


#: Subprocess timeout for the `kagura auth token` probe. The real kagura CLI is
#: heavy (kagura-ai + crypto + google-auth imports) and mints/refreshes the token
#: over the network, so a cold call measures ~25-30s on Windows — a 15s cap timed
#: out *every* time and falsely reported memory unreachable, making the agent
#: unusable. Sized with generous headroom over the observed latency; override via
#: KAGURA_MEMORY_PROBE_TIMEOUT for an unusually slow host. The gate stays
#: fail-closed (a real timeout still returns False).
_TOKEN_PROBE_TIMEOUT_SEC = 60


def _token_probe_timeout() -> float:
    """The probe timeout (seconds), env-overridable, fail-safe to the default."""
    import os

    raw = os.environ.get("KAGURA_MEMORY_PROBE_TIMEOUT", "").strip()
    if not raw:
        return _TOKEN_PROBE_TIMEOUT_SEC
    try:
        value = float(raw)
    except ValueError:
        return _TOKEN_PROBE_TIMEOUT_SEC
    return value if value > 0 else _TOKEN_PROBE_TIMEOUT_SEC


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
            timeout=_token_probe_timeout(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and bool(proc.stdout.strip())
