"""The agent runtime's view of memory-cloud — deliberately narrow.

The runtime gets append (`remember`), scoped read (`recall`, with a trust-tier
filter), one-call session start (`get_agent_bootstrap`), and `create_edge` (to
record `prevents` relationships for failure learning). It gets **no admin**
(delete/forget/merge/rollback/schema): a
prompt-injected agent must not be able to amplify a hijack into destructive
writes. `LocalMemoryClient` is the self-host backend (here in-memory; SQLite in
deployment); a real deployment composes MCP memory I/O with trusted-host REST
bootstrap behind the same surface.
"""

from __future__ import annotations

import itertools
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

#: The tier an agent's writes land in by default (#15). Read-side recall filters
#: it out of the trusted backbone (``trusted_only=True``), so a quarantined write
#: cannot pollute trusted memory until a host-side promote graduates it.
QUARANTINE_TIER = "quarantine"
TRUSTED_TIER = "trusted"

#: #165 S3 bounded boost: net-helpful feedback can move a memory's recall rank by at
#: most this many net votes — so reinforcement can't dominate match or drive runaway
#: monoculture (Δ4). Eval-tunable; the default-ON flip is gated on the #166 outcome eval.
RERANK_BOUND = 3

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
class AgentBootstrap:
    """Normalized, model-safe session-start context from ``get_agent_bootstrap``.

    Transport/audit metadata stays outside this object.  Every memory here is
    already proven to come from a trusted bootstrap lane; component failures are
    retained separately so callers can use the healthy fail-soft components while
    recording degradation.

    ``state`` is agent-scoped advisory context.  It never replaces the
    session-scoped :class:`CheckpointStore` used by ``Session`` resume.
    """

    agent_id: str | None
    context_id: str | None
    instructions: str
    pinned: tuple[Memory, ...]
    recall: tuple[Memory, ...]
    upcoming: tuple[Memory, ...]
    state: dict[str, Any]
    policy: dict[str, Any] | None
    degraded: bool
    component_failures: tuple[str, ...]
    component_statuses: tuple[tuple[str, str], ...]


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

    async def get_agent_bootstrap(
        self,
        *,
        session_id: str,
        query: str,
        recall_k: int = 5,
    ) -> AgentBootstrap:
        """Return one normalized session-start bundle.

        Cloud implementations call the production agent bootstrap REST endpoint
        once from the trusted host. Local implementations compose the same trusted
        pinned/recall lanes behind this method so callers never fan out themselves.
        """
        ...

    async def create_edge(self, src_id: str, dst_id: str, *, type: str) -> None: ...


class LocalMemoryClient:
    """Self-host backend. No admin verbs by construction."""

    def __init__(
        self,
        *,
        rerank_feedback: bool = False,
        explore_epsilon: float = 0.0,
        explore_seed: int | None = None,
    ) -> None:
        self._memories: dict[str, Memory] = {}
        self._edges: dict[str, list[tuple[str, str]]] = {}
        # #90: retrieval-quality side lane, keyed by memory_id like _edges — O(1)
        # record + lookup, and the scan stays bounded to one memory's records.
        self._feedback: dict[str, list[FeedbackRecord]] = {}
        self._ids = itertools.count(1)
        # #165 S3: bounded recall re-rank by verified feedback. DEFAULT-OFF keeps recall
        # byte-for-byte unchanged; the default-ON flip is gated on the #166 outcome eval
        # (the Δ4 research: the reinforcement mechanism is unproven until measured).
        self._rerank_feedback = rerank_feedback
        # #165 S3 Δ4 EXPLORATION FLOOR (positivity): with explore_epsilon>0, each candidate
        # has that per-recall probability of surfacing regardless of its feedback — so a
        # down-ranked memory keeps a nonzero chance to be recalled and re-evaluated rather
        # than buried forever (the feedback-loop-collapse guard). DEFAULT 0.0 (off, no RNG
        # draw -> deterministic); a nonzero floor is REQUIRED before the default-ON flip,
        # with #166 tuning the value. The RNG is seedable for reproducible tests/eval.
        self._explore_epsilon = explore_epsilon
        self._rng = random.Random(explore_seed)

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
        if self._rerank_feedback:
            # #165 S3 (default-OFF): surface verified-useful memories first by net-helpful
            # feedback clamped to ±RERANK_BOUND (bounded boost). STABLE sort, so score-0
            # memories keep insertion order and every match is still returned. The
            # exploration floor (_rerank_key) gives each candidate an explore_epsilon
            # chance to surface regardless of feedback, so a demoted memory can re-surface
            # (Δ4 positivity); with epsilon 0 the key is the plain deterministic score.
            # Per-memory attribution (vs the bounded grounded set) is the remaining Δ4
            # guardrail before the default-ON flip (#166). Feedback is the sole signal
            # over the binary text match; combining with match strength is future.
            results.sort(key=self._rerank_key, reverse=True)
        return results

    def _rerank_key(self, mem: Memory) -> int:
        # Exploration floor (Δ4): with probability explore_epsilon, surface this memory at
        # the top tier regardless of feedback (IPS positivity — a demoted memory keeps a
        # nonzero chance to be recalled and re-evaluated). Short-circuits when epsilon is 0
        # (no RNG draw), so the unexplored re-rank stays deterministic.
        if self._explore_epsilon and self._rng.random() < self._explore_epsilon:
            return RERANK_BOUND
        return self._feedback_score(mem.id)

    def _feedback_score(self, memory_id: str) -> int:
        """Net-helpful feedback for a memory, clamped to ±RERANK_BOUND (the bounded
        boost): each helpful record is +1, each unhelpful -1."""
        score = sum(1 if f.helpful else -1 for f in self._feedback.get(memory_id, []))
        return max(-RERANK_BOUND, min(RERANK_BOUND, score))

    async def load_pinned(self) -> list[Memory]:
        # Deterministic + unranked: the COMPLETE pinned set, insertion order, every
        # call. No query, no ranking, no trust filter — pinned guardrails are
        # host-curated and load whole (#88).
        return [m for m in self._memories.values() if m.delivery_mode == ALWAYS_DELIVERY]

    async def get_agent_bootstrap(
        self,
        *,
        session_id: str,
        query: str,
        recall_k: int = 5,
    ) -> AgentBootstrap:
        return await compose_agent_bootstrap(
            self, session_id=session_id, query=query, recall_k=recall_k
        )

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

        The lane is an **append-only journal** (like the ``_edges`` journal): a
        re-recorded ``(memory_id, query)`` is kept as a second record, not
        deduplicated or overwritten. A consumer that needs a single verdict must
        define its own reduction (e.g. weigh all records / last-wins) — this store
        deliberately keeps every datapoint.
        """
        if memory_id not in self._memories:
            raise KeyError(memory_id)
        self._feedback.setdefault(memory_id, []).append(
            FeedbackRecord(memory_id=memory_id, query=query, helpful=helpful)
        )

    def feedback_for(self, memory_id: str) -> list[FeedbackRecord]:
        """Host-side inspection of the feedback lane for one memory — O(1) keyed
        lookup, returns a copy (like ``edges_of``)."""
        return list(self._feedback.get(memory_id, []))

    def has_memory(self, memory_id: str) -> bool:
        """Host-side existence check (no admin leak — it reveals nothing the agent
        couldn't already learn via ``recall``). Lets the erasure cascade fail-closed
        on a bogus source id *before* it deletes any derived artifact (#93)."""
        return memory_id in self._memories

    def forget(self, memory_id: str) -> None:
        """Host-side ONLY: erase a memory and its host-side derived records (#93).

        Deliberately **NOT** on the ``MemoryClient`` protocol (the agent surface) —
        like ``promote`` / ``record_feedback``. The narrow runtime client holds no
        erasure verb on purpose (a prompt-injected agent must not amplify a hijack
        into destructive deletes); erasure is a host-side act, typically the
        agent-side half of a memory-cloud ``forget`` cascade. Removes the memory, its
        outgoing edges, any edges pointing AT it (no dangling refs), and its feedback
        lane. Unknown ``memory_id`` raises ``KeyError`` — fail-closed, no silent no-op
        that could mask an incomplete erasure (mirrors ``promote``).
        """
        del self._memories[memory_id]  # KeyError if unknown — fail-closed
        self._edges.pop(memory_id, None)
        self._feedback.pop(memory_id, None)
        # Drop dangling edges that pointed AT the erased memory, so a later
        # edges_of()/traversal never resolves a tombstone.
        for src, dsts in self._edges.items():
            kept = [(dst, etype) for (dst, etype) in dsts if dst != memory_id]
            if len(kept) != len(dsts):
                self._edges[src] = kept

    def ids_with_tag(self, tag: str) -> list[str]:
        """Host-side: ids of memories carrying ``tag`` (insertion order).

        Used by the erasure cascade to find a session's derived outcome-summaries
        (tagged ``session:<id>``). Host-side — not on the agent protocol, like the
        other admin-adjacent verbs."""
        return [mid for mid, mem in self._memories.items() if tag in mem.tags]


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

    async def get_agent_bootstrap(
        self,
        *,
        session_id: str,
        query: str,
        recall_k: int = 5,
    ) -> AgentBootstrap:
        # Read-only, server-trusted bootstrap data; confinement remains solely on
        # writes, exactly like recall/load_pinned.
        return await self._inner.get_agent_bootstrap(
            session_id=session_id, query=query, recall_k=recall_k
        )

    async def create_edge(self, src_id: str, dst_id: str, *, type: str) -> None:
        await self._inner.create_edge(src_id, dst_id, type=type)


async def compose_agent_bootstrap(
    memory: MemoryClient,
    *,
    session_id: str,
    query: str,
    recall_k: int = 5,
) -> AgentBootstrap:
    """Local/SQLite parity implementation of the server bootstrap contract.

    This helper is deliberately behind ``MemoryClient.get_agent_bootstrap``.  A
    runtime caller still performs one method call; only the cloud adapter turns it
    into one network request.  Local backends have no agent-state/upcoming/policy
    stores, so those components are explicit healthy-empty/skipped lanes.
    """
    del session_id  # correlation-only on the production service
    if not 1 <= recall_k <= 100:
        raise ValueError("recall_k must be in [1, 100]")
    pinned = tuple(
        memory_item
        for memory_item in await memory.load_pinned()
        if memory_item.trust_tier == TRUSTED_TIER
    )
    recalled = tuple((await memory.recall(query, trusted_only=True))[:recall_k])
    return AgentBootstrap(
        agent_id=None,
        context_id=None,
        instructions="",
        pinned=pinned,
        recall=recalled,
        upcoming=(),
        state={},
        policy=None,
        degraded=False,
        component_failures=(),
        component_statuses=(
            ("pinned", "ok"),
            ("recall", "ok"),
            ("upcoming", "ok"),
            ("state", "ok"),
            ("policy", "skipped"),
        ),
    )


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
    # Reject non-finite (inf/nan): inf would pass `> 0` and make subprocess.run wait
    # forever, defeating the fail-closed timeout this value exists to enforce.
    return value if (math.isfinite(value) and value > 0) else _TOKEN_PROBE_TIMEOUT_SEC


#: How many times the reachability gate retries the `kagura auth token` probe
#: before refusing. The access token is short-lived (~1h); the first run after it
#: expires forces a refresh, and a single transient hiccup at that hourly boundary
#: — a slow cold start near the timeout, a momentary network blip, a transient 401
#: mid-refresh — would otherwise hard-refuse the run on a one-shot probe. Bounded
#: retry absorbs the transient case; the gate still fails closed once exhausted.
#: Override via KAGURA_MEMORY_PROBE_ATTEMPTS.
_PROBE_ATTEMPTS = 3

#: Seconds between probe attempts. Kept small — the point is to ride out a momentary
#: blip / let an in-flight refresh settle, not to mask a real outage (which still
#: fails closed after the attempts run out). Override via KAGURA_MEMORY_PROBE_BACKOFF.
_PROBE_BACKOFF_SEC = 1.5


def _probe_attempts() -> int:
    """Probe attempt count, env-overridable (KAGURA_MEMORY_PROBE_ATTEMPTS), >= 1."""
    import os

    raw = os.environ.get("KAGURA_MEMORY_PROBE_ATTEMPTS", "").strip()
    if not raw:
        return _PROBE_ATTEMPTS
    try:
        value = int(raw)
    except ValueError:
        return _PROBE_ATTEMPTS
    return value if value >= 1 else _PROBE_ATTEMPTS


def _probe_backoff() -> float:
    """Backoff seconds between probe attempts, env-overridable, fail-safe to default."""
    import os

    raw = os.environ.get("KAGURA_MEMORY_PROBE_BACKOFF", "").strip()
    if not raw:
        return _PROBE_BACKOFF_SEC
    try:
        value = float(raw)
    except ValueError:
        return _PROBE_BACKOFF_SEC
    # Reject non-finite (inf/nan): inf would pass `>= 0` and make _sleep(inf) hang the
    # gate forever between attempts — the opposite of riding out a momentary blip.
    return value if (math.isfinite(value) and value >= 0) else _PROBE_BACKOFF_SEC


def _probe_token_once() -> bool:  # pragma: no cover - shells out to the kagura CLI
    """One `kagura auth token` call: True iff it exits 0 AND prints a token.

    A zero exit with empty stdout (e.g. a CLI that no-ops) is treated as a miss so
    the gate stays fail-closed; an OSError / subprocess error (incl. timeout) is a
    miss too. This is the only part of the gate that touches the CLI.
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


def memory_reachable(
    *,
    attempts: int | None = None,
    _probe: Callable[[], bool] = _probe_token_once,
    _sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Whether memory is reachable: can the host mint a token via the CLI?

    Asks the CLI for a short-lived access token (`kagura auth token`) and treats a
    success (exit 0 + non-empty token) as reachable. The check is retried up to
    ``attempts`` times (default: ``_probe_attempts()``) with ``_probe_backoff()``
    seconds between attempts: the access token is ~1h and the first run after expiry
    forces a refresh, so a single transient failure at that boundary (slow cold
    start, a momentary network blip, a transient 401 mid-refresh) must NOT
    hard-refuse the run. Once the attempts are exhausted the gate stays
    **fail-closed** (returns False) — a real outage or a logged-out host still
    refuses, with no silent degrade. The retry IS the handling of the
    "expired-but-refreshable" case: a refresh that needs a moment succeeds on a
    later attempt.

    **Latency note.** Each attempt can cost up to ``_token_probe_timeout()`` (60s),
    so retry multiplies the worst case on a *hung* outage. The run path opts into
    that to absorb transients; latency-sensitive *diagnostic* callers (doctor) pass
    ``attempts=1`` for a one-shot, fast probe. ``attempts`` is clamped to >= 1 so a
    stray 0/negative can never turn the loop into an unconditional "unreachable".

    The minted token is intentionally **not** threaded back into the process: this
    is a reachability gate only, and the live memory path (the kagura CLI / MCP
    proxy) owns its own refresh-aware token cache, so there is no in-process
    consumer to hand the token to — re-checking reachability per run is deliberate.

    ``_probe`` / ``_sleep`` are injected so the bounded-retry logic is unit-testable
    without shelling out (the real probe, ``_probe_token_once``, is the only part
    that touches the CLI).
    """
    n = _probe_attempts() if attempts is None else attempts
    n = max(1, n)
    backoff = _probe_backoff()
    for attempt in range(n):
        if _probe():
            return True
        if attempt + 1 < n:
            _sleep(backoff)
    return False
