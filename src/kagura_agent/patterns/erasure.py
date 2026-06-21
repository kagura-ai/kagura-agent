"""Host-side erasure cascade (#93): a memory-cloud ``forget`` must reach the
agent-side artifacts derived from a memory.

memory-cloud owns the *primary* erasure — the memory plus its server-side
embeddings/edges. But the agent derives its OWN artifacts from recalled
memories: session **checkpoints** (``patterns.checkpoint``) and
**outcome-summaries** (``continuity.remember_outcome``). A server-side ``forget``
never reaches those, so for a GDPR erasure (CSO finding C1 / ``docs/legal.md``
§3) to be complete the cascade has to delete them too.

**Host-side only, by construction.** The narrow ``MemoryClient`` the agent runs
against has no erasure verb (a hijack must not amplify into destructive deletes —
confinement by omission, like ``promote`` / ``record_feedback``). The cascade is
an operator/host action driven off a provenance trail the host records; it is
intentionally absent from the agent surface and from ``QuarantinedMemoryClient``.

Relationship to memory-cloud's own ``forget``: this does **not** reimplement it.
The server-side ``forget`` (the SDK contract) erases the primary memory and its
embeddings/edges in the backbone; ``forget_cascade`` is the agent-side companion
that erases the *derived* artifacts this process created. Run both for a complete
erasure; reference the SDK contract for the server-side scope rather than
duplicating it here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from kagura_agent.mcp.memory_cloud import LocalMemoryClient
from kagura_agent.patterns.checkpoint import CheckpointStore


@dataclass
class ProvenanceLog:
    """Host-side record of which source memories fed which sessions.

    Populated as grounding injects recalled memories into a session's prompt
    (``continuity.ground_and_run``). It is the bridge a ``forget(memory_id)`` needs
    to reach the *session-keyed* derived artifacts (checkpoints, outcome-summaries)
    that may carry the memory's content. NOT exposed to the agent — a host-side
    erasure-support structure, like ``LocalMemoryClient.promote``.

    It also captures, per session, the *trust tier* of each grounding memory
    (``record_grounding`` / ``tiers_for``) — the host evidence a run's ``input_trust``
    is derived from. Without the real tiers the input-trust gate is vacuous: a
    trusted-only read path makes "all grounding was trusted" unconditionally true.
    """

    _by_memory: dict[str, set[str]] = field(default_factory=dict)
    _tiers_by_session: dict[str, list[str]] = field(default_factory=dict)

    def record(self, session_id: str, memory_ids: Iterable[str]) -> None:
        """Note that ``session_id`` consumed each of ``memory_ids`` (idempotent)."""
        for mid in memory_ids:
            self._by_memory.setdefault(mid, set()).add(session_id)

    def record_grounding(self, session_id: str, sources: Iterable[tuple[str, str]]) -> None:
        """Record ``(memory_id, trust_tier)`` pairs that grounded ``session_id``.

        Populates BOTH the erasure trail (memory -> sessions, via ``record``) and the
        per-session tier provenance ``tiers_for`` exposes, so a run's ``input_trust``
        is derived from the memories' *actual* tiers rather than assumed. The
        grounding-site entry point; ``record`` stays the low-level ids-only primitive.

        Tiers are kept **distinct** (first-seen order): ``input_trust`` only asks "were
        all grounding memories trusted?", so re-recalling the same memory across a long
        session's turns must not grow the list unboundedly or skew tier multiplicities.
        """
        pairs = list(sources)
        self.record(session_id, [mid for mid, _ in pairs])
        for _, tier in pairs:
            tiers = self._tiers_by_session.setdefault(session_id, [])
            if tier not in tiers:
                tiers.append(tier)

    def tiers_for(self, session_id: str) -> tuple[str, ...]:
        """The distinct trust tiers that grounded ``session_id`` — a snapshot tuple.

        Immutable, so a caller cannot corrupt the log. Empty for a session with no
        recorded grounding — the fail-closed basis for ``input_trust`` (no provenance
        -> not trusted).
        """
        return tuple(self._tiers_by_session.get(session_id, ()))

    def sessions_for(self, memory_id: str) -> set[str]:
        """The sessions that consumed ``memory_id`` (a copy, never the live set)."""
        return set(self._by_memory.get(memory_id, set()))

    def memories_for(self, session_id: str) -> tuple[str, ...]:
        """The distinct memory ids that grounded ``session_id`` — the reverse of
        ``sessions_for``, so the host can reinforce a run's sources after a verified
        outcome. Empty for an unrecorded session."""
        return tuple(mid for mid, sessions in self._by_memory.items() if session_id in sessions)

    def forget_memory(self, memory_id: str) -> None:
        """Drop the source's provenance entry once it has been cascaded (idempotent)."""
        self._by_memory.pop(memory_id, None)

    def forget_session_tiers(self, session_id: str) -> None:
        """Drop a session's captured tier provenance once cascaded (idempotent)."""
        self._tiers_by_session.pop(session_id, None)


@dataclass(frozen=True)
class CascadeResult:
    """What an erasure cascade removed — returned for an audit trail (an erasure
    you cannot evidence is an erasure you cannot prove under GDPR)."""

    source_memory_id: str
    sessions: tuple[str, ...]
    forgotten_memory_ids: tuple[str, ...]
    deleted_checkpoints: tuple[str, ...]


async def forget_cascade(
    memory_id: str,
    *,
    memory: LocalMemoryClient,
    checkpoints: CheckpointStore,
    provenance: ProvenanceLog,
) -> CascadeResult:
    """Erase ``memory_id`` AND the agent-side artifacts derived from it.

    Steps (all host-side):

    1. Resolve every session that consumed the memory (``provenance``).
    2. For each session: delete its checkpoint, and forget its outcome-summary
       memories (tagged ``session:<id>``).
    3. Forget the source memory itself, and drop its provenance entry.

    **Fail-closed:** an unknown source ``memory_id`` raises ``KeyError`` *before*
    any deletion (mirrors ``promote`` / ``forget``) — a silent no-op could let an
    operator believe an erasure happened when it did not, and partially erasing the
    *derived* side for a bogus source id would be worse. The derived side is
    **idempotent** (a missing checkpoint / already-forgotten summary is fine), so a
    re-run after a partial failure completes the erasure.

    Typed against the concrete ``LocalMemoryClient`` (not the narrow ``MemoryClient``
    protocol) on purpose: the erasure verbs it uses (``forget`` / ``ids_with_tag`` /
    ``has_memory``) are host-side only and deliberately off the agent surface.
    """
    if not memory.has_memory(memory_id):  # fail-closed before touching anything
        raise KeyError(memory_id)
    sessions = sorted(provenance.sessions_for(memory_id))
    forgotten: list[str] = []
    deleted_checkpoints: list[str] = []
    for session_id in sessions:
        await checkpoints.delete(session_id)
        deleted_checkpoints.append(session_id)
        for summary_id in memory.ids_with_tag(f"session:{session_id}"):
            # Skip the source here — it is forgotten once, explicitly, below. A
            # *promoted* outcome-summary can be BOTH a session-tagged derived
            # artifact AND the erasure source; forgetting it in the loop and again
            # below would KeyError mid-cascade. The has_memory guard also makes a
            # summary that somehow surfaces under two sessions idempotent.
            if summary_id != memory_id and memory.has_memory(summary_id):
                memory.forget(summary_id)
                forgotten.append(summary_id)
        # The captured-tier provenance for this session is a derived artifact too.
        provenance.forget_session_tiers(session_id)
    # The source still exists: the top-level guard proved it and the loop skipped
    # it, so this final forget is safe and unconditional.
    memory.forget(memory_id)
    forgotten.append(memory_id)
    provenance.forget_memory(memory_id)
    return CascadeResult(
        source_memory_id=memory_id,
        sessions=tuple(sessions),
        forgotten_memory_ids=tuple(forgotten),
        deleted_checkpoints=tuple(deleted_checkpoints),
    )
