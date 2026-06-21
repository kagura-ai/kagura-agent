"""#165 S2: the OutcomeReinforcer — a verified outcome reinforces its grounding.

The producer side of the retrieval-feedback lane: after the host MEASURE step yields
a :class:`VerifiedOutcome`, this records, for each memory that grounded the run, an
*independent* helpful/unhelpful verdict via the host-only ``record_feedback`` (#90) —
turning a real, forge-resistant outcome into a recall-ranking signal.

Two hard rules, both fail-closed:

- **Only an independent verdict reinforces.** An UNVERIFIED (abstained) outcome —
  ``source == "unverified"``, i.e. no exit code and no operator approval — writes
  ZERO feedback. The agent merely finishing must never up-rank its own grounding
  (the CSO-M1 "verified, never self-report" rail, applied to the ranking signal).
- **Host-side only.** ``record_feedback`` is off the agent surface
  (``QuarantinedMemoryClient`` has no such verb), so a confined/hijacked agent can
  never drive this lane; the reinforcer runs in the trusted host, like the erasure
  cascade.

A verified *failure* (exit code != 0) down-ranks (``helpful=False``) — the grounding
led to a failed run. Best-effort over the source set: a memory erased between
grounding and reinforce is skipped (``record_feedback`` is fail-closed on an unknown
id), never a crash mid-loop.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Protocol, runtime_checkable

from kagura_agent.membrane.verified_outcome import VerifiedOutcome
from kagura_agent.patterns.erasure import ProvenanceLog
from kagura_agent.patterns.measure import measure_outcome

#: Config key (host env) holding the verify-check command; unset => no independent
#: verdict on the run path, so reinforcement stays unwired (#168).
VERIFY_CHECK_ENV = "KAGURA_AGENT_VERIFY_CHECK"
#: Optional config key for the graduation category of a verified run (default "run").
VERIFY_CATEGORY_ENV = "KAGURA_AGENT_VERIFY_CATEGORY"
_DEFAULT_CATEGORY = "run"


@runtime_checkable
class FeedbackSink(Protocol):
    """The host-side memory verbs the reinforcer needs — present (sync) on
    ``LocalMemoryClient`` and ``SqliteMemoryClient``, and deliberately OFF the agent
    surface (the ``MemoryClient`` protocol / ``QuarantinedMemoryClient`` have neither),
    so a confined/hijacked agent can never drive the ranking lane. ``runtime_checkable``
    so the run path can ``isinstance``-gate which backends reinforce — the async cloud
    ``McpMemoryClient`` lacks ``has_memory`` and is excluded (a separate async arm)."""

    def has_memory(self, memory_id: str) -> bool: ...

    def record_feedback(self, memory_id: str, query: str, *, helpful: bool) -> None: ...


class OutcomeReinforcer:
    """Records a verified outcome's independent verdict against its grounding memories.

    Typed against the :class:`FeedbackSink` protocol (``LocalMemoryClient`` /
    ``SqliteMemoryClient``), not the narrow ``MemoryClient`` protocol: ``record_feedback``
    / ``has_memory`` are host-side only and deliberately off the agent surface.
    """

    def __init__(self, memory: FeedbackSink) -> None:
        self._memory = memory

    def reinforce(
        self, outcome: VerifiedOutcome, *, query: str, source_memory_ids: Iterable[str]
    ) -> int:
        """Record ``outcome``'s verdict for each grounding memory; return the count.

        Writes nothing (returns 0) for an UNVERIFIED outcome. Skips any source id that
        no longer exists (best-effort), so an erasure between grounding and reinforce
        cannot crash the loop.
        """
        if outcome.source == "unverified":
            return 0
        written = 0
        for memory_id in source_memory_ids:
            if not self._memory.has_memory(memory_id):
                continue  # grounded memory erased before reinforce — skip, don't crash
            self._memory.record_feedback(memory_id, query, helpful=outcome.verified)
            written += 1
        return written


def reinforce_run(
    memory: FeedbackSink,
    provenance: ProvenanceLog,
    *,
    session_id: str,
    category: str,
    query: str,
    exit_code: int | None = None,
    approved: bool | None = None,
) -> VerifiedOutcome:
    """Close the loop for one finished run: MEASURE its outcome from an independent
    verdict, then reinforce the memories that grounded it. Returns the
    :class:`VerifiedOutcome` (for logging / a later graduation step).

    Host-side composition of ``measure_outcome`` + :class:`OutcomeReinforcer` over the
    session's recorded grounding. With no verdict supplied the outcome is UNVERIFIED
    and reinforcement is a no-op (zero feedback) — the fail-closed default.
    """
    outcome = measure_outcome(
        category,
        session_id=session_id,
        provenance=provenance,
        exit_code=exit_code,
        approved=approved,
    )
    OutcomeReinforcer(memory).reinforce(
        outcome, query=query, source_memory_ids=provenance.memories_for(session_id)
    )
    return outcome


def verify_and_reinforce(
    memory: FeedbackSink,
    provenance: ProvenanceLog,
    env: Mapping[str, str],
    *,
    session_id: str,
    query: str,
    run_check: Callable[[str], int],
) -> VerifiedOutcome | None:
    """The config-key independent-verdict arm for the run path.

    If ``VERIFY_CHECK_ENV`` is set, run that check host-side (via the injected
    ``run_check``: command -> exit code) after a finished run and reinforce the run's
    grounding with the exit-code verdict, returning the :class:`VerifiedOutcome`. With
    no check configured, returns ``None`` and writes nothing — the run stays UNVERIFIED
    and reinforcement stays unwired (#168). ``run_check`` is injected so the subprocess
    edge stays out of the unit-tested path.
    """
    check = env.get(VERIFY_CHECK_ENV, "").strip()
    if not check:
        return None
    category = env.get(VERIFY_CATEGORY_ENV, "").strip() or _DEFAULT_CATEGORY
    exit_code = run_check(check)
    return reinforce_run(
        memory,
        provenance,
        session_id=session_id,
        category=category,
        query=query,
        exit_code=exit_code,
    )
