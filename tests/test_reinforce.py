"""#165 S2: the OutcomeReinforcer — a verified outcome reinforces its grounding.

Only an INDEPENDENT verdict (exit code / HITL) reinforces the source memories that
grounded a run, via the host-only ``record_feedback``; an UNVERIFIED (abstained)
outcome writes ZERO feedback — the agent merely finishing must never up-rank its own
grounding. A verified failure down-ranks (helpful=False). Best-effort: a source memory
erased before reinforce is skipped, not a crash.
"""

from kagura_agent.mcp.memory_cloud import TRUSTED_TIER, LocalMemoryClient
from kagura_agent.membrane.verified_outcome import VerifiedOutcome
from kagura_agent.patterns.erasure import ProvenanceLog
from kagura_agent.patterns.reinforce import OutcomeReinforcer, reinforce_run


async def test_verified_outcome_records_helpful_feedback_for_each_source() -> None:
    memory = LocalMemoryClient()
    m1 = await memory.remember("a", trust_tier=TRUSTED_TIER)
    m2 = await memory.remember("b", trust_tier=TRUSTED_TIER)
    reinforcer = OutcomeReinforcer(memory)
    outcome = VerifiedOutcome.from_exit_code(0, "tests", input_trust=TRUSTED_TIER)

    written = reinforcer.reinforce(outcome, query="q", source_memory_ids=[m1, m2])

    assert written == 2
    assert [r.helpful for r in memory.feedback_for(m1)] == [True]
    assert [r.helpful for r in memory.feedback_for(m2)] == [True]


async def test_unverified_outcome_writes_zero_feedback() -> None:
    # The core safety property: an abstained run (no independent verdict) never
    # up-ranks its own grounding.
    memory = LocalMemoryClient()
    m1 = await memory.remember("a", trust_tier=TRUSTED_TIER)
    reinforcer = OutcomeReinforcer(memory)
    outcome = VerifiedOutcome.unverified("research", input_trust=TRUSTED_TIER)

    written = reinforcer.reinforce(outcome, query="q", source_memory_ids=[m1])

    assert written == 0
    assert memory.feedback_for(m1) == []


async def test_verified_failure_records_unhelpful_feedback() -> None:
    memory = LocalMemoryClient()
    m1 = await memory.remember("a", trust_tier=TRUSTED_TIER)
    reinforcer = OutcomeReinforcer(memory)
    outcome = VerifiedOutcome.from_exit_code(1, "tests", input_trust=TRUSTED_TIER)

    reinforcer.reinforce(outcome, query="q", source_memory_ids=[m1])

    assert [r.helpful for r in memory.feedback_for(m1)] == [False]


async def test_hitl_approval_reinforces() -> None:
    memory = LocalMemoryClient()
    m1 = await memory.remember("a", trust_tier=TRUSTED_TIER)
    reinforcer = OutcomeReinforcer(memory)
    outcome = VerifiedOutcome.from_hitl_approval(True, "research", input_trust=TRUSTED_TIER)

    reinforcer.reinforce(outcome, query="q", source_memory_ids=[m1])

    assert [r.helpful for r in memory.feedback_for(m1)] == [True]


async def test_denied_hitl_records_unhelpful_feedback() -> None:
    # A human rejection is a valid independent verdict, NOT an abstention: it must
    # down-rank (helpful=False), distinct from both the exit-code failure path and
    # the UNVERIFIED zero-write path.
    memory = LocalMemoryClient()
    m1 = await memory.remember("a", trust_tier=TRUSTED_TIER)
    reinforcer = OutcomeReinforcer(memory)
    outcome = VerifiedOutcome.from_hitl_approval(False, "research", input_trust=TRUSTED_TIER)

    written = reinforcer.reinforce(outcome, query="q", source_memory_ids=[m1])

    assert written == 1
    assert [r.helpful for r in memory.feedback_for(m1)] == [False]


async def test_forgotten_source_memory_is_skipped() -> None:
    # Best-effort: a grounded memory erased before reinforce is skipped, not a crash.
    memory = LocalMemoryClient()
    m1 = await memory.remember("a", trust_tier=TRUSTED_TIER)
    reinforcer = OutcomeReinforcer(memory)
    outcome = VerifiedOutcome.from_exit_code(0, "tests", input_trust=TRUSTED_TIER)

    written = reinforcer.reinforce(outcome, query="q", source_memory_ids=[m1, "ghost-id"])

    assert written == 1
    assert len(memory.feedback_for(m1)) == 1


async def test_empty_sources_writes_nothing() -> None:
    memory = LocalMemoryClient()
    reinforcer = OutcomeReinforcer(memory)
    outcome = VerifiedOutcome.from_exit_code(0, "tests", input_trust=TRUSTED_TIER)
    assert reinforcer.reinforce(outcome, query="q", source_memory_ids=[]) == 0


# --- reinforce_run: MEASURE + reinforce a finished run (host-side) -------------


async def test_reinforce_run_measures_and_reinforces_the_grounding() -> None:
    memory = LocalMemoryClient()
    m1 = await memory.remember("a", trust_tier=TRUSTED_TIER)
    provenance = ProvenanceLog()
    provenance.record_grounding("s", [(m1, TRUSTED_TIER)])

    outcome = reinforce_run(
        memory, provenance, session_id="s", category="tests", query="q", exit_code=0
    )

    assert outcome.verified is True
    assert outcome.input_trust == TRUSTED_TIER
    assert [r.helpful for r in memory.feedback_for(m1)] == [True]


async def test_reinforce_run_unverified_writes_no_feedback() -> None:
    memory = LocalMemoryClient()
    m1 = await memory.remember("a", trust_tier=TRUSTED_TIER)
    provenance = ProvenanceLog()
    provenance.record_grounding("s", [(m1, TRUSTED_TIER)])

    outcome = reinforce_run(memory, provenance, session_id="s", category="research", query="q")

    assert outcome.source == "unverified"
    assert memory.feedback_for(m1) == []


async def test_reinforce_run_input_trust_comes_from_provenance() -> None:
    # A run grounded on a quarantine-tier source is untrusted-input even when the
    # check passes (Δ2); the verdict is still reinforced onto the grounding.
    memory = LocalMemoryClient()
    m1 = await memory.remember("a", trust_tier=TRUSTED_TIER)
    provenance = ProvenanceLog()
    provenance.record_grounding("s", [(m1, "quarantine")])

    outcome = reinforce_run(
        memory, provenance, session_id="s", category="tests", query="q", exit_code=0
    )

    assert outcome.verified is True
    assert outcome.input_trust == "untrusted"
    assert [r.helpful for r in memory.feedback_for(m1)] == [True]


async def test_reinforce_run_failure_down_ranks_the_grounding() -> None:
    # End-to-end: a verified FAILURE (non-zero exit) down-ranks, never up-ranks.
    memory = LocalMemoryClient()
    m1 = await memory.remember("a", trust_tier=TRUSTED_TIER)
    provenance = ProvenanceLog()
    provenance.record_grounding("s", [(m1, TRUSTED_TIER)])

    outcome = reinforce_run(
        memory, provenance, session_id="s", category="tests", query="q", exit_code=1
    )

    assert outcome.verified is False
    assert [r.helpful for r in memory.feedback_for(m1)] == [False]


async def test_reinforce_run_skips_a_grounding_forgotten_from_the_store() -> None:
    # memories_for can return an id still in the provenance log but erased from the
    # store; reinforce_run must skip it (has_memory guard), not crash mid-loop.
    memory = LocalMemoryClient()
    m1 = await memory.remember("a", trust_tier=TRUSTED_TIER)
    m2 = await memory.remember("b", trust_tier=TRUSTED_TIER)
    provenance = ProvenanceLog()
    provenance.record_grounding("s", [(m1, TRUSTED_TIER), (m2, TRUSTED_TIER)])
    memory.forget(m2)  # erased from the store, still in the provenance log

    outcome = reinforce_run(
        memory, provenance, session_id="s", category="tests", query="q", exit_code=0
    )

    assert outcome.verified is True
    assert [r.helpful for r in memory.feedback_for(m1)] == [True]  # m1 reinforced
    assert memory.feedback_for(m2) == []  # m2 skipped, no crash
