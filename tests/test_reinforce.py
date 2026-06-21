"""#165 S2: the OutcomeReinforcer — a verified outcome reinforces its grounding.

Only an INDEPENDENT verdict (exit code / HITL) reinforces the source memories that
grounded a run, via the host-only ``record_feedback``; an UNVERIFIED (abstained)
outcome writes ZERO feedback — the agent merely finishing must never up-rank its own
grounding. A verified failure down-ranks (helpful=False). Best-effort: a source memory
erased before reinforce is skipped, not a crash.
"""

from kagura_agent.mcp.memory_cloud import TRUSTED_TIER, LocalMemoryClient
from kagura_agent.membrane.verified_outcome import VerifiedOutcome
from kagura_agent.patterns.reinforce import OutcomeReinforcer


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
