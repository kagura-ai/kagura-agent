"""#165 S2: the OutcomeReinforcer — a verified outcome reinforces its grounding.

Only an INDEPENDENT verdict (exit code / HITL) reinforces the source memories that
grounded a run, via the host-only ``record_feedback``; an UNVERIFIED (abstained)
outcome writes ZERO feedback — the agent merely finishing must never up-rank its own
grounding. A verified failure down-ranks (helpful=False). Best-effort: a source memory
erased before reinforce is skipped, not a crash.
"""

from pathlib import Path

from kagura_agent.mcp.mcp_memory import McpMemoryClient
from kagura_agent.mcp.memory_cloud import TRUSTED_TIER, LocalMemoryClient
from kagura_agent.mcp.memory_sqlite import SqliteMemoryClient
from kagura_agent.membrane.verified_outcome import VerifiedOutcome
from kagura_agent.patterns.erasure import ProvenanceLog
from kagura_agent.patterns.reinforce import (
    FeedbackSink,
    OutcomeReinforcer,
    reinforce_after_run,
    reinforce_run,
    verify_and_reinforce,
)


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


# --- FeedbackSink: the loop needs a PERSISTENT host-side sink ------------------


async def test_reinforcer_accepts_a_sqlite_backend(tmp_path: Path) -> None:
    # LocalMemoryClient is in-memory/throwaway; the loop only persists with a backend
    # like SqliteMemoryClient, which satisfies the FeedbackSink protocol.
    memory = SqliteMemoryClient(tmp_path / "mem.db")
    m1 = await memory.remember("a", trust_tier=TRUSTED_TIER)
    reinforcer = OutcomeReinforcer(memory)
    outcome = VerifiedOutcome.from_exit_code(0, "tests", input_trust=TRUSTED_TIER)

    reinforcer.reinforce(outcome, query="q", source_memory_ids=[m1])

    assert [r.helpful for r in memory.feedback_for(m1)] == [True]


def test_feedback_sink_includes_sync_backends_excludes_async_cloud() -> None:
    # Load-bearing: Local/Sqlite (sync) satisfy FeedbackSink; the async cloud
    # McpMemoryClient lacks has_memory and must be excluded — a sync reinforcer must
    # never be handed an async record_feedback.
    assert isinstance(LocalMemoryClient(), FeedbackSink)
    assert not hasattr(McpMemoryClient, "has_memory")


# --- verify_and_reinforce: the config-key arm ---------------------------------


async def test_verify_and_reinforce_no_check_configured_is_a_noop() -> None:
    memory = LocalMemoryClient()
    m1 = await memory.remember("a", trust_tier=TRUSTED_TIER)
    provenance = ProvenanceLog()
    provenance.record_grounding("s", [(m1, TRUSTED_TIER)])

    outcome = verify_and_reinforce(
        memory, provenance, {}, session_id="s", query="q", run_check=lambda cmd: 0
    )

    assert outcome is None  # no KAGURA_AGENT_VERIFY_CHECK -> loop stays unwired (#168)
    assert memory.feedback_for(m1) == []


async def test_verify_and_reinforce_blank_check_is_a_noop() -> None:
    # A blank-but-present env var (common from a deploy template) is treated as unset:
    # the check never runs and nothing is reinforced.
    memory = LocalMemoryClient()
    m1 = await memory.remember("a", trust_tier=TRUSTED_TIER)
    provenance = ProvenanceLog()
    provenance.record_grounding("s", [(m1, TRUSTED_TIER)])
    ran: list[str] = []

    outcome = verify_and_reinforce(
        memory,
        provenance,
        {"KAGURA_AGENT_VERIFY_CHECK": "   "},
        session_id="s",
        query="q",
        run_check=lambda cmd: ran.append(cmd) or 0,
    )

    assert outcome is None
    assert ran == []  # the check never ran
    assert memory.feedback_for(m1) == []


async def test_verify_and_reinforce_runs_the_check_and_reinforces() -> None:
    memory = LocalMemoryClient()
    m1 = await memory.remember("a", trust_tier=TRUSTED_TIER)
    provenance = ProvenanceLog()
    provenance.record_grounding("s", [(m1, TRUSTED_TIER)])
    ran: list[str] = []

    outcome = verify_and_reinforce(
        memory,
        provenance,
        {"KAGURA_AGENT_VERIFY_CHECK": "pytest -q"},
        session_id="s",
        query="q",
        run_check=lambda cmd: ran.append(cmd) or 0,
    )

    assert ran == ["pytest -q"]  # the configured check ran
    assert outcome is not None and outcome.verified is True
    assert [r.helpful for r in memory.feedback_for(m1)] == [True]


async def test_verify_and_reinforce_failed_check_down_ranks() -> None:
    memory = LocalMemoryClient()
    m1 = await memory.remember("a", trust_tier=TRUSTED_TIER)
    provenance = ProvenanceLog()
    provenance.record_grounding("s", [(m1, TRUSTED_TIER)])

    outcome = verify_and_reinforce(
        memory,
        provenance,
        {"KAGURA_AGENT_VERIFY_CHECK": "pytest"},
        session_id="s",
        query="q",
        run_check=lambda cmd: 1,
    )

    assert outcome is not None and outcome.verified is False
    assert [r.helpful for r in memory.feedback_for(m1)] == [False]


async def test_verify_and_reinforce_category_defaults_and_overrides() -> None:
    memory = LocalMemoryClient()
    provenance = ProvenanceLog()

    default = verify_and_reinforce(
        memory,
        provenance,
        {"KAGURA_AGENT_VERIFY_CHECK": "x"},
        session_id="s",
        query="q",
        run_check=lambda cmd: 0,
    )
    override = verify_and_reinforce(
        memory,
        provenance,
        {"KAGURA_AGENT_VERIFY_CHECK": "x", "KAGURA_AGENT_VERIFY_CATEGORY": "deploy"},
        session_id="s",
        query="q",
        run_check=lambda cmd: 0,
    )
    blank = verify_and_reinforce(
        memory,
        provenance,
        {"KAGURA_AGENT_VERIFY_CHECK": "x", "KAGURA_AGENT_VERIFY_CATEGORY": "  "},
        session_id="s",
        query="q",
        run_check=lambda cmd: 0,
    )

    assert default is not None and default.category == "run"
    assert override is not None and override.category == "deploy"
    assert blank is not None and blank.category == "run"  # blank-but-present -> default


# --- reinforce_after_run: the best-effort run-path hook -----------------------


async def test_reinforce_after_run_reinforces_a_sink_backend() -> None:
    memory = LocalMemoryClient()
    m1 = await memory.remember("a", trust_tier=TRUSTED_TIER)
    provenance = ProvenanceLog()
    provenance.record_grounding("s", [(m1, TRUSTED_TIER)])

    reinforce_after_run(
        memory,
        provenance,
        {"KAGURA_AGENT_VERIFY_CHECK": "x"},
        session_id="s",
        query="q",
        run_check=lambda c: 0,
    )

    assert [r.helpful for r in memory.feedback_for(m1)] == [True]


def test_reinforce_after_run_skips_a_non_sink_backend() -> None:
    # A backend without the host-side sync verbs (e.g. the async cloud client) is
    # skipped before the check even runs.
    def boom(_c: str) -> int:
        raise AssertionError("run_check must not be called for a non-sink backend")

    reinforce_after_run(
        object(),
        ProvenanceLog(),
        {"KAGURA_AGENT_VERIFY_CHECK": "x"},
        session_id="s",
        query="q",
        run_check=boom,
    )  # returns without raising


async def test_reinforce_after_run_is_best_effort_on_check_error() -> None:
    # A check that cannot spawn must not turn a completed run into a crash (logged,
    # not raised) — and records nothing.
    memory = LocalMemoryClient()
    m1 = await memory.remember("a", trust_tier=TRUSTED_TIER)
    provenance = ProvenanceLog()
    provenance.record_grounding("s", [(m1, TRUSTED_TIER)])

    def boom(_c: str) -> int:
        raise OSError("cannot spawn check")

    reinforce_after_run(
        memory,
        provenance,
        {"KAGURA_AGENT_VERIFY_CHECK": "x"},
        session_id="s",
        query="q",
        run_check=boom,
    )  # logged, not raised

    assert memory.feedback_for(m1) == []
