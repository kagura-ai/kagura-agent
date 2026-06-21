"""#165 S3: the bounded, default-OFF recall re-rank that consumes verified feedback.

When enabled, recall surfaces verified-useful trusted memories above unproven ones by
net-helpful feedback, clamped to +-RERANK_BOUND (bounded boost). The sort is stable so
never-reinforced memories keep insertion order (the cold-start floor), and nothing is
excluded. Default-OFF: recall is byte-for-byte unchanged.
"""

from pathlib import Path

from kagura_agent.mcp.memory_cloud import RERANK_BOUND, TRUSTED_TIER, LocalMemoryClient
from kagura_agent.mcp.memory_sqlite import SqliteMemoryClient


async def _seed(memory: LocalMemoryClient, *texts: str) -> list[str]:
    return [await memory.remember(t, trust_tier=TRUSTED_TIER) for t in texts]


async def test_recall_is_unchanged_when_rerank_is_off() -> None:
    memory = LocalMemoryClient()  # default: rerank off
    m1, m2 = await _seed(memory, "alpha note", "alpha memo")
    memory.record_feedback(m2, "alpha", helpful=True)  # would promote m2 if reranked

    out = await memory.recall("alpha")

    assert [m.id for m in out] == [m1, m2]  # insertion order, feedback ignored


async def test_rerank_surfaces_verified_useful_first() -> None:
    memory = LocalMemoryClient(rerank_feedback=True)
    m1, m2 = await _seed(memory, "alpha note", "alpha memo")
    memory.record_feedback(m2, "alpha", helpful=True)

    out = await memory.recall("alpha")

    assert [m.id for m in out] == [m2, m1]  # helpful m2 above unproven m1


async def test_rerank_demotes_unhelpful_below_unproven() -> None:
    memory = LocalMemoryClient(rerank_feedback=True)
    m1, m2 = await _seed(memory, "alpha note", "alpha memo")
    memory.record_feedback(m1, "alpha", helpful=False)

    out = await memory.recall("alpha")

    assert [m.id for m in out] == [m2, m1]  # m2 (0) above down-ranked m1 (negative)


async def test_rerank_cold_start_floor_keeps_unreinforced_order() -> None:
    # Never-reinforced memories keep insertion order (stable sort) and all still surface.
    memory = LocalMemoryClient(rerank_feedback=True)
    ids = await _seed(memory, "alpha a", "alpha b", "alpha c")

    out = await memory.recall("alpha")

    assert [m.id for m in out] == ids  # all score 0 -> insertion order, none dropped


async def test_rerank_boost_is_bounded() -> None:
    # A landslide of helpful votes cannot exceed the bound: a memory at the bound ties
    # with one far past it, so the extra votes buy no rank (insertion order breaks ties).
    memory = LocalMemoryClient(rerank_feedback=True)
    m1, m2 = await _seed(memory, "alpha x", "alpha y")
    for _ in range(RERANK_BOUND):
        memory.record_feedback(m1, "alpha", helpful=True)  # exactly at the bound
    for _ in range(RERANK_BOUND + 50):
        memory.record_feedback(m2, "alpha", helpful=True)  # far past the bound

    out = await memory.recall("alpha")

    assert [m.id for m in out] == [m1, m2]  # both clamp equal -> insertion order


async def test_rerank_preserves_filters_and_returns_all_matches() -> None:
    # The re-rank sorts the ALREADY-filtered set: trusted_only still applies, a
    # quarantined memory cannot be surfaced by feedback, and no match is dropped.
    memory = LocalMemoryClient(rerank_feedback=True)
    t1 = await memory.remember("alpha trusted one", trust_tier=TRUSTED_TIER)
    t2 = await memory.remember("alpha trusted two", trust_tier=TRUSTED_TIER)
    q1 = await memory.remember("alpha quarantined", trust_tier="quarantine")
    memory.record_feedback(q1, "alpha", helpful=True)  # feedback must not surface it

    out = await memory.recall("alpha", trusted_only=True)

    assert {m.id for m in out} == {t1, t2}  # quarantined excluded despite helpful feedback
    assert len(out) == 2  # both trusted matches returned, none dropped


async def test_rerank_uses_net_helpful_score() -> None:
    # Mixed votes net out: 2 helpful + 1 unhelpful = +1, still above an unproven memory.
    memory = LocalMemoryClient(rerank_feedback=True)
    m1, m2 = await _seed(memory, "alpha one", "alpha two")
    memory.record_feedback(m2, "alpha", helpful=True)
    memory.record_feedback(m2, "alpha", helpful=True)
    memory.record_feedback(m2, "alpha", helpful=False)  # net +1

    out = await memory.recall("alpha")

    assert [m.id for m in out] == [m2, m1]


# --- SqliteMemoryClient parity: the persistent backend re-ranks the SAME way --


async def test_sqlite_recall_unchanged_when_rerank_off(tmp_path: Path) -> None:
    memory = SqliteMemoryClient(tmp_path / "mem.db")  # default: rerank off
    m1 = await memory.remember("alpha one", trust_tier=TRUSTED_TIER)
    m2 = await memory.remember("alpha two", trust_tier=TRUSTED_TIER)
    memory.record_feedback(m2, "alpha", helpful=True)

    out = await memory.recall("alpha")

    assert [m.id for m in out] == [m1, m2]  # insertion order, feedback ignored
    memory.close()


async def test_sqlite_reranks_persisted_feedback_across_instances(tmp_path: Path) -> None:
    # The cross-run loop: feedback recorded by one instance re-ranks recall in a
    # FRESH instance over the same file — the headline cross-process acceptance.
    db = tmp_path / "mem.db"
    writer = SqliteMemoryClient(db)
    m1 = await writer.remember("alpha one", trust_tier=TRUSTED_TIER)
    m2 = await writer.remember("alpha two", trust_tier=TRUSTED_TIER)
    writer.record_feedback(m2, "alpha", helpful=True)
    writer.close()

    reader = SqliteMemoryClient(db, rerank_feedback=True)
    out = await reader.recall("alpha")

    assert [m.id for m in out] == [m2, m1]  # persisted feedback re-ranks the new instance
    reader.close()


async def test_sqlite_rerank_boost_is_bounded(tmp_path: Path) -> None:
    memory = SqliteMemoryClient(tmp_path / "mem.db", rerank_feedback=True)
    m1 = await memory.remember("alpha x", trust_tier=TRUSTED_TIER)
    m2 = await memory.remember("alpha y", trust_tier=TRUSTED_TIER)
    for _ in range(RERANK_BOUND):
        memory.record_feedback(m1, "alpha", helpful=True)
    for _ in range(RERANK_BOUND + 50):
        memory.record_feedback(m2, "alpha", helpful=True)

    out = await memory.recall("alpha")

    assert [m.id for m in out] == [m1, m2]  # both clamp equal -> insertion order
    memory.close()


async def test_rerank_net_negative_sinks_below_unproven() -> None:
    # And a net-negative (2 unhelpful + 1 helpful = -1) sinks below a score-0 memory.
    memory = LocalMemoryClient(rerank_feedback=True)
    m1, m2 = await _seed(memory, "alpha one", "alpha two")
    memory.record_feedback(m1, "alpha", helpful=False)
    memory.record_feedback(m1, "alpha", helpful=False)
    memory.record_feedback(m1, "alpha", helpful=True)  # net -1

    out = await memory.recall("alpha")

    assert [m.id for m in out] == [m2, m1]
