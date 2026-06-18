"""Host-side erasure cascade (#93): a forget reaches agent-side derived artifacts.

A memory-cloud ``forget`` erases the primary memory server-side; this cascade is
the agent-side companion that erases what THIS process derived from it — session
checkpoints and outcome-summaries. It is host-side ONLY: the narrow agent surface
exposes no erasure verb (confinement by omission, like ``promote``).
"""

import pytest

from kagura_agent.core.brain.base import Checkpoint
from kagura_agent.mcp.memory_cloud import (
    LocalMemoryClient,
    MemoryClient,
    QuarantinedMemoryClient,
)
from kagura_agent.patterns.checkpoint import InMemoryCheckpointStore
from kagura_agent.patterns.continuity import remember_outcome
from kagura_agent.patterns.erasure import (
    CascadeResult,
    ProvenanceLog,
    forget_cascade,
)

# --- ProvenanceLog ------------------------------------------------------------


def test_provenance_records_and_queries() -> None:
    log = ProvenanceLog()
    log.record("s1", ["m1", "m2"])
    log.record("s2", ["m2"])  # m2 fed two sessions

    assert log.sessions_for("m1") == {"s1"}
    assert log.sessions_for("m2") == {"s1", "s2"}
    assert log.sessions_for("unknown") == set()  # absent → empty, not KeyError


def test_provenance_sessions_for_returns_a_copy() -> None:
    log = ProvenanceLog()
    log.record("s1", ["m1"])
    got = log.sessions_for("m1")
    got.add("forged")  # mutating the returned set must not corrupt the log
    assert log.sessions_for("m1") == {"s1"}


# --- forget_cascade: the full erasure ----------------------------------------


async def test_cascade_erases_source_summary_and_checkpoint() -> None:
    memory = LocalMemoryClient()
    store = InMemoryCheckpointStore()
    provenance = ProvenanceLog()

    # A trusted source memory (e.g. third-party PII) recalled into session "s".
    source = await memory.remember("PII: alice@example.com", trust_tier="trusted")
    provenance.record("s", [source])
    # The run derived a checkpoint and an outcome-summary for that session.
    await store.save(Checkpoint(session_id="s", turn=2, state={"t": 2}))
    summary = await remember_outcome(memory, session_id="s", prompt="do", result="done")

    result = await forget_cascade(source, memory=memory, checkpoints=store, provenance=provenance)

    # source + summary gone from memory; checkpoint gone from the store.
    assert not memory.has_memory(source)
    assert not memory.has_memory(summary)
    assert await store.load("s") is None
    # ...and the audit trail names exactly what was removed.
    assert isinstance(result, CascadeResult)
    assert result.source_memory_id == source
    assert result.sessions == ("s",)
    assert set(result.forgotten_memory_ids) == {source, summary}
    assert result.deleted_checkpoints == ("s",)
    # provenance entry for the source is cleared too.
    assert provenance.sessions_for(source) == set()


async def test_cascade_unknown_source_is_fail_closed_before_any_delete() -> None:
    memory = LocalMemoryClient()
    store = InMemoryCheckpointStore()
    provenance = ProvenanceLog()
    # A checkpoint that must NOT be touched when the source id is bogus.
    await store.save(Checkpoint(session_id="s", turn=1, state={}))
    provenance.record("s", ["m-does-not-exist"])

    with pytest.raises(KeyError):
        await forget_cascade(
            "m-does-not-exist", memory=memory, checkpoints=store, provenance=provenance
        )

    assert await store.load("s") is not None  # nothing was erased


async def test_cascade_is_idempotent_on_missing_derived_artifacts() -> None:
    # Source fed a session whose checkpoint was already gone and which has no
    # summary: the cascade still erases the source and does not raise.
    memory = LocalMemoryClient()
    store = InMemoryCheckpointStore()
    provenance = ProvenanceLog()
    source = await memory.remember("x", trust_tier="trusted")
    provenance.record("s", [source])  # no checkpoint saved, no summary written

    result = await forget_cascade(source, memory=memory, checkpoints=store, provenance=provenance)

    assert not memory.has_memory(source)
    assert result.forgotten_memory_ids == (source,)
    assert result.deleted_checkpoints == ("s",)  # delete was a no-op, still reported


async def test_cascade_leaves_unrelated_sessions_untouched() -> None:
    memory = LocalMemoryClient()
    store = InMemoryCheckpointStore()
    provenance = ProvenanceLog()
    source = await memory.remember("erase me", trust_tier="trusted")
    provenance.record("s1", [source])
    await store.save(Checkpoint(session_id="s1", turn=1, state={}))
    await store.save(Checkpoint(session_id="s2", turn=1, state={}))  # unrelated
    other_summary = await remember_outcome(memory, session_id="s2", prompt="p", result="r")

    await forget_cascade(source, memory=memory, checkpoints=store, provenance=provenance)

    assert await store.load("s2") is not None  # unrelated checkpoint survives
    assert memory.has_memory(other_summary)  # unrelated summary survives


# --- host-side ONLY: the agent surface exposes no erasure verb ----------------


def test_erasure_verbs_are_not_on_the_memory_client_protocol() -> None:
    # The protocol IS the agent surface; the erasure verbs must not be on it (so no
    # confined client can be relied on to expose them).
    for verb in ("forget", "ids_with_tag", "has_memory"):
        assert not hasattr(MemoryClient, verb)


def test_quarantined_agent_client_has_no_erasure_path() -> None:
    # A hijacked agent must not be able to erase memories/checkpoints. forget &
    # friends are host-side only — absent from the confined client (like promote).
    agent = QuarantinedMemoryClient(LocalMemoryClient())
    for verb in ("forget", "ids_with_tag", "has_memory"):
        assert not hasattr(agent, verb)


# --- LocalMemoryClient.forget: host-side primitive ----------------------------


async def test_forget_removes_memory_edges_feedback_and_dangling_edges() -> None:
    mc = LocalMemoryClient()
    a = await mc.remember("a")
    b = await mc.remember("b")
    c = await mc.remember("c")
    await mc.create_edge(a, b, type="relates")  # a -> b (outgoing from the victim)
    await mc.create_edge(c, a, type="prevents")  # c -> a (points AT the victim)
    mc.record_feedback(a, query="q", helpful=True)

    mc.forget(a)

    assert not mc.has_memory(a)
    assert mc.edges_of(a) == []  # outgoing edges gone
    assert mc.edges_of(c) == []  # dangling edge c -> a pruned (no tombstone ref)
    assert mc.feedback_for(a) == []  # feedback lane gone
    assert mc.has_memory(b) and mc.has_memory(c)  # neighbours untouched


async def test_forget_unknown_id_is_fail_closed() -> None:
    mc = LocalMemoryClient()
    with pytest.raises(KeyError):
        mc.forget("nope")


async def test_ids_with_tag_finds_session_summaries() -> None:
    mc = LocalMemoryClient()
    s = await remember_outcome(mc, session_id="abc", prompt="p", result="r")
    await mc.remember("unrelated", tags=("other",))

    assert mc.ids_with_tag("session:abc") == [s]
    assert mc.ids_with_tag("session:none") == []
