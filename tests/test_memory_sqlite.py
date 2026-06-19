"""#107: SqliteMemoryClient — a durable, file-backed MemoryClient.

The "offline-but-durable middle tier" between the in-process LocalMemoryClient
and the trust-aware MCP cloud adapter. It must be a true drop-in for
LocalMemoryClient (the agent protocol AND the host-side admin verbs the
forget-cascade / graduation / feedback paths use), differing only in that it
persists across SEPARATE process invocations — the headline acceptance.

Resolution and trust semantics mirror LocalMemoryClient exactly (so swapping the
backend never changes behaviour): recall is any-term substring over lowercased
text in insertion order, trusted_only filters the quarantine tier, load_pinned
returns the complete always-delivery set, and an unknown id on an admin verb is a
fail-closed KeyError.
"""

from __future__ import annotations

import pytest

from kagura_agent.mcp.memory_cloud import (
    ALWAYS_DELIVERY,
    QUARANTINE_TIER,
    TRUSTED_TIER,
)
from kagura_agent.mcp.memory_sqlite import SqliteMemoryClient


def _db(tmp_path) -> str:
    return str(tmp_path / "memory.db")


# --------------------------------------------------------------------------
# remember / recall — semantics mirror LocalMemoryClient
# --------------------------------------------------------------------------


async def test_remember_then_recall_round_trips(tmp_path):
    client = SqliteMemoryClient(_db(tmp_path))
    mid = await client.remember("the cockpit drives the brain", tags=("arch",))
    assert mid  # a non-empty id
    out = await client.recall("brain")
    assert [m.text for m in out] == ["the cockpit drives the brain"]
    assert out[0].id == mid
    assert out[0].tags == ("arch",)


async def test_recall_trusted_only_filters_quarantine(tmp_path):
    client = SqliteMemoryClient(_db(tmp_path))
    await client.remember("trusted note", trust_tier=TRUSTED_TIER)
    await client.remember("quarantined note", trust_tier=QUARANTINE_TIER)
    trusted = await client.recall("note", trusted_only=True)
    assert [m.trust_tier for m in trusted] == [TRUSTED_TIER]
    both = await client.recall("note")
    assert len(both) == 2  # default: no trust filter


async def test_recall_tag_filter_is_intersection(tmp_path):
    client = SqliteMemoryClient(_db(tmp_path))
    await client.remember("alpha", tags=("x", "y"))
    await client.remember("beta", tags=("z",))
    out = await client.recall("alpha beta", tags=("y",))
    assert [m.text for m in out] == ["alpha"]


async def test_recall_is_any_term_substring_in_insertion_order(tmp_path):
    client = SqliteMemoryClient(_db(tmp_path))
    await client.remember("first apple")
    await client.remember("second banana")
    await client.remember("third apple banana")
    out = await client.recall("APPLE")  # case-insensitive
    assert [m.text for m in out] == ["first apple", "third apple banana"]


async def test_remember_rejects_unknown_delivery_mode(tmp_path):
    client = SqliteMemoryClient(_db(tmp_path))
    with pytest.raises(ValueError, match="delivery_mode"):
        await client.remember("x", delivery_mode="Always")  # typo, fail-closed


# --------------------------------------------------------------------------
# load_pinned — complete always-delivery set, unranked, no trust filter
# --------------------------------------------------------------------------


async def test_load_pinned_returns_complete_always_set_in_order(tmp_path):
    client = SqliteMemoryClient(_db(tmp_path))
    await client.remember("on-recall one")
    g1 = await client.remember("pinned guardrail", delivery_mode=ALWAYS_DELIVERY)
    await client.remember("on-recall two")
    g2 = await client.remember(
        "pinned quarantined", trust_tier=QUARANTINE_TIER, delivery_mode=ALWAYS_DELIVERY
    )
    pinned = await client.load_pinned()
    # complete set, insertion order, no trust filter (host-curated)
    assert [m.id for m in pinned] == [g1, g2]
    assert all(m.delivery_mode == ALWAYS_DELIVERY for m in pinned)


# --------------------------------------------------------------------------
# edges + host-side admin verbs (drop-in parity with LocalMemoryClient)
# --------------------------------------------------------------------------


async def test_create_edge_and_edges_of(tmp_path):
    client = SqliteMemoryClient(_db(tmp_path))
    a = await client.remember("a")
    b = await client.remember("b")
    await client.create_edge(a, b, type="prevents")
    assert client.edges_of(a) == [(b, "prevents")]
    assert client.edges_of(b) == []


async def test_promote_graduates_quarantine_to_trusted(tmp_path):
    client = SqliteMemoryClient(_db(tmp_path))
    mid = await client.remember("q", trust_tier=QUARANTINE_TIER)
    client.promote(mid)
    out = await client.recall("q", trusted_only=True)
    assert [m.id for m in out] == [mid]


def test_promote_unknown_id_is_keyerror(tmp_path):
    client = SqliteMemoryClient(_db(tmp_path))
    with pytest.raises(KeyError):
        client.promote("m999")


def test_admin_verbs_fail_closed_on_malformed_id(tmp_path):
    # An id that is not "m<int>" must resolve to absent (fail-closed) without a
    # scan or crash: has_memory False, promote/forget KeyError.
    client = SqliteMemoryClient(_db(tmp_path))
    assert client.has_memory("not-an-m-id") is False  # no "m" prefix
    assert client.has_memory("mNaN") is False  # "m" but non-integer tail
    with pytest.raises(KeyError):
        client.promote("mNaN")
    with pytest.raises(KeyError):
        client.forget("xyz")


async def test_record_feedback_is_append_only_journal(tmp_path):
    client = SqliteMemoryClient(_db(tmp_path))
    mid = await client.remember("note")
    client.record_feedback(mid, "q1", helpful=True)
    client.record_feedback(mid, "q1", helpful=False)  # same key kept, not dedup'd
    recs = client.feedback_for(mid)
    assert [(r.query, r.helpful) for r in recs] == [("q1", True), ("q1", False)]


def test_record_feedback_unknown_id_is_keyerror(tmp_path):
    client = SqliteMemoryClient(_db(tmp_path))
    with pytest.raises(KeyError):
        client.record_feedback("m999", "q", helpful=True)


async def test_has_memory(tmp_path):
    client = SqliteMemoryClient(_db(tmp_path))
    mid = await client.remember("x")
    assert client.has_memory(mid) is True
    assert client.has_memory("m999") is False


async def test_forget_drops_incoming_edges_from_multiple_sources(tmp_path):
    # The dangling-edge cleanup must drop edges pointing AT the victim from EVERY
    # source, not just one (DELETE ... WHERE dst = ?, not a single-row delete).
    client = SqliteMemoryClient(_db(tmp_path))
    victim = await client.remember("victim")
    s1 = await client.remember("s1")
    s2 = await client.remember("s2")
    await client.create_edge(s1, victim, type="r")
    await client.create_edge(s2, victim, type="r")
    client.forget(victim)
    assert client.edges_of(s1) == [] and client.edges_of(s2) == []


async def test_recall_order_preserved_after_forget(tmp_path):
    # A forgotten middle memory leaves a seq gap; recall must still return the
    # survivors in insertion order (ORDER BY seq), not reorder them.
    client = SqliteMemoryClient(_db(tmp_path))
    m1 = await client.remember("apple one")
    await client.remember("apple two")  # will be forgotten
    m3 = await client.remember("apple three")
    forget_target = (await client.recall("two"))[0].id
    client.forget(forget_target)
    assert [m.id for m in await client.recall("apple")] == [m1, m3]


async def test_tags_roundtrip_unicode_and_empty(tmp_path):
    # tags go through json.dumps/json.loads — unicode, slashes/spaces, and the
    # empty tuple must all round-trip intact.
    client = SqliteMemoryClient(_db(tmp_path))
    await client.remember("u", tags=("タグ", "a/b", "x y"))
    await client.remember("e", tags=())
    assert (await client.recall("u"))[0].tags == ("タグ", "a/b", "x y")
    assert (await client.recall("e"))[0].tags == ()


async def test_promote_does_not_change_pinned_membership(tmp_path):
    # trust_tier and delivery_mode are orthogonal (#88/#15): promoting a pinned
    # memory changes its tier but must NOT change whether it is pinned.
    client = SqliteMemoryClient(_db(tmp_path))
    g = await client.remember(
        "pinned q", trust_tier=QUARANTINE_TIER, delivery_mode=ALWAYS_DELIVERY
    )
    assert [m.id for m in await client.load_pinned()] == [g]
    client.promote(g)
    pinned = await client.load_pinned()
    assert [m.id for m in pinned] == [g]  # still pinned
    assert pinned[0].trust_tier == TRUSTED_TIER  # but now trusted
    assert pinned[0].delivery_mode == ALWAYS_DELIVERY


async def test_two_live_instances_see_each_others_writes(tmp_path):
    # Stronger than close→reopen: two LIVE instances (separate connections, no
    # intervening close) on the same file see each other's writes immediately —
    # the autocommit + busy_timeout cross-process guarantee, end to end.
    path = _db(tmp_path)
    writer = SqliteMemoryClient(path)
    reader = SqliteMemoryClient(path)
    mid = await writer.remember("live cross-connection write")
    assert [m.id for m in await reader.recall("live")] == [mid]
    writer.close()
    reader.close()


async def test_forget_erases_memory_edges_and_feedback(tmp_path):
    client = SqliteMemoryClient(_db(tmp_path))
    a = await client.remember("a")
    b = await client.remember("b")
    await client.create_edge(a, b, type="prevents")  # a -> b
    await client.create_edge(b, a, type="rel")  # b -> a (points AT a)
    client.record_feedback(a, "q", helpful=True)
    client.forget(a)
    assert client.has_memory(a) is False
    assert client.edges_of(a) == []  # outgoing gone
    assert client.edges_of(b) == []  # dangling edge pointing AT a dropped
    assert client.feedback_for(a) == []  # feedback lane gone


def test_forget_unknown_id_is_keyerror(tmp_path):
    client = SqliteMemoryClient(_db(tmp_path))
    with pytest.raises(KeyError):
        client.forget("m999")


async def test_ids_with_tag(tmp_path):
    client = SqliteMemoryClient(_db(tmp_path))
    a = await client.remember("a", tags=("session:s1",))
    await client.remember("b", tags=("other",))
    c = await client.remember("c", tags=("session:s1", "x"))
    assert client.ids_with_tag("session:s1") == [a, c]  # insertion order


# --------------------------------------------------------------------------
# THE headline: true cross-PROCESS persistence (a fresh instance on the same file)
# --------------------------------------------------------------------------


async def test_persists_across_separate_client_instances(tmp_path):
    path = _db(tmp_path)
    writer = SqliteMemoryClient(path)
    mid = await writer.remember("durable across processes", tags=("t",))
    g = await writer.remember("a guardrail", delivery_mode=ALWAYS_DELIVERY)
    writer.close()

    # A brand-new instance (models a separate process) sees the prior writes.
    reader = SqliteMemoryClient(path)
    out = await reader.recall("durable")
    assert [m.id for m in out] == [mid]
    assert out[0].tags == ("t",)
    assert [m.id for m in await reader.load_pinned()] == [g]


async def test_ids_do_not_collide_across_instances(tmp_path):
    path = _db(tmp_path)
    first = SqliteMemoryClient(path)
    id1 = await first.remember("one")
    first.close()
    second = SqliteMemoryClient(path)
    id2 = await second.remember("two")
    assert id1 != id2  # monotonic ids survive a reopen — no overwrite
    assert {m.text for m in await second.recall("one two")} == {"one", "two"}
