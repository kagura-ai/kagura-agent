"""#107: a durable, file-backed ``MemoryClient`` — the offline middle tier.

``LocalMemoryClient`` is in-process (lost when the process exits); the trust-aware
MCP cloud adapter is the production backbone (cross-repo, a deployment edge). This
sits between them: a SQLite-backed client that **persists across separate process
invocations** yet needs no network or extra dependency (``sqlite3`` is stdlib).

It is a true drop-in for :class:`~kagura_agent.mcp.memory_cloud.LocalMemoryClient`
— the agent protocol (``remember`` / ``recall`` / ``load_pinned`` / ``create_edge``)
AND the host-side admin verbs the forget-cascade (#93), graduation (#15), and
feedback (#90) paths use (``promote`` / ``forget`` / ``record_feedback`` / …). Trust
and resolution semantics mirror ``LocalMemoryClient`` exactly, so swapping the
backend never changes behaviour: ``recall`` is any-term substring over lowercased
text in insertion order, ``trusted_only`` filters the quarantine tier,
``load_pinned`` returns the complete ``always``-delivery set unranked and unfiltered,
and an unknown id on an admin verb is a fail-closed ``KeyError``.

Like ``LocalMemoryClient`` it holds **no admin verbs on the agent protocol**: a
prompt-injected agent gets append + scoped read only; ``promote`` / ``forget`` /
``record_feedback`` are host-side, off the protocol, by construction.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from kagura_agent.mcp.memory_cloud import (
    _VALID_DELIVERY,
    ALWAYS_DELIVERY,
    TRUSTED_TIER,
    FeedbackRecord,
    Memory,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    seq           INTEGER PRIMARY KEY AUTOINCREMENT,
    text          TEXT NOT NULL,
    tags          TEXT NOT NULL,   -- json array of strings
    trust_tier    TEXT NOT NULL,
    delivery_mode TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS edges (
    src  TEXT NOT NULL,
    dst  TEXT NOT NULL,
    type TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    memory_id TEXT NOT NULL,
    query     TEXT NOT NULL,
    helpful   INTEGER NOT NULL    -- 0 / 1
);
"""


class SqliteMemoryClient:
    """A durable ``MemoryClient`` backed by a single SQLite file.

    ``path`` is the database file; it (and its schema) are created on construction,
    so a bad/unwritable path fails **here**, loudly — the caller (``make_memory_client``)
    relies on that to fail closed rather than silently degrade to in-memory.
    """

    def __init__(self, path: str | Path) -> None:
        # check_same_thread stays True (default): all access is on the event-loop
        # thread. isolation_level=None → autocommit, so a single-statement write is
        # durable the moment it returns and a *separate* instance/process sees it
        # immediately (the headline cross-process guarantee).
        self._conn = sqlite3.connect(str(path), isolation_level=None)
        # Make a concurrent writer WAIT for the lock rather than fail fast with
        # "database is locked" — LocalMemoryClient never errors on a write, so this
        # keeps the durable backend a drop-in when two processes share the file.
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _seq_of(memory_id: str) -> int | None:
        """Parse a ``"m<seq>"`` id back to its integer PK, or None if malformed.

        Ids are ``f"m{seq}"`` (a pure function of the autoincrement PK, never a
        stored column), so a lookup is an O(1) keyed query on ``seq`` and a bogus
        id resolves to None → fail-closed (KeyError / False) without a scan."""
        if memory_id.startswith("m"):
            try:
                return int(memory_id[1:])
            except ValueError:
                return None
        return None

    # -- agent protocol -----------------------------------------------------

    async def remember(
        self,
        text: str,
        *,
        tags: tuple[str, ...] = (),
        trust_tier: str = TRUSTED_TIER,
        delivery_mode: str = "on_recall",
    ) -> str:
        # Fail-CLOSED at the write boundary (mirrors LocalMemoryClient): a typo'd
        # delivery mode must never be stored verbatim and silently never pin.
        if delivery_mode not in _VALID_DELIVERY:
            raise ValueError(
                f"unknown delivery_mode {delivery_mode!r} (expected one of {_VALID_DELIVERY})"
            )
        # A single atomic INSERT — the id is derived from the autoincrement PK
        # (`m{seq}`), never a separate stored column, so there is no second UPDATE
        # and therefore no window in which another connection/process could read a
        # half-written (null-id) row. AUTOINCREMENT keeps `seq` monotonic across
        # reopens (its sqlite_sequence row persists in the file), so a reopened DB
        # never reuses an id — ids stay unique across separate process invocations.
        cur = self._conn.execute(
            "INSERT INTO memories(text, tags, trust_tier, delivery_mode) VALUES (?, ?, ?, ?)",
            (text, json.dumps(list(tags)), trust_tier, delivery_mode),
        )
        return f"m{cur.lastrowid}"

    async def recall(
        self,
        query: str,
        *,
        trusted_only: bool = False,
        tags: tuple[str, ...] = (),
    ) -> list[Memory]:
        terms = [t.lower() for t in query.split()]
        want_tags = set(tags)
        results: list[Memory] = []
        for mem in self._all_memories():
            if trusted_only and mem.trust_tier != TRUSTED_TIER:
                continue
            if want_tags and not (want_tags & set(mem.tags)):
                continue
            haystack = mem.text.lower()
            if any(term in haystack for term in terms):
                results.append(mem)
        return results

    async def load_pinned(self) -> list[Memory]:
        # Deterministic + unranked: the COMPLETE pinned set, insertion order, every
        # call. No query, no ranking, no trust filter — pinned guardrails are
        # host-curated and load whole (#88).
        return [m for m in self._all_memories() if m.delivery_mode == ALWAYS_DELIVERY]

    async def create_edge(self, src_id: str, dst_id: str, *, type: str) -> None:
        # Append-only journal (like LocalMemoryClient's list): dst is stored as-is,
        # not validated to exist, so an edge can be recorded before its target.
        self._conn.execute(
            "INSERT INTO edges(src, dst, type) VALUES (?, ?, ?)", (src_id, dst_id, type)
        )

    # -- host-side admin verbs (off the agent protocol, like LocalMemoryClient) ---

    def edges_of(self, src_id: str) -> list[tuple[str, str]]:
        rows = self._conn.execute(
            "SELECT dst, type FROM edges WHERE src = ? ORDER BY rowid", (src_id,)
        ).fetchall()
        return [(str(dst), str(etype)) for dst, etype in rows]

    def promote(self, memory_id: str) -> None:
        """Host-side ONLY: graduate a quarantined memory into the trusted backbone.

        Off the agent protocol (the agent can never promote its own writes) — the
        effect of a post-graduation HITL grant (#15), applied host-side. Unknown id
        raises ``KeyError`` — fail-closed, no silent no-op that masks a bad id.
        """
        seq = self._seq_of(memory_id)
        if seq is None or not self.has_memory(memory_id):
            raise KeyError(memory_id)
        self._conn.execute(
            "UPDATE memories SET trust_tier = ? WHERE seq = ?", (TRUSTED_TIER, seq)
        )

    def record_feedback(self, memory_id: str, query: str, *, helpful: bool) -> None:
        """Host-side ONLY: record whether a recalled memory was useful (#90).

        Off the agent protocol (like ``promote``): the "helpful" verdict must come
        from an INDEPENDENT source (HITL approval / task outcome), never the agent's
        self-report. An append-only journal — a re-recorded ``(memory_id, query)`` is
        kept as a second record, not deduplicated. Unknown id → ``KeyError``.
        """
        if not self.has_memory(memory_id):
            raise KeyError(memory_id)
        self._conn.execute(
            "INSERT INTO feedback(memory_id, query, helpful) VALUES (?, ?, ?)",
            (memory_id, query, 1 if helpful else 0),
        )

    def feedback_for(self, memory_id: str) -> list[FeedbackRecord]:
        rows = self._conn.execute(
            "SELECT query, helpful FROM feedback WHERE memory_id = ? ORDER BY rowid",
            (memory_id,),
        ).fetchall()
        return [
            FeedbackRecord(memory_id=memory_id, query=str(q), helpful=bool(h)) for q, h in rows
        ]

    def has_memory(self, memory_id: str) -> bool:
        seq = self._seq_of(memory_id)
        if seq is None:
            return False
        row = self._conn.execute(
            "SELECT 1 FROM memories WHERE seq = ? LIMIT 1", (seq,)
        ).fetchone()
        return row is not None

    def forget(self, memory_id: str) -> None:
        """Host-side ONLY: erase a memory and its host-side derived records (#93).

        Off the agent protocol (a prompt-injected agent must not amplify a hijack
        into destructive deletes). Removes the memory, its outgoing edges, any edges
        pointing AT it (no dangling refs), and its feedback lane. Unknown id →
        ``KeyError`` — fail-closed, no silent no-op masking an incomplete erasure.

        The three deletes run in **one transaction**: a crash or lock mid-erasure
        rolls back wholesale rather than leaving a memory gone but its edges /
        feedback dangling — the partial-erasure state the #93 cascade must prevent.
        """
        seq = self._seq_of(memory_id)
        if seq is None or not self.has_memory(memory_id):
            raise KeyError(memory_id)
        self._conn.execute("BEGIN")
        try:
            self._conn.execute("DELETE FROM memories WHERE seq = ?", (seq,))
            self._conn.execute(
                "DELETE FROM edges WHERE src = ? OR dst = ?", (memory_id, memory_id)
            )
            self._conn.execute("DELETE FROM feedback WHERE memory_id = ?", (memory_id,))
            self._conn.execute("COMMIT")
        except Exception:  # pragma: no cover - defensive: roll back a mid-erasure failure
            self._conn.execute("ROLLBACK")
            raise

    def ids_with_tag(self, tag: str) -> list[str]:
        return [m.id for m in self._all_memories() if tag in m.tags]

    # -- internals ----------------------------------------------------------

    def _all_memories(self) -> list[Memory]:
        """Every memory in insertion (``seq``) order — the scan ``recall`` /
        ``load_pinned`` / ``ids_with_tag`` filter, mirroring LocalMemoryClient's
        dict-iteration order. Scale (indexed/server-side query) is a cloud-tier
        concern, deliberately out of scope for this offline middle tier."""
        rows = self._conn.execute(
            "SELECT seq, text, tags, trust_tier, delivery_mode FROM memories ORDER BY seq"
        ).fetchall()
        return [
            Memory(
                id=f"m{seq}",
                text=str(text),
                tags=tuple(json.loads(tags)),
                trust_tier=str(trust_tier),
                delivery_mode=str(delivery_mode),
            )
            for seq, text, tags, trust_tier, delivery_mode in rows
        ]
