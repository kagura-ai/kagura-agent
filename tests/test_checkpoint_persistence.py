"""FileCheckpointStore — cross-process checkpoint persistence (A).

This is the store that makes a *fresh* `kagura-agent run --session <id>` process
resume a prior run: run #1 saves, run #2 (new process, new store instance) loads
the same file back. Tested by round-tripping through two store instances over the
same directory, plus the traversal-safety and corrupt-file guards.
"""

from __future__ import annotations

import json

import pytest

from kagura_agent.core.brain.base import Checkpoint
from kagura_agent.patterns.checkpoint import (
    CheckpointError,
    FileCheckpointStore,
    MemoryCloudCheckpointStore,
)


async def test_save_then_load_roundtrips(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = FileCheckpointStore(tmp_path / "state")
    cp = Checkpoint(session_id="work", turn=3, state={"resume": "abc", "budget": 2})
    await store.save(cp)

    loaded = await store.load("work")
    assert loaded == cp


async def test_load_missing_session_is_none(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = FileCheckpointStore(tmp_path / "state")
    assert await store.load("never-saved") is None


async def test_persists_across_store_instances(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # The cross-PROCESS guarantee, simulated as two independent store objects over
    # the same dir: a new `run` process must see the previous process's checkpoint.
    base = tmp_path / "state"
    await FileCheckpointStore(base).save(Checkpoint(session_id="s", turn=1, state={"k": "v"}))

    reopened = await FileCheckpointStore(base).load("s")
    assert reopened is not None
    assert reopened.turn == 1 and reopened.state == {"k": "v"}


async def test_save_overwrites_prior_checkpoint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = FileCheckpointStore(tmp_path / "state")
    await store.save(Checkpoint(session_id="s", turn=1, state={"step": "a"}))
    await store.save(Checkpoint(session_id="s", turn=2, state={"step": "b"}))

    loaded = await store.load("s")
    assert loaded is not None and loaded.turn == 2 and loaded.state == {"step": "b"}


async def test_distinct_sessions_do_not_collide(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = FileCheckpointStore(tmp_path / "state")
    await store.save(Checkpoint(session_id="alpha", turn=1, state={"who": "a"}))
    await store.save(Checkpoint(session_id="beta", turn=9, state={"who": "b"}))

    a = await store.load("alpha")
    b = await store.load("beta")
    assert a is not None and a.state == {"who": "a"}
    assert b is not None and b.state == {"who": "b"}


async def test_traversal_session_id_stays_in_base_dir(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # A crafted session id with path separators must NOT escape base_dir — the
    # hashed filename neutralizes traversal. Nothing is written outside base.
    base = tmp_path / "state"
    store = FileCheckpointStore(base)
    await store.save(Checkpoint(session_id="../../evil", turn=1, state={}))

    # round-trips by the same (unsanitized) id...
    assert (await store.load("../../evil")) is not None
    # ...and every file produced lives inside base_dir, none above it.
    assert not list(tmp_path.glob("evil*"))
    assert all(base in p.parents or p.parent == base for p in base.iterdir())


async def test_corrupt_checkpoint_raises_not_silently_none(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # A present-but-unreadable checkpoint must surface (CheckpointError), never be
    # mistaken for "no checkpoint" (which would silently restart the task fresh).
    base = tmp_path / "state"
    store = FileCheckpointStore(base)
    await store.save(Checkpoint(session_id="s", turn=1, state={"k": "v"}))
    # clobber the on-disk file with garbage
    [f] = list(base.glob("*.json"))
    f.write_text("not json {", encoding="utf-8")

    with pytest.raises(CheckpointError, match="corrupt"):
        await store.load("s")


async def test_save_cleans_up_temp_file_on_write_failure(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # If the atomic rename fails (e.g. disk full / cross-device), the error must
    # propagate AND the sibling temp file must not be left behind to accumulate.
    from kagura_agent.patterns import checkpoint as cp_mod

    base = tmp_path / "state"
    store = FileCheckpointStore(base)

    def _boom(*_a: object, **_k: object) -> None:
        raise OSError("rename failed")

    monkeypatch.setattr(cp_mod.os, "replace", _boom)
    with pytest.raises(OSError, match="rename failed"):
        await store.save(Checkpoint(session_id="s", turn=1, state={}))

    assert list(base.glob("*.tmp")) == []  # temp file cleaned up, none leaked


async def test_checkpoint_missing_field_raises(tmp_path) -> None:  # type: ignore[no-untyped-def]
    base = tmp_path / "state"
    store = FileCheckpointStore(base)
    await store.save(Checkpoint(session_id="s", turn=1, state={}))
    [f] = list(base.glob("*.json"))
    f.write_text(json.dumps({"session_id": "s"}), encoding="utf-8")  # no turn/state

    with pytest.raises(CheckpointError):
        await store.load("s")


# --- MemoryCloudCheckpointStore (B2): checkpoint over the KV state API --------


class _FakeStateBackend:
    """In-memory stand-in for memory-cloud's get_state/set_state."""

    def __init__(self) -> None:
        self.kv: dict[str, str] = {}

    async def get_state(self, key: str) -> str | None:
        return self.kv.get(key)

    async def set_state(self, key: str, value: str) -> None:
        self.kv[key] = value


async def test_memorycloud_store_roundtrips() -> None:
    backend = _FakeStateBackend()
    store = MemoryCloudCheckpointStore(backend)
    cp = Checkpoint(session_id="work", turn=2, state={"resume": "tok"})

    await store.save(cp)
    assert await store.load("work") == cp


async def test_memorycloud_store_namespaces_and_overwrites() -> None:
    backend = _FakeStateBackend()
    store = MemoryCloudCheckpointStore(backend)
    await store.save(Checkpoint(session_id="s", turn=1, state={"v": 1}))
    await store.save(Checkpoint(session_id="s", turn=2, state={"v": 2}))

    # keys are namespaced so they can't collide with other agent state...
    assert all(k.startswith("checkpoint:") for k in backend.kv)
    # ...and save overwrites rather than appends.
    loaded = await store.load("s")
    assert loaded is not None and loaded.turn == 2


async def test_memorycloud_store_missing_is_none() -> None:
    assert await MemoryCloudCheckpointStore(_FakeStateBackend()).load("nope") is None


async def test_memorycloud_store_corrupt_raises() -> None:
    backend = _FakeStateBackend()
    backend.kv["checkpoint:s"] = "not json {"
    with pytest.raises(CheckpointError, match="corrupt"):
        await MemoryCloudCheckpointStore(backend).load("s")
