"""Long-task resume: persist a provider's opaque state between turns.

v1 ships three stores behind one tiny `CheckpointStore` protocol:

- `InMemoryCheckpointStore` — process-local (tests + a long-lived cockpit/REPL).
- `FileCheckpointStore` — on-disk JSON, so a *fresh* `kagura-agent run --session`
  process resumes a prior run's checkpoint (the cross-process continuity the
  one-shot CLI otherwise lacks).
- `MemoryCloudCheckpointStore` — backed by the memory backbone (see
  `kagura_agent.patterns.continuity`), so a checkpoint is durable knowledge the
  agent can recall, on-thesis with memory-as-backbone.

The store is deliberately tiny — the session is the only writer, and
`Checkpoint.state` is opaque to it (the provider's resume token / granted budget,
never live credentials).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Protocol

from kagura_agent.core.brain.base import Checkpoint


class CheckpointError(RuntimeError):
    """A persisted checkpoint exists but could not be read back (corrupt /
    unreadable). Surfaced rather than silently treated as "no checkpoint", so a
    resume does not quietly lose the task's context and start over."""


def _decode_checkpoint(raw: str, *, where: str) -> Checkpoint:
    """Parse a persisted checkpoint payload into a Checkpoint, fail-closed.

    Raises ``CheckpointError`` on anything malformed — non-JSON, a non-object top
    level, a missing key, OR a **wrong-typed field** (a valid JSON object whose
    ``turn`` is a string or ``state`` is a list). The type checks matter: without
    them a wrong-typed payload would build a Checkpoint whose ``state`` is not a
    dict and crash only later, far away, when a brain does ``state.get(...)`` on
    resume. Shared by every serializing store so they validate identically.
    """
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise CheckpointError(f"checkpoint {where} is corrupt (not JSON): {exc}") from exc
    if not isinstance(data, dict):
        raise CheckpointError(f"checkpoint {where} is corrupt: top level is not a JSON object")
    try:
        session_id = data["session_id"]
        turn = data["turn"]
        state = data["state"]
    except KeyError as exc:
        raise CheckpointError(f"checkpoint {where} is corrupt (missing {exc} field)") from exc
    # bool is an int subclass — exclude it so a JSON `true`/`false` turn is rejected.
    if (
        not isinstance(session_id, str)
        or not isinstance(turn, int)
        or isinstance(turn, bool)
        or not isinstance(state, dict)
    ):
        raise CheckpointError(
            f"checkpoint {where} is corrupt: expected {{session_id:str, turn:int, "
            f"state:object}}, got {{session_id:{type(session_id).__name__}, "
            f"turn:{type(turn).__name__}, state:{type(state).__name__}}}"
        )
    return Checkpoint(session_id=session_id, turn=turn, state=state)


class CheckpointStore(Protocol):
    async def save(self, checkpoint: Checkpoint) -> None: ...

    async def load(self, session_id: str) -> Checkpoint | None: ...


class InMemoryCheckpointStore:
    """Process-local checkpoint store (tests + the cockpit's hot path)."""

    def __init__(self) -> None:
        self._by_session: dict[str, Checkpoint] = {}

    async def save(self, checkpoint: Checkpoint) -> None:
        self._by_session[checkpoint.session_id] = checkpoint

    async def load(self, session_id: str) -> Checkpoint | None:
        return self._by_session.get(session_id)


def _checkpoint_filename(session_id: str) -> str:
    """A traversal-safe filename for a session id.

    The session id is operator/transport-supplied (a thread id, a ``--session``
    value) and may contain path separators or ``..`` — using it raw as a filename
    would let a crafted id escape the store directory. Hashing yields a fixed,
    filesystem-safe name AND closes that traversal vector; the real id is stored
    *inside* the file (load reads it back), so no reverse mapping is needed.
    """
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    return f"{digest}.json"


class FileCheckpointStore:
    """On-disk JSON checkpoint store — one file per session under ``base_dir``.

    This is what makes consecutive ``kagura-agent run --session <id>`` invocations
    (separate processes) continue one another: run #1 saves here, run #2 loads it
    back. Writes are atomic (temp file + ``os.replace``) so a crash mid-write
    never leaves a half-written checkpoint that ``load`` would choke on.
    """

    def __init__(self, base_dir: str | os.PathLike[str]) -> None:
        self._base = Path(base_dir)

    def _path(self, session_id: str) -> Path:
        return self._base / _checkpoint_filename(session_id)

    async def save(self, checkpoint: Checkpoint) -> None:
        self._base.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "session_id": checkpoint.session_id,
                "turn": checkpoint.turn,
                "state": checkpoint.state,
            }
        )
        target = self._path(checkpoint.session_id)
        # Atomic publish: write a sibling temp file, then rename onto the target.
        # os.replace is atomic + overwrites on the same filesystem, so a reader
        # never sees a partially written file. This rests on the documented
        # single-writer contract (the session is the only writer): on Windows,
        # os.replace raises a sharing violation if the target is concurrently
        # open, so a same-session concurrent writer is out of contract by design.
        fd, tmp = tempfile.mkstemp(dir=self._base, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp, target)
        except BaseException:
            # Don't leak the temp file if the write/replace failed.
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    async def load(self, session_id: str) -> Checkpoint | None:
        path = self._path(session_id)
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None  # genuinely absent — not an error
        except OSError as exc:
            # Present but unreadable (a directory at the path, a permission / AV
            # lock, …) is NOT "no checkpoint" — surface it rather than silently
            # starting the task over. FileNotFoundError is a *sibling* OSError, so
            # this clause must come AFTER it to keep absent → None.
            raise CheckpointError(
                f"checkpoint for session {session_id!r} at {path} is unreadable: {exc}"
            ) from exc
        return _decode_checkpoint(raw, where=f"for session {session_id!r} at {path}")


class StateBackend(Protocol):
    """memory-cloud's key→value state API (the `get_state`/`set_state` tools).

    A tiny KV surface — exactly what a checkpoint needs (exact-key get + overwrite),
    unlike the fuzzy recall/remember surface. The real backend shells to the kagura
    CLI / SDK (deployment edge); the store below is pure over this protocol so it
    unit-tests against a fake.
    """

    async def get_state(self, key: str) -> str | None: ...

    async def set_state(self, key: str, value: str) -> None: ...


class MemoryCloudCheckpointStore:
    """Checkpoint store backed by memory-cloud's KV state API.

    On-thesis with memory-as-backbone: a resumable task's state lives in the same
    durable store the agent already trusts, so a checkpoint survives process death
    AND is owned by the backbone rather than a local file. Semantics match
    `FileCheckpointStore` (overwrite on save, corrupt → `CheckpointError`).
    """

    #: Namespacing prefix so checkpoint keys never collide with other agent state.
    _PREFIX = "checkpoint:"

    def __init__(self, backend: StateBackend) -> None:
        self._backend = backend

    async def save(self, checkpoint: Checkpoint) -> None:
        await self._backend.set_state(
            self._PREFIX + checkpoint.session_id,
            json.dumps(
                {
                    "session_id": checkpoint.session_id,
                    "turn": checkpoint.turn,
                    "state": checkpoint.state,
                }
            ),
        )

    async def load(self, session_id: str) -> Checkpoint | None:
        raw = await self._backend.get_state(self._PREFIX + session_id)
        if raw is None:
            return None
        return _decode_checkpoint(
            raw, where=f"for session {session_id!r} in memory-cloud state"
        )
