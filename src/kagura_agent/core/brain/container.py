"""#102 (PR1): brain-in-container IPC + a BrainProvider that drives a container.

#92 sealed the launch path (egress in-path, hardened `docker run`); the remaining
half is the **execution-model change** — running the brain's tool loop *inside*
that hardened, egress-sealed container instead of as a host subprocess.

The seam that keeps this from being a rewrite: :class:`ContainerBrainProvider`
implements the existing :class:`~kagura_agent.core.brain.base.BrainProvider`
protocol, so ``Session._drive`` (the ``async for event in brain.run(...)`` loop)
is **unchanged** — it drives a container rather than an in-process brain. The
provider also surfaces the container id the moment the container starts, so the
cockpit (PR2) can register it for ``/kill`` before the run finishes.

Wire protocol (deliberately the simplest thing that works):
  - **host → container** — a single JSON object on stdin: the ``Task`` plus an
    optional resume ``Checkpoint`` (``encode_run_input`` / ``decode_run_input``).
  - **container → host** — **JSON Lines** on stdout, one ``BrainEvent`` per line
    (``encode_event`` / ``decode_event``). stdout is the *pure* event channel;
    container logs go to stderr, so any non-JSON line on stdout is protocol
    corruption and fails closed rather than being silently dropped.

This module is the transport-agnostic core: no Docker, no cockpit wiring, and no
auth decision (the in-container entrypoint runs whatever ``make_brain(env)``
yields — the auth model is decided separately before the deployment wiring). The
real container streaming and the agent-image entrypoint are PR2/PR3.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol

from kagura_agent.core.brain.base import (
    BrainCaps,
    BrainEvent,
    BrainProvider,
    Checkpoint,
    DoneEvent,
    MessageEvent,
    Task,
)

_MESSAGE = "message"
_DONE = "done"

#: Defense-in-depth cap on a single decoded event line. The host decodes output
#: from the *less-trusted* container, so a hijacked brain emitting one gigantic
#: line is refused rather than materialised. This is a backstop only: the real
#: streaming transport (PR3) must ALSO bound its reads (see BrainContainerSession)
#: — the OOM is prevented at read time; this guards anything that slips through.
_MAX_EVENT_CHARS = 8 * 1024 * 1024


# --------------------------------------------------------------------------
# wire protocol
# --------------------------------------------------------------------------


def encode_run_input(task: Task, resume: Checkpoint | None) -> bytes:
    """Encode the host→container run input (a ``Task`` + optional resume) as one
    UTF-8 JSON line for the container's stdin."""
    payload: dict[str, Any] = {
        "task": {"prompt": task.prompt, "session_id": task.session_id},
        "resume": None
        if resume is None
        else {"session_id": resume.session_id, "turn": resume.turn, "state": resume.state},
    }
    return (json.dumps(payload) + "\n").encode("utf-8")


def decode_run_input(data: bytes) -> tuple[Task, Checkpoint | None]:
    """Decode the container's run input back into ``(Task, resume)``.

    Fail-closed and UNIFORMLY, like :func:`decode_event`: malformed JSON, a non-object
    payload, a missing required field, or a field of the wrong type all raise a single
    ``ValueError`` rather than a raw ``KeyError``/``TypeError`` — the entrypoint must
    not run on a garbled input, and a wrong-typed ``turn``/``state`` must not build a
    malformed ``Checkpoint`` that breaks the brain mid-run (e.g. ``state.get(...)`` on
    a list)."""
    try:
        obj = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"malformed run input (not JSON): {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"run input must be a JSON object, got {type(obj).__name__}")
    t = obj.get("task")
    if not isinstance(t, dict):
        raise ValueError("run input 'task' must be an object")
    task = Task(
        prompt=_require_str(t, "prompt", ctx="run input task"),
        session_id=_require_str(t, "session_id", ctx="run input task"),
    )
    r = obj.get("resume")
    if r is None:
        return task, None
    if not isinstance(r, dict):
        raise ValueError("run input 'resume' must be an object or null")
    turn = r.get("turn")
    if not isinstance(turn, int) or isinstance(turn, bool):
        raise ValueError(f"run input resume 'turn' must be an int, got {type(turn).__name__}")
    state = r.get("state")
    if not isinstance(state, dict):
        raise ValueError(f"run input resume 'state' must be an object, got {type(state).__name__}")
    resume = Checkpoint(
        session_id=_require_str(r, "session_id", ctx="run input resume"), turn=turn, state=state
    )
    return task, resume


def encode_event(event: BrainEvent) -> str:
    """Encode one ``BrainEvent`` as a single JSON line (no trailing newline)."""
    if isinstance(event, MessageEvent):
        return json.dumps({"kind": _MESSAGE, "text": event.text})
    if isinstance(event, DoneEvent):
        return json.dumps({"kind": _DONE, "result": event.result, "state": event.state})
    raise TypeError(f"unencodable brain event: {type(event).__name__}")


def _require_str(obj: dict[str, Any], field: str, *, ctx: str = "brain event") -> str:
    value = obj.get(field)
    if not isinstance(value, str):
        raise ValueError(
            f"{ctx} field {field!r} must be a string, got {type(value).__name__}"
        )
    return value


def decode_event(line: str) -> BrainEvent | None:
    """Decode one container stdout line into a ``BrainEvent``.

    The host decodes output from the *less-trusted* container, so this is a
    fail-closed trust boundary. A blank line is a no-op (``None``) — a keepalive,
    never an event. Every other line must be a well-formed known event; anything
    malformed — over the size cap, non-JSON, not a JSON object, an unknown
    ``kind``, a missing field, or a field of the wrong type (a ``text``/``result``
    that isn't a string, a ``state`` that isn't an object) — raises a single
    ``ValueError``. Failing closed (uniformly, not a raw ``KeyError``) means a
    corrupted or hostile stream can never (a) silently drop a real event such as
    the terminal DoneEvent and hang the run, nor (b) inject a malformed event
    whose non-string ``text`` / non-dict ``state`` would later break ``Session``
    mid-run (e.g. ``done.state.get(...)`` on an int)."""
    stripped = line.strip()
    if not stripped:
        return None
    if len(stripped) > _MAX_EVENT_CHARS:
        raise ValueError(f"brain event line exceeds {_MAX_EVENT_CHARS} chars — refusing")
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed brain event line (not JSON): {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"brain event must be a JSON object, got {type(obj).__name__}")
    kind = obj.get("kind")
    if kind == _MESSAGE:
        return MessageEvent(text=_require_str(obj, "text"))
    if kind == _DONE:
        state = obj.get("state", {})
        if not isinstance(state, dict):
            raise ValueError(
                f"brain event 'state' must be an object, got {type(state).__name__}"
            )
        return DoneEvent(result=_require_str(obj, "result"), state=state)
    raise ValueError(f"unknown brain event kind {kind!r}")


# --------------------------------------------------------------------------
# ContainerBrainProvider — the BrainProvider that drives a container
# --------------------------------------------------------------------------


class BrainContainerSession(Protocol):
    """A started brain container: its id plus its stdout event-line stream.

    The transport seam — a fake in tests, a real streaming ``docker run`` in PR3.
    ``container_id`` is available immediately (the moment the container starts);
    ``events()`` yields the raw stdout lines (decoded by the provider).

    **Transport contract (the less-trusted container is the producer).** A real
    implementation MUST bound the stream against a hijacked container that never
    terminates or floods stdout — the *pure* protocol here cannot enforce time/IO
    limits, so they live in the transport:

    - bound each read so one giant line cannot OOM the host before ``decode_event``
      ever sees it (the decoder's char cap is only a backstop);
    - enforce an idle / wall-clock timeout so an endless stream (e.g. infinite
      keepalive lines, or output that never yields a terminal DoneEvent) ends the
      run instead of hanging ``Session._drive`` forever.

    **Teardown.** ``container_id`` surfaces via the provider's ``on_start`` the
    instant the container starts, so the cockpit (PR2) registers it *before* the
    run completes — that registration is the single seam through which ``/kill``
    and restart reconciliation reap a container left running after a mid-stream
    failure (this generator deliberately holds no teardown of its own)."""

    container_id: str

    def events(self) -> AsyncIterator[str]: ...


#: Start a brain container with the given encoded run input (stdin payload) and
#: return its session. Injected so the provider is unit-testable without Docker.
StartContainer = Callable[[bytes], Awaitable[BrainContainerSession]]


class ContainerBrainProvider:
    """A :class:`BrainProvider` whose agentic loop runs inside a container.

    Implements the same protocol as the in-process ``ClaudeBrain``, so ``Session``
    drives it unchanged: it encodes the task, starts the container via the injected
    ``start`` transport, and re-materialises the container's stdout JSON lines into
    ``BrainEvent``s. ``on_start`` (optional) is called with the container id the
    instant the container starts — the cockpit (PR2) wires it to the session
    registry so ``/kill`` can tear down the real container mid-run.
    """

    def __init__(
        self,
        start: StartContainer,
        *,
        caps: BrainCaps,
        on_start: Callable[[str], None] | None = None,
    ) -> None:
        self._start = start
        self.caps = caps
        self._on_start = on_start

    async def run(
        self, task: Task, *, resume: Checkpoint | None = None
    ) -> AsyncIterator[BrainEvent]:
        session = await self._start(encode_run_input(task, resume))
        if self._on_start is not None:
            self._on_start(session.container_id)
        async for line in session.events():
            event = decode_event(line)
            if event is not None:
                yield event


# --------------------------------------------------------------------------
# entrypoint core — runs inside the container (inverse of decode_event)
# --------------------------------------------------------------------------


async def stream_brain_events(
    input_bytes: bytes,
    brain: BrainProvider,
    emit: Callable[[str], None],
) -> None:
    """Run the in-container brain for the encoded run input, emitting each event
    as a JSON line via ``emit``.

    The container half of the protocol: ``decode_run_input`` the stdin payload,
    drive the real brain's loop, and ``encode_event`` every event to stdout. Pure
    and transport-agnostic — the real entrypoint (PR3) wires ``input_bytes`` from
    ``sys.stdin``, ``brain`` from ``make_brain(os.environ)``, and ``emit`` to a
    line-buffered ``sys.stdout``."""
    task, resume = decode_run_input(input_bytes)
    async for event in brain.run(task, resume=resume):
        emit(encode_event(event))
