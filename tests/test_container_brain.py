"""#102 (PR1): brain-in-container IPC + the ContainerBrainProvider seam.

The execution-model change (#92 finding ①) moves the brain's tool execution into
the hardened, egress-sealed container. The seam that keeps this from being a
rewrite: `ContainerBrainProvider` implements the existing `BrainProvider`
protocol, so `Session._drive` (the `async for event in brain.run(...)` loop) is
unchanged — it just drives a container instead of an in-process brain.

This PR is the **pure, transport-agnostic core**: the JSON-lines wire protocol
(host→container Task/resume in, container→host events out), the provider that
re-materialises container stdout lines into `BrainEvent`s and surfaces the
container id, and the in-container entrypoint core that runs the real brain and
emits those lines. No Docker, no cockpit wiring, no auth decision — those are
PR2/PR3.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from kagura_agent.core.brain.base import (
    BrainCaps,
    BrainProvider,
    Checkpoint,
    DoneEvent,
    MessageEvent,
    Task,
)
from kagura_agent.core.brain.container import (
    ContainerBrainProvider,
    decode_event,
    decode_run_input,
    encode_event,
    encode_run_input,
    stream_brain_events,
)

_CAPS = BrainCaps(name="claude", auth_modes=("subscription",))


# --------------------------------------------------------------------------
# wire protocol — host -> container input (Task + optional resume)
# --------------------------------------------------------------------------


def test_run_input_round_trips_without_resume():
    task = Task(prompt="do a thing", session_id="s1")
    decoded_task, decoded_resume = decode_run_input(encode_run_input(task, None))
    assert decoded_task == task
    assert decoded_resume is None


def test_run_input_round_trips_with_resume():
    task = Task(prompt="continue", session_id="s1")
    resume = Checkpoint(session_id="s1", turn=3, state={"turn": 3, "k": "v"})
    decoded_task, decoded_resume = decode_run_input(encode_run_input(task, resume))
    assert decoded_task == task
    assert decoded_resume == resume


def test_encode_run_input_is_bytes_one_line():
    payload = encode_run_input(Task(prompt="x", session_id="s"), None)
    assert isinstance(payload, bytes)
    assert payload.endswith(b"\n")


def test_decode_run_input_malformed_is_fail_closed():
    with pytest.raises(Exception):  # noqa: B017 — any decode failure must surface, not pass
        decode_run_input(b"not json")


def test_decode_run_input_missing_field_raises_valueerror():
    # #123: fail-closed UNIFORMLY (a single ValueError), like decode_event — never a
    # raw KeyError that a caller catching ValueError (the documented contract) misses.
    import json

    with pytest.raises(ValueError):
        decode_run_input(json.dumps({"task": {"prompt": "x"}}).encode())  # no session_id
    with pytest.raises(ValueError):
        decode_run_input(json.dumps({"not_task": 1}).encode())  # no task
    with pytest.raises(ValueError, match="must be a JSON object"):
        decode_run_input(json.dumps([1, 2]).encode())  # not an object envelope


def test_decode_run_input_wrong_typed_resume_raises_valueerror():
    # #123: a non-int turn / non-dict state must not build a malformed Checkpoint
    # that breaks the brain mid-run (e.g. resume.state.get(...) on a list). Symmetric
    # with decode_event's hardening, which the input side previously lacked.
    import json

    task = {"prompt": "x", "session_id": "s"}
    bad_turn = {"task": task, "resume": {"session_id": "s", "turn": "three", "state": {}}}
    bad_state = {"task": task, "resume": {"session_id": "s", "turn": 1, "state": [1, 2]}}
    with pytest.raises(ValueError):
        decode_run_input(json.dumps(bad_turn).encode())
    with pytest.raises(ValueError):
        decode_run_input(json.dumps(bad_state).encode())
    with pytest.raises(ValueError, match="'resume' must be an object"):
        decode_run_input(json.dumps({"task": task, "resume": 5}).encode())  # resume not an object


# --------------------------------------------------------------------------
# wire protocol — container -> host events (JSON lines)
# --------------------------------------------------------------------------


def test_encode_decode_message_event_round_trips():
    ev = decode_event(encode_event(MessageEvent(text="hello — world")))
    assert ev == MessageEvent(text="hello — world")


def test_encode_decode_done_event_round_trips_with_state():
    ev = decode_event(encode_event(DoneEvent(result="final", state={"turn": 2})))
    assert ev == DoneEvent(result="final", state={"turn": 2})


def test_decode_event_blank_line_is_skipped():
    assert decode_event("   ") is None
    assert decode_event("\n") is None


def test_decode_event_non_json_is_fail_closed():
    # stdout is the pure event channel (logs go to stderr); a non-JSON line means
    # protocol corruption and must raise, never be silently dropped.
    with pytest.raises(Exception):  # noqa: B017
        decode_event("INFO: starting up")


def test_decode_event_unknown_kind_is_fail_closed():
    with pytest.raises(ValueError, match="unknown brain event kind"):
        decode_event('{"kind": "tool_call", "name": "rm"}')


def test_decode_event_missing_field_is_valueerror_not_keyerror():
    # Fail-closed must be UNIFORM: a missing field is protocol corruption and must
    # raise ValueError (like unknown-kind), not leak a raw KeyError a ValueError-
    # catching caller would miss.
    with pytest.raises(ValueError, match="must be a string"):
        decode_event('{"kind": "message"}')  # no "text"
    with pytest.raises(ValueError, match="must be a string"):
        decode_event('{"kind": "done"}')  # no "result"


def test_decode_event_rejects_non_string_text_and_result():
    # A hijacked container yielding a non-string text/result must be refused at the
    # boundary, not handed to Session (which would append a dict as "narration"
    # or set SessionResult.text to a list).
    with pytest.raises(ValueError, match="must be a string"):
        decode_event('{"kind": "message", "text": {"a": 1}}')
    with pytest.raises(ValueError, match="must be a string"):
        decode_event('{"kind": "done", "result": [1, 2]}')


def test_decode_event_rejects_non_dict_state():
    # state flows verbatim into Checkpoint.state; a non-object must fail closed so
    # `done.state.get("turn", 0)` can't AttributeError mid-run.
    with pytest.raises(ValueError, match="'state' must be an object"):
        decode_event('{"kind": "done", "result": "ok", "state": 42}')


def test_decode_event_rejects_non_object_json():
    # A bare JSON scalar/array is not an event envelope.
    with pytest.raises(ValueError, match="must be a JSON object"):
        decode_event("42")


def test_decode_event_rejects_oversize_line():
    # Defense-in-depth: an absurdly large line is refused rather than materialised.
    from kagura_agent.core.brain.container import _MAX_EVENT_CHARS

    giant = '{"kind": "message", "text": "' + "A" * (_MAX_EVENT_CHARS + 1) + '"}'
    with pytest.raises(ValueError, match="exceeds"):
        decode_event(giant)


def test_encode_event_rejects_unknown_event_type():
    class WeirdEvent:
        pass

    with pytest.raises(TypeError, match="unencodable"):
        encode_event(WeirdEvent())  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# ContainerBrainProvider — drives an injected container session
# --------------------------------------------------------------------------


class _FakeSession:
    """A fake container session: a fixed container id + a fixed line stream, plus an
    aclose() that records whether the provider reaped it (the teardown contract)."""

    def __init__(self, container_id: str, lines: list[str]) -> None:
        self.container_id = container_id
        self._lines = lines
        self.closed = False

    async def events(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line

    async def aclose(self) -> None:
        self.closed = True


def _provider(lines: list[str], *, on_start=None) -> ContainerBrainProvider:
    async def start(_payload: bytes) -> _FakeSession:
        return _FakeSession("c0ntainer1d", lines)

    return ContainerBrainProvider(start, caps=_CAPS, on_start=on_start)


def _provider_capturing_session(
    lines: list[str], *, on_start=None
) -> tuple[ContainerBrainProvider, list[_FakeSession]]:
    """A provider whose started session is captured so a test can assert it was
    reaped (aclose) on every exit path."""
    captured: list[_FakeSession] = []

    async def start(_payload: bytes) -> _FakeSession:
        session = _FakeSession("c0ntainer1d", lines)
        captured.append(session)
        return session

    return ContainerBrainProvider(start, caps=_CAPS, on_start=on_start), captured


async def test_provider_is_a_brain_provider():
    assert isinstance(_provider([]), BrainProvider)


async def test_provider_rematerialises_events_in_order():
    lines = [
        encode_event(MessageEvent(text="step 1")),
        "",  # a blank keepalive line must be skipped, not break the stream
        encode_event(MessageEvent(text="step 2")),
        encode_event(DoneEvent(result="done", state={"turn": 1})),
    ]
    out = [ev async for ev in _provider(lines).run(Task(prompt="p", session_id="s"))]
    assert out == [
        MessageEvent(text="step 1"),
        MessageEvent(text="step 2"),
        DoneEvent(result="done", state={"turn": 1}),
    ]


async def test_provider_surfaces_container_id_on_start():
    # The container id must surface the moment the container starts (so the cockpit
    # in PR2 can register it for /kill BEFORE the run finishes), not only at the end.
    seen: list[str] = []
    provider = _provider(
        [encode_event(DoneEvent(result="ok"))], on_start=seen.append
    )
    started_before_first_event = False
    async for _ev in provider.run(Task(prompt="p", session_id="s")):
        started_before_first_event = bool(seen)  # on_start already fired
        break
    assert started_before_first_event
    assert seen == ["c0ntainer1d"]


async def test_provider_forwards_the_encoded_task_to_start():
    captured: dict[str, bytes] = {}

    async def start(payload: bytes) -> _FakeSession:
        captured["payload"] = payload
        return _FakeSession("cid", [encode_event(DoneEvent(result="ok"))])

    provider = ContainerBrainProvider(start, caps=_CAPS)
    task = Task(prompt="hello", session_id="s9")
    resume = Checkpoint(session_id="s9", turn=1, state={"turn": 1})
    [_ev async for _ev in provider.run(task, resume=resume)]
    # The payload the container received decodes back to exactly our task+resume.
    assert decode_run_input(captured["payload"]) == (task, resume)


async def test_provider_reaps_session_on_normal_completion():
    provider, captured = _provider_capturing_session([encode_event(DoneEvent(result="ok"))])
    [_ev async for _ev in provider.run(Task(prompt="p", session_id="s"))]
    assert captured[0].closed is True  # reaped after a clean run


async def test_provider_reaps_session_synchronously_on_decode_failure():
    # #123 regression: a malformed line raises ValueError INSIDE run's async-for body.
    # run's finally must reap the container synchronously (not leak it until GC) — the
    # exact mid-stream-failure leak the seam contract warns about.
    provider, captured = _provider_capturing_session(["this is not valid json"])
    with pytest.raises(ValueError):
        [_ev async for _ev in provider.run(Task(prompt="p", session_id="s"))]
    assert captured[0].closed is True


async def test_provider_reaps_session_when_on_start_raises():
    # #123 regression: if on_start throws after the container started but before
    # events() is iterated, the container must still be reaped (not leaked with no
    # teardown path).
    def boom(_cid: str) -> None:
        raise RuntimeError("registry write failed")

    provider, captured = _provider_capturing_session(
        [encode_event(DoneEvent(result="ok"))], on_start=boom
    )
    with pytest.raises(RuntimeError, match="registry write failed"):
        [_ev async for _ev in provider.run(Task(prompt="p", session_id="s"))]
    assert captured[0].closed is True


# --------------------------------------------------------------------------
# entrypoint core — runs the real brain, emits JSON lines (inverse of decode)
# --------------------------------------------------------------------------


class _FakeBrain:
    caps = _CAPS

    def __init__(self, events: list) -> None:
        self._events = events
        self.seen: tuple | None = None

    async def run(self, task: Task, *, resume: Checkpoint | None = None) -> AsyncIterator:
        self.seen = (task, resume)
        for ev in self._events:
            yield ev


async def test_stream_brain_events_emits_decodable_lines():
    brain = _FakeBrain([MessageEvent(text="m"), DoneEvent(result="r", state={"turn": 1})])
    emitted: list[str] = []
    payload = encode_run_input(Task(prompt="p", session_id="s"), None)
    await stream_brain_events(payload, brain, emitted.append)
    # The entrypoint's emitted lines are the exact inverse of the provider's decode.
    assert [decode_event(line) for line in emitted] == [
        MessageEvent(text="m"),
        DoneEvent(result="r", state={"turn": 1}),
    ]


async def test_stream_brain_events_passes_task_and_resume_to_brain():
    brain = _FakeBrain([DoneEvent(result="r")])
    task = Task(prompt="p", session_id="s")
    resume = Checkpoint(session_id="s", turn=2, state={"turn": 2})
    await stream_brain_events(encode_run_input(task, resume), brain, lambda _l: None)
    assert brain.seen == (task, resume)


# --------------------------------------------------------------------------
# end-to-end seam: ContainerBrainProvider plugs into the REAL Session unchanged
# --------------------------------------------------------------------------


async def test_container_provider_drives_a_real_session_unchanged():
    # The whole point of the seam: Session._drive's `async for event in brain.run`
    # loop works against the container provider with NO change to Session. We feed
    # the lines an in-container brain would emit and assert Session resolves them.
    from kagura_agent.core.session import Session
    from kagura_agent.patterns.checkpoint import InMemoryCheckpointStore

    lines = [
        encode_event(MessageEvent(text="thinking")),
        encode_event(DoneEvent(result="the answer", state={"turn": 1})),
    ]
    session = Session(_provider(lines), InMemoryCheckpointStore())
    result = await session.run(Task(prompt="q", session_id="s1"))
    assert result.text == "the answer"
    assert result.messages == ["thinking"]
