"""#102 (PR2): cockpit wires LAUNCH through the hardened container.

The execution-model change lands at the cockpit: when a container backend is
wired, a LAUNCH builds a `LaunchSpec`, validates it at the membrane gate, runs
the brain INSIDE the container (via PR1's `ContainerBrainProvider`), records the
returned `container_id` in the session registry so `/kill` tears down a real
container, and reconciles dead containers on startup. With no backend the cockpit
keeps today's in-process behaviour, unchanged.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Set

from kagura_agent.cockpit.core import Cockpit
from kagura_agent.cockpit.registry import SessionRegistry
from kagura_agent.cockpit.transports.base import Event
from kagura_agent.cockpit.transports.cli import CliTransport
from kagura_agent.core.brain.base import BrainCaps, BrainEvent, Checkpoint, DoneEvent, Task
from kagura_agent.core.brain.container import encode_event
from kagura_agent.membrane.launcher import LaunchSpec
from kagura_agent.patterns.checkpoint import InMemoryCheckpointStore


class _HostBrain:
    """The in-process brain — its result is distinguishable from the container's,
    so a test can prove which one actually ran."""

    caps = BrainCaps(name="claude")

    async def run(
        self, task: Task, *, resume: Checkpoint | None = None
    ) -> AsyncIterator[BrainEvent]:
        yield DoneEvent(result=f"host: {task.prompt}", state={"turn": 1})


class _FakeContainerSession:
    def __init__(self, container_id: str, lines: list[str]) -> None:
        self.container_id = container_id
        self._lines = lines
        self.closed = False

    async def events(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line

    async def aclose(self) -> None:
        self.closed = True


class _FakeBackend:
    """A fake container backend: returns a fixed spec and a fixed in-container
    event stream, recording what it was asked to start."""

    def __init__(self, spec: LaunchSpec, *, live: Set[str] = frozenset()) -> None:
        self.project_root = "/proj"
        self._spec = spec
        self._live = live
        self.started: list[tuple[LaunchSpec, bytes]] = []

    def spec_for(self, session_id: str) -> LaunchSpec:
        return self._spec

    async def start(self, spec: LaunchSpec, stdin: bytes) -> _FakeContainerSession:
        self.started.append((spec, stdin))
        return _FakeContainerSession(
            "cid-123", [encode_event(DoneEvent(result="in-container ok", state={"turn": 1}))]
        )

    async def live_container_ids(self) -> Set[str]:
        return self._live


class _FakeKiller:
    def __init__(self) -> None:
        self.killed: list[str] = []

    async def kill(self, container_id: str) -> None:
        self.killed.append(container_id)


def _launch(thread_id: str = "t1", text: str = "do work") -> Event:
    return Event(thread_id=thread_id, text=text, is_thread_reply=False)


# --------------------------------------------------------------------------


async def test_launch_runs_in_container_and_registers_container_id() -> None:
    backend = _FakeBackend(LaunchSpec(image="kagura-agent:python"))
    registry = SessionRegistry()
    transport = CliTransport(inbox=[_launch()])
    cockpit = Cockpit(
        transport, _HostBrain(), InMemoryCheckpointStore(), registry=registry, container=backend
    )

    await cockpit.serve()

    # The IN-CONTAINER brain's result was returned, not the host brain's.
    assert transport.sent[0] == ("t1", "in-container ok")
    assert len(backend.started) == 1
    started_spec, _stdin = backend.started[0]
    assert started_spec.image == "kagura-agent:python"
    rec = registry.get("t1")
    assert rec is not None
    assert rec.container_id == "cid-123"  # the launched container id, recorded
    assert rec.image == "kagura-agent:python"
    assert rec.status == "running"


async def test_membrane_violation_refuses_before_any_container_starts() -> None:
    # A spec that validate_spec rejects (a wildcard egress entry) must refuse the
    # run at the membrane gate — NO container is started, nothing is registered.
    backend = _FakeBackend(LaunchSpec(image="img", egress_allow=("*.evil.com",)))
    registry = SessionRegistry()
    transport = CliTransport(inbox=[_launch()])
    cockpit = Cockpit(
        transport, _HostBrain(), InMemoryCheckpointStore(), registry=registry, container=backend
    )

    await cockpit.serve()

    assert backend.started == []  # the membrane refused before start
    assert registry.get("t1") is None  # never registered
    assert "internal error" in transport.sent[0][1]  # serve() surfaced the refusal


async def test_kill_tears_down_the_in_container_session() -> None:
    backend = _FakeBackend(LaunchSpec(image="img"))
    killer = _FakeKiller()
    registry = SessionRegistry()
    transport = CliTransport(
        inbox=[_launch(), Event(thread_id="t1", text="/kill", is_thread_reply=True)]
    )
    cockpit = Cockpit(
        transport,
        _HostBrain(),
        InMemoryCheckpointStore(),
        registry=registry,
        container=backend,
        launcher=killer,
    )

    await cockpit.serve()

    assert killer.killed == ["cid-123"]  # /kill killed the real launched container
    rec = registry.get("t1")
    assert rec is not None and rec.status == "closed"


async def test_reconcile_marks_vanished_container_dead_on_start() -> None:
    # A session from before the restart whose container is no longer live must be
    # marked dead on startup, so a follow-up isn't routed to a stale CONTINUE.
    backend = _FakeBackend(LaunchSpec(image="img"), live=frozenset())  # nothing alive
    registry = SessionRegistry()
    registry.add("t1", container_id="old-cid")
    cockpit = Cockpit(
        CliTransport(inbox=[]), _HostBrain(), InMemoryCheckpointStore(),
        registry=registry, container=backend,
    )

    await cockpit.serve()  # reconciles before the (empty) event loop

    rec = registry.get("t1")
    assert rec is not None and rec.status == "dead"


async def test_reconcile_is_best_effort_when_enumeration_fails() -> None:
    # If listing live containers fails (DockerRuntime.list fails closed), leave
    # records running rather than risk marking a live container dead.
    class _BoomBackend(_FakeBackend):
        async def live_container_ids(self) -> Set[str]:
            raise RuntimeError("docker ps failed")

    registry = SessionRegistry()
    registry.add("t1", container_id="c1")
    cockpit = Cockpit(
        CliTransport(inbox=[]), _HostBrain(), InMemoryCheckpointStore(),
        registry=registry, container=_BoomBackend(LaunchSpec(image="img")),
    )

    await cockpit.serve()

    rec = registry.get("t1")
    assert rec is not None and rec.status == "running"  # not marked dead on uncertainty


async def test_failed_in_container_run_leaves_a_killable_record() -> None:
    # If the in-container brain ends WITHOUT a terminal DoneEvent the run fails,
    # but on_start already registered the container — so the record isn't silently
    # lost: the operator can /kill it (and restart-reconcile would mark it dead).
    class _NoDoneBackend(_FakeBackend):
        async def start(self, spec: LaunchSpec, stdin: bytes) -> _FakeContainerSession:
            self.started.append((spec, stdin))
            return _FakeContainerSession("cid-x", [])  # no events → no DoneEvent

    registry = SessionRegistry()
    transport = CliTransport(inbox=[_launch()])
    cockpit = Cockpit(
        transport, _HostBrain(), InMemoryCheckpointStore(),
        registry=registry, container=_NoDoneBackend(LaunchSpec(image="img")),
    )

    await cockpit.serve()

    assert "internal error" in transport.sent[0][1]  # the run failed (no DoneEvent)
    rec = registry.get("t1")
    assert rec is not None and rec.container_id == "cid-x"  # tracked for /kill, not lost


def test_set_container_updates_in_place_and_preserves_granted_caps() -> None:
    # on_start uses set_container, not add: a re-launch must keep granted_caps
    # (a full-replace add() would silently wipe them once PR3 records grants).
    registry = SessionRegistry()
    registry.add("t1", container_id="old", image="img", granted_caps=frozenset({"memory:read"}))
    registry.set_container("t1", container_id="new", image="img2")
    rec = registry.get("t1")
    assert rec is not None
    assert rec.container_id == "new" and rec.image == "img2" and rec.status == "running"
    assert rec.granted_caps == frozenset({"memory:read"})  # preserved across the re-launch


def test_set_container_creates_record_when_absent() -> None:
    registry = SessionRegistry()
    registry.set_container("t1", container_id="cid")  # fresh LAUNCH, no prior record
    rec = registry.get("t1")
    assert rec is not None and rec.container_id == "cid" and rec.status == "running"


def test_set_container_without_image_keeps_existing_image() -> None:
    # A re-launch that doesn't restate the image must not wipe it to None.
    registry = SessionRegistry()
    registry.add("t1", container_id="old", image="kagura-agent:python")
    registry.set_container("t1", container_id="new")  # no image arg
    rec = registry.get("t1")
    assert rec is not None and rec.container_id == "new" and rec.image == "kagura-agent:python"


async def test_no_backend_keeps_in_process_execution() -> None:
    # Backward compatible: with no container backend the brain runs in-process and
    # the session is registered without a container id (today's behaviour).
    registry = SessionRegistry()
    transport = CliTransport(inbox=[_launch(text="hi")])
    cockpit = Cockpit(transport, _HostBrain(), InMemoryCheckpointStore(), registry=registry)

    await cockpit.serve()

    assert transport.sent[0] == ("t1", "host: hi")  # ran on the host
    rec = registry.get("t1")
    assert rec is not None
    assert rec.container_id is None and rec.status == "running"
