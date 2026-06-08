"""v0.3: the cockpit handles status and kill intents, not just launch/continue."""

from collections.abc import AsyncIterator

from kagura_agent.cockpit.core import Cockpit
from kagura_agent.cockpit.registry import SessionRegistry
from kagura_agent.cockpit.transports.base import Event
from kagura_agent.cockpit.transports.cli import CliTransport
from kagura_agent.core.brain.base import BrainCaps, BrainEvent, Checkpoint, DoneEvent, Task
from kagura_agent.patterns.checkpoint import InMemoryCheckpointStore


class OneShotBrain:
    caps = BrainCaps(name="oneshot")

    async def run(
        self, task: Task, *, resume: Checkpoint | None = None
    ) -> AsyncIterator[BrainEvent]:
        yield DoneEvent(result=f"ok: {task.prompt}", state={"turn": 1})


class FakeLauncher:
    def __init__(self) -> None:
        self.killed: list[str] = []

    async def kill(self, container_id: str) -> None:
        self.killed.append(container_id)


async def test_status_reports_running_session() -> None:
    transport = CliTransport(
        inbox=[
            Event(thread_id="t1", text="do work", is_thread_reply=False),
            Event(thread_id="t1", text="/status", is_thread_reply=True),
        ]
    )
    cockpit = Cockpit(transport, OneShotBrain(), InMemoryCheckpointStore())

    await cockpit.serve()

    assert transport.sent[0] == ("t1", "ok: do work")
    assert "running" in transport.sent[1][1]


async def test_kill_terminates_container_and_closes_session() -> None:
    registry = SessionRegistry()
    registry.add("t1", container_id="c1")
    launcher = FakeLauncher()
    transport = CliTransport(inbox=[Event(thread_id="t1", text="/kill", is_thread_reply=True)])
    cockpit = Cockpit(
        transport,
        OneShotBrain(),
        InMemoryCheckpointStore(),
        registry=registry,
        launcher=launcher,
    )

    await cockpit.serve()

    assert launcher.killed == ["c1"]
    assert registry.get("t1").status == "closed"
    assert "killed" in transport.sent[0][1].lower()
