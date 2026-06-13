"""Regression tests for ultrareview findings (bug_001/003/019/021/026/029,
merged_006/007/009). Each test reproduces a reported bug; the fix makes it pass.
"""

import inspect
from collections.abc import AsyncIterator

import pytest

from kagura_agent.cockpit.core import Cockpit
from kagura_agent.cockpit.hitl import CapabilityRequest, HitlGate
from kagura_agent.cockpit.transports.base import Event
from kagura_agent.cockpit.transports.cli import CliTransport
from kagura_agent.cockpit.transports.discord import DiscordTransport
from kagura_agent.cockpit.transports.slack import SlackTransport, normalize_slack_event
from kagura_agent.core.brain.base import BrainCaps, BrainEvent, Checkpoint, DoneEvent, Task
from kagura_agent.mcp.memory_cloud import LocalMemoryClient
from kagura_agent.membrane.graduation import GraduationEngine, GraduationPolicy
from kagura_agent.membrane.launcher import LaunchSpec, MembraneViolation, Mount, validate_spec
from kagura_agent.membrane.lease import Budget, CredentialBroker, LeaseLedger
from kagura_agent.patterns.checkpoint import InMemoryCheckpointStore

DAY = 86400


def _clock() -> float:
    return 1000.0


class _Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class RecordingBrain:
    caps = BrainCaps(name="rec")

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def run(
        self, task: Task, *, resume: Checkpoint | None = None
    ) -> AsyncIterator[BrainEvent]:
        self.prompts.append(task.prompt)
        yield DoneEvent(result=f"ok: {task.prompt}", state={"turn": 1})


# --- bug_021: membrane symlink bypass ------------------------------------

def test_symlink_inside_project_root_to_outside_is_rejected(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "secret"
    outside.mkdir()
    link = project / "escape"
    link.symlink_to(outside)  # lexically inside project, really outside

    spec = LaunchSpec(image="x", mounts=(Mount(source=str(link), target="/x"),))
    with pytest.raises(MembraneViolation):
        validate_spec(spec, project_root=str(project))


# --- bug_003: graduation must recover after failure + cooldown -----------

def test_failure_then_cooldown_then_resuccess_requalifies() -> None:
    clock = _Clock()
    engine = GraduationEngine(GraduationPolicy(cooldown_seconds=7 * DAY), clock=clock)
    for i in range(5):
        engine.record_success("apt", task_id=f"t{i % 3}", verified=True)

    engine.record_failure("apt")
    assert engine.should_propose("apt", input_trust="trusted") is False  # demoted

    clock.t += 7 * DAY + 1  # cooldown elapses
    for i in range(5):  # re-accrue verified successes
        engine.record_success("apt", task_id=f"r{i % 3}", verified=True)

    assert engine.should_propose("apt", input_trust="trusted") is True  # recovered


# --- merged_006: closed session must not route as CONTINUE ----------------

class LaunchOnlyBrain:
    caps = BrainCaps(name="launchonly")

    async def run(
        self, task: Task, *, resume: Checkpoint | None = None
    ) -> AsyncIterator[BrainEvent]:
        if resume is not None:
            raise AssertionError("must not resume a killed session")
        yield DoneEvent(result=f"ok: {task.prompt}", state={"turn": 1})


async def test_reply_after_kill_launches_fresh_not_resume() -> None:
    transport = CliTransport(
        inbox=[
            Event(thread_id="t1", text="do work", is_thread_reply=False),
            Event(thread_id="t1", text="/kill", is_thread_reply=True),
            Event(thread_id="t1", text="new task", is_thread_reply=True),
        ]
    )
    cockpit = Cockpit(transport, LaunchOnlyBrain(), InMemoryCheckpointStore())
    await cockpit.serve()
    assert transport.sent[2] == ("t1", "ok: new task")  # fresh launch, no resume


# --- bug_001: typed /approve must not launch a brain ----------------------

async def test_typed_approve_does_not_launch_brain() -> None:
    transport = CliTransport(
        inbox=[
            Event(thread_id="t1", text="do work", is_thread_reply=False),
            Event(thread_id="t1", text="/approve", is_thread_reply=True),
        ]
    )
    brain = RecordingBrain()
    cockpit = Cockpit(transport, brain, InMemoryCheckpointStore())
    await cockpit.serve()

    assert brain.prompts == ["do work"]  # /approve did NOT spin a brain run
    assert "no pending approval" in transport.sent[1][1].lower()


# --- bug_026: HitlGate fails closed on any transport error ----------------

class BoomTransport:
    async def listen(self) -> AsyncIterator[Event]:  # pragma: no cover
        if False:
            yield Event("", "", False)

    async def send(self, thread_id: str, text: str) -> None:  # pragma: no cover
        pass

    async def ask(self, thread_id: str, question: str, options: list[str]) -> str:
        raise RuntimeError("slack down")


async def test_hitl_transport_error_fails_closed_and_records() -> None:
    memory = LocalMemoryClient()
    gate = HitlGate(transport=BoomTransport(), memory=memory)

    decision = await gate.review(CapabilityRequest("t1", "aws:s3:write", "upload"))

    assert decision.approved is False
    trail = await memory.recall("aws:s3:write", tags=("graduation-trail",))
    assert trail  # audit entry written despite the transport error


# --- bug_029: one bad event must not kill the cockpit loop ----------------

class FlakyBrain:
    caps = BrainCaps(name="flaky")

    async def run(
        self, task: Task, *, resume: Checkpoint | None = None
    ) -> AsyncIterator[BrainEvent]:
        if task.prompt == "boom":
            raise RuntimeError("brain blew up")
        yield DoneEvent(result=f"ok: {task.prompt}", state={"turn": 1})


async def test_serve_survives_a_handler_exception() -> None:
    transport = CliTransport(
        inbox=[
            Event(thread_id="t1", text="boom", is_thread_reply=False),
            Event(thread_id="t2", text="fine", is_thread_reply=False),
        ]
    )
    cockpit = Cockpit(transport, FlakyBrain(), InMemoryCheckpointStore())
    await cockpit.serve()  # must not raise
    assert ("t2", "ok: fine") in transport.sent


# --- merged_009: stateful lease lifecycle ---------------------------------

class FakeStatefulProvider:
    stateful = True

    def __init__(self) -> None:
        self.minted = 0
        self.revoked: list[str] = []

    async def mint(self, scope: str, ttl: int) -> tuple[str, str | None]:
        self.minted += 1
        return f"cf-token-{self.minted}", f"cf-handle-{self.minted}"

    async def revoke(self, handle: str | None) -> None:
        assert handle is not None
        self.revoked.append(handle)


async def test_renew_revokes_old_stateful_handle() -> None:
    stateful = FakeStatefulProvider()
    broker = CredentialBroker({"cf": stateful}, clock=_clock)
    lease = await broker.acquire("cf", scope="z", ttl=300, budget=Budget(3600))

    await broker.renew(lease, ttl=300)

    assert stateful.revoked == ["cf-handle-1"]  # old token revoked, not leaked


class FlakyRevokeProvider:
    stateful = True

    def __init__(self) -> None:
        self.minted = 0
        self.revoked: list[str] = []

    async def mint(self, scope: str, ttl: int) -> tuple[str, str | None]:
        self.minted += 1
        return f"t{self.minted}", f"h{self.minted}"

    async def revoke(self, handle: str | None) -> None:
        if handle == "h1":
            raise RuntimeError("404 token not found")
        assert handle is not None
        self.revoked.append(handle)


async def test_sweep_continues_past_a_failing_revoke() -> None:
    provider = FlakyRevokeProvider()
    broker = CredentialBroker({"cf": provider}, clock=_clock, ledger=LeaseLedger())
    await broker.acquire("cf", scope="z", ttl=300, budget=Budget(3600))  # h1 (will fail)
    await broker.acquire("cf", scope="z", ttl=300, budget=Budget(3600))  # h2

    await broker.sweep()  # must not raise despite h1 failing

    assert "h2" in provider.revoked
    await broker.sweep()  # h1 was forgotten — second sweep does not re-hit it


# --- bug_019: Slack subtype events must be dropped ------------------------

def test_slack_message_changed_is_ignored() -> None:
    payload = {
        "type": "message",
        "subtype": "message_changed",
        "channel": "C1",
        "ts": "2.0",
        "message": {"text": "edited", "user": "U1"},
    }
    assert normalize_slack_event(payload, bot_user_id="UBOT") is None


def test_slack_channel_join_is_ignored() -> None:
    payload = {
        "type": "message",
        "subtype": "channel_join",
        "user": "U1",
        "text": "has joined",
        "ts": "3.0",
    }
    assert normalize_slack_event(payload, bot_user_id="UBOT") is None


# --- merged_007: listen() must be an async generator; send() honest -------

def test_transport_listen_is_async_generator() -> None:
    assert inspect.isasyncgenfunction(SlackTransport.listen)
    assert inspect.isasyncgenfunction(DiscordTransport.listen)


class _FakeSlackClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []  # type: ignore[type-arg]

    async def chat_postMessage(self, **kw: object) -> None:
        self.calls.append(dict(kw))


class _FakeSlackApp:
    """Minimal stand-in for slack_bolt's AsyncApp: no-op decorators + a client."""

    def __init__(self) -> None:
        self.client = _FakeSlackClient()

    def event(self, _name: str):  # type: ignore[no-untyped-def]
        return lambda fn: fn

    def action(self, _spec: object):  # type: ignore[no-untyped-def]
        return lambda fn: fn


async def test_slack_send_posts_to_recorded_channel() -> None:
    app = _FakeSlackApp()
    transport = SlackTransport(app=app, bot_user_id="U", channel_map={"1.0": "C9"})
    await transport.send("1.0", "hello")
    assert app.client.calls == [{"channel": "C9", "thread_ts": "1.0", "text": "hello"}]


async def test_slack_send_unknown_thread_fails_loud() -> None:
    # No recorded channel for the thread → raise, never post to a guessed channel.
    transport = SlackTransport(app=_FakeSlackApp(), bot_user_id="U")
    with pytest.raises(KeyError):
        await transport.send("ghost", "hi")
