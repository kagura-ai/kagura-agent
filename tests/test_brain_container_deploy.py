"""#102 (PR3): the deployment edge — brain entrypoint + the LaunchSpec builder.

The CI-testable core of PR3 is the *spec* (where #113's BYOK auth decision lands)
and the in-container *entrypoint core* (runs the real brain on the stdin run
input). The real streaming ``docker run`` and the agent-image Dockerfile are the
deployment edge (``# pragma: no cover`` / build glue), like ``DockerRuntime``.

Auth model (#113): the containerized brain authenticates with BYOK
(``ANTHROPIC_API_KEY``) inside the egress-sealed container — subscription auth
stays the in-process default and never enters the container.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping

import pytest

from kagura_agent.core.brain.base import (
    BrainCaps,
    BrainEvent,
    Checkpoint,
    DoneEvent,
    MessageEvent,
    Task,
)
from kagura_agent.core.brain.container import decode_event, encode_run_input
from kagura_agent.core.brain.container_main import run_brain_entrypoint
from kagura_agent.membrane.brain_container import (
    DockerBrainBackend,
    _DockerBrainSession,
    build_brain_launch_spec,
)
from kagura_agent.membrane.launcher import Mount, validate_spec

# --------------------------------------------------------------------------
# build_brain_launch_spec — where #113's BYOK decision is wired
# --------------------------------------------------------------------------


def test_spec_has_image_ro_mount_byok_env_and_anthropic_egress():
    spec = build_brain_launch_spec(
        image="kagura-agent:python", project_root="/proj", byok_key="sk-ant-xyz"
    )
    assert spec.image == "kagura-agent:python"
    assert spec.mounts == (Mount(source="/proj", target="/workspace", read_only=True),)
    assert spec.env["ANTHROPIC_API_KEY"] == "sk-ant-xyz"  # BYOK, the #113 decision
    assert "CLAUDE_CODE_SUBSCRIPTION" not in spec.env  # subscription never enters the container
    assert "api.anthropic.com" in spec.egress_allow  # the brain can reach its LLM endpoint


def test_spec_merges_extra_egress_and_tool_creds():
    spec = build_brain_launch_spec(
        image="img",
        project_root="/p",
        byok_key="k",
        egress_allow=("github.com",),
        tool_creds_env={"GH_TOKEN": "t"},
    )
    assert "api.anthropic.com" in spec.egress_allow and "github.com" in spec.egress_allow
    assert spec.env["ANTHROPIC_API_KEY"] == "k" and spec.env["GH_TOKEN"] == "t"


def test_spec_does_not_duplicate_anthropic_in_egress():
    spec = build_brain_launch_spec(
        image="img", project_root="/p", byok_key="k", egress_allow=("api.anthropic.com",)
    )
    assert spec.egress_allow.count("api.anthropic.com") == 1


@pytest.mark.parametrize("auth_var", ["ANTHROPIC_API_KEY", "CLAUDE_CODE_SUBSCRIPTION"])
def test_spec_rejects_auth_vars_smuggled_via_tool_creds(auth_var):
    # The membrane must enforce its own #113 invariant: a tool credential must not
    # carry an auth-mode var — that could override the validated BYOK key with an
    # attacker/wrong key, or sneak subscription auth into the container.
    with pytest.raises(ValueError, match="auth-mode"):
        build_brain_launch_spec(
            image="img", project_root="/p", byok_key="k", tool_creds_env={auth_var: "evil"}
        )


def test_byok_key_is_authoritative_over_tool_creds_order():
    # Even constructed normally, the BYOK key the host resolved is the one that ends
    # up in the env (defense-in-depth: it is set last).
    spec = build_brain_launch_spec(
        image="img", project_root="/p", byok_key="host-key", tool_creds_env={"GH_TOKEN": "t"}
    )
    assert spec.env["ANTHROPIC_API_KEY"] == "host-key"
    assert spec.env["GH_TOKEN"] == "t"


def test_spec_fails_closed_without_byok_key():
    # The in-container brain CANNOT use subscription auth, so a missing BYOK key is
    # a fail-closed refusal, not a silent fallback to ambient/none.
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_brain_launch_spec(image="img", project_root="/p", byok_key="   ")


def test_spec_passes_the_membrane_validate_gate(tmp_path):
    # The spec the backend hands the cockpit must clear validate_spec (the cockpit
    # validates before launch); the project-root RO mount resolves and is allowed.
    spec = build_brain_launch_spec(image="img", project_root=str(tmp_path), byok_key="k")
    validated = validate_spec(spec, project_root=str(tmp_path))
    assert validated.mounts[0].read_only is True
    assert "api.anthropic.com" in validated.egress_allow


def test_spec_carries_byok_and_ro_mount_into_real_docker_args(tmp_path):
    # End-to-end at the argv level: the BYOK key must actually reach docker as an
    # `-e ANTHROPIC_API_KEY=...` env flag, and the project mount as a read-only
    # `-v ...:/workspace:ro` — the security-critical payload of #113, all pure.
    from kagura_agent.membrane.launcher import docker_run_args

    spec = build_brain_launch_spec(image="img", project_root=str(tmp_path), byok_key="sk-key")
    args = docker_run_args(validate_spec(spec, project_root=str(tmp_path)))
    assert "ANTHROPIC_API_KEY=sk-key" in args  # the BYOK key reaches the container env
    assert any(a.endswith(":/workspace:ro") for a in args)  # project root mounted read-only


def test_byok_key_is_stripped_before_injection():
    # A key read from a file/env often has a trailing newline; storing it verbatim
    # would 401 at Anthropic. The stored key must be stripped.
    spec = build_brain_launch_spec(image="img", project_root="/p", byok_key="  sk-key\n")
    assert spec.env["ANTHROPIC_API_KEY"] == "sk-key"


def test_spec_dedups_anthropic_case_insensitively():
    # A case-variant of the always-allowed host must not produce a redundant entry.
    spec = build_brain_launch_spec(
        image="img", project_root="/p", byok_key="k", egress_allow=("API.ANTHROPIC.COM",)
    )
    assert sum(1 for h in spec.egress_allow if h.lower() == "api.anthropic.com") == 1


# --------------------------------------------------------------------------
# DockerBrainBackend — spec_for resolves the BYOK key host-side
# --------------------------------------------------------------------------


def test_backend_spec_for_resolves_byok_key():
    backend = DockerBrainBackend(
        image="kagura-agent:python", project_root="/p", resolve_byok=lambda: "sk-host-key"
    )
    spec = backend.spec_for("s1")
    assert spec.env["ANTHROPIC_API_KEY"] == "sk-host-key"
    assert backend.project_root == "/p"
    assert spec.image == "kagura-agent:python"


def test_backend_spec_for_fails_closed_when_byok_absent():
    backend = DockerBrainBackend(image="img", project_root="/p", resolve_byok=lambda: "")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        backend.spec_for("s1")


# --------------------------------------------------------------------------
# entrypoint core — runs the real brain inside the container
# --------------------------------------------------------------------------


class _FakeBrain:
    caps = BrainCaps(name="claude", auth_modes=("byok",))

    async def run(
        self, task: Task, *, resume: Checkpoint | None = None
    ) -> AsyncIterator[BrainEvent]:
        yield MessageEvent(text=f"prompt={task.prompt}")
        yield DoneEvent(result="ok", state={"turn": 1})


async def test_entrypoint_propagates_make_brain_failure_without_emitting():
    # A brain that can't be built (e.g. BrainUnavailable / AuthError on a bad BYOK
    # key) must fail the run, never emit a spurious DoneEvent that would look like a
    # clean result — the container exits non-zero, the host Session fails closed.
    def make_brain(env: Mapping[str, str]) -> _FakeBrain:
        raise RuntimeError("brain construction failed")

    emitted: list[str] = []
    payload = encode_run_input(Task(prompt="p", session_id="s"), None)
    with pytest.raises(RuntimeError, match="brain construction failed"):
        await run_brain_entrypoint(payload, make_brain=make_brain, env={}, emit=emitted.append)
    assert emitted == []  # nothing emitted on a build failure


async def test_entrypoint_threads_resume_into_the_brain():
    # A dropped resume would silently restart a resumed session from scratch — the
    # entrypoint must thread the resume Checkpoint into brain.run(resume=...).
    seen: dict[str, Checkpoint | None] = {}

    class _ResumeBrain:
        caps = BrainCaps(name="claude")

        async def run(
            self, task: Task, *, resume: Checkpoint | None = None
        ) -> AsyncIterator[BrainEvent]:
            seen["resume"] = resume
            yield DoneEvent(result="ok")

    resume = Checkpoint(session_id="s", turn=2, state={"turn": 2})
    payload = encode_run_input(Task(prompt="p", session_id="s"), resume)
    await run_brain_entrypoint(
        payload, make_brain=lambda env: _ResumeBrain(), env={}, emit=lambda _line: None
    )
    assert seen["resume"] == resume


async def test_entrypoint_builds_brain_from_env_and_emits_the_protocol():
    seen_env: dict[str, str] = {}

    def make_brain(env: Mapping[str, str]) -> _FakeBrain:
        seen_env.update(env)
        return _FakeBrain()

    emitted: list[str] = []
    payload = encode_run_input(Task(prompt="hello", session_id="s"), None)
    await run_brain_entrypoint(
        payload, make_brain=make_brain, env={"ANTHROPIC_API_KEY": "k"}, emit=emitted.append
    )

    # The entrypoint built the brain from the injected (BYOK) env and emitted the
    # brain's events as the wire protocol the host decodes.
    assert seen_env["ANTHROPIC_API_KEY"] == "k"
    assert [decode_event(line) for line in emitted] == [
        MessageEvent(text="prompt=hello"),
        DoneEvent(result="ok", state={"turn": 1}),
    ]


# --------------------------------------------------------------------------
# _DockerBrainSession — the BOUNDED stdout stream (#123): idle/wall-clock
# timeout, oversized-line guard, and guaranteed teardown. proc + feeder are
# injected, so the bounding/teardown logic is unit-tested with fakes (only the
# real `docker run` in DockerBrainBackend.start is the deployment edge).
# --------------------------------------------------------------------------


class _FakeStdout:
    def __init__(self, lines, *, stall=False, raise_value_error=False):
        self._lines = list(lines)
        self._stall = stall
        self._raise = raise_value_error

    async def readline(self) -> bytes:
        if self._stall:
            await asyncio.Event().wait()  # never resolves → exercises the idle timeout
        if self._raise:
            raise ValueError("Separator is not found, and chunk exceed the limit")
        return self._lines.pop(0) if self._lines else b""  # b"" == EOF


class _FakeProc:
    def __init__(self, stdout, *, returncode=None, wait_stalls=False):
        self.stdout = stdout
        self.returncode = returncode
        self.terminated = 0
        self.killed = False
        self.waited = 0
        self._wait_stalls = wait_stalls

    def terminate(self) -> None:
        self.terminated += 1  # does NOT set returncode (a container may ignore SIGTERM)

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        self.waited += 1
        if self._wait_stalls and not self.killed:
            await asyncio.Event().wait()  # SIGTERM ignored — only kill() ends the wait
        if self.returncode is None:
            self.returncode = -15
        return self.returncode


def _session(lines=(), *, returncode=None, stall=False, raise_value_error=False,
             wait_stalls=False, **kw):
    proc = _FakeProc(
        _FakeStdout(lines, stall=stall, raise_value_error=raise_value_error),
        returncode=returncode, wait_stalls=wait_stalls,
    )
    feeder = asyncio.create_task(asyncio.sleep(3600))  # a real cancellable stdin-feeder
    return _DockerBrainSession("cid-1", proc, feeder=feeder, **kw), proc, feeder


# events() — bounded streaming (teardown is the separate aclose())


async def test_session_streams_lines_then_eof():
    sess, proc, feeder = _session([b"line1\n", b"line2\n"])
    out = [line async for line in sess.events()]
    assert out == ["line1", "line2"]
    await sess.aclose()  # the provider always does this; here we drive it explicitly
    assert feeder.cancelled() or feeder.done()


async def test_session_idle_timeout_fails_closed():
    sess, proc, feeder = _session(stall=True, idle_timeout_s=0.02)
    with pytest.raises(TimeoutError, match="no output"):
        async for _ in sess.events():
            pass
    await sess.aclose()


async def test_session_wall_clock_deadline_fails_closed():
    calls = {"n": 0}

    def clock() -> float:
        calls["n"] += 1
        return 0.0 if calls["n"] == 1 else 100.0  # first sets deadline; next is past it

    sess, proc, feeder = _session([b"x\n"], wall_clock_s=10.0, clock=clock)
    with pytest.raises(TimeoutError, match="wall-clock"):
        async for _ in sess.events():
            pass
    await sess.aclose()


async def test_session_oversized_line_fails_closed_with_clear_error():
    sess, proc, feeder = _session(raise_value_error=True)
    with pytest.raises(ValueError, match="oversized"):
        async for _ in sess.events():
            pass
    await sess.aclose()


# aclose() — guaranteed, idempotent, non-hanging teardown


async def test_aclose_cancels_feeder_and_terminates_proc():
    sess, proc, feeder = _session([])
    await sess.aclose()
    assert proc.terminated == 1 and proc.waited >= 1  # reaped
    assert feeder.cancelled() or feeder.done()
    assert proc.killed is False  # graceful SIGTERM sufficed


async def test_aclose_skips_terminate_when_proc_already_exited():
    sess, proc, feeder = _session([], returncode=0)  # already exited
    await sess.aclose()
    assert proc.terminated == 0  # returncode already set → no terminate/wait
    assert feeder.cancelled() or feeder.done()


async def test_aclose_escalates_to_sigkill_when_terminate_ignored():
    # A container that ignores SIGTERM must not let teardown hang forever — escalate
    # to SIGKILL after the grace period.
    sess, proc, feeder = _session([], wait_stalls=True, teardown_wait_s=0.02)
    await sess.aclose()
    assert proc.terminated == 1 and proc.killed is True


async def test_aclose_is_idempotent():
    sess, proc, feeder = _session([])
    await sess.aclose()
    await sess.aclose()  # second call is a no-op
    assert proc.terminated == 1  # not terminated twice
