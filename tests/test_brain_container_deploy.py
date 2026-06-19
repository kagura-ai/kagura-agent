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
from kagura_agent.membrane.brain_container import DockerBrainBackend, build_brain_launch_spec
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
