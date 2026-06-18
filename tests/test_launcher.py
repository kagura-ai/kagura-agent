"""v0.2: the Launcher ties the membrane to a container runtime, fail-closed.

A bad spec must be rejected *before* the runtime is ever touched.
"""

import pytest

from kagura_agent.membrane.launcher import LaunchSpec, MembraneViolation, Mount
from kagura_agent.membrane.runtime import Launcher


class FakeRuntime:
    def __init__(self) -> None:
        self.ran: list[list[str]] = []
        self.killed: list[str] = []

    async def run(self, args: list[str]) -> str:
        self.ran.append(args)
        return f"container-{len(self.ran)}"

    async def list(self) -> list[str]:
        return [f"container-{i + 1}" for i in range(len(self.ran))]

    async def kill(self, container_id: str) -> None:
        self.killed.append(container_id)


async def test_launch_validates_then_runs() -> None:
    runtime = FakeRuntime()
    launcher = Launcher(runtime=runtime, project_root="/work/project")
    spec = LaunchSpec(
        image="kagura-agent:python",
        mounts=(Mount(source="/work/project/src", target="/work/src"),),
    )

    container_id = await launcher.launch(spec)

    assert container_id == "container-1"
    assert runtime.ran and runtime.ran[0][:2] == ["docker", "run"]


async def test_launch_resolves_each_mount_once(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # validate -> run on the same resolved path: through the Launcher a symlink is
    # resolved once, so the path validated is exactly the path mounted (no second
    # resolution a swap could redirect = no validate->run TOCTOU).
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    runtime = FakeRuntime()
    launcher = Launcher(runtime=runtime, project_root=str(tmp_path))
    spec = LaunchSpec(image="x", mounts=(Mount(source=str(link), target="/w"),))

    await launcher.launch(spec)

    assert f"{real}:/w:ro" in runtime.ran[0]  # the validated, resolved path is mounted


async def test_launch_enforces_egress_in_path_end_to_end() -> None:
    # Integration (#92): an egress-granted launch goes validate_spec -> docker_run_args
    # and the args the runtime receives carry the FULL in-path enforcement — joins
    # the proxy network (not host/none), is routed through the proxy at the app layer,
    # and carries its per-run allowlist label. Proves the membrane launch path itself
    # enforces egress, not just the standalone arg builder.
    from kagura_agent.membrane.egress import EGRESS_ALLOW_LABEL, EGRESS_NETWORK

    runtime = FakeRuntime()
    launcher = Launcher(runtime=runtime, project_root="/work/project")
    spec = LaunchSpec(image="kagura-agent:python", egress_allow=("api.anthropic.com",))

    await launcher.launch(spec)

    args = runtime.ran[0]
    joined = " ".join(args)
    assert f"--network {EGRESS_NETWORK}" in joined and "--network host" not in joined
    assert "-e" in args and "HTTP_PROXY=http://egress-proxy:3128" in args
    assert f"{EGRESS_ALLOW_LABEL}=api.anthropic.com" in args


async def test_launcher_reconcile_and_kill_delegate_to_runtime() -> None:
    runtime = FakeRuntime()
    launcher = Launcher(runtime=runtime, project_root="/work/project")
    await launcher.launch(LaunchSpec(image="x"))

    assert await launcher.reconcile() == ["container-1"]  # cockpit restart reconciliation
    await launcher.kill("container-1")
    assert runtime.killed == ["container-1"]


async def test_launch_rejects_bad_spec_without_touching_runtime() -> None:
    runtime = FakeRuntime()
    launcher = Launcher(runtime=runtime, project_root="/work/project")
    spec = LaunchSpec(
        image="x",
        mounts=(Mount(source="/var/run/docker.sock", target="/var/run/docker.sock"),),
    )

    with pytest.raises(MembraneViolation):
        await launcher.launch(spec)

    assert runtime.ran == []  # fail-closed: nothing launched
