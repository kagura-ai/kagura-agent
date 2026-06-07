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
