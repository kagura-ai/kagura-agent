"""Container runtime seam + the Launcher that drives it through the membrane.

`ContainerRuntime` is a protocol so the core never imports the Docker SDK; the
real adapter (`DockerRuntime`) shells out to `docker`. The `Launcher` enforces
the membrane (validate → hardened args) before the runtime is ever touched.
"""

from __future__ import annotations

from typing import Protocol

from kagura_agent.membrane.launcher import LaunchSpec, docker_run_args, validate_spec


class ContainerRuntime(Protocol):
    async def run(self, args: list[str]) -> str: ...

    async def list(self) -> list[str]: ...

    async def kill(self, container_id: str) -> None: ...


class Launcher:
    def __init__(self, runtime: ContainerRuntime, *, project_root: str) -> None:
        self._runtime = runtime
        self._project_root = project_root

    async def launch(self, spec: LaunchSpec) -> str:
        validate_spec(spec, project_root=self._project_root)  # fail-closed
        return await self._runtime.run(docker_run_args(spec))

    async def reconcile(self) -> list[str]:
        """Container ids currently alive (cockpit restart reconciliation)."""
        return await self._runtime.list()

    async def kill(self, container_id: str) -> None:
        await self._runtime.kill(container_id)


class DockerRuntime:  # pragma: no cover - shells out to docker
    """Real runtime: runs `docker` as a subprocess from the trusted cockpit."""

    async def run(self, args: list[str]) -> str:
        import asyncio

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"docker run failed: {err.decode().strip()}")
        return out.decode().strip()

    async def list(self) -> list[str]:
        import asyncio

        proc = await asyncio.create_subprocess_exec(
            "docker", "ps", "--filter", "label=kagura-agent", "-q",
            stdout=asyncio.subprocess.PIPE,
        )
        out, _ = await proc.communicate()
        return [line for line in out.decode().splitlines() if line]

    async def kill(self, container_id: str) -> None:
        import asyncio

        proc = await asyncio.create_subprocess_exec("docker", "kill", container_id)
        await proc.communicate()
