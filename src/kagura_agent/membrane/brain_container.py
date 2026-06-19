"""#102 (PR3): the deployment edge that runs the brain in a real container.

`build_brain_launch_spec` is the pure, testable heart — it builds the per-run
:class:`LaunchSpec` and is where #113's BYOK auth decision lands. `DockerBrainBackend`
is the cockpit's container backend (the `spec_for` / `start` / `live_container_ids`
shape `cockpit.core` expects): `spec_for` is tested, while `start` (the streaming
`docker run`) and `live_container_ids` (`docker ps`) are the deployment edge,
`# pragma: no cover` like `membrane.runtime.DockerRuntime`.

Auth (#113): the containerized brain authenticates with **BYOK** —
`ANTHROPIC_API_KEY` is injected into the egress-sealed container, whose allowlist
always includes `api.anthropic.com` so the brain can reach its LLM endpoint.
Subscription auth stays the in-process default and never enters the container.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Set

from kagura_agent.core.brain.container import BrainContainerSession
from kagura_agent.membrane.launcher import LaunchSpec, Mount

log = logging.getLogger(__name__)

#: The container path the project root is mounted at (read-only).
_WORKSPACE = "/workspace"
#: The brain's own LLM endpoint — always egress-allowed so the BYOK call succeeds.
_ANTHROPIC_HOST = "api.anthropic.com"
#: Auth-mode env vars (the ones `core/brain/auth.py` reads). The brain's auth is
#: BYOK via the dedicated seam ONLY, so deployer-supplied tool credentials must not
#: carry these — else a tool cred could override the validated BYOK key or sneak
#: subscription auth into the container (which must never happen, #113).
_RESERVED_AUTH_ENV = frozenset({"ANTHROPIC_API_KEY", "CLAUDE_CODE_SUBSCRIPTION"})


def build_brain_launch_spec(
    *,
    image: str,
    project_root: str,
    byok_key: str,
    egress_allow: tuple[str, ...] = (),
    tool_creds_env: Mapping[str, str] | None = None,
) -> LaunchSpec:
    """Build the per-run :class:`LaunchSpec` for the in-container brain (#102/#113).

    - the project root is mounted **read-only** at ``/workspace``;
    - ``ANTHROPIC_API_KEY`` (BYOK) is injected — the in-container brain cannot use
      subscription auth, so a missing/blank key is a **fail-closed** ``ValueError``
      rather than a silent fallback to ambient/no auth;
    - the egress allowlist always includes ``api.anthropic.com`` (so the BYOK call
      reaches Anthropic), merged with any caller-supplied hosts the tools need;
    - ``tool_creds_env`` (leased tool credentials) is merged into the env.
    """
    # Store the STRIPPED key: a key read from a file/env often has a trailing
    # newline, which would otherwise be injected verbatim and 401 at Anthropic.
    key = byok_key.strip()
    if not key:
        raise ValueError(
            "in-container brain requires a BYOK ANTHROPIC_API_KEY — subscription auth "
            "does not run inside the container (set ANTHROPIC_API_KEY or run in-process)"
        )
    creds = dict(tool_creds_env or {})
    # Fail-closed: the brain's auth is BYOK only. Deployer-supplied tool creds must
    # not carry an auth-mode var — that could override the validated BYOK key with
    # an attacker/wrong key, or sneak subscription auth into the container (#113).
    reserved = _RESERVED_AUTH_ENV & creds.keys()
    if reserved:
        raise ValueError(
            f"tool_creds_env must not set auth-mode variable(s) {sorted(reserved)}: the "
            "in-container brain authenticates via the BYOK ANTHROPIC_API_KEY seam only"
        )
    # BYOK key set LAST so it is authoritative even if the reserved check ever missed
    # a variant; tool creds fill the rest of the env.
    env: dict[str, str] = {**creds, "ANTHROPIC_API_KEY": key}
    # Always allow the brain's LLM endpoint; drop any case-variant duplicate of it
    # from the caller's list (EgressPolicy lowercases hosts at enforcement).
    extra = tuple(h for h in egress_allow if h.lower() != _ANTHROPIC_HOST)
    egress = (_ANTHROPIC_HOST, *extra)
    return LaunchSpec(
        image=image,
        mounts=(Mount(source=project_root, target=_WORKSPACE, read_only=True),),
        egress_allow=egress,
        env=env,
    )


class DockerBrainBackend:
    """The cockpit's container backend, backed by real ``docker``.

    ``resolve_byok`` reads the host-side BYOK key at spec-build time (so the key is
    never held longer than a run needs it); ``egress_allow`` / ``tool_creds_env``
    are the per-deploy tool egress + leased credentials. ``start`` and
    ``live_container_ids`` shell out to docker and are the deployment edge.
    """

    def __init__(
        self,
        *,
        image: str,
        project_root: str,
        resolve_byok: Callable[[], str],
        egress_allow: tuple[str, ...] = (),
        tool_creds_env: Mapping[str, str] | None = None,
    ) -> None:
        self.project_root = project_root
        self._image = image
        self._resolve_byok = resolve_byok
        self._egress_allow = tuple(egress_allow)
        self._tool_creds_env = dict(tool_creds_env or {})

    def spec_for(self, session_id: str) -> LaunchSpec:
        return build_brain_launch_spec(
            image=self._image,
            project_root=self.project_root,
            byok_key=self._resolve_byok(),
            egress_allow=self._egress_allow,
            tool_creds_env=self._tool_creds_env,
        )

    async def start(  # pragma: no cover - real streaming `docker run`
        self, spec: LaunchSpec, stdin: bytes
    ) -> BrainContainerSession:
        """Run the brain container ATTACHED, feeding the run input on stdin and
        streaming its stdout event lines. The container is named up front so its id
        is known before the stream starts (the cockpit registers it for /kill)."""
        import asyncio
        import uuid

        from kagura_agent.membrane.launcher import docker_run_args, validate_spec

        # Defense-in-depth: never launch an unvalidated spec at this boundary, even
        # though the cockpit validates first — re-validate and run the resolved spec.
        resolved = validate_spec(spec, project_root=self.project_root)
        name = f"kagura-brain-{uuid.uuid4().hex[:12]}"
        args = docker_run_args(resolved)
        # Splice in -i (the entrypoint reads the run input on stdin) and a stable
        # --name (so container_id is known up front), after the `docker run` verb.
        args = [args[0], args[1], "-i", "--name", name, *args[2:]]
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            # stderr INHERITS the cockpit's stderr (not a pipe): container logs stay
            # visible AND an undrained stderr buffer can never deadlock the run.
        )
        # Feed stdin in the background so a large run input cannot deadlock against a
        # container that floods stdout before it has consumed all of stdin (the
        # stdout reader in events() runs concurrently with this feeder).
        async def _feed() -> None:
            assert proc.stdin is not None
            try:
                proc.stdin.write(stdin)
                await proc.stdin.drain()
                proc.stdin.write_eof()
            except (BrokenPipeError, ConnectionResetError, OSError):
                # The container closed stdin early (read its input then exited, or
                # crashed). Not fatal — the stdout stream + Session's no-DoneEvent
                # fail-closed decide the outcome; log so the feed failure isn't a
                # silent unretrieved-task exception.
                log.warning("brain container %s: feeding run input failed", name)

        return _DockerBrainSession(name, proc, feeder=asyncio.create_task(_feed()))

    async def live_container_ids(self) -> Set[str]:  # pragma: no cover - real `docker ps`
        # Reuse the single docker-enumeration seam (DockerRuntime.list, fail-closed
        # on a docker error) rather than a second `docker ps` copy that could drift
        # from the launcher's reconcile on a label/filter change.
        from kagura_agent.membrane.runtime import DockerRuntime

        return frozenset(await DockerRuntime().list())


class _DockerBrainSession:  # pragma: no cover - wraps a live docker subprocess
    """A live brain container: its id (the --name) + its stdout event-line stream."""

    def __init__(self, container_id: str, proc: object, *, feeder: object) -> None:
        self.container_id = container_id
        self._proc = proc
        self._feeder = feeder  # keep the stdin-feeder task referenced (no GC mid-run)

    async def events(self):  # type: ignore[no-untyped-def]
        stdout = self._proc.stdout  # type: ignore[attr-defined]
        async for raw in stdout:
            yield raw.decode().rstrip("\n")
