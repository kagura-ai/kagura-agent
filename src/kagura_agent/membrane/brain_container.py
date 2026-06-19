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

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Callable, Mapping, Set
from typing import Any

from kagura_agent.core.brain.container import BrainContainerSession
from kagura_agent.membrane.launcher import LaunchSpec, Mount

log = logging.getLogger(__name__)

#: Bound a single stdout read so a dead/hung container ends the run instead of
#: hanging ``Session._drive`` — and, because the cockpit is a SINGLE consumer, the
#: whole cockpit — forever. Generous: a slow brain (a long tool call) is normal;
#: only a container that emits NOTHING for this long is treated as hung.
_IDLE_TIMEOUT_S = 600.0
#: Absolute wall-clock backstop for one run, so a container that streams forever
#: (an endless keepalive loop that never yields a terminal DoneEvent) still ends.
_WALL_CLOCK_S = 6 * 60 * 60.0
#: StreamReader byte limit for the container's stdout. The asyncio default (64 KiB)
#: would truncate a legitimate large event (e.g. a long DoneEvent.result) with an
#: opaque error BEFORE decode_event's char cap applies; size it above that cap so the
#: decoder's clean, uniform ValueError is the binding constraint.
_STDOUT_LIMIT_BYTES = 32 * 1024 * 1024
#: Grace period for a graceful SIGTERM at teardown before escalating to SIGKILL, so
#: teardown — the very thing that bounds a hung container — can never itself hang.
_TEARDOWN_WAIT_S = 10.0

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
            # Raise the stdout line limit above decode_event's char cap so a large but
            # legitimate event isn't truncated by the 64 KiB default before the decoder
            # (the intended bound) sees it.
            limit=_STDOUT_LIMIT_BYTES,
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


class _DockerBrainSession:
    """A live brain container: its id (the --name) + its BOUNDED stdout event stream.

    Honours the :class:`BrainContainerSession` transport contract the pure protocol
    cannot enforce: each read is bounded by an idle timeout, the run is bounded by a
    wall-clock deadline, and ``events()`` ALWAYS tears down — cancelling the stdin
    feeder and terminating + reaping the docker subprocess — so a finished, failed,
    timed-out, or abandoned run never leaks a live (egress-capable) container or an
    unretrieved task. ``proc`` (the docker subprocess) and ``feeder`` (the stdin task)
    are injected, so this bounding/teardown logic is unit-tested with fakes; only the
    real ``docker run`` in :meth:`DockerBrainBackend.start` is the deployment edge.
    """

    def __init__(
        self,
        container_id: str,
        proc: Any,
        *,
        feeder: asyncio.Task[None],
        idle_timeout_s: float = _IDLE_TIMEOUT_S,
        wall_clock_s: float = _WALL_CLOCK_S,
        teardown_wait_s: float = _TEARDOWN_WAIT_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.container_id = container_id
        self._proc = proc
        self._feeder = feeder  # keep the stdin-feeder task referenced (no GC mid-run)
        self._idle_timeout_s = idle_timeout_s
        self._wall_clock_s = wall_clock_s
        self._teardown_wait_s = teardown_wait_s
        self._clock = clock
        self._closed = False

    async def events(self) -> AsyncIterator[str]:
        deadline = self._clock() + self._wall_clock_s
        while True:
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise TimeoutError(
                    f"brain container {self.container_id} exceeded the "
                    f"{self._wall_clock_s:.0f}s wall-clock limit"
                )
            try:
                raw = await asyncio.wait_for(
                    self._proc.stdout.readline(),
                    timeout=min(self._idle_timeout_s, remaining),
                )
            except TimeoutError:
                raise TimeoutError(
                    f"brain container {self.container_id} produced no output for "
                    f"{self._idle_timeout_s:.0f}s — treating it as hung"
                ) from None
            except (ValueError, asyncio.LimitOverrunError) as exc:
                # readline raises when a single line exceeds the StreamReader limit.
                # Fail closed with a clear message (decode_event's char cap is the
                # intended bound; this byte limit is only the backstop).
                raise ValueError(
                    f"brain container {self.container_id} emitted an oversized "
                    f"stdout line: {exc}"
                ) from exc
            if not raw:  # EOF — the container closed stdout / exited
                return
            yield raw.decode().rstrip("\n")

    async def aclose(self) -> None:
        """Reap the run: cancel the stdin feeder, then terminate + reap the docker
        subprocess. Idempotent (the provider calls it in a finally on every path).
        Cannot itself hang: SIGTERM is escalated to SIGKILL if the container ignores
        it, and every step is suppressed so teardown never masks the run's real error."""
        if self._closed:
            return
        self._closed = True
        self._feeder.cancel()
        await asyncio.gather(self._feeder, return_exceptions=True)
        if self._proc.returncode is not None:
            return  # already exited — nothing to reap
        with contextlib.suppress(Exception):
            self._proc.terminate()
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=self._teardown_wait_s)
        except Exception:
            # SIGTERM ignored (or wait failed) — escalate to SIGKILL so teardown,
            # the very thing that bounds a hung container, can never itself hang. The
            # final reap is also time-bounded so even a pathological child can't wedge
            # teardown (SIGKILL is uncatchable, so this normally returns immediately).
            with contextlib.suppress(Exception):
                self._proc.kill()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._proc.wait(), timeout=self._teardown_wait_s)
