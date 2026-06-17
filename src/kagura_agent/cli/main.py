"""`kagura-agent run "task description"` — the local debug entrypoint.

Argument parsing and `--mcp-config` loading are real logic (tested). The wiring
in `main()` constructs the real subscription-backed brain and is exercised end
to end by the smoke path rather than unit tests (it needs the SDK + a
subscription).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from kagura_agent.cli.doctor import (
    DOCTOR_FAIL_EXIT,
    FAIL,
    format_report,
    overall_status,
    run_doctor,
)
from kagura_agent.core.brain.base import BrainInvocationError, BrainUnavailable
from kagura_agent.core.session import SessionError
from kagura_agent.membrane.registry import GrantSet, ProviderSpec, parse_grants
from kagura_agent.patterns.checkpoint import (
    CheckpointError,
    CheckpointStore,
    FileCheckpointStore,
    InMemoryCheckpointStore,
)

log = logging.getLogger(__name__)


class CredentialSetupError(RuntimeError):
    """A run's credential provisioning failed on operator input (a malformed /
    missing registry, a --grant naming a provider absent from it, or a provider
    kind needing deployment wiring). Distinct from a ValueError raised later by
    the agent run, so main() surfaces only the former as a clean exit-2 message."""

#: Lease tuning for the run path's granted credentials (#65). A short TTL plus a
#: renewable budget keeps a leaked cred short-lived; the run releases on exit.
_LEASE_TTL_SEC = 900
_LEASE_BUDGET_SEC = 3600

#: Decorative Unicode glyphs the CLI prints in help / doctor / setup output.
#: A legacy OEM code page — notably the Japanese-Windows cp932 console — can't
#: encode some of these (the em-dash and arrows aren't in cp932), so writing
#: them raises UnicodeEncodeError and crashes before any output appears. These
#: ASCII equivalents keep such a console readable instead of mojibake.
_ASCII_FALLBACKS = {
    ord("—"): "-",  # — em dash
    ord("–"): "-",  # – en dash
    ord("‘"): "'",  # ‘ left single quote
    ord("’"): "'",  # ’ right single quote
    ord("“"): '"',  # “ left double quote
    ord("”"): '"',  # ” right double quote
    ord("…"): "...",  # … horizontal ellipsis
    ord("→"): "->",  # → rightwards arrow
    ord("↳"): "->",  # ↳ downwards arrow with tip rightwards
}

#: One sample of every decorative glyph, used to probe a stream's encoding.
_DECORATIVE_GLYPHS = "".join(chr(cp) for cp in _ASCII_FALLBACKS)


class _AsciiFallbackStream:
    """A text-stream proxy that transliterates the CLI's decorative glyphs to
    ASCII before writing, so a legacy console shows readable text (``-``, ``->``)
    instead of crashing or printing mojibake. Every other attribute delegates to
    the wrapped stream, so it stays a drop-in for ``sys.stdout``/``sys.stderr``.
    """

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def write(self, text: str) -> int:
        self._stream.write(text.translate(_ASCII_FALLBACKS))
        return len(text)  # chars consumed, per the TextIOBase.write contract

    def writelines(self, lines: Any) -> None:
        # Route through our write() so glyphs are transliterated; the inherited
        # delegation would otherwise hit the raw stream untranslated.
        for line in lines:
            self.write(line)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _encoding_handles_glyphs(stream: Any) -> bool:
    """True if ``stream``'s encoding can render every decorative glyph as-is."""
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return True  # a str-only stream (e.g. StringIO) has no byte encoding to fail
    try:
        _DECORATIVE_GLYPHS.encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def configure_output_stream(stream: Any) -> Any:
    """Return a crash-proof, readable version of a CLI output stream.

    If the stream's encoding already covers the CLI's decorative glyphs (UTF-8
    and friends) it is returned unchanged — the nice glyphs are preserved. On a
    legacy code page such as cp932 the stream is set to ``errors="replace"`` (so
    an unexpected character degrades to ``?`` rather than crashing) and wrapped
    so the decorative glyphs transliterate to ASCII.
    """
    if stream is None or isinstance(stream, _AsciiFallbackStream):
        return stream  # already None, or already a fallback stream — don't nest
    if _encoding_handles_glyphs(stream):
        return stream
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(errors="replace")  # keep the code page; never crash on stray chars
        except (ValueError, OSError):  # already-detached / mid-write / unsupported
            pass
    return _AsciiFallbackStream(stream)


def resolve_grants(grant_specs: list[str] | None) -> GrantSet:
    """Parse ``--grant PROVIDER:SCOPE`` specs into an enforced :class:`GrantSet`.

    v0.7 (#65): the GrantSet is now **enforced** by
    :class:`~kagura_agent.membrane.granted_broker.GrantedBroker` in the run path.
    A malformed spec is a fail-closed ``ValueError``. An empty/absent list yields
    an empty (deny-all) GrantSet — default-deny: with no grant the run builds no
    broker and acquires no credential (the empty lease plan falls out for free).
    """
    return parse_grants(grant_specs or [])


def plan_granted_specs(
    registry: Iterable[ProviderSpec], grants: GrantSet
) -> list[ProviderSpec]:
    """Select only the registry specs a grant references (deterministic order).

    Building **only** the granted providers is least-privilege and keeps the run
    honest: an ungranted provider — which may need deployment wiring the default
    factory cannot supply — is never constructed, so it can't abort a run that
    never asked for it (this is also why ``doctor`` predicts the run: doctor only
    resolves references, never builds an ungranted provider). Fail-closed: a
    granted provider with no matching spec is a clean ``ValueError`` here, not a
    later ``KeyError`` deep inside the broker.
    """
    by_name = {spec.name: spec for spec in registry}
    granted_providers = sorted({g.provider for g in grants.grants})
    missing = [p for p in granted_providers if p not in by_name]
    if missing:
        raise ValueError(
            f"--grant names provider(s) not in the registry: {', '.join(missing)}"
        )
    return [by_name[p] for p in granted_providers]


#: Default on-disk home for persisted session checkpoints (relative to cwd). An
#: operator can relocate it with KAGURA_AGENT_STATE_DIR (e.g. an XDG path).
_DEFAULT_STATE_DIR = ".kagura-agent"

#: The session id a one-shot `run` (no --session) uses — its checkpoint lives in
#: a throwaway in-memory store, so it never persists (old one-shot behaviour).
_ONESHOT_SESSION = "cli"

#: Default session for `repl` when --session is omitted.
_DEFAULT_REPL_SESSION = "repl"


def resolve_state_dir(env: Mapping[str, str] | None = None) -> Path:
    """Where persisted checkpoints live: ``$KAGURA_AGENT_STATE_DIR/checkpoints``
    or ``./.kagura-agent/checkpoints``. A set-but-blank override is treated as
    unset (so an empty env var doesn't resolve to the filesystem root)."""
    environ = os.environ if env is None else env
    override = environ.get("KAGURA_AGENT_STATE_DIR", "").strip()
    base = Path(override) if override else Path(_DEFAULT_STATE_DIR)
    return base / "checkpoints"


def make_run_store(session_id: str | None) -> tuple[CheckpointStore, str]:
    """Pick the checkpoint store + effective session id for a ``run``.

    No ``--session`` → a throwaway in-memory store under a fixed id: a pure
    one-shot with no cross-run memory (unchanged legacy behaviour). A named
    ``--session ID`` → a persistent on-disk store, so a later ``run --session ID``
    in a fresh process resumes this one's checkpoint (the continuity feature).
    """
    if session_id is None:
        return InMemoryCheckpointStore(), _ONESHOT_SESSION
    return FileCheckpointStore(resolve_state_dir()), session_id


def _nonempty_task(value: str) -> str:
    """Reject an empty/whitespace-only task at parse time.

    A blank prompt would otherwise spin a billed empty-prompt brain run (the
    transports already drop empty inbound messages; the CLI is the other entry)."""
    if not value.strip():
        raise argparse.ArgumentTypeError("task must not be empty")
    return value


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="kagura-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a single task")
    run.add_argument("task", type=_nonempty_task, help="natural-language task description")
    # Memory is CLI-primary; these knobs are for *other* MCP servers, mirroring
    # Claude Code's own flags (orthogonal to memory — v0.2-A6).
    run.add_argument(
        "--mcp-config",
        dest="mcp_config",
        default=None,
        help="path to a JSON file of MCP server configs (non-memory MCP servers)",
    )
    run.add_argument(
        "--strict-mcp-config",
        dest="strict_mcp_config",
        action="store_true",
        help="reject MCP servers not present in --mcp-config (no silent passthrough)",
    )
    run.add_argument(
        "--grant",
        dest="grants",
        action="append",
        default=None,
        metavar="PROVIDER:SCOPE",
        help="grant PROVIDER:SCOPE (repeatable). Default-deny: only granted "
        "(provider, scope) pairs are reachable; with no --grant the run acquires "
        "no credentials.",
    )
    run.add_argument(
        "--registry",
        default="kagura-agent.toml",
        help="provider registry TOML the granted credentials are minted from "
        "(default: kagura-agent.toml; only read when --grant is given)",
    )
    run.add_argument(
        "--session",
        default=None,
        metavar="ID",
        help="continue (or start) a named, persisted session: a later "
        "`run --session ID` resumes this run's checkpoint. Omit for a one-shot "
        "run that keeps no cross-run context.",
    )
    repl = sub.add_parser(
        "repl", help="interactive session: each line continues the same context"
    )
    repl.add_argument(
        "--session",
        default=_DEFAULT_REPL_SESSION,
        metavar="ID",
        help=f"session id to drive (default: {_DEFAULT_REPL_SESSION}); its "
        "checkpoint persists, so re-entering resumes where you left off",
    )
    repl.add_argument(
        "--mcp-config",
        dest="mcp_config",
        default=None,
        help="path to a JSON file of MCP server configs (non-memory MCP servers)",
    )
    repl.add_argument(
        "--strict-mcp-config",
        dest="strict_mcp_config",
        action="store_true",
        help="reject MCP servers not present in --mcp-config (no silent passthrough)",
    )
    doctor = sub.add_parser(
        "doctor", help="preflight check: memory / claude / docker / egress (+ providers)"
    )
    doctor.add_argument(
        "--registry",
        default="kagura-agent.toml",
        help="provider registry TOML to diagnose (default: kagura-agent.toml; skipped if absent)",
    )
    doctor.add_argument(
        "--probe",
        action="store_true",
        help="opt-in: dry-mint a short-lived scoped token per provider, then revoke "
        "(performs a real mint against the live provider)",
    )
    setup = sub.add_parser(
        "setup", help="operator-gated wizard: guidance for memory / transport auth"
    )
    setup.add_argument(
        "topic",
        nargs="?",
        choices=["memory", "transport"],
        help="show CLI-first guidance for memory or transport auth (default: both)",
    )
    return parser.parse_args(list(argv))


def load_mcp_config(value: str | None) -> dict[str, Any] | None:
    """Load an `--mcp-config` JSON file into an SDK `mcp_servers` mapping.

    Accepts the Claude Code convention `{"mcpServers": {...}}` (returns the inner
    map) as well as a bare `{name: config}` map. Returns None when no path is
    given. A missing file raises (fail-loud: the operator asked for a config that
    is not there).
    """
    if value is None:
        return None
    with open(value, encoding="utf-8") as fh:
        data = json.load(fh)
    servers = data.get("mcpServers", data) if isinstance(data, dict) else data
    if not isinstance(servers, dict):
        raise ValueError(
            f"--mcp-config {value!r}: expected a JSON object of MCP servers "
            f"(got {type(servers).__name__})"
        )
    return dict(servers)


def make_memory_client() -> Any | None:
    """The grounding seam (B): the MemoryClient used to recall prior context and
    persist task summaries around a run.

    Returns ``None`` for now → ``ground_and_run`` degrades to plain checkpoint
    resume (A). A real client MUST honour ``recall(trusted_only=True)`` so an
    externally-ingested / quarantined memory is never fed back as behaviour-
    influencing context (membrane memory-provenance rule). The kagura CLI's
    ``recall`` exposes neither a machine-readable output nor a trust-tier filter,
    so a trust-aware MCP/SDK-backed adapter is the remaining deployment edge; this
    factory is where it gets wired in.
    """
    return None


async def _run_task(  # pragma: no cover - needs SDK + subscription
    task: str,
    *,
    session_id: str | None = None,
    grants: GrantSet | None = None,
    registry_path: str = "kagura-agent.toml",
    mcp_servers: dict[str, Any] | None = None,
    strict_mcp_config: bool = False,
) -> str:
    import time

    from kagura_agent.core.brain.select import make_brain
    from kagura_agent.mcp.memory_cloud import ensure_memory_reachable, memory_reachable
    from kagura_agent.membrane.cloud_transports import build_broker
    from kagura_agent.membrane.granted_broker import GrantedBroker, lease_requests
    from kagura_agent.membrane.lease import Budget, Lease
    from kagura_agent.membrane.registry_io import load_registry
    from kagura_agent.patterns.continuity import ground_and_run

    # Redefined startup gate (v0.2-A6): memory must be reachable via the CLI,
    # independent of the brain. Fail-closed; no silent memory-less degrade.
    ensure_memory_reachable(reachable=memory_reachable())

    # v0.7 (#65): provision granted credentials host-side. Default-deny falls out
    # for free — with no --grant the lease plan is empty, so no broker is built,
    # no lease is acquired, and nothing is injected. The granted (provider, scope)
    # pairs are enforced at the chokepoint by GrantedBroker. The leased creds are
    # injected as the run's env (the launcher / spawned tools inherit them) and
    # restored + released on exit so a scoped, time-boxed cred never outlives the
    # task — even on failure (finally).
    reqs = (
        lease_requests(grants, ttl=_LEASE_TTL_SEC, budget_seconds=_LEASE_BUDGET_SEC)
        if grants is not None
        else ()
    )
    broker: GrantedBroker | None = None
    leases: list[Lease] = []
    env_restore: dict[str, str | None] = {}
    try:
        if reqs:
            assert grants is not None  # a non-empty plan implies --grant was given
            # Build ONLY the granted providers (plan_granted_specs) — an ungranted,
            # possibly deployment-incomplete provider in the registry must never
            # abort a run that did not ask for it. Wrap the config-construction
            # ValueErrors (bad registry, unknown granted provider, unsupported
            # kind) in CredentialSetupError so main() can give a clean exit-2
            # message WITHOUT swallowing an unrelated ValueError raised later by
            # the agent run (cockpit.serve / the brain).
            try:
                specs = plan_granted_specs(load_registry(registry_path), grants)
                inner = build_broker(specs, clock=time.monotonic)
            except ValueError as exc:
                raise CredentialSetupError(str(exc)) from exc
            broker = GrantedBroker(inner, grants)
            for req in reqs:
                leases.append(
                    await broker.acquire(
                        req.provider,
                        scope=req.scope,
                        ttl=req.ttl,
                        budget=Budget(req.budget_seconds),
                    )
                )
            for key, value in broker.container_env(leases).items():
                env_restore[key] = os.environ.get(key)
                os.environ[key] = value

        brain = make_brain(os.environ, mcp_servers=mcp_servers, strict_mcp_config=strict_mcp_config)
        # A named --session uses a persistent on-disk store (resume across runs);
        # a one-shot run uses a throwaway in-memory store (no persisted context).
        store, sid = make_run_store(session_id)
        result = await ground_and_run(
            brain, store, make_memory_client(), session_id=sid, prompt=task
        )
        return result.text
    finally:
        # Restore env first, then release every acquired lease. Per-lease guarded
        # so one failing revoke does not skip the rest (the broker's sweep retries
        # anything left tracked).
        for key, prior in env_restore.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior
        if broker is not None:
            for lease in leases:
                try:
                    await broker.release(lease)
                except Exception:
                    log.exception(
                        "release of lease for %s failed; left tracked for sweep", lease.provider
                    )


def _run_probes(registry: Any) -> list[Any]:  # pragma: no cover - deployment edge (live mint)
    """Dry-mint each derivable-scope provider (--probe), per-provider.

    Built one provider at a time so a kind that needs a deployment-supplied
    callable (cloudflare/memory_cloud) reports its own FAIL without blocking the
    others (e.g. aws_sts still probes). Kinds whose probe scope is not derivable
    from the registry are skipped with a WARN.
    """
    import asyncio
    import time

    from kagura_agent.cli.doctor import (
        FAIL,
        WARN,
        CheckResult,
        _probe_scope,
        probe_provider,
    )
    from kagura_agent.membrane.cloud_transports import build_broker
    from kagura_agent.membrane.registry_io import SecretRefError

    async def _all() -> list[Any]:
        out: list[Any] = []
        for spec in registry:
            cname = f"probe:{spec.name}"
            scope = _probe_scope(spec)
            if scope is None:
                out.append(
                    CheckResult(cname, WARN, "probe scope not derivable for this kind — skipped")
                )
                continue
            try:
                # build_broker resolves secrets host-side, so it can raise both a
                # ValueError (unsupported kind / ambiguous) and a SecretRefError
                # (unresolved ref) — both become a per-provider FAIL, not a crash.
                broker = build_broker([spec], clock=time.monotonic)
            except (ValueError, SecretRefError) as exc:
                out.append(
                    CheckResult(cname, FAIL, "could not build provider for --probe", hint=str(exc))
                )
                continue
            out.append(await probe_provider(broker, spec.name, scope=scope))
        return out

    return asyncio.run(_all())


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - glue
    # Before argparse (or any handler) prints help/doctor/setup text containing
    # non-cp932 glyphs, install crash-proof output streams: UTF-8 consoles keep
    # the nice glyphs; a legacy code page (cp932) gets ASCII transliteration so
    # it neither crashes nor renders mojibake.
    sys.stdout = configure_output_stream(sys.stdout)
    sys.stderr = configure_output_stream(sys.stderr)
    ns = parse_args(sys.argv[1:] if argv is None else argv)
    if ns.command == "doctor":
        from pathlib import Path

        from kagura_agent.membrane.registry_io import load_registry

        registry = None
        reg_path = Path(ns.registry)
        if reg_path.exists():
            try:
                registry = load_registry(reg_path)
            except ValueError as exc:
                print(f"registry error: {exc}", file=sys.stderr)
                return 2
        results = run_doctor(registry=registry)
        if ns.probe and registry:
            results = results + _run_probes(registry)
        print(format_report(results))  # one coherent report, one overall verdict
        return DOCTOR_FAIL_EXIT if overall_status(results) == FAIL else 0
    if ns.command == "setup":
        from kagura_agent.cli.setup import setup_memory_guidance, setup_transport_guidance

        if ns.topic == "memory":
            print(setup_memory_guidance())
        elif ns.topic == "transport":
            print(setup_transport_guidance())
        else:
            print(setup_memory_guidance())
            print()
            print(setup_transport_guidance())
        return 0
    if ns.command == "run":
        from kagura_agent.core.brain.kagura_brain_engine import resolve_kagura_brain_backend
        from kagura_agent.core.brain.select import resolve_brain_backend

        try:
            # Validate KAGURA_AGENT_BRAIN (+ _BACKEND for the kagura-brain backend)
            # up front so a typo fails closed with a clean exit 2, not deep in _run_task.
            if resolve_brain_backend(os.environ) == "kagura-brain":
                resolve_kagura_brain_backend(os.environ)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        try:
            grants = resolve_grants(ns.grants)
        except ValueError as exc:
            # Malformed --grant spec — fail-closed with a clean message (exit 2).
            print(f"--grant: {exc}", file=sys.stderr)
            return 2
        try:
            mcp_servers = load_mcp_config(ns.mcp_config)
        except (OSError, ValueError) as exc:
            # Missing file / bad JSON / wrong shape: surface a clean, actionable
            # message instead of a raw traceback. Exit 2 (operator input error).
            print(f"--mcp-config {ns.mcp_config!r}: {exc}", file=sys.stderr)
            return 2
        try:
            result = asyncio.run(
                _run_task(
                    ns.task,
                    session_id=ns.session,
                    grants=grants,
                    registry_path=ns.registry,
                    mcp_servers=mcp_servers,
                    strict_mcp_config=ns.strict_mcp_config,
                )
            )
        except BrainUnavailable as exc:
            # Expected setup condition (optional brain not installed) — surface the
            # actionable install hint, not a raw traceback or generic "internal error".
            # Exit 3 (not 2) so a wrapping script can tell this apart from argparse's
            # own usage-error exit code (2).
            print(str(exc), file=sys.stderr)
            return 3
        except CredentialSetupError as exc:
            # Credential-provisioning input error (a malformed/missing registry, a
            # --grant naming a provider absent from it, or a kind needing deploy
            # wiring) — fail-closed with a clean message and exit 2, the operator-
            # input code (mirrors doctor and --mcp-config), never a raw traceback.
            # NOT a bare `except ValueError`: a ValueError raised later by the
            # agent run must surface as itself, not be mislabeled a --grant error.
            print(f"--grant/--registry: {exc}", file=sys.stderr)
            return 2
        except (CheckpointError, SessionError, BrainInvocationError) as exc:
            # A corrupt/unreadable persisted checkpoint (CheckpointError), a brain
            # that ended without a terminal result (SessionError), or a brain
            # invocation that failed/timed out (BrainInvocationError) — surface a
            # clean one-line message + exit 2, never a raw traceback. Restores the
            # clean failure surface the old cockpit.serve() path gave before this
            # command drove the Session directly.
            print(f"run failed: {exc}", file=sys.stderr)
            return 2
        print(result)
        return 0
    if ns.command == "repl":
        from kagura_agent.core.brain.kagura_brain_engine import resolve_kagura_brain_backend
        from kagura_agent.core.brain.select import resolve_brain_backend

        try:
            if resolve_brain_backend(os.environ) == "kagura-brain":
                resolve_kagura_brain_backend(os.environ)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        try:
            mcp_servers = load_mcp_config(ns.mcp_config)
        except (OSError, ValueError) as exc:
            print(f"--mcp-config {ns.mcp_config!r}: {exc}", file=sys.stderr)
            return 2
        try:
            asyncio.run(
                _run_repl(
                    session_id=ns.session,
                    mcp_servers=mcp_servers,
                    strict_mcp_config=ns.strict_mcp_config,
                )
            )
        except BrainUnavailable as exc:
            print(str(exc), file=sys.stderr)
            return 3
        except (CheckpointError, SessionError, BrainInvocationError) as exc:
            # run_repl isolates per-turn errors, but a failure outside the loop
            # (store setup, a pre-loop load) still surfaces cleanly, not as a
            # traceback.
            print(f"repl failed: {exc}", file=sys.stderr)
            return 2
        return 0
    return 1


def _stdin_lines(prompt: str) -> Iterable[str]:  # pragma: no cover - interactive stdin
    """Yield typed lines until EOF (Ctrl-D / Ctrl-Z), printing a prompt each turn."""
    while True:
        try:
            yield input(prompt)
        except EOFError:
            return


async def _run_repl(  # pragma: no cover - needs SDK + subscription + interactive stdin
    *,
    session_id: str,
    mcp_servers: dict[str, Any] | None = None,
    strict_mcp_config: bool = False,
) -> None:
    from kagura_agent.core.brain.select import make_brain
    from kagura_agent.mcp.memory_cloud import ensure_memory_reachable, memory_reachable
    from kagura_agent.patterns.continuity import run_repl

    ensure_memory_reachable(reachable=memory_reachable())
    brain = make_brain(os.environ, mcp_servers=mcp_servers, strict_mcp_config=strict_mcp_config)
    store = FileCheckpointStore(resolve_state_dir())
    print(f"kagura-agent repl — session {session_id!r}. /exit to quit.")
    await run_repl(
        brain,
        store,
        _stdin_lines(f"[{session_id}] "),
        print,
        session_id=session_id,
        memory=make_memory_client(),
    )
