"""`kagura-agent run "task description"` — the local debug entrypoint.

Argument parsing and `--mcp-config` loading are real logic (tested). The wiring
in `main()` constructs the real subscription-backed brain and is exercised end
to end by the smoke path rather than unit tests (it needs the SDK + a
subscription).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import math
import os
import sys
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kagura_agent.core.brain.sdk_engine import PermissionMode
    from kagura_agent.membrane.brain_container import DockerBrainBackend

from kagura_agent.cli.doctor import (
    DOCTOR_FAIL_EXIT,
    FAIL,
    format_report,
    overall_status,
    run_doctor,
)
from kagura_agent.core.brain.base import BrainInvocationError, BrainUnavailable
from kagura_agent.core.session import SessionError
from kagura_agent.mcp.memory_cloud import MemoryClient, MemoryUnreachableError
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


class TransportUnavailable(RuntimeError):
    """A `serve` transport's optional extra (slack-bolt / discord.py) is not
    installed. Surfaced by the serve gate as a clean exit-3 message (like the
    brain's `BrainUnavailable`), never a raw `ModuleNotFoundError` traceback."""


#: serve --transport NAME → (the module to probe for importability, the pip extra
#: that provides it). Used by `require_transport_sdk` to fail closed before the
#: real SDK import in `_build_transport`.
_TRANSPORT_SDKS: dict[str, tuple[str, str]] = {
    "slack": ("slack_bolt", "slack"),
    "discord": ("discord", "discord"),
}


def require_transport_sdk(
    name: str, *, find_spec: Callable[[str], object | None] = importlib.util.find_spec
) -> None:
    """Raise `TransportUnavailable` with an install hint if the transport SDK is absent.

    Mirrors the brain's `require_claude_sdk`: pure and SDK-free (inject `find_spec`
    to unit-test; the real `find_spec` only inspects import metadata, never importing
    the heavy SDK). Called at the top of `_build_transport` so a missing
    `[slack]`/`[discord]` extra fails closed with a clean message instead of a raw
    `ModuleNotFoundError` once `serve` is wired. `name` is one of the argparse
    `--transport` choices, so the dict lookup never misses in practice."""
    module, extra = _TRANSPORT_SDKS[name]
    if find_spec(module) is None:
        raise TransportUnavailable(
            f"The {name} transport requires the optional '{extra}' extra ({module}), "
            "which is not installed. Install it with:\n"
            f"  pip install 'kagura-agent[{extra}]'"
        )

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
        translated = text.translate(_ASCII_FALLBACKS)
        encoding = getattr(self._stream, "encoding", None)
        if encoding:
            # Last-resort safety net (#82): a non-decorative char that the stream's
            # code page still can't encode — and which reconfigure(errors="replace")
            # may have failed to make safe — would crash a strict stream. Degrade
            # any such char to "?" so the wrapper is crash-proof regardless of
            # whether the earlier reconfigure succeeded.
            translated = translated.encode(encoding, "replace").decode(encoding, "replace")
        self._stream.write(translated)
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
        # No declared encoding. A pure-text stream (e.g. StringIO — no byte buffer
        # underneath) can never raise UnicodeEncodeError, so the glyphs are safe.
        # But a *byte*-backed stream (has a .buffer) that simply didn't report an
        # encoding could be a legacy code page underneath — we can't claim it
        # handles the glyphs, so fail closed (#82) and let the fallback wrap it.
        return not hasattr(stream, "buffer")
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


#: Rejected at parse time (_nonempty_task) AND after a file/stdin read
#: (resolve_run_prompt): a blank prompt would spin a billed empty-prompt run.
_EMPTY_TASK_MSG = "task must not be empty"


def _nonempty_task(value: str) -> str:
    """Reject an empty/whitespace-only task at parse time.

    A blank prompt would otherwise spin a billed empty-prompt brain run (the
    transports already drop empty inbound messages; the CLI is the other entry)."""
    if not value.strip():
        raise argparse.ArgumentTypeError(_EMPTY_TASK_MSG)
    return value


def _read_stdin() -> str:  # pragma: no cover - trivial stdin shim, injected in tests
    return sys.stdin.read()


def resolve_run_prompt(
    task: str | None,
    prompt_file: str | None,
    *,
    stdin_read: Callable[[], str] = _read_stdin,
) -> str:
    """Resolve the run's prompt body from the inline task, --prompt-file, or stdin.

    Exactly one source is expected — argparse's required mutually-exclusive group
    enforces that at parse time — but this stays a *total* function (every branch
    defined, defensive on "neither") so it is unit-tested in isolation. A ``-`` in
    either the positional or ``--prompt-file`` means "read stdin" (the universal
    convention). The resolved body is returned verbatim; only an empty/whitespace
    result is rejected, so an empty file or empty stdin can't spin a billed
    empty-prompt brain run (the same guard ``_nonempty_task`` gives the inline
    positional). File/stdin read failures are folded into a clean ``ValueError``
    so ``main()`` surfaces a one-line exit-2 message, never a raw traceback (#142)."""
    if task == "-" or prompt_file == "-":
        text = stdin_read()
        source = "stdin"
    elif prompt_file is not None:
        try:
            text = Path(prompt_file).read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"could not read --prompt-file {prompt_file!r}: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise ValueError(f"--prompt-file {prompt_file!r} is not valid UTF-8 text") from exc
        source = f"--prompt-file {prompt_file!r}"
    elif task is not None:
        text = task
        source = "task argument"
    else:  # neither source — argparse prevents this; defensive total-function branch
        raise ValueError("provide a task argument, --prompt-file PATH, or '-' for stdin")
    if not text.strip():
        raise ValueError(f"{_EMPTY_TASK_MSG} (read from {source})")
    return text


def _add_observability_args(p: argparse.ArgumentParser) -> None:
    """Shared --verbose / --log-level for run + repl (#105), so they stay in lockstep."""
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="stream the run's step-by-step narration (SDK engine) to stderr as it "
        "progresses; the final result still goes to stdout. One-shot backends "
        "(kagura-brain) have no mid-stream narration, so this simply shows less.",
    )
    p.add_argument(
        "--log-level",
        dest="log_level",
        default=None,
        choices=["debug", "info", "warning", "error", "critical"],
        help="surface internal logs at this level (or set KAGURA_LOG); default: quiet",
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="kagura-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a single task")
    # The task body comes from exactly one source: the inline positional, a file
    # (--prompt-file), or stdin ('-' in either). A *required* mutually-exclusive
    # group makes argparse enforce "exactly one" at parse time — neither and both
    # fail closed with exit 2 (#142).
    src = run.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "task",
        nargs="?",
        type=_nonempty_task,
        help="natural-language task description, or '-' to read it from stdin",
    )
    src.add_argument(
        "--prompt-file",
        dest="prompt_file",
        default=None,
        metavar="PATH",
        help="read the task body from PATH ('-' for stdin); mutually exclusive with "
        "the inline task argument",
    )
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
    _add_observability_args(run)
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
    _add_observability_args(repl)
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
    serve = sub.add_parser("serve", help="run the cockpit serve loop on a chat transport")
    serve.add_argument(
        # Choices are the _TRANSPORT_SDKS keys, so a transport can never be a valid
        # --transport value without a matching missing-extra guard (no drift → no
        # fail-open KeyError in require_transport_sdk).
        "--transport", choices=tuple(_TRANSPORT_SDKS), required=True,
        help="chat transport to serve (the bot token is read from the host environment)",
    )
    serve.add_argument(
        "--container", action="store_true",
        help="run the brain INSIDE a hardened, egress-sealed container (#102); requires a "
        "BYOK ANTHROPIC_API_KEY (subscription auth does not run in-container)",
    )
    serve.add_argument(
        "--image", default="kagura-agent:agent",
        help="agent brain image for --container (default: kagura-agent:agent)",
    )
    serve.add_argument(
        "--project-root", default=".",
        help="project root mounted READ-ONLY into the container (default: cwd)",
    )
    serve.add_argument(
        "--egress", action="append", default=[], metavar="HOST",
        help="additional egress-allowed host for --container (repeatable; api.anthropic.com "
        "is always allowed)",
    )
    serve.add_argument(
        "--operator-id", default=None,
        help="restrict /kill and /approve to this sender id (single-user CLI: omit)",
    )
    serve.add_argument(
        "--mcp-config", dest="mcp_config", default=None,
        help="path to a JSON file of MCP server configs (non-memory MCP servers)",
    )
    serve.add_argument(
        "--strict-mcp-config", dest="strict_mcp_config", action="store_true",
        help="reject MCP servers not present in --mcp-config (no silent passthrough)",
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


_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def resolve_log_level(arg: str | None, env: Mapping[str, str]) -> int:
    """The logging level for a run (#105): ``--log-level`` wins, else ``KAGURA_LOG``,
    else WARNING (quiet).

    Internal logs (``log = logging.getLogger(__name__)``) are silent by default —
    there is no ``basicConfig`` until a run configures one at this level, so an
    operator opts in with ``--log-level debug`` / ``KAGURA_LOG=debug`` to see them.
    An unrecognized value falls back to the quiet default rather than erroring.
    """
    raw = (arg or env.get("KAGURA_LOG", "")).strip().lower()
    return _LOG_LEVELS.get(raw, logging.WARNING)


#: Env var naming the durable memory DB file. When set (non-blank), the grounding
#: seam uses the SQLite middle tier so memory persists across separate `run`
#: invocations; unset → the in-process LocalMemoryClient (today's default).
_MEMORY_DB_ENV = "KAGURA_AGENT_MEMORY_DB"

#: Env var naming the kagura-memory MCP context UUID. When set (non-blank), the
#: grounding seam uses the trust-aware MCP cloud backbone (#111) — the strongest
#: tier. The server command is in _MEMORY_MCP_SERVER_ENV.
_MEMORY_MCP_CONTEXT_ENV = "KAGURA_AGENT_MEMORY_MCP_CONTEXT"
#: Env var naming the kagura-memory MCP server command (stdio). Required when the
#: cloud context is set — read by build_mcp_call_tool.
_MEMORY_MCP_SERVER_ENV = "KAGURA_AGENT_MEMORY_MCP_SERVER"

#: #165 S3: opt-in flag for the bounded recall re-rank by verified feedback. DEFAULT-OFF
#: (unset/falsy) — recall is byte-for-byte unchanged. Applies to the host-side sync
#: backends (Local/Sqlite); the default-ON flip stays gated on the #166 outcome eval.
_RECALL_RERANK_ENV = "KAGURA_AGENT_RECALL_RERANK"

#: #165 S3: the Δ4 exploration-floor epsilon (float in [0, 1]) for the re-rank — the
#: per-recall probability a candidate surfaces regardless of feedback, so a demoted
#: memory can re-surface. Blank/invalid -> 0.0 (off); a nonzero floor is required before
#: the default-ON flip (#166 tunes it).
_RECALL_EXPLORE_ENV = "KAGURA_AGENT_RECALL_EXPLORE"


def _parse_explore_epsilon(value: str) -> float:
    """``KAGURA_AGENT_RECALL_EXPLORE`` -> a float in [0, 1]; blank/invalid -> 0.0 (off)."""
    try:
        eps = float(value.strip())
    except ValueError:
        return 0.0
    if not math.isfinite(eps):  # nan/inf are not a probability -> off (never full)
        return 0.0
    return max(0.0, min(1.0, eps))


def make_memory_client(env: Mapping[str, str] | None = None) -> MemoryClient:
    """The grounding seam (B): the MemoryClient used to recall prior context and
    persist task summaries around a run.

    **Always returns a client, never ``None`` (#104).** The old ``None`` made
    ``ground_and_run`` silently degrade to a plain checkpoint resume — so a run paid
    the reachability gate's friction (``ensure_memory_reachable``) yet got none of
    the backbone's benefit. Memory is now *always present*; only the backend's
    strength differs, the seam never disappears.

    Backend selection (strongest configured wins; all honour ``trusted_only`` so
    grounding stays provenance-safe):

    - ``KAGURA_AGENT_MEMORY_MCP_CONTEXT`` set → :class:`McpMemoryClient`, the
      **trust-aware MCP cloud** backbone (#111): cross-host persistence + the real
      ``recall(trusted_only=True)`` so an externally-ingested / quarantined memory
      is never fed back as behaviour-influencing context (the membrane provenance
      rule). **Fail-closed**: a malformed context (not a UUID) raises rather than
      silently degrading. The connection itself is lazy — a missing ``mcp`` SDK /
      unreachable server fails closed on first use, not a silent memory-less run.
    - else ``KAGURA_AGENT_MEMORY_DB`` set → :class:`SqliteMemoryClient`, the
      **durable** middle tier (#107): persists across separate process invocations
      with no network/extra dependency. **Fail-closed** on an unopenable DB path.
    - otherwise → :class:`LocalMemoryClient` (in-process; today's default).
    """
    # MemoryUnreachableError (a RuntimeError subclass) is the CLI-handled config-error
    # exception that run/repl/serve already catch → clean exit 3 + message, never a raw
    # traceback. A bare RuntimeError would escape every handler as an unhandled crash.
    from kagura_agent.mcp.memory_cloud import LocalMemoryClient, MemoryUnreachableError

    environ = os.environ if env is None else env
    rerank = environ.get(_RECALL_RERANK_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
    explore = _parse_explore_epsilon(environ.get(_RECALL_EXPLORE_ENV, ""))
    mcp_context = environ.get(_MEMORY_MCP_CONTEXT_ENV, "").strip()
    if mcp_context:
        import uuid

        from kagura_agent.mcp.mcp_memory import McpMemoryClient, build_mcp_call_tool

        try:
            uuid.UUID(mcp_context)
        except ValueError as exc:
            # Fail-closed: a misconfigured cloud context must refuse, not fall back.
            raise MemoryUnreachableError(
                f"{_MEMORY_MCP_CONTEXT_ENV}={mcp_context!r} is not a valid context UUID "
                f"— fix it or unset {_MEMORY_MCP_CONTEXT_ENV}"
            ) from exc
        if not environ.get(_MEMORY_MCP_SERVER_ENV, "").strip():
            # Fail-closed with a clear message at construction, not an opaque stdio
            # spawn error on the first recall/remember deep in the run loop.
            raise MemoryUnreachableError(
                f"{_MEMORY_MCP_CONTEXT_ENV} is set but {_MEMORY_MCP_SERVER_ENV} (the "
                "kagura-memory MCP server command) is not — set it or unset the context"
            )
        return McpMemoryClient(build_mcp_call_tool(environ), context_id=mcp_context)
    db_path = environ.get(_MEMORY_DB_ENV, "").strip()
    if db_path:
        import sqlite3

        from kagura_agent.mcp.memory_sqlite import SqliteMemoryClient

        try:
            return SqliteMemoryClient(db_path, rerank_feedback=rerank, explore_epsilon=explore)
        except (sqlite3.Error, OSError) as exc:
            # Gated fail-closed: the operator opted into durable memory but the DB
            # is unusable — refuse rather than silently fall back to ephemeral memory.
            raise MemoryUnreachableError(
                f"{_MEMORY_DB_ENV}={db_path!r} could not be opened as a memory database: "
                f"{exc} — fix the path/permissions or unset {_MEMORY_DB_ENV} to use "
                "in-memory storage"
            ) from exc
    return LocalMemoryClient(rerank_feedback=rerank, explore_epsilon=explore)


def build_container_backend(
    env: Mapping[str, str],
    *,
    enabled: bool,
    image: str,
    project_root: str,
    egress_allow: tuple[str, ...] = (),
) -> DockerBrainBackend | None:
    """Build the brain-in-container backend for ``serve`` (#102/#116), or None.

    With ``enabled`` false the cockpit runs the brain in-process (today's default).
    When enabled it returns a :class:`DockerBrainBackend` whose ``resolve_byok``
    reads ``ANTHROPIC_API_KEY`` from ``env`` host-side.

    **Fail-closed (#113):** container execution authenticates BYOK — subscription
    auth cannot run inside the container — so ``--container`` without an
    ``ANTHROPIC_API_KEY`` refuses to start rather than silently falling back to
    in-process and dropping the isolation the operator asked for.
    """
    if not enabled:
        return None
    if not env.get("ANTHROPIC_API_KEY", "").strip():
        raise ValueError(
            "--container requires a BYOK ANTHROPIC_API_KEY: the in-container brain cannot "
            "use subscription auth (set ANTHROPIC_API_KEY, or drop --container to run "
            "the brain in-process)"
        )
    from kagura_agent.membrane.brain_container import DockerBrainBackend

    return DockerBrainBackend(
        image=image,
        project_root=project_root,
        resolve_byok=lambda: env.get("ANTHROPIC_API_KEY", ""),
        egress_allow=egress_allow,
    )


def _narrate(text: str) -> None:  # pragma: no cover - I/O
    """Stream a narration line to stderr (keeps stdout = the final result, so a
    script capturing the result is unaffected by --verbose)."""
    print(text, file=sys.stderr, flush=True)


def _verify_check(command: str) -> int:  # pragma: no cover - host subprocess edge
    """Run the configured verify check (KAGURA_AGENT_VERIFY_CHECK) on the host and
    return its exit code. shell=True: the command is operator-supplied config, run at
    the operator's own privilege, like the rest of the run path."""
    import subprocess

    return subprocess.run(command, shell=True).returncode


async def _run_task(  # pragma: no cover - needs SDK + subscription
    task: str,
    *,
    session_id: str | None = None,
    grants: GrantSet | None = None,
    registry_path: str = "kagura-agent.toml",
    mcp_servers: dict[str, Any] | None = None,
    strict_mcp_config: bool = False,
    default_permission_mode: PermissionMode = "default",
    verbose: bool = False,
) -> str:
    import time

    from kagura_agent.core.brain.select import make_brain
    from kagura_agent.mcp.memory_cloud import ensure_memory_reachable, memory_reachable
    from kagura_agent.membrane.cloud_transports import build_broker
    from kagura_agent.membrane.granted_broker import GrantedBroker, lease_requests
    from kagura_agent.membrane.lease import Budget, Lease
    from kagura_agent.membrane.registry_io import load_registry
    from kagura_agent.patterns.continuity import ground_and_run
    from kagura_agent.patterns.erasure import ProvenanceLog
    from kagura_agent.patterns.reinforce import reinforce_after_run

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

        brain = make_brain(
            os.environ,
            default_permission_mode=default_permission_mode,
            mcp_servers=mcp_servers,
            strict_mcp_config=strict_mcp_config,
        )
        # A named --session uses a persistent on-disk store (resume across runs);
        # a one-shot run uses a throwaway in-memory store (no persisted context).
        store, sid = make_run_store(session_id)
        memory = make_memory_client()
        provenance = ProvenanceLog()
        result = await ground_and_run(
            brain,
            store,
            memory,
            session_id=sid,
            prompt=task,
            provenance=provenance,
            on_message=_narrate if verbose else None,
        )
        # Independent-verdict arm (#165 S2): when KAGURA_AGENT_VERIFY_CHECK is set and
        # the backend is a host-side sync sink, run the check and reinforce the run's
        # grounding. Best-effort — a check that cannot spawn must not fail a done run.
        reinforce_after_run(
            memory, provenance, os.environ, session_id=sid, query=task, run_check=_verify_check
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
        from kagura_agent.core.brain.sdk_engine import resolve_permission_mode
        from kagura_agent.core.brain.select import resolve_brain_backend

        # Wire logging once, up front (#105): internal logs stay quiet unless
        # --log-level / KAGURA_LOG asks for them.
        logging.basicConfig(level=resolve_log_level(ns.log_level, os.environ))
        try:
            # Validate KAGURA_AGENT_BRAIN (+ _BACKEND for the kagura-brain backend)
            # and the SDK-only KAGURA_AGENT_PERMISSION_MODE up front so a typo fails
            # closed with a clean exit 2, not a raw traceback deep in make_brain.
            if resolve_brain_backend(os.environ) == "kagura-brain":
                resolve_kagura_brain_backend(os.environ)
            else:
                resolve_permission_mode(os.environ, default="acceptEdits")
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
            # Resolve the task body from the inline positional, --prompt-file, or
            # stdin ('-'). A missing/unreadable/non-UTF-8 file or an empty file/
            # stdin fails closed with a clean message (exit 2), never a traceback.
            task_text = resolve_run_prompt(ns.task, ns.prompt_file)
        except ValueError as exc:
            print(f"run: {exc}", file=sys.stderr)
            return 2
        try:
            result = asyncio.run(
                _run_task(
                    task_text,
                    session_id=ns.session,
                    grants=grants,
                    registry_path=ns.registry,
                    mcp_servers=mcp_servers,
                    strict_mcp_config=ns.strict_mcp_config,
                    default_permission_mode="acceptEdits",
                    verbose=ns.verbose,
                )
            )
        except BrainUnavailable as exc:
            # Expected setup condition (optional brain not installed) — surface the
            # actionable install hint, not a raw traceback or generic "internal error".
            # Exit 3 (not 2) so a wrapping script can tell this apart from argparse's
            # own usage-error exit code (2).
            print(str(exc), file=sys.stderr)
            return 3
        except MemoryUnreachableError as exc:
            # The startup gate (memory must be reachable + authenticated via the
            # kagura CLI) — surface the actionable message cleanly (exit 3, a
            # setup-not-ready code like BrainUnavailable), never a raw traceback.
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
        from kagura_agent.core.brain.sdk_engine import resolve_permission_mode
        from kagura_agent.core.brain.select import resolve_brain_backend

        logging.basicConfig(level=resolve_log_level(ns.log_level, os.environ))
        try:
            if resolve_brain_backend(os.environ) == "kagura-brain":
                resolve_kagura_brain_backend(os.environ)
            else:
                resolve_permission_mode(os.environ, default="acceptEdits")
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
                    default_permission_mode="acceptEdits",
                    verbose=ns.verbose,
                )
            )
        except BrainUnavailable as exc:
            print(str(exc), file=sys.stderr)
            return 3
        except MemoryUnreachableError as exc:
            print(str(exc), file=sys.stderr)
            return 3
        except (CheckpointError, SessionError, BrainInvocationError) as exc:
            # run_repl isolates per-turn errors, but a failure outside the loop
            # (store setup, a pre-loop load) still surfaces cleanly, not as a
            # traceback.
            print(f"repl failed: {exc}", file=sys.stderr)
            return 2
        return 0
    if ns.command == "serve":
        return _serve(ns)
    return 1


def _serve(ns: argparse.Namespace) -> int:  # pragma: no cover - glue (real transport SDKs + loop)
    """Assemble the cockpit and run its serve loop on the chosen chat transport.

    Deployment edge: builds the brain (``make_brain``), a persistent checkpoint
    store, the session registry, a ``Launcher`` (for ``/kill``), memory, and —
    when ``--container`` — the :class:`DockerBrainBackend` (#102/#116), then runs
    ``cockpit.serve()`` concurrently with the transport's own connection loop. The
    BYOK fail-closed gate (``build_container_backend``) is the one tested seam; the
    transport SDK construction needs the real slack-bolt / discord.py + tokens.
    """
    import asyncio
    import contextlib

    from kagura_agent.cockpit.core import Cockpit
    from kagura_agent.cockpit.registry import SessionRegistry
    from kagura_agent.core.brain.kagura_brain_engine import resolve_kagura_brain_backend
    from kagura_agent.core.brain.select import make_brain, resolve_brain_backend
    from kagura_agent.mcp.memory_cloud import ensure_memory_reachable, memory_reachable
    from kagura_agent.membrane.runtime import DockerRuntime, Launcher
    from kagura_agent.patterns.checkpoint import FileCheckpointStore

    logging.basicConfig(level=resolve_log_level(getattr(ns, "log_level", None), os.environ))
    project_root = os.path.abspath(ns.project_root)
    # The SAME run/repl startup preamble (so the three entry points share one
    # fail-closed contract): validate the brain backend, the redefined memory gate
    # (no silent memory-less degrade), and load MCP config — all up front, before
    # the long-lived loop accepts any traffic.
    try:
        if resolve_brain_backend(os.environ) == "kagura-brain":
            resolve_kagura_brain_backend(os.environ)
        mcp_servers = load_mcp_config(ns.mcp_config)
        container = build_container_backend(
            os.environ,
            enabled=ns.container,
            image=ns.image,
            project_root=project_root,
            egress_allow=tuple(ns.egress),
        )
        transport, transport_run = _build_transport(
            ns.transport, os.environ, operator_id=ns.operator_id
        )
        ensure_memory_reachable(reachable=memory_reachable())
        brain = make_brain(
            os.environ, mcp_servers=mcp_servers, strict_mcp_config=ns.strict_mcp_config
        )
        memory = make_memory_client()
        checkpoints = FileCheckpointStore(resolve_state_dir(os.environ))
    except (BrainUnavailable, MemoryUnreachableError, TransportUnavailable) as exc:
        # A missing runtime dependency (brain or transport extra) / unreachable memory:
        # exit 3 (matches run/repl). TransportUnavailable folds the would-be raw
        # ModuleNotFoundError from a missing slack-bolt / discord.py into this gate.
        print(str(exc), file=sys.stderr)
        return 3
    except (ValueError, KeyError, OSError) as exc:
        # A config error (bad BYOK key / brain backend / mcp-config / transport
        # token): exit 2 with a clean message, never a raw traceback.
        print(f"serve: {exc}", file=sys.stderr)
        return 2

    # Cockpit / registry / launcher construction is pure wiring (no config errors),
    # so it lives outside the gate — keeping the try narrow to the config sources
    # rather than masking a programming bug there as a config error.
    cockpit = Cockpit(
        transport,
        brain,
        checkpoints,
        registry=SessionRegistry(),
        launcher=Launcher(DockerRuntime(), project_root=project_root),
        memory=memory,
        operator_id=ns.operator_id,
        container=container,
    )

    async def _run() -> None:
        # The transport's connection loop and the cockpit consumer run together;
        # when EITHER finishes (returns or raises) cancel the other — gather() would
        # instead hang on a still-running sibling after the first one ends.
        tasks = [asyncio.ensure_future(transport_run()), asyncio.ensure_future(cockpit.serve())]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
            # Await the cancellation so the transport's connection (e.g. the discord
            # websocket / aiohttp session) tears down cleanly instead of leaking with
            # a "Task was destroyed but it is pending" warning.
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for task in done:
            task.result()  # surface an exception from whichever finished first

    asyncio.run(_run())
    return 0


def _build_transport(  # pragma: no cover - needs slack-bolt / discord.py + a live workspace
    name: str, env: Mapping[str, str], *, operator_id: str | None
) -> tuple[Any, Callable[[], Awaitable[Any]]]:
    """Construct the chat transport + a coroutine that runs its connection loop.

    Tokens and the bot user id are read from the host environment (never baked):
    Slack needs ``SLACK_BOT_TOKEN`` / ``SLACK_APP_TOKEN`` / ``SLACK_BOT_USER_ID``;
    Discord needs ``DISCORD_BOT_TOKEN`` / ``DISCORD_BOT_USER_ID``. Reading the bot
    id from config sidesteps the connect-then-discover ordering the SDKs impose.
    """
    # Fail closed BEFORE the SDK import: a missing extra becomes a clean
    # TransportUnavailable (exit 3 at the serve gate), not a raw ModuleNotFoundError.
    require_transport_sdk(name)
    if name == "slack":
        from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
        from slack_bolt.async_app import AsyncApp

        from kagura_agent.cockpit.transports.slack import SlackTransport

        app = AsyncApp(token=env["SLACK_BOT_TOKEN"])
        slack = SlackTransport(app, env["SLACK_BOT_USER_ID"], operator_id=operator_id)
        handler = AsyncSocketModeHandler(app, env["SLACK_APP_TOKEN"])
        return slack, handler.start_async

    import discord

    from kagura_agent.cockpit.transports.discord import DiscordTransport

    # Read the token EAGERLY (like slack) so a missing one is a build-time KeyError
    # the serve gate catches — not a raw traceback once the loop has started.
    token = env["DISCORD_BOT_TOKEN"]
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    discord_tp = DiscordTransport(client, int(env["DISCORD_BOT_USER_ID"]), operator_id=operator_id)
    return discord_tp, lambda: client.start(token)


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
    default_permission_mode: PermissionMode = "default",
    verbose: bool = False,
) -> None:
    from kagura_agent.core.brain.select import make_brain
    from kagura_agent.mcp.memory_cloud import ensure_memory_reachable, memory_reachable
    from kagura_agent.patterns.continuity import run_repl

    ensure_memory_reachable(reachable=memory_reachable())
    brain = make_brain(
        os.environ,
        default_permission_mode=default_permission_mode,
        mcp_servers=mcp_servers,
        strict_mcp_config=strict_mcp_config,
    )
    store = FileCheckpointStore(resolve_state_dir())
    print(f"kagura-agent repl — session {session_id!r}. /exit to quit.")
    await run_repl(
        brain,
        store,
        _stdin_lines(f"[{session_id}] "),
        print,
        session_id=session_id,
        memory=make_memory_client(),
        on_message=_narrate if verbose else None,
    )
