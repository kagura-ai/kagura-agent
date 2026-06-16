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
import sys
from collections.abc import Sequence
from typing import Any

from kagura_agent.cli.doctor import (
    DOCTOR_FAIL_EXIT,
    FAIL,
    format_report,
    overall_status,
    run_doctor,
)
from kagura_agent.core.brain.base import BrainUnavailable
from kagura_agent.membrane.registry import GrantSet, parse_grants

#: --grant is parse-only in v0.6; this loud warning prevents an operator from
#: mistaking a parsed grant for an enforced one (enforcement lands in v0.7).
GRANT_NOT_ENFORCED_WARNING = (
    "warning: --grant is parsed for validation only and is NOT yet enforced "
    "(enforcement lands in v0.7); all configured providers remain reachable."
)


def resolve_grants(grant_specs: list[str] | None) -> tuple[GrantSet, str | None]:
    """Parse ``--grant PROVIDER:SCOPE`` specs into a :class:`GrantSet`.

    **Parse-only in v0.6**: the GrantSet is validated (a malformed spec is a
    fail-closed ``ValueError``) and returned, but it is NOT enforced yet — so
    when any grant is given, a loud "not enforced" warning is returned alongside
    it for the caller to surface. Returns ``(GrantSet, warning_or_None)``.
    """
    grants = parse_grants(grant_specs or [])
    warning = GRANT_NOT_ENFORCED_WARNING if grant_specs else None
    return grants, warning


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
        help="grant PROVIDER:SCOPE (repeatable). PARSED FOR VALIDATION ONLY in this "
        "release — NOT yet enforced; enforcement lands in v0.7.",
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


async def _run_task(  # pragma: no cover - needs SDK + subscription
    task: str,
    *,
    mcp_servers: dict[str, Any] | None = None,
    strict_mcp_config: bool = False,
) -> str:
    from kagura_agent.cockpit.core import Cockpit
    from kagura_agent.cockpit.transports.base import Event
    from kagura_agent.cockpit.transports.cli import CliTransport
    from kagura_agent.core.brain.claude import make_default_brain
    from kagura_agent.mcp.memory_cloud import ensure_memory_reachable, memory_reachable
    from kagura_agent.patterns.checkpoint import InMemoryCheckpointStore

    # Redefined startup gate (v0.2-A6): memory must be reachable via the CLI,
    # independent of the brain. Fail-closed; no silent memory-less degrade.
    ensure_memory_reachable(reachable=memory_reachable())

    brain = make_default_brain(
        mcp_servers=mcp_servers, strict_mcp_config=strict_mcp_config
    )

    transport = CliTransport(
        inbox=[Event(thread_id="cli", text=task, is_thread_reply=False)]
    )
    cockpit = Cockpit(transport, brain, InMemoryCheckpointStore())
    await cockpit.serve()
    return transport.sent[-1][1] if transport.sent else ""


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
        try:
            _grants, grant_warning = resolve_grants(ns.grants)
        except ValueError as exc:
            # Malformed --grant spec — fail-closed with a clean message (exit 2).
            print(f"--grant: {exc}", file=sys.stderr)
            return 2
        if grant_warning is not None:
            print(grant_warning, file=sys.stderr)  # never let a parsed grant look enforced
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
        print(result)
        return 0
    return 1
