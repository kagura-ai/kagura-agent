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


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="kagura-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a single task")
    run.add_argument("task", help="natural-language task description")
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
    sub.add_parser("doctor", help="preflight check: memory / claude / docker / egress")
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


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - glue
    ns = parse_args(sys.argv[1:] if argv is None else argv)
    if ns.command == "doctor":
        results = run_doctor()
        print(format_report(results))
        return DOCTOR_FAIL_EXIT if overall_status(results) == FAIL else 0
    if ns.command == "run":
        mcp_servers = load_mcp_config(ns.mcp_config)
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
