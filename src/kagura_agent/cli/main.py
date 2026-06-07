"""`kagura-agent run "task description"` — the local debug entrypoint.

Argument parsing is real logic (tested). The wiring in `main()` constructs the
real subscription-backed brain and is exercised end to end by the smoke path
rather than unit tests (it needs the SDK + a subscription).
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="kagura-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a single task")
    run.add_argument("task", help="natural-language task description")
    return parser.parse_args(list(argv))


async def _run_task(task: str) -> str:  # pragma: no cover - needs SDK + subscription
    from kagura_agent.cockpit.core import Cockpit
    from kagura_agent.cockpit.transports.base import Event
    from kagura_agent.cockpit.transports.cli import CliTransport
    from kagura_agent.core.brain.claude import make_default_brain
    from kagura_agent.core.brain.startup import ensure_startable
    from kagura_agent.mcp.memory_cloud import mcp_available
    from kagura_agent.patterns.checkpoint import InMemoryCheckpointStore

    brain = make_default_brain()
    ensure_startable(brain.caps, mcp_available=mcp_available())

    transport = CliTransport(
        inbox=[Event(thread_id="cli", text=task, is_thread_reply=False)]
    )
    cockpit = Cockpit(transport, brain, InMemoryCheckpointStore())
    await cockpit.serve()
    return transport.sent[-1][1] if transport.sent else ""


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - glue
    import sys

    ns = parse_args(sys.argv[1:] if argv is None else argv)
    if ns.command == "run":
        print(asyncio.run(_run_task(ns.task)))
        return 0
    return 1
