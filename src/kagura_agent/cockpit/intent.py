"""Structural-first intent routing.

Routing must not depend on a language model: a top-level message is a launch, a
reply inside a session we know is a continuation. v0.1 ships launch/continue;
v0.3 adds status/approve/kill (which a sandboxed classifier may disambiguate
when structure is ambiguous).
"""

from __future__ import annotations

from collections.abc import Set
from enum import Enum

from kagura_agent.cockpit.transports.base import Event


class Intent(Enum):
    LAUNCH = "launch"
    CONTINUE = "continue"
    STATUS = "status"
    APPROVE = "approve"
    KILL = "kill"


# Explicit slash-commands override structural routing. Approve normally arrives
# as a button answer inside the HITL flow, but a typed "/approve" is honored too.
_COMMANDS = {
    "/status": Intent.STATUS,
    "/kill": Intent.KILL,
    "/approve": Intent.APPROVE,
}


def classify(event: Event, *, known_sessions: Set[str]) -> Intent:
    command = event.text.strip().split(maxsplit=1)[0].lower() if event.text.strip() else ""
    if command in _COMMANDS:
        return _COMMANDS[command]
    if event.is_thread_reply and event.thread_id in known_sessions:
        return Intent.CONTINUE
    return Intent.LAUNCH
