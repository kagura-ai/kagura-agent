"""The `Transport` protocol — the cockpit's one boundary to the outside.

Slack (Bolt, Socket Mode), Discord (discord.py), and a CLI adapter all
normalize to a single `Event`; the core never imports a transport SDK. The
protocol is intentionally tiny: listen for events, send a reply, and `ask` the
human a HITL question (the cockpit's reason to exist).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Event:
    """A normalized inbound message from any transport.

    `is_thread_reply` is the structural signal the intent router uses to tell a
    fresh launch from a continuation — no language understanding required.

    `sender` is the transport-specific identity of who sent the message (e.g. a
    Slack user id), used by the cockpit to enforce operator-identity on HITL
    approvals (#14). It is optional: the single-user CLI adapter leaves it None,
    which the cockpit treats as "no operator gate" (backward-compatible).
    """

    thread_id: str
    text: str
    is_thread_reply: bool
    sender: str | None = None


class Transport(Protocol):
    def listen(self) -> AsyncIterator[Event]: ...

    async def send(self, thread_id: str, text: str) -> None: ...

    async def ask(self, thread_id: str, question: str, options: list[str]) -> str: ...


def click_authorized(clicker: str | None, operator_id: str | None) -> bool:
    """Whether a HITL button click may resolve a pending request.

    The button (`ask`) path is the synchronous twin of the typed `/approve`
    event; it must enforce the same operator-identity gate (#14). With no
    operator configured (single-user CLI), any click is allowed; otherwise only
    the operator's click qualifies — so a non-operator (e.g. a hijacked agent)
    cannot self-approve by clicking ✅.
    """
    return operator_id is None or clicker == operator_id
