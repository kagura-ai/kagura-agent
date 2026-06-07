"""Slack transport — Bolt, Socket Mode (no public URL). A pure addition.

The testable core is `normalize_slack_event`: it maps a Slack events-API
`message` payload onto the shared `Event`, deriving the structural
launch/continue signal and dropping the bot's own messages. The Bolt wiring is
lazy-imported glue, exercised in deployment rather than unit tests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from kagura_agent.cockpit.transports.base import Event


def normalize_slack_event(payload: dict[str, Any], *, bot_user_id: str) -> Event | None:
    if payload.get("type") != "message":
        return None
    if payload.get("subtype"):
        # message_changed / message_deleted / channel_join / thread_broadcast /
        # file_share etc. arrive as type=message with a subtype; the user often
        # lives under payload["message"], so the bot/user check below misses them.
        # Dropping them prevents empty-prompt (billed) brain runs.
        return None
    if payload.get("bot_id") or payload.get("user") == bot_user_id:
        return None  # ignore our own / other bots' messages

    ts = payload["ts"]
    thread_ts = payload.get("thread_ts")
    thread_id = thread_ts or ts
    is_thread_reply = thread_ts is not None and thread_ts != ts
    return Event(thread_id=thread_id, text=payload.get("text", ""), is_thread_reply=is_thread_reply)


class SlackTransport:  # pragma: no cover - requires slack-bolt + a workspace
    """Socket-Mode adapter. Holds the bot token; lives only in the cockpit."""

    def __init__(self, app: Any, bot_user_id: str) -> None:
        self._app = app
        self._bot_user_id = bot_user_id

    async def listen(self) -> AsyncIterator[Event]:
        raise NotImplementedError("wired via Bolt's event handlers in deployment")
        yield  # unreachable — makes this an async generator, not a coroutine

    async def send(self, thread_id: str, text: str) -> None:
        # Honest stub: `Event` does not yet carry the Slack channel id (it keys on
        # thread ts), so a real chat_postMessage needs a thread_id->channel map
        # wired at deployment. Raise rather than ship a send that posts to the
        # wrong channel.
        raise NotImplementedError("wired via Bolt with a thread->channel map in deployment")

    async def ask(self, thread_id: str, question: str, options: list[str]) -> str:
        # posts ✅/❌ buttons; resolved by an interactive-action handler
        raise NotImplementedError("wired via Bolt's action handlers in deployment")
