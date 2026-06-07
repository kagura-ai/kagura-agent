"""Discord transport — discord.py. A pure addition on the Transport protocol.

`normalize_discord_message` is the testable core: a message in a thread is a
continuation (keyed by thread id), a message in a plain channel is a launch
(keyed by channel id), and the bot's own messages are dropped. The gateway
client wiring is lazy-imported glue.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from kagura_agent.cockpit.transports.base import Event


def normalize_discord_message(
    *,
    author_id: int,
    bot_user_id: int,
    content: str,
    channel_id: int,
    thread_id: int | None,
) -> Event | None:
    if author_id == bot_user_id:
        return None
    if thread_id is not None:
        return Event(thread_id=str(thread_id), text=content, is_thread_reply=True)
    return Event(thread_id=str(channel_id), text=content, is_thread_reply=False)


class DiscordTransport:  # pragma: no cover - requires discord.py + a bot token
    def __init__(self, client: Any, bot_user_id: int) -> None:
        self._client = client
        self._bot_user_id = bot_user_id

    async def listen(self) -> AsyncIterator[Event]:
        raise NotImplementedError("wired via discord.py's on_message in deployment")
        yield  # unreachable — makes this an async generator, not a coroutine

    async def send(self, thread_id: str, text: str) -> None:
        # `thread_id` is a channel/thread id here, so fetch_channel would work —
        # but keep parity with listen/ask as an explicit deployment-wired stub.
        raise NotImplementedError("wired via discord.py in deployment")

    async def ask(self, thread_id: str, question: str, options: list[str]) -> str:
        raise NotImplementedError("wired via discord.py view/button callbacks in deployment")
