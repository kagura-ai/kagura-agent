"""Discord transport — discord.py. A pure addition on the Transport protocol.

Mirrors the Slack split: pure, unit-tested helpers (`normalize_discord_message`
— now carrying the sender for the operator-identity gate; `target_channel_id`;
the HITL button custom-id round-trip) and a `DiscordTransport` whose gateway I/O
needs discord.py + a bot token, so it is exercised at deployment (`# pragma: no
cover`).

Unlike Slack, an `Event.thread_id` here already *is* the channel/thread id, so
no thread→channel map is needed — `send`/`ask` fetch the channel directly.

Operator identity (#14): `normalize_discord_message` puts the author id on
`Event.sender`; the deployer constructs `Cockpit(operator_id="<id>")` so only
that user's `/approve` resolves a pending request.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from kagura_agent.cockpit.transports.base import Event

# Prefix on each HITL button's custom_id, so the callback can tell an approval
# click apart from any other component and recover the chosen option.
APPROVAL_CUSTOM_ID_PREFIX = "kagura_hitl:"


def normalize_discord_message(
    *,
    author_id: int,
    bot_user_id: int,
    content: str,
    channel_id: int,
    thread_id: int | None,
) -> Event | None:
    """Map a discord.py message onto the shared `Event`.

    A message in a thread is a continuation (keyed by thread id); one in a plain
    channel is a launch (keyed by channel id). The bot's own messages are
    dropped, and the author id rides on `sender` for the operator-identity gate.
    """
    if author_id == bot_user_id:
        return None
    if not content.strip():
        # Empty / whitespace-only message (attachment-only, sticker, embed). Drop
        # it so it cannot become a billed empty-prompt brain LAUNCH.
        return None
    sender = str(author_id)
    if thread_id is not None:
        return Event(
            thread_id=str(thread_id), text=content, is_thread_reply=True, sender=sender
        )
    return Event(
        thread_id=str(channel_id), text=content, is_thread_reply=False, sender=sender
    )


def target_channel_id(thread_id: str) -> int:
    """The Discord channel/thread id to post a reply to.

    `Event.thread_id` is a stringified Discord snowflake; parse it back. Fail
    loud on a non-numeric id rather than fetch a wrong/None channel.
    """
    try:
        return int(thread_id)
    except (TypeError, ValueError):
        raise ValueError(f"thread_id {thread_id!r} is not a Discord channel id") from None


def custom_id_for(option: str) -> str:
    return f"{APPROVAL_CUSTOM_ID_PREFIX}{option}"


def option_from_custom_id(custom_id: str) -> str:
    """Recover the chosen option from a button custom_id (inverse of `custom_id_for`)."""
    if not custom_id.startswith(APPROVAL_CUSTOM_ID_PREFIX):
        raise ValueError(f"custom_id {custom_id!r} is not a kagura HITL button")
    return custom_id[len(APPROVAL_CUSTOM_ID_PREFIX):]


class DiscordTransport:  # pragma: no cover - requires discord.py + a bot token
    """Gateway adapter. Holds the bot token; lives only in the cockpit.

    `client` is a `discord.Client`. Inbound messages are normalized by the
    `on_message` listener into a queue `listen()` drains; `ask` posts a button
    View and blocks on a future its callbacks resolve.
    """

    def __init__(self, client: Any, bot_user_id: int) -> None:
        self._client = client
        self._bot_user_id = bot_user_id
        self._inbox: asyncio.Queue[Event] = asyncio.Queue()
        self._client.add_listener(self._on_message, "on_message")

    async def _on_message(self, message: Any) -> None:
        import discord  # local: optional dep

        channel = message.channel
        is_thread = isinstance(channel, discord.Thread)
        normalized = normalize_discord_message(
            author_id=message.author.id,
            bot_user_id=self._bot_user_id,
            content=message.content,
            channel_id=channel.parent_id if is_thread else channel.id,
            thread_id=channel.id if is_thread else None,
        )
        if normalized is not None:
            await self._inbox.put(normalized)

    async def listen(self) -> AsyncIterator[Event]:
        while True:
            yield await self._inbox.get()

    async def send(self, thread_id: str, text: str) -> None:
        channel = await self._client.fetch_channel(target_channel_id(thread_id))
        await channel.send(text)

    async def ask(self, thread_id: str, question: str, options: list[str]) -> str:
        import discord  # local: optional dep

        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        view = discord.ui.View(timeout=None)
        for option in options:
            button = discord.ui.Button(label=option, custom_id=custom_id_for(option))

            async def _callback(interaction: Any, _option: str = option) -> None:
                await interaction.response.defer()
                if not future.done():
                    future.set_result(_option)

            button.callback = _callback
            view.add_item(button)
        channel = await self._client.fetch_channel(target_channel_id(thread_id))
        await channel.send(question, view=view)
        return await future
