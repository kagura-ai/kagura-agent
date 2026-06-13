"""#38: Discord transport pure helpers (gateway I/O is deployment-exercised)."""

from __future__ import annotations

import pytest

from kagura_agent.cockpit.transports.base import Event
from kagura_agent.cockpit.transports.discord import (
    APPROVAL_CUSTOM_ID_PREFIX,
    DiscordTransport,
    custom_id_for,
    normalize_discord_message,
    option_from_custom_id,
    target_channel_id,
)

# --- sender (operator-identity gate, #14) -----------------------------------

def test_normalize_channel_message_carries_sender() -> None:
    event = normalize_discord_message(
        author_id=1, bot_user_id=99, content="build it", channel_id=555, thread_id=None
    )
    assert event == Event(thread_id="555", text="build it", is_thread_reply=False, sender="1")


def test_normalize_thread_message_carries_sender() -> None:
    event = normalize_discord_message(
        author_id=7, bot_user_id=99, content="more", channel_id=555, thread_id=777
    )
    assert event == Event(thread_id="777", text="more", is_thread_reply=True, sender="7")


def test_normalize_drops_empty_content_message() -> None:
    # Attachment-only / sticker messages have empty content — drop, don't LAUNCH.
    assert normalize_discord_message(
        author_id=1, bot_user_id=99, content="", channel_id=555, thread_id=None
    ) is None
    assert normalize_discord_message(
        author_id=1, bot_user_id=99, content="   ", channel_id=555, thread_id=777
    ) is None


def test_normalize_drops_bot_own_message() -> None:
    assert normalize_discord_message(
        author_id=99, bot_user_id=99, content="hi", channel_id=555, thread_id=None
    ) is None


# --- channel id parsing -----------------------------------------------------

def test_target_channel_id_parses_snowflake() -> None:
    assert target_channel_id("555") == 555


def test_target_channel_id_rejects_non_numeric() -> None:
    with pytest.raises(ValueError, match="not a Discord channel id"):
        target_channel_id("not-a-snowflake")


# --- HITL button custom_id round-trip ---------------------------------------

def test_custom_id_round_trips_option() -> None:
    cid = custom_id_for("/approve")
    assert cid.startswith(APPROVAL_CUSTOM_ID_PREFIX)
    assert option_from_custom_id(cid) == "/approve"


def test_option_from_custom_id_rejects_foreign_id() -> None:
    with pytest.raises(ValueError, match="not a kagura HITL button"):
        option_from_custom_id("some_other_component")


# --- send (against a fake discord.py client) --------------------------------

class _FakeChannel:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, text: str, **_: object) -> None:
        self.sent.append(text)


class _FakeDiscordClient:
    def __init__(self) -> None:
        self.channel = _FakeChannel()
        self.last_fetch: int | None = None

    def add_listener(self, _fn: object, _name: str) -> None:
        pass

    async def fetch_channel(self, channel_id: int) -> _FakeChannel:
        self.last_fetch = channel_id
        return self.channel


async def test_discord_send_fetches_channel_and_posts() -> None:
    client = _FakeDiscordClient()
    transport = DiscordTransport(client=client, bot_user_id=99)
    await transport.send("555", "hello")
    assert client.last_fetch == 555
    assert client.channel.sent == ["hello"]


async def test_discord_send_rejects_bad_thread_id() -> None:
    transport = DiscordTransport(client=_FakeDiscordClient(), bot_user_id=99)
    with pytest.raises(ValueError):
        await transport.send("not-a-snowflake", "hi")
