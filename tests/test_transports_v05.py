"""v0.5: Slack + Discord normalize to the same `Event` the cockpit already eats.

This is the proof that v0.5 is a *pure addition*: no core change, no new intent —
each transport just maps its native payload onto `Event`, including the
structural launch/continue signal (top-level = launch, thread reply = continue)
and dropping the bot's own messages. The core never imports a transport SDK.
"""

from kagura_agent.cockpit.intent import Intent, classify
from kagura_agent.cockpit.transports.base import Event
from kagura_agent.cockpit.transports.discord import normalize_discord_message
from kagura_agent.cockpit.transports.slack import normalize_slack_event

# --- Slack ----------------------------------------------------------------

def test_slack_top_level_message_is_launch_event() -> None:
    payload = {"type": "message", "user": "U1", "text": "build it", "channel": "D1", "ts": "100.1"}
    event = normalize_slack_event(payload, bot_user_id="UBOT")
    assert event == Event(thread_id="100.1", text="build it", is_thread_reply=False, sender="U1")
    assert classify(event, known_sessions=set()) is Intent.LAUNCH


def test_slack_thread_reply_is_continue_event() -> None:
    payload = {
        "type": "message", "user": "U1", "text": "more", "channel": "D1",
        "ts": "101.2", "thread_ts": "100.1",
    }
    event = normalize_slack_event(payload, bot_user_id="UBOT")
    assert event == Event(thread_id="100.1", text="more", is_thread_reply=True, sender="U1")
    assert classify(event, known_sessions={"100.1"}) is Intent.CONTINUE


def test_slack_ignores_bot_own_messages() -> None:
    assert normalize_slack_event(
        {"type": "message", "user": "UBOT", "text": "hi", "ts": "1"}, bot_user_id="UBOT"
    ) is None
    assert normalize_slack_event(
        {"type": "message", "bot_id": "B1", "text": "hi", "ts": "1"}, bot_user_id="UBOT"
    ) is None


def test_slack_ignores_non_message_events() -> None:
    assert normalize_slack_event(
        {"type": "reaction_added", "user": "U1"}, bot_user_id="UBOT"
    ) is None


# --- Discord --------------------------------------------------------------

def test_discord_channel_message_is_launch_event() -> None:
    event = normalize_discord_message(
        author_id=1, bot_user_id=99, content="build it", channel_id=555, thread_id=None
    )
    assert event == Event(thread_id="555", text="build it", is_thread_reply=False, sender="1")


def test_discord_thread_message_is_continue_event() -> None:
    event = normalize_discord_message(
        author_id=1, bot_user_id=99, content="more", channel_id=555, thread_id=777
    )
    assert event == Event(thread_id="777", text="more", is_thread_reply=True, sender="1")


def test_discord_ignores_bot_own_messages() -> None:
    assert normalize_discord_message(
        author_id=99, bot_user_id=99, content="hi", channel_id=555, thread_id=None
    ) is None
