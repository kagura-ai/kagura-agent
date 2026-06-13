"""#37: Slack transport pure helpers (the Bolt I/O is deployment-exercised).

Covers the unit-testable layer: the normalizer now carrying the sender (for the
operator-identity gate), channel resolution, and the HITL button block / click
round-trip.
"""

from __future__ import annotations

import pytest

from kagura_agent.cockpit.transports.base import Event
from kagura_agent.cockpit.transports.slack import (
    APPROVAL_ACTION_ID,
    action_value,
    approval_blocks,
    normalize_slack_event,
    resolve_channel,
)

# --- sender (operator-identity gate, #14) -----------------------------------

def test_normalize_carries_sender() -> None:
    event = normalize_slack_event(
        {"type": "message", "user": "U123", "text": "hi", "channel": "D1", "ts": "1.0"},
        bot_user_id="UBOT",
    )
    assert event == Event(thread_id="1.0", text="hi", is_thread_reply=False, sender="U123")


def test_normalize_thread_reply_carries_sender() -> None:
    event = normalize_slack_event(
        {
            "type": "message", "user": "U9", "text": "more", "channel": "D1",
            "ts": "2.0", "thread_ts": "1.0",
        },
        bot_user_id="UBOT",
    )
    assert event == Event(thread_id="1.0", text="more", is_thread_reply=True, sender="U9")


# --- channel resolution -----------------------------------------------------

def test_resolve_channel_returns_recorded_channel() -> None:
    assert resolve_channel("1.0", {"1.0": "C42"}) == "C42"


def test_resolve_channel_unknown_thread_fails_loud() -> None:
    # Posting to a guessed channel is worse than raising.
    with pytest.raises(KeyError, match="no channel known"):
        resolve_channel("nope", {"1.0": "C42"})


# --- HITL buttons -----------------------------------------------------------

def test_approval_blocks_one_button_per_option_value_roundtrips() -> None:
    blocks = approval_blocks("approve?", ["/approve", "/deny"])
    actions = next(b for b in blocks if b["type"] == "actions")
    assert actions["block_id"] == APPROVAL_ACTION_ID
    values = [el["value"] for el in actions["elements"]]
    assert values == ["/approve", "/deny"]
    # the question text is present as a section
    assert any(b.get("type") == "section" for b in blocks)


def test_action_value_extracts_clicked_option() -> None:
    payload = {"actions": [{"action_id": f"{APPROVAL_ACTION_ID}:/approve", "value": "/approve"}]}
    assert action_value(payload) == "/approve"


def test_action_value_empty_actions_raises() -> None:
    with pytest.raises(ValueError, match="no actions"):
        action_value({"actions": []})
