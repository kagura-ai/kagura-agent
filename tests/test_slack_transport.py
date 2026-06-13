"""#37: Slack transport pure helpers (the Bolt I/O is deployment-exercised).

Covers the unit-testable layer: the normalizer now carrying the sender (for the
operator-identity gate), channel resolution, and the HITL button block / click
round-trip.
"""

from __future__ import annotations

import asyncio

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


def test_normalize_drops_empty_text_message() -> None:
    # A plain type=message with no text (attachment-only) must not become a
    # billed empty-prompt LAUNCH.
    assert normalize_slack_event(
        {"type": "message", "user": "U1", "channel": "D1", "ts": "1.0"},
        bot_user_id="UBOT",
    ) is None
    assert normalize_slack_event(
        {"type": "message", "user": "U1", "channel": "D1", "ts": "1.0", "text": "   "},
        bot_user_id="UBOT",
    ) is None


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


# --- HITL ask: operator gate + no supersede hang ----------------------------

class _FakeSlackClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []  # type: ignore[type-arg]

    async def chat_postMessage(self, **kw: object) -> None:
        self.calls.append(dict(kw))


class _CapturingApp:
    """Fake AsyncApp that captures the registered message/action handlers."""

    def __init__(self) -> None:
        self.client = _FakeSlackClient()
        self.action_handler = None  # type: ignore[var-annotated]

    def event(self, _name: str):  # type: ignore[no-untyped-def]
        return lambda fn: fn

    def action(self, _spec: object):  # type: ignore[no-untyped-def]
        def deco(fn):  # type: ignore[no-untyped-def]
            self.action_handler = fn
            return fn

        return deco


async def _noop_ack() -> None:
    return None


async def test_ask_button_resolves_only_for_operator() -> None:
    from kagura_agent.cockpit.transports.slack import SlackTransport

    app = _CapturingApp()
    t = SlackTransport(app, "UBOT", channel_map={"1.0": "C1"}, operator_id="op")
    pending = asyncio.create_task(t.ask("1.0", "approve?", ["/approve", "/deny"]))
    await asyncio.sleep(0)  # let ask register the pending future + post

    # A non-operator click is ignored (request stays pending).
    await app.action_handler(  # type: ignore[misc]
        ack=_noop_ack,
        body={"user": {"id": "attacker"}, "container": {"thread_ts": "1.0"},
              "actions": [{"value": "/approve"}]},
    )
    await asyncio.sleep(0)
    assert not pending.done()

    # The operator's click resolves it.
    await app.action_handler(  # type: ignore[misc]
        ack=_noop_ack,
        body={"user": {"id": "op"}, "container": {"thread_ts": "1.0"},
              "actions": [{"value": "/approve"}]},
    )
    assert await asyncio.wait_for(pending, timeout=1) == "/approve"


async def test_ask_supersede_fails_old_future_instead_of_hanging() -> None:
    from kagura_agent.cockpit.transports.slack import SlackTransport

    app = _CapturingApp()
    t = SlackTransport(app, "UBOT", channel_map={"1.0": "C1"})
    first = asyncio.create_task(t.ask("1.0", "q1", ["/approve"]))
    await asyncio.sleep(0)
    second = asyncio.create_task(t.ask("1.0", "q2", ["/approve"]))
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="superseded"):
        await asyncio.wait_for(first, timeout=1)  # old awaiter unblocks, not hangs
    second.cancel()
