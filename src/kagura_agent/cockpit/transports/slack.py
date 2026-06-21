"""Slack transport — Bolt, Socket Mode (no public URL). A pure addition.

Split into two layers, matching the codebase convention:

- **pure, unit-tested helpers** — `normalize_slack_event` (payload -> `Event`,
  now carrying the sender for the operator-identity gate), `resolve_channel`
  (thread_ts -> channel id), `approval_blocks` / `action_value` (the ✅/❌ HITL
  buttons and reading the click back).
- **`SlackTransport`** — the Bolt/WebClient I/O that wires those helpers to a
  live workspace. It needs slack-bolt + a workspace, so it is exercised at
  deployment, not in unit tests (`# pragma: no cover`).

Operator identity (#14): `normalize_slack_event` puts the Slack user id on
`Event.sender`; the deployer constructs `Cockpit(operator_id="<U…>")` so only
that user's `/approve` resolves a pending request. A separate bot id
(`@kagura-agent`) keeps this cockpit distinct from any ingestion bot.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from kagura_agent.cockpit.transports.base import Event, click_authorized

# Stable identifier for the HITL approval action block, so the action handler can
# tell an approval click apart from any other interactive component.
APPROVAL_ACTION_ID = "kagura_hitl_choice"


def normalize_slack_event(payload: dict[str, Any], *, bot_user_id: str) -> Event | None:
    """Map a Slack events-API `message` payload onto the shared `Event`.

    Derives the structural launch/continue signal, drops the bot's own messages,
    and carries the sender (`payload["user"]`) so the cockpit can enforce
    operator identity on HITL approvals (#14).
    """
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

    text = payload.get("text", "")
    if not text.strip():
        # Empty / whitespace-only body (attachment-only, sticker, share). A plain
        # type=message with no text slips past the subtype guard above; dropping it
        # prevents a billed empty-prompt brain LAUNCH.
        return None

    ts = payload["ts"]
    thread_ts = payload.get("thread_ts")
    thread_id = thread_ts or ts
    is_thread_reply = thread_ts is not None and thread_ts != ts
    return Event(
        thread_id=thread_id,
        text=text,
        is_thread_reply=is_thread_reply,
        sender=payload.get("user"),
    )


def resolve_channel(thread_id: str, channel_map: dict[str, str]) -> str:
    """The Slack channel a reply on `thread_id` must go to.

    `Event` keys on the thread ts and does not carry the channel, so the
    transport records thread_id -> channel as events arrive. Fail loud on an
    unknown thread rather than guess a channel — posting to the wrong channel is
    worse than raising.
    """
    try:
        return channel_map[thread_id]
    except KeyError:
        raise KeyError(
            f"no channel known for thread {thread_id!r}; cannot post (the thread "
            "must have been observed via listen() first)"
        ) from None


def approval_blocks(question: str, options: list[str]) -> list[dict[str, Any]]:
    """Block Kit: the question plus one button per option (value = the option).

    The click is read back by `action_value`; the option string round-trips
    through the button `value`, so the cockpit's HITL contract (`ask` returns the
    chosen option) holds without a side lookup table.
    """
    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": question}},
        {
            "type": "actions",
            "block_id": APPROVAL_ACTION_ID,
            "elements": [
                {
                    "type": "button",
                    "action_id": f"{APPROVAL_ACTION_ID}:{opt}",
                    "text": {"type": "plain_text", "text": opt},
                    "value": opt,
                }
                for opt in options
            ],
        },
    ]


def action_value(payload: dict[str, Any]) -> str:
    """Extract the clicked option from a Slack interactive-action payload."""
    actions = payload.get("actions") or []
    if not actions:
        raise ValueError("Slack action payload has no actions")
    return str(actions[0]["value"])


class SlackTransport:  # pragma: no cover - requires slack-bolt + a workspace
    """Socket-Mode adapter. Holds the bot token; lives only in the cockpit.

    `app` is a `slack_bolt.async_app.AsyncApp`. Channels are learned from inbound
    events (`listen`) and reused by `send`/`ask`, so a reply always lands on the
    thread it belongs to. `ask` posts ✅/❌ buttons and blocks on a future the
    action handler resolves.
    """

    def __init__(
        self,
        app: Any,
        bot_user_id: str,
        channel_map: dict[str, str] | None = None,
        *,
        operator_id: str | None = None,
        require_operator: bool = False,
    ) -> None:
        self._app = app
        self._bot_user_id = bot_user_id
        self._channels: dict[str, str] = dict(channel_map or {})
        self._pending: dict[str, asyncio.Future[str]] = {}
        self._inbox: asyncio.Queue[Event] = asyncio.Queue()
        self._operator_id = operator_id
        self._require_operator = require_operator
        self._wire_handlers()

    def _wire_handlers(self) -> None:
        # Registered via explicit calls rather than `@self._app.event(...)`
        # decorator syntax: `self._app` is Any (Bolt is an optional dep), and
        # decorator syntax on an untyped callable trips mypy strict's
        # disallow_untyped_decorators. A plain call of an Any is fine.
        async def _on_message(event: dict[str, Any], **_: Any) -> None:
            normalized = normalize_slack_event(event, bot_user_id=self._bot_user_id)
            if normalized is None:
                return
            channel = event.get("channel")
            if channel is not None:
                self._channels[normalized.thread_id] = channel
            await self._inbox.put(normalized)

        async def _on_choice(ack: Any, body: dict[str, Any], **_: Any) -> None:
            await ack()
            # Operator-identity gate (#14) for the button path: ignore a click
            # from anyone but the operator (leave the request pending).
            clicker = (body.get("user") or {}).get("id")
            if not click_authorized(
                clicker, self._operator_id, require_operator=self._require_operator
            ):
                return
            # Resolve the pending future by the thread the prompt was keyed under
            # (thread_ts); or, if the prompt was NOT threaded (Slack posted it
            # top-level, e.g. a stale thread_id), by the prompt message's own ts
            # (container.message_ts / message.ts), which `ask` also keyed the future
            # under. Only a payload with no usable identifier at all is ignored (the
            # awaiter then fails closed on its own timeout) rather than guessed.
            msg = body.get("message") or {}
            container = body.get("container") or {}
            key = (
                msg.get("thread_ts")
                or container.get("thread_ts")
                or container.get("message_ts")
                or msg.get("ts")
            )
            if key is None:
                return
            # Extract the clicked value BEFORE popping, so a malformed payload
            # (action_value raising) leaves the future pending — re-clickable, and
            # failing closed on its own timeout — rather than popped-but-unresolved
            # (which would hang the awaiter).
            value = action_value(body)
            future = self._pending.pop(key, None)
            if future is not None and not future.done():
                future.set_result(value)

        self._app.event("message")(_on_message)
        self._app.action({"block_id": APPROVAL_ACTION_ID})(_on_choice)

    async def listen(self) -> AsyncIterator[Event]:
        while True:
            yield await self._inbox.get()

    async def send(self, thread_id: str, text: str) -> None:
        await self._app.client.chat_postMessage(
            channel=resolve_channel(thread_id, self._channels),
            thread_ts=thread_id,
            text=text,
        )

    async def ask(self, thread_id: str, question: str, options: list[str]) -> str:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        # Fail an outstanding request for the same thread instead of silently
        # overwriting it — its awaiter would otherwise hang forever.
        superseded = self._pending.get(thread_id)
        if superseded is not None and not superseded.done():
            superseded.set_exception(
                RuntimeError("approval superseded by a newer request on this thread")
            )
        self._pending[thread_id] = future
        resp = await self._app.client.chat_postMessage(
            channel=resolve_channel(thread_id, self._channels),
            thread_ts=thread_id,
            text=question,
            blocks=approval_blocks(question, options),
        )
        # Also key the future by the POSTED prompt's own ts so a button click resolves
        # even when its payload carries no thread_ts (Slack posted the prompt
        # top-level) — _on_choice falls back to container.message_ts / message.ts.
        posted_ts = resp.get("ts") if hasattr(resp, "get") else None
        if posted_ts and posted_ts != thread_id:
            self._pending[posted_ts] = future
        try:
            return await future
        finally:
            # Drop our own keys when the request completes (resolved / superseded /
            # awaiter cancelled), but only if they still point at THIS future — so a
            # newer request that reused thread_id is never evicted.
            for key in (thread_id, posted_ts):
                if key is not None and self._pending.get(key) is future:
                    del self._pending[key]
