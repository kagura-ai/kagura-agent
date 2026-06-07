"""Real `ClaudeEngine` over the Claude Agent SDK / Claude Code CLI subprocess.

Isolated here so the SDK import never reaches the core. Exercised by the live
smoke path, not unit tests (it needs the SDK installed and a subscription).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from kagura_agent.core.brain.claude import RawTurn


class SdkEngine:  # pragma: no cover - requires claude-agent-sdk + subscription
    """Adapter: translate Claude Agent SDK messages into `RawTurn`s.

    The SDK owns the agentic loop (tool calls, MCP, sub-agents); we only relay
    its messages. Resume state is threaded through the SDK's session id.
    """

    async def query(
        self, prompt: str, *, resume_state: dict[str, Any] | None
    ) -> AsyncIterator[RawTurn]:
        from claude_agent_sdk import ClaudeAgentOptions, query  # type: ignore

        options = ClaudeAgentOptions(
            resume=(resume_state or {}).get("sdk_session_id"),
            permission_mode="default",
        )

        last_text = ""
        session_id: str | None = None
        async for message in query(prompt=prompt, options=options):
            text = _message_text(message)
            session_id = getattr(message, "session_id", session_id)
            if _is_result(message):
                yield RawTurn(
                    kind="result",
                    text=text or last_text,
                    state={"sdk_session_id": session_id},
                )
            elif text:
                last_text = text
                yield RawTurn(kind="message", text=text)


def _message_text(message: Any) -> str:  # pragma: no cover
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(getattr(block, "text", "") for block in content)
    return getattr(message, "result", "") or ""


def _is_result(message: Any) -> bool:  # pragma: no cover
    return type(message).__name__ == "ResultMessage"
