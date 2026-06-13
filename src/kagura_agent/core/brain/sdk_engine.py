"""Real `ClaudeEngine` over the Claude Agent SDK / Claude Code CLI subprocess.

Isolated here so the SDK import never reaches the core. Exercised by the live
smoke path, not unit tests (it needs the SDK installed and a subscription).
"""

from __future__ import annotations

import importlib.util
from collections.abc import AsyncIterator, Callable
from typing import Any

from kagura_agent.core.brain.base import BrainUnavailable
from kagura_agent.core.brain.claude import RawTurn

_CLAUDE_SDK_MODULE = "claude_agent_sdk"
_INSTALL_HINT = (
    "The Claude brain requires the optional 'claude' extra (claude-agent-sdk), "
    "which is not installed. Install it with one of:\n"
    "  uv run --extra claude kagura-agent run ...\n"
    "  pip install 'kagura-agent[claude]'"
)


def claude_sdk_available(
    *, find_spec: Callable[[str], object | None] = importlib.util.find_spec
) -> bool:
    """Whether the Claude Agent SDK is importable, *without* importing it.

    Pure and SDK-free (like `_mcp_option_kwargs`) so it is unit-testable by
    injecting `find_spec`; the real `find_spec` only inspects import metadata and
    never triggers the heavy `claude_agent_sdk` import.

    Known limit: this confirms the module is *findable*, not that it imports
    cleanly. A corrupt/partial install would pass here yet still raise at the real
    import in `query()` (falling back to the generic error path). The dominant
    case — the `claude` extra simply not installed — is fully covered.
    """
    return find_spec(_CLAUDE_SDK_MODULE) is not None


def require_claude_sdk(
    *, find_spec: Callable[[str], object | None] = importlib.util.find_spec
) -> None:
    """Raise `BrainUnavailable` with an actionable install hint if the SDK is absent.

    Called at brain construction (fail-fast) so the missing-extra condition is
    surfaced before the agentic loop runs, never as a raw `ModuleNotFoundError`.
    """
    if not claude_sdk_available(find_spec=find_spec):
        raise BrainUnavailable(_INSTALL_HINT)


def _mcp_option_kwargs(
    mcp_servers: dict[str, Any] | None, strict_mcp_config: bool
) -> dict[str, Any]:
    """Build the MCP-related kwargs for ClaudeAgentOptions (v0.2-A6).

    Module-level and SDK-free (outside the no-cover SdkEngine) so the branch
    logic is unit-tested without the SDK. Two independent rules:
    - pass `mcp_servers` only when configured, so the default path is byte-for-byte
      the pre-A6 options (memory needs no MCP server here);
    - pass `strict_mcp_config` whenever requested — including WITHOUT --mcp-config,
      a valid intent ("reject all ambient MCP servers") that must not be lost.
    """
    extra: dict[str, Any] = {}
    if mcp_servers is not None:
        extra["mcp_servers"] = mcp_servers
    if strict_mcp_config:
        extra["strict_mcp_config"] = True
    return extra


class SdkEngine:  # pragma: no cover - requires claude-agent-sdk + subscription
    """Adapter: translate Claude Agent SDK messages into `RawTurn`s.

    The SDK owns the agentic loop (tool calls, MCP, sub-agents); we only relay
    its messages. Resume state is threaded through the SDK's session id.

    `mcp_servers` is orthogonal to memory (v0.2-A6): memory is CLI-primary, so
    this carries *other* MCP servers through to the SDK's `mcp_servers` option
    (the `--mcp-config` knob). `strict_mcp_config` maps to the SDK's strict flag
    so unknown servers are rejected rather than silently ignored.
    """

    def __init__(
        self,
        *,
        mcp_servers: dict[str, Any] | None = None,
        strict_mcp_config: bool = False,
    ) -> None:
        self._mcp_servers = mcp_servers
        self._strict_mcp_config = strict_mcp_config

    async def query(
        self, prompt: str, *, resume_state: dict[str, Any] | None
    ) -> AsyncIterator[RawTurn]:
        from claude_agent_sdk import ClaudeAgentOptions, query  # type: ignore

        options = ClaudeAgentOptions(
            resume=(resume_state or {}).get("sdk_session_id"),
            permission_mode="default",
            **_mcp_option_kwargs(self._mcp_servers, self._strict_mcp_config),
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
