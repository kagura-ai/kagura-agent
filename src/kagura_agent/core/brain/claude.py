"""ClaudeBrain — v1's only brain. Wraps the Agent SDK / Claude Code CLI.

It **owns its agentic loop**: the SDK runs tool calls internally; ClaudeBrain
only translates the raw turns into `BrainEvent`s. All Claude-specific surface is
isolated to this module — `session.py` never imports it.

The SDK is injected as a `ClaudeEngine` so the core (and tests) run without the
SDK installed or any network. `SdkEngine` is the real adapter, imported lazily.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from kagura_agent.core.brain.sdk_engine import PermissionMode

from kagura_agent.core.brain.base import (
    BrainCaps,
    BrainEvent,
    Checkpoint,
    DoneEvent,
    MessageEvent,
    Task,
)


@dataclass(frozen=True)
class RawTurn:
    """One step out of the engine's own loop, pre-translation."""

    kind: str  # "message" | "result"
    text: str
    state: dict[str, Any] = field(default_factory=dict)


class ClaudeEngine(Protocol):
    def query(
        self, prompt: str, *, resume_state: dict[str, Any] | None
    ) -> AsyncIterator[RawTurn]: ...


class ClaudeBrain:
    caps = BrainCaps(
        name="claude",
        auth_modes=("subscription", "byok"),
    )

    def __init__(self, engine: ClaudeEngine) -> None:
        self._engine = engine

    async def run(
        self, task: Task, *, resume: Checkpoint | None = None
    ) -> AsyncIterator[BrainEvent]:
        resume_state = resume.state if resume is not None else None
        async for turn in self._engine.query(task.prompt, resume_state=resume_state):
            if turn.kind == "message":
                yield MessageEvent(text=turn.text)
            elif turn.kind == "result":
                yield DoneEvent(result=turn.text, state=turn.state)
            # unknown kinds are ignored — forward-compatible with new SDK turns


def make_default_brain(  # pragma: no cover - requires the SDK
    *,
    mcp_servers: dict[str, Any] | None = None,
    strict_mcp_config: bool = False,
    permission_mode: PermissionMode = "default",
) -> ClaudeBrain:
    """Construct ClaudeBrain over the real SDK engine (lazy import).

    `mcp_servers` (from `--mcp-config`) is threaded into the SDK engine for
    non-memory MCP servers; memory itself is reached via the CLI, not here.
    `permission_mode` (resolved from `KAGURA_AGENT_PERMISSION_MODE` by `make_brain`,
    which is the only caller and always passes it explicitly) is the headless
    tool-approval policy. The literal default here is the SAFE `default`; the live
    per-path value (operator-typed `run`/`repl` -> `acceptEdits`) comes from
    `make_brain`.
    """

    from kagura_agent.core.brain.sdk_engine import SdkEngine, require_claude_sdk

    # Fail-fast with an actionable BrainUnavailable if the optional `claude` extra
    # is missing — before the cockpit loop runs, so the operator gets an install
    # hint instead of a raw ModuleNotFoundError surfaced as "internal error".
    require_claude_sdk()
    return ClaudeBrain(
        engine=SdkEngine(
            mcp_servers=mcp_servers,
            strict_mcp_config=strict_mcp_config,
            permission_mode=permission_mode,
        )
    )
