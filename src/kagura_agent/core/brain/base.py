"""The brain seam — the one abstraction v1 must protect.

`core/session.py` never calls the Claude Agent SDK directly; it depends only on
`BrainProvider`. The provider **owns its agentic loop** and yields high-level
`BrainEvent`s; the session orchestrates tasks and checkpoints, never individual
tool calls. This is what makes a future Codex/OpenAI brain a pure Phase-2
addition rather than a v1 rewrite. The protocol's *shape* is the entire
insurance policy — nothing more is abstracted (no brain-selection knob in v1).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class BrainCaps:
    """What a brain declares about itself at startup.

    `requires_mcp` is load-bearing: memory-cloud is an MCP server, so a brain
    that cannot speak MCP is *rejected at startup* (we do not silently degrade).
    """

    name: str
    requires_mcp: bool = True
    auth_modes: tuple[str, ...] = ("subscription",)


@dataclass(frozen=True)
class Task:
    """A unit of work handed to a brain. `session_id` ties it to a thread."""

    prompt: str
    session_id: str


@dataclass(frozen=True)
class Checkpoint:
    """Opaque, provider-owned resumable state for a long-running task.

    `state` is provider-private (the session never interprets it) and, per the
    Lease design, carries the renewable *budget* — never live credentials.
    """

    session_id: str
    turn: int
    state: dict[str, Any] = field(default_factory=dict)


class BrainEvent:
    """Base type for everything a brain yields while running its loop."""


@dataclass(frozen=True)
class MessageEvent(BrainEvent):
    """Human-facing narration from the brain."""

    text: str


@dataclass(frozen=True)
class DoneEvent(BrainEvent):
    """Terminal event: the brain finished this task.

    `state` becomes the next `Checkpoint.state`; the session wraps it.
    """

    result: str
    state: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BrainProvider(Protocol):
    """The seam. v1 has exactly one implementation: `ClaudeBrain`."""

    caps: BrainCaps

    def run(
        self, task: Task, *, resume: Checkpoint | None = None
    ) -> AsyncIterator[BrainEvent]:
        """Run the full agentic loop for `task`, optionally resuming.

        Returns an async iterator of events; the final `DoneEvent` carries the
        provider state to checkpoint.
        """
        ...
