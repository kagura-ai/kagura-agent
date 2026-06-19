"""v0.1: per-provider auth and ClaudeBrain translation.

Key invariants under test:
- subscription auth injects **no secret** into the container.
- ClaudeBrain owns its loop; it translates a `ClaudeEngine`'s raw turns into
  `BrainEvent`s without `session.py` ever seeing the SDK.

The memory startup gate is no longer a brain concern (v0.2-A6): memory is
brain-independent, so its reachability gate lives in `mcp/memory_cloud.py` and
is exercised by `test_memory.py`.
"""

from collections.abc import AsyncIterator

import pytest

from kagura_agent.core.brain.auth import AuthError, resolve_auth
from kagura_agent.core.brain.base import (
    BrainUnavailable,
    Checkpoint,
    DoneEvent,
    MessageEvent,
    Task,
)
from kagura_agent.core.brain.claude import ClaudeBrain, RawTurn
from kagura_agent.core.brain.sdk_engine import claude_sdk_available, require_claude_sdk

# --- per-provider auth ----------------------------------------------------

def test_subscription_auth_carries_no_secret() -> None:
    res = resolve_auth(("subscription",), env={"CLAUDE_CODE_SUBSCRIPTION": "1"})
    assert res.mode == "subscription"
    assert res.secret is None


def test_byok_auth_carries_the_key() -> None:
    res = resolve_auth(("subscription", "key"), env={"ANTHROPIC_API_KEY": "sk-xyz"})
    assert res.mode == "key"
    assert res.secret == "sk-xyz"


def test_byok_mode_is_preserved_not_collapsed_to_key() -> None:
    # #123: when the satisfied mode is byok, the resolution must report mode="byok",
    # not "key". A caller branching on the future SaaS BYOK path must be able to tell
    # BYOK from key auth — the distinction this module exists to preserve.
    res = resolve_auth(("byok",), env={"ANTHROPIC_API_KEY": "sk-xyz"})
    assert res.mode == "byok"
    assert res.secret == "sk-xyz"


def test_subscription_preferred_over_key_when_both_present() -> None:
    res = resolve_auth(
        ("subscription", "key"),
        env={"CLAUDE_CODE_SUBSCRIPTION": "1", "ANTHROPIC_API_KEY": "sk-xyz"},
    )
    assert res.mode == "subscription"
    assert res.secret is None


def test_auth_raises_when_no_mode_satisfiable() -> None:
    with pytest.raises(AuthError):
        resolve_auth(("subscription",), env={})


# --- ClaudeBrain translation ----------------------------------------------

class FakeEngine:
    """Stand-in for the Claude Agent SDK / Claude Code CLI subprocess."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict | None]] = []

    async def query(
        self, prompt: str, *, resume_state: dict | None
    ) -> AsyncIterator[RawTurn]:
        self.calls.append((prompt, resume_state))
        yield RawTurn(kind="message", text="working on it")
        yield RawTurn(kind="result", text="all done", state={"turn": 1})


def test_claude_brain_declares_subscription_auth() -> None:
    brain = ClaudeBrain(engine=FakeEngine())
    assert "subscription" in brain.caps.auth_modes
    assert not hasattr(brain.caps, "requires_mcp")  # memory decoupled from brain


async def test_claude_brain_translates_engine_turns_to_events() -> None:
    brain = ClaudeBrain(engine=FakeEngine())
    events = [e async for e in brain.run(Task(prompt="hi", session_id="s1"))]

    assert isinstance(events[0], MessageEvent)
    assert events[0].text == "working on it"
    assert isinstance(events[-1], DoneEvent)
    assert events[-1].result == "all done"
    assert events[-1].state == {"turn": 1}


async def test_claude_brain_passes_resume_state_to_engine() -> None:
    engine = FakeEngine()
    brain = ClaudeBrain(engine=engine)
    cp = Checkpoint(session_id="s1", turn=1, state={"turn": 1})

    _ = [e async for e in brain.run(Task(prompt="more", session_id="s1"), resume=cp)]

    assert engine.calls == [("more", {"turn": 1})]


# --- #28: optional Claude brain (claude-agent-sdk) availability -----------
# The SDK is an optional extra. Detection is a pure, SDK-free helper (find_spec
# injected) so it is testable without installing the 73MB SDK — mirroring the
# `_mcp_option_kwargs` pattern. Absence raises a typed, actionable BrainUnavailable
# (NOT a raw ModuleNotFoundError that downstream would render as "internal error").

def test_claude_sdk_available_reflects_find_spec() -> None:
    assert claude_sdk_available(find_spec=lambda name: object()) is True
    assert claude_sdk_available(find_spec=lambda name: None) is False


def test_require_claude_sdk_silent_when_present() -> None:
    require_claude_sdk(find_spec=lambda name: object())  # must not raise


def test_require_claude_sdk_raises_actionable_when_absent() -> None:
    with pytest.raises(BrainUnavailable) as ei:
        require_claude_sdk(find_spec=lambda name: None)
    msg = str(ei.value)
    assert "claude" in msg.lower()  # names the optional extra
    # surfaces a concrete install command the operator can run
    assert "--extra claude" in msg or "kagura-agent[claude]" in msg
