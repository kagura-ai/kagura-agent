"""Per-deploy brain-backend selection — `sdk` (default) | `kagura-brain`.

Two ClaudeEngines live behind the same protocol (`SdkEngine`, `KaguraBrainEngine`);
this picks which one a run uses from `KAGURA_AGENT_BRAIN`. The selection is a pure
function (`resolve_brain_backend`); `make_brain` is the thin dispatcher the CLI
calls. The default is `sdk`, so existing runs are unchanged and the cross-repo
kagura-brain dependency is strictly opt-in.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from kagura_agent.core.brain.claude import ClaudeBrain, make_default_brain
from kagura_agent.core.brain.kagura_brain_engine import make_kagura_brain

#: Env var selecting the brain backend.
BRAIN_ENV = "KAGURA_AGENT_BRAIN"
_SDK = "sdk"
_KAGURA_BRAIN = "kagura-brain"
_VALID = (_SDK, _KAGURA_BRAIN)


def resolve_brain_backend(env: Mapping[str, str]) -> str:
    """Resolve the selected backend, default ``sdk``. Pure (env in, value out).

    A set-but-blank value is treated as unset (→ sdk). An explicit unknown value
    is a fail-closed ``ValueError`` rather than a silent fallback — a typo like
    ``KAGURA_AGENT_BRAIN=kagura_brain`` must not quietly run the wrong (default)
    backend.
    """
    raw = env.get(BRAIN_ENV, "").strip().lower()
    if not raw:
        return _SDK
    if raw not in _VALID:
        raise ValueError(
            f"{BRAIN_ENV}={raw!r} is not a known brain backend "
            f"(expected one of: {', '.join(_VALID)})"
        )
    return raw


def make_brain(
    env: Mapping[str, str],
    *,
    mcp_servers: dict[str, Any] | None = None,
    strict_mcp_config: bool = False,
    sdk_factory: Callable[..., ClaudeBrain] = make_default_brain,
    kagura_factory: Callable[[Mapping[str, str]], ClaudeBrain] = make_kagura_brain,
) -> ClaudeBrain:
    """Build the brain for the selected backend.

    The `--mcp-config` knobs (`mcp_servers`, `strict_mcp_config`) are SDK-specific
    and forwarded only to the SDK factory; the kagura-brain backend does not wire
    in-task MCP here (memory is CLI-primary; grounding is the agent's own layer),
    so those knobs are intentionally not threaded into it. Factories are injected
    so the dispatch is unit-tested without the real SDK / brain extra installed.
    """
    if resolve_brain_backend(env) == _KAGURA_BRAIN:
        return kagura_factory(env)
    return sdk_factory(mcp_servers=mcp_servers, strict_mcp_config=strict_mcp_config)
