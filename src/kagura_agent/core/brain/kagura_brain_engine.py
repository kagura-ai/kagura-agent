"""Alternate `ClaudeEngine` over `kagura-brain` — a second engine, not a replacement.

`ClaudeBrain` injects a `ClaudeEngine` (`query(prompt, *, resume_state) ->
AsyncIterator[RawTurn]`). `SdkEngine` wraps claude-agent-sdk; this wraps the
sibling **kagura-brain** library (the same one-shot claude/codex wrapper
kagura-engineer uses), so a deploy can pick the lighter, subscription-friendly
backend via `KAGURA_AGENT_BRAIN=kagura-brain` without touching `ClaudeBrain` /
`session.py` / `BrainProvider`. The SDK stays the default.

Capability note: kagura-brain's `handle.invoke` is a **synchronous, one-shot**
call returning a single `BrainResult` — there is no mid-stream narration and no
native resume. So this engine yields exactly one terminal `RawTurn` and ignores
`resume_state` (cross-turn continuity then rests on kagura-agent's own checkpoint
+ grounding layer, not the brain's session). The blocking `invoke` is run in a
thread so it never stalls the event loop.

The real `kagura_brain` import is isolated here and lazy (like `SdkEngine`), so
the core and tests run without the optional `brain` extra installed.
"""

from __future__ import annotations

import asyncio
import importlib.util
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path
from typing import Any

from kagura_agent.core.brain.base import BrainInvocationError, BrainUnavailable
from kagura_agent.core.brain.claude import ClaudeBrain, RawTurn

_KAGURA_BRAIN_MODULE = "kagura_brain"
_INSTALL_HINT = (
    "The kagura-brain backend requires the optional 'brain' extra (kagura-brain), "
    "which is not installed. Install it with one of:\n"
    "  uv run --extra brain kagura-agent run ...\n"
    "  pip install 'kagura-agent[brain]'\n"
    "or select the default SDK backend (unset KAGURA_AGENT_BRAIN)."
)

#: kagura-brain reads its API key from this env var (the library owns the name).
#: We pass it through to `select` consumer-side; the library never reads env.
_BRAIN_API_KEY_ENV = "KAGURA_BRAIN_API_KEY"
#: kagura-agent-side knobs for the alternate backend.
_BACKEND_ENV = "KAGURA_AGENT_BRAIN_BACKEND"  # claude | codex
_ENDPOINT_ENV = "KAGURA_AGENT_BRAIN_ENDPOINT"  # BYO endpoint (paired with the key)
#: Default per-invoke timeout (seconds) for the one-shot brain call.
_DEFAULT_TIMEOUT_SEC = 1800


def kagura_brain_available(
    *, find_spec: Callable[[str], object | None] = importlib.util.find_spec
) -> bool:
    """Whether the kagura-brain library is importable, *without* importing it.

    Pure and dependency-free (mirrors `claude_sdk_available`) so it is unit-tested
    by injecting `find_spec`; the real `find_spec` only inspects import metadata.
    """
    return find_spec(_KAGURA_BRAIN_MODULE) is not None


def require_kagura_brain(
    *, find_spec: Callable[[str], object | None] = importlib.util.find_spec
) -> None:
    """Raise `BrainUnavailable` with an actionable hint if the brain extra is absent.

    Called at brain construction (fail-fast), so the missing-extra condition is a
    clean install hint rather than a raw `ModuleNotFoundError` deep in the loop.
    """
    if not kagura_brain_available(find_spec=find_spec):
        raise BrainUnavailable(_INSTALL_HINT)


def resolve_kagura_brain_backend(env: Mapping[str, str]) -> str:
    """kagura-brain's own backend selector (claude | codex), default claude.

    Pure (env in, value out). Fail-closed on an unknown value — consistent with
    the ``KAGURA_AGENT_BRAIN`` selector (`select.resolve_brain_backend`): a typo
    like ``KAGURA_AGENT_BRAIN_BACKEND=codx`` must not silently run claude. A
    set-but-blank value is unset (→ claude).
    """
    raw = env.get(_BACKEND_ENV, "").strip().lower()
    if not raw:
        return "claude"
    if raw not in ("claude", "codex"):
        raise ValueError(
            f"{_BACKEND_ENV}={raw!r} is not a known kagura-brain backend "
            "(expected one of: claude, codex)"
        )
    return raw


def _result_to_turn(result: Any) -> RawTurn:
    """Translate a kagura-brain ``BrainResult`` into a terminal ``RawTurn``, fail-closed.

    Two contract points the live invoke must honour (both proven against the
    kagura-engineer consumer, the authoritative reference):

    - The model's reply is ``result.stdout`` (already a str). NOT ``as_text(result)``
      — kagura-brain's ``as_text`` is a ``bytes|str|None`` stream normalizer, not a
      ``BrainResult`` accessor, so ``as_text(result)`` would yield the dataclass
      itself / a repr, never the answer.
    - A non-zero exit or a timeout (``returncode != 0`` / ``timed_out``) is a real
      failure — RAISE ``BrainInvocationError`` rather than relay it as a successful
      turn. Otherwise a failed/timed-out CLI call would be recorded as a completed
      run + checkpoint + grounding summary (a silent swallow).

    One-shot has no resume token, so ``state`` is empty.
    """
    if result.returncode != 0 or result.timed_out:
        raise BrainInvocationError(f"kagura-brain invocation failed: {result.detail()}")
    return RawTurn(kind="result", text=result.stdout, state={})


def kagura_brain_select_kwargs(env: Mapping[str, str]) -> dict[str, Any]:
    """Build the `kagura_brain.select(...)` kwargs from env, fail-closed on a
    half-configured BYO endpoint/key pair.

    Pure and SDK-free so the env→kwargs mapping is unit-tested. Mirrors
    kagura-engineer's rule: a BYO endpoint requires a key and vice-versa (the
    library would otherwise surface the half-config only at the first invoke,
    mid-run). With neither set, both are None → the claude backend inherits the
    Pro/Max subscription via `claude -p` (no key needed).
    """
    endpoint = env.get(_ENDPOINT_ENV, "").strip() or None
    api_key = env.get(_BRAIN_API_KEY_ENV, "").strip() or None
    if endpoint and api_key is None:
        raise BrainUnavailable(
            f"{_ENDPOINT_ENV} is set but {_BRAIN_API_KEY_ENV} is not — a BYO "
            "endpoint needs an API key (or unset the endpoint to use subscription)."
        )
    if api_key and endpoint is None:
        raise BrainUnavailable(
            f"{_BRAIN_API_KEY_ENV} is set but {_ENDPOINT_ENV} is not — set the "
            "endpoint or unset the key to use the subscription instead."
        )
    return {
        "backend": resolve_kagura_brain_backend(env),
        "endpoint": endpoint,
        "api_key": api_key,
    }


class KaguraBrainEngine:  # pragma: no cover - requires kagura-brain + subscription/key
    """`ClaudeEngine` adapter over a kagura-brain handle.

    Selects the handle lazily on first query (so construction is I/O-free and the
    library import stays off the import path until a run actually uses it), then
    runs the synchronous one-shot `invoke` in a worker thread and relays its text
    as a single terminal `RawTurn`.
    """

    def __init__(
        self,
        *,
        select_kwargs: dict[str, Any],
        cwd: Path | None = None,
        timeout: int = _DEFAULT_TIMEOUT_SEC,
    ) -> None:
        self._select_kwargs = select_kwargs
        self._cwd = cwd
        self._timeout = timeout
        self._handle: Any | None = None

    async def query(
        self, prompt: str, *, resume_state: dict[str, Any] | None
    ) -> AsyncIterator[RawTurn]:
        # Lazy import; a partial / version-skewed install where kagura_brain is
        # top-level findable but the API surface is missing surfaces as the
        # actionable BrainUnavailable hint, not a raw ImportError deep in the loop.
        try:
            import kagura_brain
        except ImportError as exc:  # pragma: no cover - broken-install edge
            raise BrainUnavailable(_INSTALL_HINT) from exc

        if self._handle is None:
            self._handle = kagura_brain.select(**self._select_kwargs)
        # kagura-brain is one-shot: no native resume, so resume_state is ignored
        # (cross-turn continuity rests on the agent's checkpoint + grounding layer).
        # invoke() is blocking — run it off the event loop. _result_to_turn extracts
        # result.stdout and fails closed on a non-zero/timed-out invoke.
        result = await asyncio.to_thread(
            self._handle.invoke, prompt, cwd=self._cwd, timeout=self._timeout
        )
        yield _result_to_turn(result)


def make_kagura_brain(
    env: Mapping[str, str], *, cwd: Path | None = None
) -> ClaudeBrain:  # pragma: no cover - requires the brain extra
    """Construct `ClaudeBrain` over `KaguraBrainEngine` from env config.

    Fail-fast with an actionable `BrainUnavailable` if the optional `brain` extra
    is missing (before the loop runs). `cwd` defaults to the current directory.
    """
    require_kagura_brain()
    return ClaudeBrain(
        engine=KaguraBrainEngine(
            select_kwargs=kagura_brain_select_kwargs(env),
            cwd=cwd if cwd is not None else Path.cwd(),
        )
    )
