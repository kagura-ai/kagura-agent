"""KaguraBrainEngine pure seams — availability, backend resolve, select kwargs.

The live `KaguraBrainEngine.query` / `make_kagura_brain` need the real
kagura-brain lib + a subscription (pragma no cover, like SdkEngine); the pure
config logic below is unit-tested by injecting find_spec / env.
"""

from __future__ import annotations

import pytest

from kagura_agent.core.brain.base import BrainUnavailable
from kagura_agent.core.brain.kagura_brain_engine import (
    kagura_brain_available,
    kagura_brain_select_kwargs,
    require_kagura_brain,
    resolve_kagura_brain_backend,
)


def test_kagura_brain_available_true_when_findable() -> None:
    assert kagura_brain_available(find_spec=lambda name: object()) is True


def test_kagura_brain_available_false_when_absent() -> None:
    assert kagura_brain_available(find_spec=lambda name: None) is False


def test_require_kagura_brain_raises_actionable_when_absent() -> None:
    with pytest.raises(BrainUnavailable, match="brain"):
        require_kagura_brain(find_spec=lambda name: None)


def test_require_kagura_brain_passes_when_present() -> None:
    require_kagura_brain(find_spec=lambda name: object())  # no raise


def test_resolve_backend_defaults_to_claude() -> None:
    assert resolve_kagura_brain_backend({}) == "claude"
    assert resolve_kagura_brain_backend({"KAGURA_AGENT_BRAIN_BACKEND": "claude"}) == "claude"


def test_resolve_backend_codex_explicit() -> None:
    assert resolve_kagura_brain_backend({"KAGURA_AGENT_BRAIN_BACKEND": "codex"}) == "codex"
    assert resolve_kagura_brain_backend({"KAGURA_AGENT_BRAIN_BACKEND": "CODEX"}) == "codex"


def test_resolve_backend_unknown_falls_to_claude() -> None:
    # Conservative default: anything not explicitly codex is claude.
    assert resolve_kagura_brain_backend({"KAGURA_AGENT_BRAIN_BACKEND": "weird"}) == "claude"


def test_select_kwargs_subscription_default_has_no_endpoint_or_key() -> None:
    # No endpoint/key → claude backend inherits the subscription (both None).
    kw = kagura_brain_select_kwargs({})
    assert kw == {"backend": "claude", "endpoint": None, "api_key": None}


def test_select_kwargs_byo_endpoint_and_key() -> None:
    kw = kagura_brain_select_kwargs(
        {
            "KAGURA_AGENT_BRAIN_ENDPOINT": "https://brain.example",
            "KAGURA_BRAIN_API_KEY": "sk-xyz",
            "KAGURA_AGENT_BRAIN_BACKEND": "codex",
        }
    )
    assert kw == {
        "backend": "codex",
        "endpoint": "https://brain.example",
        "api_key": "sk-xyz",
    }


def test_select_kwargs_endpoint_without_key_fails_closed() -> None:
    with pytest.raises(BrainUnavailable, match="needs an API key"):
        kagura_brain_select_kwargs({"KAGURA_AGENT_BRAIN_ENDPOINT": "https://brain.example"})


def test_select_kwargs_key_without_endpoint_fails_closed() -> None:
    with pytest.raises(BrainUnavailable, match="endpoint"):
        kagura_brain_select_kwargs({"KAGURA_BRAIN_API_KEY": "sk-xyz"})


def test_select_kwargs_blank_values_treated_as_unset() -> None:
    kw = kagura_brain_select_kwargs(
        {"KAGURA_AGENT_BRAIN_ENDPOINT": "  ", "KAGURA_BRAIN_API_KEY": ""}
    )
    assert kw == {"backend": "claude", "endpoint": None, "api_key": None}
