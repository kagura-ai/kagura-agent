"""KaguraBrainEngine pure seams — availability, backend resolve, select kwargs.

The live `KaguraBrainEngine.query` / `make_kagura_brain` need the real
kagura-brain lib + a subscription (pragma no cover, like SdkEngine); the pure
config logic below is unit-tested by injecting find_spec / env.
"""

from __future__ import annotations

import pytest

from kagura_agent.core.brain.base import BrainInvocationError, BrainUnavailable
from kagura_agent.core.brain.kagura_brain_engine import (
    _result_to_turn,
    kagura_brain_available,
    kagura_brain_select_kwargs,
    require_kagura_brain,
    resolve_kagura_brain_backend,
)


class _FakeBrainResult:
    """Duck-typed kagura-brain BrainResult for translating without the lib."""

    def __init__(self, *, returncode=0, stdout="", timed_out=False, detail="detail-text"):  # type: ignore[no-untyped-def]
        self.returncode = returncode
        self.stdout = stdout
        self.timed_out = timed_out
        self._detail = detail

    def detail(self) -> str:
        return self._detail


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


def test_resolve_backend_unknown_fails_closed() -> None:
    # Fail-closed (consistent with the KAGURA_AGENT_BRAIN selector): a typo must not
    # silently run claude.
    with pytest.raises(ValueError, match="not a known kagura-brain backend"):
        resolve_kagura_brain_backend({"KAGURA_AGENT_BRAIN_BACKEND": "codx"})


def test_resolve_backend_blank_is_claude() -> None:
    assert resolve_kagura_brain_backend({"KAGURA_AGENT_BRAIN_BACKEND": "  "}) == "claude"


# --- _result_to_turn: stdout extraction + fail-closed on a bad invoke ---------


def test_result_to_turn_extracts_stdout() -> None:
    # The model's reply is result.stdout — NOT as_text(result).
    turn = _result_to_turn(_FakeBrainResult(returncode=0, stdout="the answer"))
    assert turn.kind == "result"
    assert turn.text == "the answer"
    assert turn.state == {}


def test_result_to_turn_raises_on_nonzero_exit() -> None:
    with pytest.raises(BrainInvocationError, match="detail-text"):
        _result_to_turn(_FakeBrainResult(returncode=1, stdout="partial", detail="detail-text"))


def test_result_to_turn_raises_on_timeout() -> None:
    with pytest.raises(BrainInvocationError):
        _result_to_turn(_FakeBrainResult(returncode=0, timed_out=True))


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
