"""Brain-backend selection: resolve_brain_backend + make_brain dispatch."""

from __future__ import annotations

import pytest

from kagura_agent.core.brain.select import make_brain, resolve_brain_backend


def test_resolve_defaults_to_sdk() -> None:
    assert resolve_brain_backend({}) == "sdk"


def test_resolve_blank_is_sdk() -> None:
    assert resolve_brain_backend({"KAGURA_AGENT_BRAIN": "   "}) == "sdk"


def test_resolve_explicit_sdk_and_kagura_brain() -> None:
    assert resolve_brain_backend({"KAGURA_AGENT_BRAIN": "sdk"}) == "sdk"
    assert resolve_brain_backend({"KAGURA_AGENT_BRAIN": "kagura-brain"}) == "kagura-brain"
    assert resolve_brain_backend({"KAGURA_AGENT_BRAIN": "KAGURA-BRAIN"}) == "kagura-brain"


def test_resolve_unknown_is_fail_closed() -> None:
    # A typo must not silently run the default backend.
    with pytest.raises(ValueError, match="not a known brain backend"):
        resolve_brain_backend({"KAGURA_AGENT_BRAIN": "kagura_brain"})


# --- make_brain dispatch (factories injected so no SDK / brain extra needed) ---


def _sdk_factory_spy(calls):  # type: ignore[no-untyped-def]
    def _factory(*, mcp_servers=None, strict_mcp_config=False, permission_mode="acceptEdits"):
        calls.append(("sdk", mcp_servers, strict_mcp_config, permission_mode))
        return "SDK_BRAIN"
    return _factory


def _kagura_factory_spy(calls):  # type: ignore[no-untyped-def]
    def _factory(env):
        calls.append(("kagura", dict(env)))
        return "KAGURA_BRAIN"
    return _factory


def test_make_brain_defaults_to_sdk_factory() -> None:
    calls: list = []
    brain = make_brain(
        {},
        mcp_servers={"fs": {}},
        strict_mcp_config=True,
        sdk_factory=_sdk_factory_spy(calls),
        kagura_factory=_kagura_factory_spy(calls),
    )
    assert brain == "SDK_BRAIN"
    # mcp knobs + the SAFE default permission mode forwarded to the SDK factory.
    assert calls == [("sdk", {"fs": {}}, True, "default")]


def test_make_brain_uses_default_permission_mode_param() -> None:
    # Operator-typed callers (run/repl) raise the per-path default to acceptEdits.
    calls: list = []
    make_brain(
        {},
        default_permission_mode="acceptEdits",
        sdk_factory=_sdk_factory_spy(calls),
        kagura_factory=_kagura_factory_spy(calls),
    )
    assert calls == [("sdk", None, False, "acceptEdits")]


def test_make_brain_env_overrides_default_permission_mode_param() -> None:
    # An explicit KAGURA_AGENT_PERMISSION_MODE wins over the per-path default.
    calls: list = []
    make_brain(
        {"KAGURA_AGENT_PERMISSION_MODE": "plan"},
        default_permission_mode="acceptEdits",
        sdk_factory=_sdk_factory_spy(calls),
        kagura_factory=_kagura_factory_spy(calls),
    )
    assert calls == [("sdk", None, False, "plan")]


def test_make_brain_fail_closed_on_invalid_permission_mode() -> None:
    with pytest.raises(ValueError, match="not a known permission mode"):
        make_brain({"KAGURA_AGENT_PERMISSION_MODE": "nope"})


def test_make_brain_routes_to_kagura_when_selected() -> None:
    calls: list = []
    env = {"KAGURA_AGENT_BRAIN": "kagura-brain"}
    brain = make_brain(
        env,
        mcp_servers={"fs": {}},  # must NOT be forwarded to the kagura factory
        sdk_factory=_sdk_factory_spy(calls),
        kagura_factory=_kagura_factory_spy(calls),
    )
    assert brain == "KAGURA_BRAIN"
    assert calls == [("kagura", env)]  # kagura factory got the env, no mcp knobs


def test_make_brain_propagates_invalid_backend() -> None:
    with pytest.raises(ValueError, match="not a known brain backend"):
        make_brain({"KAGURA_AGENT_BRAIN": "bogus"})
