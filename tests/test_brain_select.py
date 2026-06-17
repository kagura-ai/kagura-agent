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
    def _factory(*, mcp_servers=None, strict_mcp_config=False):
        calls.append(("sdk", mcp_servers, strict_mcp_config))
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
    # mcp knobs forwarded to the SDK factory only
    assert calls == [("sdk", {"fs": {}}, True)]


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
