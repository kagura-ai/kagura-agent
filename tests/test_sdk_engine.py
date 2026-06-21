"""v0.2-A6: the testable seam of the SDK engine — MCP option kwargs.

`SdkEngine.query` itself needs the live claude-agent-sdk (`# pragma: no cover`),
but the decision of WHICH mcp options to pass is pure logic. It is extracted to
`_mcp_option_kwargs` so the four branches are regression-tested without the SDK —
in particular the `--strict-mcp-config`-without-`--mcp-config` case (a bug fixed
in code review that previously had no test).
"""

import pytest

from kagura_agent.core.brain.sdk_engine import (
    _mcp_option_kwargs,
    resolve_permission_mode,
)


def test_mcp_kwargs_empty_when_unconfigured() -> None:
    # Default path: no mcp keys, so ClaudeAgentOptions stays byte-for-byte pre-A6.
    assert _mcp_option_kwargs(None, False) == {}


def test_mcp_kwargs_servers_only() -> None:
    assert _mcp_option_kwargs({"fs": {"command": "srv"}}, False) == {
        "mcp_servers": {"fs": {"command": "srv"}}
    }


def test_mcp_kwargs_strict_only_is_not_dropped() -> None:
    # The regression: --strict-mcp-config without --mcp-config must still thread
    # strict ("reject all ambient MCP servers"), not silently lose it.
    assert _mcp_option_kwargs(None, True) == {"strict_mcp_config": True}


def test_mcp_kwargs_both() -> None:
    assert _mcp_option_kwargs({"fs": {}}, True) == {
        "mcp_servers": {"fs": {}},
        "strict_mcp_config": True,
    }


# --- permission mode (the headless `run` write-block fix) -------------------
# A headless `query()` with permission_mode="default" and no approval channel
# dead-ends every mutating tool. The SAFE module default is "default"; the
# `default` arg lets operator-typed callers (run/repl) opt into "acceptEdits"
# so a self-host run can write files, while serve keeps the safe default.


def test_permission_mode_defaults_to_safe_default() -> None:
    # Unset + no override → the safe "default" (dead-end), NOT auto-accept: a
    # blanket auto-write default is unsafe on the unsealed, operator-gateless
    # in-process serve brain (see the cockpit task path).
    assert resolve_permission_mode({}) == "default"


def test_permission_mode_default_arg_is_used_when_unset() -> None:
    # Operator-typed run/repl pass default="acceptEdits" so the run can write.
    assert resolve_permission_mode({}, default="acceptEdits") == "acceptEdits"


def test_permission_mode_blank_uses_the_default_arg() -> None:
    # A set-but-blank value is treated as unset, like the brain selector.
    assert (
        resolve_permission_mode({"KAGURA_AGENT_PERMISSION_MODE": "   "}, default="acceptEdits")
        == "acceptEdits"
    )


def test_permission_mode_env_overrides_the_default_arg() -> None:
    # An explicit operator setting wins over the per-path default, on every path.
    assert (
        resolve_permission_mode({"KAGURA_AGENT_PERMISSION_MODE": "plan"}, default="acceptEdits")
        == "plan"
    )


def test_permission_mode_exposes_all_sdk_modes() -> None:
    # The curated set matches the SDK's own PermissionMode Literal so we never
    # reject a value the SDK accepts — including the stricter `dontAsk` and `auto`.
    for mode in ("default", "acceptEdits", "plan", "bypassPermissions", "dontAsk", "auto"):
        assert resolve_permission_mode({"KAGURA_AGENT_PERMISSION_MODE": mode}) == mode


def test_permission_mode_is_case_insensitive_to_canonical() -> None:
    # Env vars are case-fragile; accept any case, return the SDK's exact spelling.
    assert resolve_permission_mode({"KAGURA_AGENT_PERMISSION_MODE": "dontask"}) == "dontAsk"


def test_permission_mode_unknown_is_fail_closed() -> None:
    # A typo must not silently fall back to a dead-end or an over-permissive
    # mode — fail closed, like resolve_brain_backend.
    with pytest.raises(ValueError, match="not a known permission mode"):
        resolve_permission_mode({"KAGURA_AGENT_PERMISSION_MODE": "yolo"})
