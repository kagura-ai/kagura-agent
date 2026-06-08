"""v0.2-A6: the testable seam of the SDK engine — MCP option kwargs.

`SdkEngine.query` itself needs the live claude-agent-sdk (`# pragma: no cover`),
but the decision of WHICH mcp options to pass is pure logic. It is extracted to
`_mcp_option_kwargs` so the four branches are regression-tested without the SDK —
in particular the `--strict-mcp-config`-without-`--mcp-config` case (a bug fixed
in code review that previously had no test).
"""

from kagura_agent.core.brain.sdk_engine import _mcp_option_kwargs


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
