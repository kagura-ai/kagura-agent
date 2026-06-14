"""v0.1: CLI argument parsing for `kagura-agent run "task"`.

v0.2-A6 adds the orthogonal MCP knobs: `--mcp-config` (memory is CLI-primary;
this flag is for *other* MCP servers, mirroring Claude Code's own flag) and
`--strict-mcp-config`.
"""

import json

import pytest

from kagura_agent.cli.main import load_mcp_config, main, parse_args
from kagura_agent.core.brain.base import BrainUnavailable


def test_parse_run_with_task() -> None:
    ns = parse_args(["run", "build me a thing"])
    assert ns.command == "run"
    assert ns.task == "build me a thing"


def test_parse_run_requires_task() -> None:
    with pytest.raises(SystemExit):
        parse_args(["run"])


def test_parse_no_command_exits() -> None:
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_rejects_empty_task() -> None:
    # A blank prompt would spin a billed empty-prompt brain run.
    with pytest.raises(SystemExit):
        parse_args(["run", ""])


def test_parse_rejects_whitespace_task() -> None:
    with pytest.raises(SystemExit):
        parse_args(["run", "   "])


# --- v0.2-A6: --mcp-config / --strict-mcp-config --------------------------

def test_parse_run_defaults_have_no_mcp_config() -> None:
    ns = parse_args(["run", "t"])
    assert ns.mcp_config is None
    assert ns.strict_mcp_config is False


def test_parse_run_accepts_mcp_config_path() -> None:
    ns = parse_args(["run", "t", "--mcp-config", "/etc/mcp.json"])
    assert ns.mcp_config == "/etc/mcp.json"
    assert ns.strict_mcp_config is False


def test_parse_run_accepts_strict_mcp_config() -> None:
    ns = parse_args(["run", "t", "--mcp-config", "/etc/mcp.json", "--strict-mcp-config"])
    assert ns.strict_mcp_config is True


def test_load_mcp_config_none_returns_none() -> None:
    assert load_mcp_config(None) is None


def test_load_mcp_config_extracts_mcp_servers_wrapper(tmp_path) -> None:
    # Claude Code convention: {"mcpServers": {...}}. The SDK wants the inner map.
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps({"mcpServers": {"fs": {"command": "srv"}}}))
    assert load_mcp_config(str(p)) == {"fs": {"command": "srv"}}


def test_load_mcp_config_accepts_bare_server_map(tmp_path) -> None:
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps({"fs": {"command": "srv"}}))
    assert load_mcp_config(str(p)) == {"fs": {"command": "srv"}}


def test_load_mcp_config_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_mcp_config(str(tmp_path / "nope.json"))


def test_load_mcp_config_rejects_non_object_json(tmp_path) -> None:
    # An operator typo like `{"mcpServers": null}` (or a bare array) must fail
    # loud, not crash later with a cryptic TypeError on dict().
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps({"mcpServers": None}))
    with pytest.raises(ValueError, match="expected a JSON object"):
        load_mcp_config(str(p))


# --- #28: missing Claude brain surfaces an actionable error, not "internal error" ---

def test_main_run_surfaces_brain_unavailable(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    # When the optional `claude` extra is absent, brain construction raises
    # BrainUnavailable; `main` must surface it as an actionable message + non-zero
    # exit, NOT let it fall through to a raw traceback / generic "internal error".
    from kagura_agent.cli import main as cli_main

    async def _boom(*_a, **_k) -> str:
        raise BrainUnavailable(
            "The Claude brain requires the optional 'claude' extra "
            "(claude-agent-sdk). Install it with: uv run --extra claude ..."
        )

    monkeypatch.setattr(cli_main, "_run_task", _boom)
    rc = main(["run", "do a thing"])

    assert rc == 3  # distinct from argparse's usage-error code (2)
    err = capsys.readouterr().err
    assert "claude" in err.lower()
    assert "--extra claude" in err or "kagura-agent[claude]" in err
    assert "internal error" not in err.lower()  # the failure mode this issue fixes


# --- --mcp-config load failures surface cleanly, not as a raw traceback ---

def test_main_run_clean_error_on_missing_mcp_config(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    rc = main(["run", "do a thing", "--mcp-config", str(tmp_path / "nope.json")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--mcp-config" in err
    assert "Traceback" not in err


def test_main_run_clean_error_on_invalid_mcp_json(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    bad = tmp_path / "bad.json"
    bad.write_text("not json{")
    rc = main(["run", "do a thing", "--mcp-config", str(bad)])
    assert rc == 2
    assert "--mcp-config" in capsys.readouterr().err
