"""v0.1: CLI argument parsing for `kagura-agent run "task"`.

v0.2-A6 adds the orthogonal MCP knobs: `--mcp-config` (memory is CLI-primary;
this flag is for *other* MCP servers, mirroring Claude Code's own flag) and
`--strict-mcp-config`.
"""

import io
import json

import pytest

from kagura_agent.cli.main import (
    configure_output_stream,
    load_mcp_config,
    main,
    parse_args,
    resolve_grants,
)
from kagura_agent.core.brain.base import BrainUnavailable
from kagura_agent.membrane.registry import GrantSet

# --- #65: --grant is now ENFORCED; resolve_grants returns a GrantSet ----------


def test_resolve_grants_parses_to_grantset() -> None:
    grants = resolve_grants(["aws:s3:read", "cf:zone:purge"])
    assert isinstance(grants, GrantSet)
    assert grants.allows("aws", "s3:read")
    assert grants.allows("cf", "zone:purge")
    assert not grants.allows("aws", "s3:write")  # exact-match, default-deny


def test_resolve_grants_none_is_deny_all() -> None:
    # No --grant → empty GrantSet: default-deny, nothing reachable (no broker).
    grants = resolve_grants(None)
    assert isinstance(grants, GrantSet)
    assert grants.grants == frozenset()


def test_resolve_grants_malformed_is_fail_closed() -> None:
    with pytest.raises(ValueError):
        resolve_grants(["no-colon-here"])


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


# --- console output: CLI text uses em-dash/arrow glyphs a cp932 console can't encode ---


def test_configure_output_stream_keeps_utf8_stream_and_glyphs() -> None:
    # A UTF-8-capable stream renders the decorative glyphs natively, so it is
    # returned unchanged — no needless wrapping, glyphs preserved.
    stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    result = configure_output_stream(stream)

    assert result is stream
    result.write("em-dash — and arrow ↳")
    result.flush()
    assert "—" in stream.buffer.getvalue().decode("utf-8")  # not transliterated


def test_configure_output_stream_transliterates_glyphs_on_cp932() -> None:
    # A cp932 console can't encode — / ↳; instead of crashing, the wrapper maps
    # them to readable ASCII (-, ->) so output is never mojibake.
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp932")
    result = configure_output_stream(stream)

    assert result is not stream  # wrapped
    result.write("em — dash, arrow ↳, ellipsis …")
    result.flush()
    out = raw.getvalue().decode("cp932")
    assert "—" not in out and "↳" not in out and "…" not in out
    assert "-" in out and "->" in out and "..." in out


def test_configure_output_stream_cp932_does_not_crash_on_other_nonencodable() -> None:
    # A char that is neither ASCII nor in our table nor in cp932 (e.g. an emoji)
    # must degrade to a replacement char, never raise UnicodeEncodeError.
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp932")
    result = configure_output_stream(stream)

    result.write("rocket 🚀 done")  # must NOT raise
    result.flush()
    assert raw.getvalue()  # something was written


def test_configure_output_stream_writelines_also_transliterates() -> None:
    # writelines() must not bypass translation (else it would hit the raw cp932
    # stream via attribute delegation and crash on a decorative glyph).
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp932")
    result = configure_output_stream(stream)

    result.writelines(["arrow ↳ one\n", "dash — two\n"])  # must NOT raise
    result.flush()
    out = raw.getvalue().decode("cp932")
    assert "↳" not in out and "—" not in out
    assert "->" in out and "-" in out


def test_configure_output_stream_is_idempotent() -> None:
    # main() runs once, but re-applying must not nest wrappers (cp932 encoding
    # delegates through, so a naive re-wrap would stack translators).
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp932")
    once = configure_output_stream(stream)
    twice = configure_output_stream(once)
    assert twice is once  # already a fallback stream — returned as-is


def test_configure_output_stream_none_is_passthrough() -> None:
    # A headless process (pythonw) can have sys.stdout is None; don't crash.
    assert configure_output_stream(None) is None


def test_configure_output_stream_wrapper_delegates_attributes() -> None:
    # The wrapper must be a drop-in for sys.stdout: undefined attributes
    # delegate to the wrapped stream (encoding, flush, isatty, ...).
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp932")
    result = configure_output_stream(stream)

    assert result.encoding.lower() in ("cp932", "shift_jis", "ms932")
    assert callable(result.flush)
    assert result.writable() is True


def test_main_help_renders_ascii_on_cp932_console(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Regression: `kagura-agent run --help` printed the em-dash in the --grant
    # help text straight to a cp932 stdout and died with UnicodeEncodeError.
    # main() must install the fallback stream before argparse prints anything.
    import sys

    out = io.TextIOWrapper(io.BytesIO(), encoding="cp932")
    err = io.TextIOWrapper(io.BytesIO(), encoding="cp932")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    with pytest.raises(SystemExit) as exc:
        main(["run", "--help"])

    assert exc.value.code == 0  # --help is a clean exit, not a crash
    out.flush()
    rendered = out.buffer.getvalue().decode("cp932")
    assert "--grant" in rendered  # the help body actually made it out
    assert "—" not in rendered  # em-dash transliterated to ASCII, not mojibake
