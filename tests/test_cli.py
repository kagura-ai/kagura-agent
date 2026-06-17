"""v0.1: CLI argument parsing for `kagura-agent run "task"`.

v0.2-A6 adds the orthogonal MCP knobs: `--mcp-config` (memory is CLI-primary;
this flag is for *other* MCP servers, mirroring Claude Code's own flag) and
`--strict-mcp-config`.
"""

import io
import json
from pathlib import Path

import pytest

from kagura_agent.cli.main import (
    configure_output_stream,
    load_mcp_config,
    main,
    make_memory_client,
    make_run_store,
    parse_args,
    plan_granted_specs,
    resolve_grants,
    resolve_state_dir,
)
from kagura_agent.core.brain.base import BrainUnavailable
from kagura_agent.membrane.registry import GrantSet, parse_grants, parse_registry
from kagura_agent.patterns.checkpoint import FileCheckpointStore, InMemoryCheckpointStore

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


# --- #65: plan_granted_specs builds ONLY granted providers, fail-closed -------


def _registry():
    return parse_registry(
        {
            "aws": {"kind": "aws_sts", "role_arn": "arn:aws:iam::1:role/a"},
            "mem": {"kind": "memory_cloud", "parent_token_env": "MEM"},
        }
    )


def test_plan_granted_specs_selects_only_granted_providers() -> None:
    # An UNGRANTED provider (mem, which needs deployment wiring) must NOT be in
    # the build set — only the granted one (aws) is constructed, so an
    # ungranted/incomplete provider can never abort a run that didn't ask for it.
    specs = plan_granted_specs(_registry(), parse_grants(["aws:arn:aws:iam::1:role/a"]))
    assert [s.name for s in specs] == ["aws"]


def test_plan_granted_specs_empty_grants_selects_nothing() -> None:
    assert plan_granted_specs(_registry(), GrantSet(frozenset())) == []


def test_plan_granted_specs_unknown_granted_provider_is_fail_closed() -> None:
    # A --grant naming a provider absent from the registry fails closed with a
    # clean message here, not a later KeyError deep in the broker.
    with pytest.raises(ValueError, match="not in the registry"):
        plan_granted_specs(_registry(), parse_grants(["typo:scope"]))


def test_plan_granted_specs_deterministic_order() -> None:
    specs = plan_granted_specs(
        _registry(), parse_grants(["mem:memory:read", "aws:arn:aws:iam::1:role/a"])
    )
    assert [s.name for s in specs] == ["aws", "mem"]  # sorted by provider name


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


# --- context continuity: --session, repl, store selection ----------------


def test_parse_run_session_defaults_none() -> None:
    assert parse_args(["run", "t"]).session is None


def test_parse_run_accepts_session() -> None:
    assert parse_args(["run", "t", "--session", "work"]).session == "work"


def test_parse_repl_command_defaults() -> None:
    ns = parse_args(["repl"])
    assert ns.command == "repl"
    assert ns.session == "repl"  # default session id
    assert ns.mcp_config is None and ns.strict_mcp_config is False


def test_parse_repl_accepts_session_and_mcp() -> None:
    ns = parse_args(["repl", "--session", "work", "--mcp-config", "/m.json", "--strict-mcp-config"])
    assert ns.session == "work"
    assert ns.mcp_config == "/m.json" and ns.strict_mcp_config is True


def test_make_run_store_oneshot_is_ephemeral() -> None:
    # No --session → a throwaway in-memory store under the fixed one-shot id, so a
    # plain `run` keeps no cross-run context (unchanged legacy behaviour).
    store, sid = make_run_store(None)
    assert isinstance(store, InMemoryCheckpointStore)
    assert sid == "cli"


def test_make_run_store_named_is_persistent() -> None:
    store, sid = make_run_store("work")
    assert isinstance(store, FileCheckpointStore)
    assert sid == "work"


def test_resolve_state_dir_default() -> None:
    assert resolve_state_dir({}) == Path(".kagura-agent") / "checkpoints"


def test_resolve_state_dir_env_override() -> None:
    out = resolve_state_dir({"KAGURA_AGENT_STATE_DIR": "/var/lib/ka"})
    assert out == Path("/var/lib/ka") / "checkpoints"


def test_resolve_state_dir_blank_env_is_default() -> None:
    # A set-but-blank override must not resolve to the filesystem root.
    out = resolve_state_dir({"KAGURA_AGENT_STATE_DIR": "   "})
    assert out == Path(".kagura-agent") / "checkpoints"


def test_make_memory_client_is_none_seam() -> None:
    # The grounding seam (B): None until a trust-aware adapter is wired, so
    # ground_and_run degrades to plain checkpoint resume.
    assert make_memory_client() is None


def test_main_run_rejects_invalid_brain_backend(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    # An unknown KAGURA_AGENT_BRAIN must fail closed up front (exit 2), never run
    # the default backend silently and never reach brain construction.
    monkeypatch.setenv("KAGURA_AGENT_BRAIN", "kagura_brain")  # typo (underscore)
    rc = main(["run", "do a thing"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "KAGURA_AGENT_BRAIN" in err
    assert "Traceback" not in err


def test_main_repl_rejects_invalid_brain_backend(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("KAGURA_AGENT_BRAIN", "bogus")
    rc = main(["repl"])
    assert rc == 2
    assert "KAGURA_AGENT_BRAIN" in capsys.readouterr().err


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


def test_main_run_clean_error_on_corrupt_checkpoint(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    # A corrupt persisted checkpoint raises CheckpointError; the run handler must
    # surface a clean exit-2 message, never a raw traceback (matches the rest of
    # the CLI and replaces the old cockpit.serve() isolation).
    from kagura_agent.cli import main as cli_main
    from kagura_agent.patterns.checkpoint import CheckpointError

    async def _boom(*_a, **_k) -> str:
        raise CheckpointError("checkpoint for session 'work' at ... is corrupt: ...")

    monkeypatch.setattr(cli_main, "_run_task", _boom)
    rc = main(["run", "do a thing", "--session", "work"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "run failed" in err and "corrupt" in err
    assert "Traceback" not in err


def test_main_run_clean_error_on_session_error(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    # A brain that ends without a terminal result raises SessionError; surface it
    # cleanly (exit 2), not as a raw traceback.
    from kagura_agent.cli import main as cli_main
    from kagura_agent.core.session import SessionError

    async def _boom(*_a, **_k) -> str:
        raise SessionError("brain ended without DoneEvent for session 'cli'")

    monkeypatch.setattr(cli_main, "_run_task", _boom)
    rc = main(["run", "do a thing"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "run failed" in err
    assert "Traceback" not in err


def test_main_run_clean_error_on_credential_provisioning_failure(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    # A credential-provisioning failure (bad/missing registry, or a --grant naming
    # a provider absent from it) raises CredentialSetupError and must surface as a
    # clean exit-2 message — never a raw traceback (matches doctor's posture).
    from kagura_agent.cli import main as cli_main

    async def _boom(*_a, **_k) -> str:
        raise cli_main.CredentialSetupError(
            "--grant names provider(s) not in the registry: typo"
        )

    monkeypatch.setattr(cli_main, "_run_task", _boom)
    rc = main(["run", "do a thing", "--grant", "typo:scope"])

    assert rc == 2  # operator-input error, same code as malformed --grant / --mcp-config
    err = capsys.readouterr().err
    assert "registry" in err
    assert "Traceback" not in err


def test_main_run_does_not_mislabel_an_unrelated_value_error(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # A ValueError raised LATER by the agent run (cockpit.serve / the brain) must
    # NOT be caught and mislabeled as a --grant/--registry error — it propagates
    # as itself. Guards against the over-broad `except ValueError` (#code-review).
    from kagura_agent.cli import main as cli_main

    async def _boom(*_a, **_k) -> str:
        raise ValueError("a deep run-time error unrelated to credentials")

    monkeypatch.setattr(cli_main, "_run_task", _boom)
    with pytest.raises(ValueError, match="deep run-time error"):
        main(["run", "do a thing"])


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


def test_configure_output_stream_stringio_passthrough() -> None:
    # #82 item 8: a pure-text stream (StringIO — no byte .buffer underneath) can
    # never raise UnicodeEncodeError, so it is returned unchanged (glyphs kept).
    s = io.StringIO()
    assert configure_output_stream(s) is s


class _FakeByteStreamNoEncoding:
    """A byte-backed stream (has .buffer) that reports no encoding and has no
    reconfigure — models a stream we cannot verify, so the fallback must wrap it
    (#82 item 8) and write() must tolerate the unknown encoding without crashing."""

    encoding = None

    def __init__(self) -> None:
        self.buffer = io.BytesIO()
        self.written: list[str] = []

    def write(self, text: str) -> int:
        self.written.append(text)
        return len(text)


def test_configure_output_stream_wraps_byte_stream_with_no_encoding() -> None:
    # #82 item 8: a byte-backed stream that didn't report an encoding could be a
    # legacy code page underneath — fail closed and wrap it (not pass it through),
    # and the wrapper must not crash when the encoding is unknown.
    fake = _FakeByteStreamNoEncoding()
    result = configure_output_stream(fake)
    assert result is not fake  # wrapped, not passed through
    result.write("dash — arrow ↳")  # must not raise despite unknown encoding
    assert "".join(fake.written) == "dash - arrow ->"  # decorative glyphs transliterated


class _ReconfigureFailsStream:
    """A cp932 stream whose reconfigure() raises (e.g. detached/mid-write) and
    which encodes strictly — models the #82 item 7 case where the errors='replace'
    reconfigure is swallowed, so the wrapper itself must stay crash-proof."""

    encoding = "cp932"

    def __init__(self) -> None:
        self.buffer = io.BytesIO()

    def reconfigure(self, **_kw: object) -> None:
        raise OSError("cannot reconfigure a detached stream")

    def write(self, text: str) -> int:
        self.buffer.write(text.encode("cp932"))  # strict: raises on a non-cp932 char
        return len(text)


def test_configure_output_stream_stays_crash_proof_when_reconfigure_fails() -> None:
    # #82 item 7: when reconfigure(errors="replace") fails (swallowed), the stream
    # is still strict — so the wrapper's own encode-safe net must degrade any char
    # the code page can't encode (beyond the 9 glyphs) to "?" rather than crash.
    stream = _ReconfigureFailsStream()
    result = configure_output_stream(stream)
    assert result is not stream  # wrapped despite the reconfigure failure
    result.write("rocket 🚀 and dash —")  # must NOT raise
    out = stream.buffer.getvalue().decode("cp932")
    assert "—" not in out and "🚀" not in out  # em-dash transliterated, emoji replaced
    assert "-" in out  # the em-dash degraded to ASCII


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
