"""#36: `kagura-agent doctor` — preflight environment check.

The check *logic* is pure and injectable (each `check_*` takes the probed value,
never probes itself), mirroring `memory_reachable()`/`ensure_memory_reachable()`
and `claude_sdk_available(find_spec=...)`. The live probes (docker/egress) are
thin and exercised at deployment, not here.
"""

from __future__ import annotations

from kagura_agent.cli.doctor import (
    FAIL,
    OK,
    WARN,
    check_brain,
    check_docker,
    check_egress,
    check_memory,
    format_report,
    overall_status,
    run_doctor,
)
from kagura_agent.cli.main import main, parse_args

# --- memory -----------------------------------------------------------------

def test_check_memory_ok() -> None:
    r = check_memory(reachable=True)
    assert r.status == OK
    assert "memory" in r.name


def test_check_memory_fail_is_actionable() -> None:
    r = check_memory(reachable=False)
    assert r.status == FAIL
    assert r.hint is not None
    assert "kagura auth login" in r.hint


# --- brain (sdk + auth) -----------------------------------------------------

def test_check_brain_fail_when_sdk_missing() -> None:
    r = check_brain(sdk_available=False, env={})
    assert r.status == FAIL
    assert r.hint is not None
    assert "--extra claude" in r.hint


def test_check_brain_ok_with_subscription() -> None:
    r = check_brain(sdk_available=True, env={"CLAUDE_CODE_SUBSCRIPTION": "1"})
    assert r.status == OK
    assert "subscription" in r.detail.lower()


def test_check_brain_warns_when_api_key_overrides_subscription() -> None:
    # ANTHROPIC_API_KEY overrides subscription auth (README L127) — surface it.
    r = check_brain(
        sdk_available=True,
        env={"ANTHROPIC_API_KEY": "sk-x", "CLAUDE_CODE_SUBSCRIPTION": "1"},
    )
    assert r.status == WARN
    assert "ANTHROPIC_API_KEY" in r.detail


def test_check_brain_warns_when_api_key_only() -> None:
    r = check_brain(sdk_available=True, env={"ANTHROPIC_API_KEY": "sk-x"})
    assert r.status == WARN
    assert "ANTHROPIC_API_KEY" in r.detail


def test_check_brain_warns_when_auth_unverifiable() -> None:
    # SDK present but neither env signal: a CLI subscription cache may still exist,
    # so this is "can't verify" (WARN), not a hard FAIL.
    r = check_brain(sdk_available=True, env={})
    assert r.status == WARN
    assert r.hint is not None
    assert "claude" in r.hint.lower()


def test_check_brain_kagura_backend_fail_when_extra_missing() -> None:
    # backend=kagura-brain: the dependency that matters is the 'brain' extra, NOT
    # the SDK (even with the SDK present). Doctor predicts the selected run.
    r = check_brain(
        backend="kagura-brain", sdk_available=True, kagura_brain_available=False, env={}
    )
    assert r.status == FAIL
    assert r.hint is not None and "--extra brain" in r.hint


def test_check_brain_kagura_backend_fail_when_cli_missing() -> None:
    # Extra installed but the underlying claude/codex CLI is not on PATH → FAIL
    # (the run-blocking dep), not a lenient WARN.
    r = check_brain(
        backend="kagura-brain",
        sdk_available=True,
        kagura_brain_available=True,
        kagura_backend="claude",
        kagura_cli_present=False,
        env={},
    )
    assert r.status == FAIL
    assert "claude" in r.detail and "PATH" in r.detail


def test_check_brain_kagura_backend_warns_when_present_and_cli_on_path() -> None:
    r = check_brain(
        backend="kagura-brain",
        sdk_available=False,
        kagura_brain_available=True,
        kagura_backend="claude",
        kagura_cli_present=True,
        env={},
    )
    assert r.status == WARN  # present; auth via underlying CLI not verifiable here
    assert "kagura-brain" in r.detail


# --- docker -----------------------------------------------------------------

def test_check_docker_ok() -> None:
    assert check_docker(available=True).status == OK


def test_check_docker_fail_is_actionable() -> None:
    r = check_docker(available=False)
    assert r.status == FAIL
    assert r.hint is not None


# --- egress -----------------------------------------------------------------

def test_check_egress_ok() -> None:
    assert check_egress(configured=True).status == OK


def test_check_egress_warns_when_unconfigured() -> None:
    # Egress proxy is a deploy-time concern; its absence is a warning, not a hard
    # fail of the local toolchain.
    r = check_egress(configured=False)
    assert r.status == WARN


# --- overall ----------------------------------------------------------------

def test_overall_fail_when_any_fail() -> None:
    results = [
        check_memory(reachable=True),
        check_docker(available=False),  # FAIL
    ]
    assert overall_status(results) == FAIL


def test_overall_warn_when_any_warn_but_no_fail() -> None:
    results = [
        check_memory(reachable=True),
        check_egress(configured=False),  # WARN
    ]
    assert overall_status(results) == WARN


def test_overall_ok_when_all_ok() -> None:
    results = [check_memory(reachable=True), check_docker(available=True)]
    assert overall_status(results) == OK


# --- run_doctor wiring (probes injected) ------------------------------------

def test_run_doctor_threads_probes_into_checks() -> None:
    results = run_doctor(
        memory_probe=lambda: True,
        sdk_probe=lambda: True,
        docker_probe=lambda: False,
        egress_probe=lambda: True,
        env={"CLAUDE_CODE_SUBSCRIPTION": "1"},
    )
    by_name = {r.name: r.status for r in results}
    assert by_name["memory"] == OK
    assert by_name["docker"] == FAIL  # probe returned False
    assert overall_status(results) == FAIL


def test_run_doctor_kagura_backend_checks_brain_extra() -> None:
    # With KAGURA_AGENT_BRAIN=kagura-brain, the brain check follows the brain extra
    # (absent here → FAIL) regardless of SDK presence.
    results = run_doctor(
        memory_probe=lambda: True,
        sdk_probe=lambda: True,  # SDK present, but irrelevant for this backend
        kagura_brain_probe=lambda: False,  # brain extra absent
        kagura_cli_probe=lambda name: True,
        docker_probe=lambda: True,
        egress_probe=lambda: True,
        env={"KAGURA_AGENT_BRAIN": "kagura-brain"},
    )
    by_name = {r.name: r.status for r in results}
    assert by_name["brain"] == FAIL


def test_run_doctor_kagura_backend_fails_when_cli_absent() -> None:
    # Extra present but the claude CLI it shells out to is not on PATH → FAIL.
    results = run_doctor(
        memory_probe=lambda: True,
        sdk_probe=lambda: True,
        kagura_brain_probe=lambda: True,  # extra present
        kagura_cli_probe=lambda name: False,  # but `claude` not on PATH
        docker_probe=lambda: True,
        egress_probe=lambda: True,
        env={"KAGURA_AGENT_BRAIN": "kagura-brain"},
    )
    brain = next(r for r in results if r.name == "brain")
    assert brain.status == FAIL
    assert "PATH" in brain.detail


def test_run_doctor_invalid_backend_is_brain_fail_not_crash() -> None:
    results = run_doctor(
        memory_probe=lambda: True,
        sdk_probe=lambda: True,
        docker_probe=lambda: True,
        egress_probe=lambda: True,
        env={"KAGURA_AGENT_BRAIN": "bogus"},
    )
    brain = next(r for r in results if r.name == "brain")
    assert brain.status == FAIL
    assert "KAGURA_AGENT_BRAIN" in brain.detail


def test_run_doctor_invalid_kagura_backend_is_brain_fail() -> None:
    results = run_doctor(
        memory_probe=lambda: True,
        sdk_probe=lambda: True,
        kagura_brain_probe=lambda: True,
        kagura_cli_probe=lambda name: True,
        docker_probe=lambda: True,
        egress_probe=lambda: True,
        env={"KAGURA_AGENT_BRAIN": "kagura-brain", "KAGURA_AGENT_BRAIN_BACKEND": "codx"},
    )
    brain = next(r for r in results if r.name == "brain")
    assert brain.status == FAIL
    assert "KAGURA_AGENT_BRAIN_BACKEND" in brain.detail


def test_format_report_contains_each_check_and_overall() -> None:
    results = run_doctor(
        memory_probe=lambda: True,
        sdk_probe=lambda: True,
        docker_probe=lambda: True,
        egress_probe=lambda: True,
        env={"CLAUDE_CODE_SUBSCRIPTION": "1"},
    )
    text = format_report(results)
    assert "memory" in text
    assert "brain" in text
    assert "overall" in text.lower()


# --- CLI wiring -------------------------------------------------------------

def test_parse_doctor_command() -> None:
    ns = parse_args(["doctor"])
    assert ns.command == "doctor"
    assert ns.registry == "kagura-agent.toml"  # default registry path
    assert ns.probe is False  # dry-mint is opt-in


def test_parse_doctor_probe_flag() -> None:
    ns = parse_args(["doctor", "--probe", "--registry", "custom.toml"])
    assert ns.probe is True
    assert ns.registry == "custom.toml"


def test_main_doctor_returns_4_on_fail(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    from kagura_agent.cli import main as cli_main

    def _failing(**_kwargs) -> list:  # type: ignore[type-arg, no-untyped-def]
        return [check_docker(available=False)]

    monkeypatch.setattr(cli_main, "run_doctor", _failing)
    rc = main(["doctor"])
    assert rc == 4  # distinct from run's exit 3 and argparse's exit 2
    out = capsys.readouterr().out
    assert "docker" in out


def test_main_doctor_returns_0_when_only_warn(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from kagura_agent.cli import main as cli_main

    def _warn_only(**_kwargs) -> list:  # type: ignore[type-arg, no-untyped-def]
        return [check_memory(reachable=True), check_egress(configured=False)]

    monkeypatch.setattr(cli_main, "run_doctor", _warn_only)
    rc = main(["doctor"])
    assert rc == 0  # warnings do not fail the gate
