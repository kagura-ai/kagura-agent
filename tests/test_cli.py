"""v0.1: CLI argument parsing for `kagura-agent run "task"`."""

import pytest

from kagura_agent.cli.main import parse_args


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
