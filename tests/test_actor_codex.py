"""Codex actor adapter pure seams — payload parsing, prompt build, run() exits.

The live `main()` / `_codex_invoke` need the real kagura-brain lib + a ChatGPT
subscription (pragma no cover, like KaguraBrainEngine.query); the pure contract
below is unit-tested by injecting a duck-typed invoke.
"""

from __future__ import annotations

import pytest

from kagura_agent.core.brain.base import BrainUnavailable
from kagura_agent.eval.actor_codex import (
    ActorPayload,
    PayloadError,
    build_prompt,
    parse_payload,
    run,
)


class _FakeBrainResult:
    """Duck-typed kagura-brain BrainResult for exercising run() without the lib."""

    def __init__(  # type: ignore[no-untyped-def]
        self, *, returncode=0, stdout="", timed_out=False, detail="detail-text"
    ):
        self.returncode = returncode
        self.stdout = stdout
        self.timed_out = timed_out
        self._detail = detail

    def detail(self) -> str:
        return self._detail


_HAPPY = (
    '{"task": {"id": "code-01", "category": "code", '
    '"query": "Orion failure retry behavior", '
    '"prompt": "State the current retry count."}, '
    '"bootstrap_context": "Recall:\\n- [mem-code-01] 4 attempts", "seed": 7}'
)


# --- parse_payload ---


def test_parse_payload_happy() -> None:
    payload = parse_payload(_HAPPY)
    assert payload == ActorPayload(
        task_id="code-01",
        prompt="State the current retry count.",
        bootstrap_context="Recall:\n- [mem-code-01] 4 attempts",
        seed=7,
    )


def test_parse_payload_rejects_invalid_json() -> None:
    with pytest.raises(PayloadError, match="JSON"):
        parse_payload("{not json")


def test_parse_payload_rejects_non_object() -> None:
    with pytest.raises(PayloadError, match="object"):
        parse_payload('["task"]')


def test_parse_payload_rejects_missing_task() -> None:
    with pytest.raises(PayloadError, match="task"):
        parse_payload('{"bootstrap_context": "", "seed": 1}')


def test_parse_payload_rejects_blank_task_prompt() -> None:
    with pytest.raises(PayloadError, match="prompt"):
        parse_payload('{"task": {"id": "t", "prompt": "  "}, "bootstrap_context": "", "seed": 1}')


def test_parse_payload_rejects_non_string_context() -> None:
    with pytest.raises(PayloadError, match="bootstrap_context"):
        parse_payload('{"task": {"id": "t", "prompt": "p"}, "bootstrap_context": 3, "seed": 1}')


def test_parse_payload_rejects_non_int_seed() -> None:
    with pytest.raises(PayloadError, match="seed"):
        parse_payload('{"task": {"id": "t", "prompt": "p"}, "bootstrap_context": "", "seed": "1"}')


def test_parse_payload_rejects_bool_seed() -> None:
    # bool is an int subclass; a stray true must not pass as a seed.
    with pytest.raises(PayloadError, match="seed"):
        parse_payload('{"task": {"id": "t", "prompt": "p"}, "bootstrap_context": "", "seed": true}')


# --- build_prompt ---


def test_build_prompt_embeds_context_and_task() -> None:
    payload = parse_payload(_HAPPY)
    prompt = build_prompt(payload)
    assert "Recall:\n- [mem-code-01] 4 attempts" in prompt
    assert "State the current retry count." in prompt
    # The grounding instruction is the load-bearing line of the adapter.
    assert "ONLY" in prompt


def test_build_prompt_marks_empty_context() -> None:
    payload = ActorPayload(task_id="t", prompt="p", bootstrap_context="  ", seed=1)
    assert "(no memory context provided)" in build_prompt(payload)


# --- run ---


def test_run_happy_relays_answer() -> None:
    code, out, err = run(_HAPPY, invoke=lambda prompt: _FakeBrainResult(stdout="4 attempts"))
    assert (code, out, err) == (0, "4 attempts", "")


def test_run_bad_payload_exits_2() -> None:
    code, out, err = run("{not json", invoke=lambda prompt: _FakeBrainResult(stdout="x"))
    assert code == 2
    assert out == ""
    assert "payload" in err


def test_run_brain_unavailable_exits_3() -> None:
    def unavailable(prompt: str) -> _FakeBrainResult:
        raise BrainUnavailable("install the brain extra")

    code, out, err = run(_HAPPY, invoke=unavailable)
    assert code == 3
    assert out == ""
    assert "install the brain extra" in err


def test_run_nonzero_invoke_exits_4_with_detail() -> None:
    code, out, err = run(
        _HAPPY,
        invoke=lambda prompt: _FakeBrainResult(returncode=1, stdout="partial", detail="exit 1"),
    )
    assert code == 4
    assert out == ""
    assert "exit 1" in err


def test_run_timed_out_invoke_exits_4() -> None:
    code, out, err = run(
        _HAPPY,
        invoke=lambda prompt: _FakeBrainResult(timed_out=True, detail="timed out after 270s"),
    )
    assert code == 4
    assert "timed out" in err


def test_run_invoke_crash_exits_4() -> None:
    def crash(prompt: str) -> _FakeBrainResult:
        raise RuntimeError("codex binary vanished")

    code, out, err = run(_HAPPY, invoke=crash)
    assert code == 4
    assert "codex binary vanished" in err


def test_run_blank_answer_fails_closed() -> None:
    # An always-empty adapter would silently zero both arms; refuse to score it.
    code, out, err = run(_HAPPY, invoke=lambda prompt: _FakeBrainResult(stdout="   "))
    assert code == 4
    assert "empty" in err


def test_run_passes_built_prompt_to_invoke() -> None:
    seen: list[str] = []

    def capture(prompt: str) -> _FakeBrainResult:
        seen.append(prompt)
        return _FakeBrainResult(stdout="ok")

    run(_HAPPY, invoke=capture)
    assert len(seen) == 1
    assert "State the current retry count." in seen[0]
