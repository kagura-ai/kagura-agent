"""Codex (ChatGPT-subscription) actor adapter for the #188 bootstrap A/B runner.

The runner's `CommandObjectiveActor` spawns this module once per trial with one
JSON object on stdin — ``{"task": {…}, "bootstrap_context": "…", "seed": N}`` —
and scores whatever it writes to stdout (docs/bootstrap-eval.md). The gold check
never reaches this process.

Auth: the runner strips ``KAGURA_*`` credentials from this child's env; codex
auth is the file-based ``codex login`` (``~/.codex/auth.json``), inherited
through kagura-brain's codex backend, which itself strips ``OPENAI_*`` /
``CODEX_*`` from *its* child so the ChatGPT subscription always wins.

Determinism: ``seed`` is validated but not forwarded — ``codex exec`` exposes no
sampling-seed control; pairing validity rests on the runner's fixed corpus and
identical per-arm inputs, not actor-side seeding.

Exit codes: 0 answer produced · 2 bad stdin payload · 3 kagura-brain/codex
unavailable · 4 the codex invocation failed (non-zero, timed out, crashed, or an
empty answer — an always-empty adapter would silently zero both arms).
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kagura_agent.core.brain.base import BrainUnavailable

#: Optional model pin for the actor (`--model` on codex); stamped by the
#: operator in the manifest's ``versions.actor_model``.
_MODEL_ENV = "KAGURA_EVAL_ACTOR_MODEL"
#: Inner invoke timeout (seconds). Kept below the runner's actor timeout
#: (300s default) so a slow codex surfaces as this adapter's exit 4 + detail,
#: not an opaque runner-side kill.
_TIMEOUT_ENV = "KAGURA_EVAL_ACTOR_TIMEOUT"
_DEFAULT_TIMEOUT_SEC = 270


class PayloadError(ValueError):
    """The stdin payload does not match the runner's actor contract."""


@dataclass(frozen=True)
class ActorPayload:
    task_id: str
    prompt: str
    bootstrap_context: str
    seed: int


def parse_payload(raw: str) -> ActorPayload:
    """Parse and validate the runner's stdin payload, fail-closed on any drift."""
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PayloadError(f"stdin is not valid JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise PayloadError("stdin payload must be a JSON object")
    task = decoded.get("task")
    if not isinstance(task, dict):
        raise PayloadError("payload.task must be an object")
    task_id = task.get("id")
    if not isinstance(task_id, str) or not task_id:
        raise PayloadError("payload.task.id must be a non-empty string")
    prompt = task.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise PayloadError("payload.task.prompt must be a non-empty string")
    bootstrap_context = decoded.get("bootstrap_context")
    if not isinstance(bootstrap_context, str):
        raise PayloadError("payload.bootstrap_context must be a string")
    seed = decoded.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PayloadError("payload.seed must be an integer")
    return ActorPayload(
        task_id=task_id,
        prompt=prompt,
        bootstrap_context=bootstrap_context,
        seed=seed,
    )


def build_prompt(payload: ActorPayload) -> str:
    """Render the single grounded-answer prompt sent to codex.

    The load-bearing instruction is answering from ONLY the bootstrap context —
    the experiment measures whether ranking surfaced the right memories, so an
    actor that answers from world knowledge would blur the arms together.
    """
    context = payload.bootstrap_context.strip() or "(no memory context provided)"
    return (
        "You are an agent answering one task from your memory bootstrap.\n"
        "Answer using ONLY the facts in the memory context below. Quote exact\n"
        "figures, counts, names, and technical terms from the context verbatim.\n"
        "Reply with the answer text only — no preamble, no markdown headings.\n\n"
        f"Memory context:\n{context}\n\n"
        f"Task:\n{payload.prompt}"
    )


def run(stdin_text: str, *, invoke: Callable[[str], Any]) -> tuple[int, str, str]:
    """Pure adapter core: stdin text in, (exit code, stdout, stderr) out.

    ``invoke`` takes the built prompt and returns a kagura-brain ``BrainResult``
    duck (``returncode`` / ``timed_out`` / ``stdout`` / ``detail()``) — injected
    so the contract is unit-tested without the lib or a subscription.
    """
    try:
        payload = parse_payload(stdin_text)
    except PayloadError as exc:
        return 2, "", f"bad actor payload: {exc}"
    try:
        result = invoke(build_prompt(payload))
    except BrainUnavailable as exc:
        return 3, "", f"codex brain unavailable: {exc}"
    except Exception as exc:  # noqa: BLE001 - any invoke crash must fail the trial
        return 4, "", f"codex invocation crashed: {exc}"
    if result.returncode != 0 or result.timed_out:
        return 4, "", f"codex invocation failed: {result.detail()}"
    answer = str(result.stdout)
    if not answer.strip():
        return 4, "", "codex invocation returned an empty answer"
    return 0, answer, ""


def _codex_invoke(prompt: str) -> Any:  # pragma: no cover - needs kagura-brain + codex login
    """Live invoke: kagura-brain codex backend on the ChatGPT subscription."""
    try:
        import kagura_brain
    except ImportError as exc:
        raise BrainUnavailable(
            "the codex actor requires the optional 'brain' extra "
            "(pip install 'kagura-agent[brain]')"
        ) from exc
    handle = kagura_brain.select(backend="codex", endpoint=None, api_key=None)
    kwargs: dict[str, Any] = {}
    model = os.environ.get(_MODEL_ENV, "").strip()
    if model:
        kwargs["model"] = model
    timeout = int(os.environ.get(_TIMEOUT_ENV, "").strip() or _DEFAULT_TIMEOUT_SEC)
    return handle.invoke(prompt, timeout=timeout, **kwargs)


def main() -> int:  # pragma: no cover - thin stdin/stdout wiring over run()
    code, out, err = run(sys.stdin.read(), invoke=_codex_invoke)
    if out:
        sys.stdout.write(out)
    if err:
        sys.stderr.write(err + "\n")
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
