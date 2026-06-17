"""Cross-run context continuity — the pieces the CLI wires together so tasks
remember one another.

Three layers, all pure and brain-agnostic (testable with a fake brain + the
in-memory stores):

- **drive_task** (A) — resume an existing session's checkpoint, else launch
  fresh. This is what turns ``kagura-agent run --session work`` from a one-shot
  into a continuation.
- **ground_prompt / remember_outcome / ground_and_run** (B) — recall relevant
  prior memories into the prompt before running, and persist a task summary after.
  Brain-native resume (A) carries the *in-session* thread; the memory backbone
  here carries *durable knowledge* across sessions, brain restarts, even brain
  swaps. This is the memory-as-backbone thesis applied to the run path.
- **run_repl** (C) — a long-lived loop so consecutive lines continue one session
  within a single process; combined with a persistent store it also survives
  restarts.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

from kagura_agent.core.brain.base import BrainProvider, Task
from kagura_agent.core.session import Session, SessionResult
from kagura_agent.mcp.memory_cloud import QUARANTINE_TIER, MemoryClient
from kagura_agent.patterns.checkpoint import CheckpointStore

log = logging.getLogger(__name__)

#: How many recalled memories to inject as prior context (keeps the preamble
#: bounded so grounding never crowds out the actual task).
_MAX_GROUNDING = 5

#: Lines that end a REPL session.
_REPL_QUIT = frozenset({"/exit", "/quit"})


async def drive_task(
    brain: BrainProvider,
    store: CheckpointStore,
    *,
    session_id: str,
    prompt: str,
) -> SessionResult:
    """Resume ``session_id`` if it has a saved checkpoint, else launch it fresh.

    The single decision that gives the CLI cross-run continuity: a prior
    checkpoint (from an earlier process, via a persistent store) means *continue*;
    no checkpoint means *new task*. Mirrors the cockpit's LAUNCH/CONTINUE split
    without needing a session registry the one-shot CLI process doesn't carry over.

    The checkpoint is loaded **once** and handed straight to ``Session.drive``
    (resume when present, launch when None) — never re-loaded by ``Session.resume``.
    That halves the store reads per resume turn AND closes the load-decide /
    re-load race (a checkpoint vanishing between the two reads no longer crashes
    the run with a spurious "no checkpoint to resume").
    """
    prior = await store.load(session_id)
    session = Session(brain, store)
    return await session.drive(Task(prompt=prompt, session_id=session_id), resume=prior)


async def ground_prompt(memory: MemoryClient, prompt: str) -> str:
    """Prepend relevant trusted prior memories to ``prompt`` (else return as-is).

    Only the *trusted* backbone is recalled (``trusted_only=True``): an
    externally-ingested / quarantined memory must never be silently promoted into
    a behaviour-influencing preamble (OWASP LLM01/LLM03 — the membrane's memory-
    provenance rule applied at the run path). With nothing relevant, the prompt is
    returned unchanged so a first-ever task is not padded with an empty block.
    """
    memories = await memory.recall(prompt, trusted_only=True)
    if not memories:
        return prompt
    lines = [f"- {m.text}" for m in memories[:_MAX_GROUNDING]]
    preamble = "Relevant context from prior work:\n" + "\n".join(lines)
    return f"{preamble}\n\nTask:\n{prompt}"


async def remember_outcome(
    memory: MemoryClient,
    *,
    session_id: str,
    prompt: str,
    result: str,
) -> str:
    """Persist a one-line task summary so a later run can recall what happened.

    Returns the new memory id. Tagged with the session so a session's history is
    filterable. The summary is **explicitly** written to the quarantine tier
    (``trust_tier=QUARANTINE_TIER``) — it is agent/tool-derived output and must not
    be trusted just because a permissive client defaults writes to ``trusted``.
    ``ground_prompt`` only reads *trusted* memories, so an unreviewed summary
    cannot feed itself back as behaviour-influencing context until host-side
    promotion (OWASP LLM01/LLM03). Mirrors ``failure_learning.py``'s quarantine of
    the same untrusted-provenance writes — never rely on the client's default tier.
    """
    text = f"Task (session {session_id}): {prompt}\nOutcome: {result}"
    return await memory.remember(
        text, tags=("task-summary", f"session:{session_id}"), trust_tier=QUARANTINE_TIER
    )


async def ground_and_run(
    brain: BrainProvider,
    store: CheckpointStore,
    memory: MemoryClient | None,
    *,
    session_id: str,
    prompt: str,
) -> SessionResult:
    """``drive_task`` wrapped in B's memory grounding when a memory client is given.

    Recall → run/resume → remember. With ``memory=None`` it degrades to a plain
    ``drive_task`` (A only), so the CLI can run with or without the backbone wired.

    Persisting the summary is **best-effort**: the task already succeeded and its
    checkpoint is durable by this point, so a memory-write failure is logged, not
    raised — otherwise a backbone hiccup would surface a *completed* run as failed
    and a retry would needlessly resume (re-execute) the finished turn.
    """
    effective = await ground_prompt(memory, prompt) if memory is not None else prompt
    result = await drive_task(brain, store, session_id=session_id, prompt=effective)
    if memory is not None:
        try:
            await remember_outcome(
                memory, session_id=session_id, prompt=prompt, result=result.text
            )
        except Exception:
            log.exception(
                "remember_outcome failed for session %s (summary not persisted)", session_id
            )
    return result


async def run_repl(
    brain: BrainProvider,
    store: CheckpointStore,
    lines: Iterable[str],
    emit: Callable[[str], None],
    *,
    session_id: str,
    memory: MemoryClient | None = None,
) -> None:
    """Drive a session over a stream of input ``lines`` until exhausted or quit.

    Each non-empty line continues the SAME ``session_id``: the first line launches,
    every later line resumes (the store now has a checkpoint), so context is held
    across turns within the process. A blank line is ignored; ``/exit`` or
    ``/quit`` ends the loop. ``lines``/``emit`` are injected so the loop is unit-
    testable without real stdin/stdout.

    Each turn is isolated: a failing turn (a transient brain error, a corrupt
    checkpoint on resume, …) is reported to the user and the loop CONTINUES — one
    bad turn must never kill the interactive session and strand the context the
    user is building (the same discipline the cockpit's ``serve`` loop applies
    per event).
    """
    for raw in lines:
        text = raw.strip()
        if not text:
            continue
        if text in _REPL_QUIT:
            return
        try:
            result = await ground_and_run(
                brain, store, memory, session_id=session_id, prompt=text
            )
        except Exception as exc:
            log.exception("repl turn failed for session %s", session_id)
            emit(f"error: {exc} (session continues — type /exit to quit)")
            continue
        emit(result.text)
