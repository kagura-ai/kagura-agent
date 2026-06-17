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

from collections.abc import Callable, Iterable

from kagura_agent.core.brain.base import BrainProvider, Task
from kagura_agent.core.session import Session, SessionResult
from kagura_agent.mcp.memory_cloud import MemoryClient
from kagura_agent.patterns.checkpoint import CheckpointStore

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
    """
    session = Session(brain, store)
    if await store.load(session_id) is not None:
        return await session.resume(session_id, prompt=prompt)
    return await session.run(Task(prompt=prompt, session_id=session_id))


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
    filterable. Writes go to the agent's default (quarantine) tier by the client's
    own policy — ``ground_prompt`` only reads *trusted* memories, so an unreviewed
    summary cannot feed itself back as trusted context until host-side promotion.
    """
    text = f"Task (session {session_id}): {prompt}\nOutcome: {result}"
    return await memory.remember(text, tags=("task-summary", f"session:{session_id}"))


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
    """
    effective = await ground_prompt(memory, prompt) if memory is not None else prompt
    result = await drive_task(brain, store, session_id=session_id, prompt=effective)
    if memory is not None:
        await remember_outcome(memory, session_id=session_id, prompt=prompt, result=result.text)
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
    """
    for raw in lines:
        text = raw.strip()
        if not text:
            continue
        if text in _REPL_QUIT:
            return
        result = await ground_and_run(
            brain, store, memory, session_id=session_id, prompt=text
        )
        emit(result.text)
