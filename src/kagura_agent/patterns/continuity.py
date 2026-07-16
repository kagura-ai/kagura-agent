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

import hashlib
import json
import logging
import re
from collections.abc import Callable, Iterable

from kagura_agent.core.brain.base import BrainProvider, Task
from kagura_agent.core.session import Session, SessionResult
from kagura_agent.mcp.memory_cloud import (
    QUARANTINE_TIER,
    TRUSTED_TIER,
    AgentBootstrap,
    Memory,
    MemoryClient,
)
from kagura_agent.patterns.checkpoint import CheckpointStore
from kagura_agent.patterns.erasure import ProvenanceLog

log = logging.getLogger(__name__)

#: How many recalled memories to inject as prior context (keeps the preamble
#: bounded so grounding never crowds out the actual task).
_MAX_GROUNDING = 5

_BOOTSTRAP_QUERY_MAX = 1024
_BOOTSTRAP_SESSION_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

#: Lines that end a REPL session.
_REPL_QUIT = frozenset({"/exit", "/quit"})


async def drive_task(
    brain: BrainProvider,
    store: CheckpointStore,
    *,
    session_id: str,
    prompt: str,
    on_message: Callable[[str], None] | None = None,
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

    ``on_message`` streams each narration event live (the ``--verbose`` hook, #105).
    """
    prior = await store.load(session_id)
    session = Session(brain, store)
    return await session.drive(
        Task(prompt=prompt, session_id=session_id), resume=prior, on_message=on_message
    )


async def ground_prompt(memory: MemoryClient, prompt: str) -> str:
    """Prepend relevant trusted prior memories to ``prompt`` (else return as-is).

    Only the *trusted* backbone is recalled (``trusted_only=True``): an
    externally-ingested / quarantined memory must never be silently promoted into
    a behaviour-influencing preamble (OWASP LLM01/LLM03 — the membrane's memory-
    provenance rule applied at the run path). With nothing relevant, the prompt is
    returned unchanged so a first-ever task is not padded with an empty block.
    """
    grounded, _ = await _grounded_with_sources(memory, prompt)
    return grounded


async def _grounded_with_sources(
    memory: MemoryClient, prompt: str
) -> tuple[str, list[Memory]]:
    """``ground_prompt`` + the memories it injected.

    Returns ``(grounded_prompt, used_memories)`` so the caller (``ground_and_run``)
    can record provenance — which source memories fed this session — without a
    second recall. On a recall miss it returns ``(prompt, [])`` (prompt unchanged,
    nothing to attribute). Same trusted-only provenance gate as ``ground_prompt``.
    """
    memories = await memory.recall(prompt, trusted_only=True)
    if not memories:
        return prompt, []
    used = memories[:_MAX_GROUNDING]
    lines = [f"- {m.text}" for m in used]
    preamble = "Relevant context from prior work:\n" + "\n".join(lines)
    return f"{preamble}\n\nTask:\n{prompt}", used


async def load_guardrails(memory: MemoryClient) -> str:
    """Render the deterministic pinned Goal/Guardrail set as a prompt preamble (#88).

    Unlike ``ground_prompt`` (probabilistic recall of *relevant* context), this loads
    the **complete pinned set every turn** via ``load_pinned`` — standing guardrails
    must never be missed by a recall miss. Returns "" when nothing is pinned.

    **Trust-gated, like every behaviour-influencing read:** only *trusted*-tier
    pinned memories become a standing guardrail. ``load_pinned`` itself is the raw
    primitive (the complete pinned set, SDK-faithful), but this lane is the most
    authoritative slot in the prompt, so it applies the same provenance gate
    ``ground_prompt`` uses (``trusted_only``) — defence in depth, so a host that
    pins a non-trusted (e.g. externally-ingested) memory cannot turn it into an
    always-apply instruction (OWASP LLM01/LLM03). The confinement that stops a
    *confined agent* from pinning at all lives in ``QuarantinedMemoryClient`` (it
    forces ``on_recall``); this gate is the read-side backstop, not a substitute
    for wiring a confined client.
    """
    pinned = [m for m in (await memory.load_pinned()) if m.trust_tier == TRUSTED_TIER]
    if not pinned:
        return ""
    lines = [f"- {m.text}" for m in pinned]
    return "Standing guardrails (always apply):\n" + "\n".join(lines)


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


def _bootstrap_correlation_id(session_id: str) -> str:
    """Return a server-valid opaque bootstrap correlation id.

    Transport thread ids are not guaranteed to use the bootstrap contract's
    restricted character set.  Preserve already-valid ids; otherwise send a
    stable digest, never a lossy/ambiguous character replacement.
    """
    if _BOOTSTRAP_SESSION_RE.fullmatch(session_id):
        return session_id
    digest = hashlib.sha256(session_id.encode()).hexdigest()[:32]
    return f"session-{digest}"


def _dedupe_memories(*groups: tuple[Memory, ...]) -> list[Memory]:
    seen: set[str] = set()
    out: list[Memory] = []
    for group in groups:
        for memory in group:
            if memory.id not in seen:
                seen.add(memory.id)
                out.append(memory)
    return out


def _render_bootstrap_prompt(bootstrap: AgentBootstrap, prompt: str) -> str:
    blocks: list[str] = []
    seen_memory_ids: set[str] = set()

    def unique(group: tuple[Memory, ...]) -> list[Memory]:
        lane: list[Memory] = []
        for memory in group:
            if memory.id not in seen_memory_ids:
                seen_memory_ids.add(memory.id)
                lane.append(memory)
        return lane

    if bootstrap.instructions.strip():
        blocks.append(f"Bootstrap instructions:\n{bootstrap.instructions.strip()}")
    pinned = unique(bootstrap.pinned)
    recalled = unique(bootstrap.recall)
    upcoming = unique(bootstrap.upcoming)
    if pinned:
        blocks.append(
            "Standing guardrails (always apply):\n"
            + "\n".join(f"- {memory.text}" for memory in pinned)
        )
    if recalled:
        blocks.append(
            "Relevant context from prior work:\n"
            + "\n".join(f"- {memory.text}" for memory in recalled)
        )
    if upcoming:
        blocks.append(
            "Upcoming time memories:\n"
            + "\n".join(f"- {memory.text}" for memory in upcoming)
        )
    if bootstrap.state:
        blocks.append(
            "Agent state (advisory; session checkpoint remains authoritative):\n"
            + json.dumps(
                bootstrap.state,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    if not blocks:
        return prompt
    return "\n\n".join((*blocks, f"Task:\n{prompt}"))


async def prepare_bootstrap_prompt(
    memory: MemoryClient,
    *,
    session_id: str,
    prompt: str,
) -> tuple[str, list[Memory], AgentBootstrap]:
    """Fetch one session-start bundle and render only its model-safe fields.

    The complete user prompt still reaches the brain. Only the recall query is
    bounded to the server contract's 1024 characters. Component errors are
    fail-soft and logged; total identity/contract errors raised by the client stay
    fail-closed and abort the task.
    """
    bootstrap = await memory.get_agent_bootstrap(
        session_id=_bootstrap_correlation_id(session_id),
        query=prompt[:_BOOTSTRAP_QUERY_MAX],
        recall_k=_MAX_GROUNDING,
    )
    if bootstrap.degraded:
        log.warning(
            "agent bootstrap degraded for session %s (components=%s)",
            session_id,
            ",".join(bootstrap.component_failures),
        )
    used = _dedupe_memories(bootstrap.pinned, bootstrap.recall, bootstrap.upcoming)
    return _render_bootstrap_prompt(bootstrap, prompt), used, bootstrap


async def ground_and_run(
    brain: BrainProvider,
    store: CheckpointStore,
    memory: MemoryClient,
    *,
    session_id: str,
    prompt: str,
    provenance: ProvenanceLog | None = None,
    on_message: Callable[[str], None] | None = None,
) -> SessionResult:
    """``drive_task`` wrapped in B's memory grounding.

    One ``get_agent_bootstrap`` read (context guide + trusted pinned/recall/upcoming
    + agent-state) → run/resume → remember. Local backends compose the same lanes
    behind that method; cloud performs one trusted-host REST call.

    ``memory`` is **always present** (#104): ``make_memory_client`` never returns
    ``None`` — the seam never disappears, only its backend strength differs
    (``LocalMemoryClient`` for test/dev, a trust-aware cloud client in deployment).
    This closed the worst-of-both gap where a run paid the reachability gate's
    friction yet got none of the backbone's benefit (the old ``memory is None``
    degrade branch is gone).

    When a ``provenance`` log is given, the source memories injected into this
    session's prompt are recorded against ``session_id`` (#93) — the bridge a later
    host-side ``forget_cascade`` needs to reach the checkpoint / outcome-summary
    this run derives. The log is host-side and optional, so the run path works with
    or without erasure-cascade tracking wired.

    Persisting the summary is **best-effort**: the task already succeeded and its
    checkpoint is durable by this point, so a memory-write failure is logged, not
    raised — otherwise a backbone hiccup would surface a *completed* run as failed
    and a retry would needlessly resume (re-execute) the finished turn.
    """
    effective, used, _bootstrap = await prepare_bootstrap_prompt(
        memory, session_id=session_id, prompt=prompt
    )
    if provenance is not None and used:
        # Record which source memories (and their tiers) fed this session: the tier
        # capture is the host evidence for the run's input_trust, and the memory ids
        # let a later erasure of any of them cascade to this run's derived artifacts.
        provenance.record_grounding(session_id, [(m.id, m.trust_tier) for m in used])
    result = await drive_task(
        brain, store, session_id=session_id, prompt=effective, on_message=on_message
    )
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
    memory: MemoryClient,
    on_message: Callable[[str], None] | None = None,
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
                brain, store, memory, session_id=session_id, prompt=text,
                on_message=on_message,
            )
        except Exception as exc:
            log.exception("repl turn failed for session %s", session_id)
            emit(f"error: {exc} (session continues — type /exit to quit)")
            continue
        emit(result.text)
