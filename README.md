# kagura-agent

> **Private repository.** Part of the Kagura Memory Cloud commercial offering.
>
> **Status: Future placeholder.** No runnable code yet. This README is the
> canonical design document until development begins. No fixed launch trigger —
> development starts when memory-cloud + the chat ingestion / dataset /
> embeddings workers have accumulated enough customer signal to justify
> building a first-party agent on top.

Autonomous AI agent built on the
[Claude Agent SDK (Python)](https://docs.claude.com/en/api/agent-sdk).
Uses [Kagura Memory Cloud](https://github.com/kagura-ai/memory-cloud) as its
**long-term memory backbone** while orchestrating filesystem, shell, infra,
and custom MCP servers to execute real, long-running tasks autonomously.

---

## What kagura-agent is — and is not

A common misread is "it's an AI that calls `recall` / `reference` / `explore`".
That captures only the memory-reader role. The agent's defining capability is
the combination of **memory + actor**:

| | Narrow read (commodity) | What kagura-agent actually is |
|---|---|---|
| **Brain** | Generic chat LLM | Claude (via Agent SDK Python, subscription or API key) |
| **Memory** | Stateless or session-only | Kagura Memory Cloud as persistent backbone — accumulates across runs |
| **Hands** | None — just answers | shell exec / filesystem / git / Docker / Cloudflare / cloud APIs (via MCP) |
| **Time horizon** | One conversation | Long-running tasks resumable from memory state |
| **Differentiation** | Anyone with Claude Desktop + memory MCP plugin matches it | Cost-aware planning + failure-mode learning + sub-agent dispatch with memory handoff |

The agent is an **actor** in the topology — it lives entirely outside `memory-cloud`,
which it treats as one of its MCP servers (the most important one). It can be run
as a CLI, a daemon, or as a managed SaaS lane.

---

## Architecture

```
                    ┌───────────────────────────────────┐
                    │   Claude Agent SDK (Python)        │
                    │   subprocess-wraps Claude Code CLI │
                    │   → subscription auth inherits     │
                    └───────────────────────────────────┘
                                      │
                          orchestrates tool calls
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                ▼            ▼            ▼               ▼
   ┌─────────┐   ┌───────────┐  ┌─────────┐  ┌─────────┐    ┌────────┐
   │ memory- │   │filesystem │  │  shell  │  │cloudflare│    │ custom │
   │ cloud   │   │   MCP     │  │   MCP   │  │   MCP   │    │  MCP   │
   │  MCP    │   │           │  │ (Docker │  │  / aws  │    │ per-   │
   │         │   │           │  │ -isolat)│  │  / gcp  │    │ tenant │
   └─────────┘   └───────────┘  └─────────┘  └─────────┘    └────────┘
        │
   recall / reference / explore  ← read past learnings
   remember / create_edge       ← write what was learned
   ingest_events                ← push cost ledger / task tracker
```

### Auth model (from memory `31e85a92`)

- **Python Agent SDK** wraps the Claude Code CLI as a subprocess →
  the user's Pro/Max subscription is inherited automatically.
- **Self-hosted / single-user mode**: flat subscription cost regardless
  of agent load (within Anthropic's cap-based metering: 5h rolling +
  weekly 7-day rolling). No per-token tracking required from kagura side.
- **SaaS / multi-tenant mode**: per-tenant subscription is not viable.
  Falls back to `workspace.external_api_keys` (BYOK) for Claude API key,
  unifying with the chat-ingestion worker's BYOK auth path.
- `ANTHROPIC_API_KEY` env, if set, overrides subscription auth.

This dual-path (subscription for self-hosted, API key for SaaS) is the
same pattern memory-cloud already uses for LLM provider keys.

---

## Differentiating capabilities

The four capabilities below are what make memory + actor worth more than
their sum. None of them are achievable with a stateless agent or a
memory-less actor.

### 1. Cost-aware planning

Before kicking off a multi-step task, the agent recalls past similar
tasks' actual cost (token spend, time, retries, failure modes) and
adjusts plan + budget accordingly. Example:

```
user: "deploy v0.16.1 to staging"

agent:
  → recall("deploy staging", filters={status: "failed"})
  → finds 2 past failures (Caddyfile permission trap, env-file omission)
  → adds explicit pre-flight checks for both before deploy
  → reserves 30% buffer over avg historical cost
```

### 2. Long-running task resume

Context window dies; the agent doesn't. Task state is checkpointed to
memory-cloud and resumed cleanly in a fresh session:

```
session 1 (interrupted):
  remember(type="task-checkpoint", details={step: 4, pending: ["test", "deploy"]})

session 2 (resumed):
  recall("task-checkpoint", filters={task_id: ...}, k=1)
  → "continuing from step 4 of N — pending: [test, deploy]"
```

### 3. Failure-mode learning

Every failure becomes a memory with a `prevents` edge to its fix:

```
remember(
  summary="Caddyfile cp fails when root-owned",
  type="bug-fix",
  details={trigger: "...", fix: "sudo chown ... && retry"}
)
create_edge(from=fix_memory, to=task_memory, type="prevents")
```

Next time the agent plans a similar task, the recall surfaces this fix
preemptively. Failure cost → 0 over time on recurring patterns.

### 4. Sub-agent dispatch with memory handoff

A large task spawns child agents; context is passed not as prompt
text but as memory IDs:

```
parent agent:
  remember(summary="task context for child", scope="working", ttl=3600)
  → returns memory_id

  dispatch(child_agent, prompt="recall memory_id=<...> and execute")

child agent:
  recall(memory_id=<...>)
  → child works on it, writes its own memories, finishes
  → parent recalls child's output memories to continue
```

Parent context window stays small; complex pipelines become composable.

---

## Phase 1 scope (when development begins)

Tightly scoped to validate the "memory + actor" thesis before broadening:

| Capability | In | Out |
|---|---|---|
| shell exec (Docker-isolated) | ✅ | host shell ❌ |
| filesystem read/write (project root) | ✅ | system-wide fs ❌ |
| git ops (clone, commit, push) | ✅ | rebase / force-push ❌ |
| `.env` and config file mgmt | ✅ | OS pkg install ❌ |
| Cloudflare DNS read | ✅ | DNS write (Phase 2) |
| `sudo apt install` etc. | ❌ Phase 2 | (surprisingly dangerous) |
| memory-cloud full MCP toolset | ✅ | — |

The "surprisingly dangerous" note on `sudo apt install` reflects a real
concern: package installation has the widest blast radius of any common
ops action. It gets gated behind explicit Phase 2 review.

---

## What kagura-agent is NOT

To prevent scope creep, several adjacent things are explicitly out of scope:

- **NOT a chat interface for memory-cloud.** That's covered by the existing
  MCP server in memory-cloud + any MCP-capable client (Claude Desktop etc.).
- **NOT a fine-tuned model.** It runs base Claude, not a customer-specific LLM.
  Custom-model concerns belong to `kagura-memory-dataset-worker` Layer 2.
- **NOT an ingestion source.** Slack / Teams chat ingestion belongs to
  `kagura-memory-ai-worker`. The agent may *use* those memories, not produce them.
- **NOT a memory analyzer.** broadlistening lives in memory-cloud.
- **NOT a domain LLM** (Layer 3 rejected — see dataset-worker README).

The agent's job is **autonomous task execution with persistent memory**,
nothing more, nothing less.

---

## Phase and launch trigger

This repository is **not yet under active development**. No fixed launch date.
Reasonable triggers to revisit:

1. `kagura-memory-ai-worker` Phase 1+2 in production, accumulating
   non-trivial customer memory.
2. Internal dogfooding signal: the team has noticed they want
   "agent that remembers past failures" while operating memory-cloud itself.
3. Customer ask: at least one Enterprise customer explicitly wants
   "an agent that uses our memory autonomously" (vs just an MCP client).

Unlike the dataset / embeddings workers, this agent's launch is not
tied to a specific quantitative break-even — it's a qualitative
"the value of memory-as-backbone is clear enough to invest in an actor
that depends on it" call.

---

## Repository layout (planned)

```
kagura-agent/
├── README.md                       # this file
├── pyproject.toml
├── src/
│   ├── core/
│   │   ├── session.py              # Claude Agent SDK wrapper
│   │   ├── auth.py                 # subscription vs API key resolver
│   │   └── budget.py               # cost-aware planning loop
│   ├── mcp/
│   │   ├── memory_cloud.py         # primary MCP — recall/remember/etc.
│   │   ├── filesystem.py
│   │   ├── shell_docker.py
│   │   └── infra/
│   │       ├── cloudflare.py
│   │       ├── aws.py
│   │       └── gcp.py
│   ├── patterns/
│   │   ├── checkpoint.py           # long-task resume
│   │   ├── failure_learning.py     # remember(prevents) edges
│   │   └── subagent_dispatch.py    # memory-id handoff
│   └── cli/
│       └── main.py                 # `kagura-agent run "task description"`
├── tests/
└── deploy/
    ├── Dockerfile
    └── compose.yml                 # single-user self-hosted
```

Python, Claude Agent SDK Python, subprocess-wrapped Claude Code CLI.

---

## Related repositories

| Repo | Role | Relationship to agent |
|---|---|---|
| [`kagura-ai/memory-cloud`](https://github.com/kagura-ai/memory-cloud) | Persistence + MCP server | **The backbone.** Agent's primary MCP. |
| [`kagura-ai/kagura-memory-python-sdk`](https://github.com/kagura-ai/kagura-memory-python-sdk) | Primitive SDK | Used by the memory MCP client wrapper inside the agent. |
| [`kagura-ai/kagura-memory-ai-worker`](https://github.com/kagura-ai/kagura-memory-ai-worker) | Chat ingestion | Produces memories the agent later reads. Not in the agent's execution path. |
| [`kagura-ai/kagura-memory-dataset-worker`](https://github.com/kagura-ai/kagura-memory-dataset-worker) | Export + fine-tune | Independent. May export agent-produced memories as datasets. |
| [`kagura-ai/kagura-embeddings-worker`](https://github.com/kagura-ai/kagura-embeddings-worker) | Sovereign embeddings | Indirect — agent's `recall` quality depends on which embeddings lane the workspace uses. |

---

## License

Proprietary — © Kagura AI. Not for redistribution. See `LICENSE` for terms.
