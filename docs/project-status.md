# Project status and scope

This document records the current implementation boundary, remaining production work, repository map, and milestone coverage.

## What kagura-agent is NOT

To prevent scope creep, several adjacent things are explicitly out of scope:

- **NOT a chat front-end for memory-cloud.** Querying memory-cloud as a chat is
  covered by its existing MCP server + any MCP-capable client (Claude Desktop
  etc.). The agent's Slack/Discord surface is a **cockpit for driving the agent**
  (launch / steer / approve / kill tasks — see [Control surface internals](design.md#control-surface-internals-cockpit)), not
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

The v0.1–v0.7 **skeleton is built and tested** — the [design](design.md) is implemented as
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

## Bootstrap ranking evaluation

The outcome-level gate for feedback-influenced bootstrap ranking is documented
in [bootstrap-eval.md](bootstrap-eval.md). It uses a committed fixed
corpus, isolated memory-cloud contexts, and a task-level paired confidence
interval; retrieval-only lift cannot authorize a default change.

---

## Repository layout

The structure the [design](design.md) maps onto — **as built** (v0.1–v0.7):

```
kagura-agent/
├── README.md                       # concise project overview and quickstart
├── docs/
│   ├── README.md                   # documentation index
│   ├── getting-started.md          # install, authentication, usage, troubleshooting
│   ├── design.md                   # canonical architecture and security design
│   ├── project-status.md           # scope, roadmap, repository map, milestones
│   ├── operations.md               # incident runbook (hijack / key rotation, cred lifecycle)
│   ├── extending.md                # new API hands (custom MCP / egress / cred + secret backend)
│   ├── bootstrap-eval.md           # #188 outcome A/B + long-horizon ranking safety gate
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

## Implementation status (v0.1–v0.7 skeleton)

The pure-Python core of every milestone is implemented and tested (58 test modules, `mypy --strict`, ≥95% coverage); the infrastructure edges (real Docker,
cloud STS/Cloudflare, the Slack/Discord/SDK clients) sit behind protocol seams with
their adapters wired for deployment.

**Running it.** See [Getting started](getting-started.md) for the full
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
