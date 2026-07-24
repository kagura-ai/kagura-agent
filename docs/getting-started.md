# Getting started

> kagura-agent is on PyPI — `pip install 'kagura-agent[claude]'` — and needs **two
> logins** before it will run. This guide covers the full first-run setup; `kagura-agent doctor`
> tells you exactly what is still missing.

## Prerequisites

- **Python ≥ 3.11.**
- **Memory login — required to start.** Every `run` / `repl` / `serve` checks that Kagura
  Memory Cloud is reachable *before doing anything else* and **refuses to start** if it is
  not (a run is rejected, never silently degraded). Authenticate with the **separate
  `kagura` CLI** — `kagura auth login`. There is no fully-offline mode: `KAGURA_AGENT_MEMORY_DB`
  changes only *where* memories are stored, not this gate.
- **A brain.** The default `sdk` brain needs the `claude` extra **and** the Claude Code CLI
  signed in to your Pro/Max plan (or `ANTHROPIC_API_KEY`, which overrides subscription
  auth). The `brain` extra's **codex** backend likewise inherits your **ChatGPT
  subscription** via `codex login`. The bare core ships with **no brain** — you pick an extra.
- **Docker** — only for `serve --container` and the live membrane.

## Get running

```bash
# 1. Create and activate a virtual environment (macOS/Linux):
python -m venv .venv
source .venv/bin/activate
```

```powershell
# Windows PowerShell:
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Then install a brain extra; the bare core intentionally includes none:

```bash
pip install 'kagura-agent[claude]'    # default — Claude Agent SDK
# pip install 'kagura-agent[brain]'   # alternate — kagura-brain (Claude/Codex/Ollama)

# 2. Authenticate (both logins are real prerequisites):
kagura auth login                 # Kagura Memory Cloud — the separate `kagura` CLI
claude                            # sign the Claude Code CLI into your plan…
# export ANTHROPIC_API_KEY=sk-…   # …or bring your own key (overrides subscription)
# codex login                     # [brain] extra + KAGURA_AGENT_BRAIN_BACKEND=codex —
#                                 # the codex brain runs on your ChatGPT subscription

# 3. Preflight — reports exactly what is still missing:
kagura-agent doctor

# 4. Run a task:
kagura-agent run "summarize the repository layout"
```

## More ways to drive it

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

## Troubleshooting

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
(`sdk` default, or `kagura-brain`) — see [Brain-provider seam](design.md#brain-provider-seam).
Contributors install from a clone: `git clone` + `pip install -e '.[dev]'`, then `pytest`
and `mypy` (strict) — see [CONTRIBUTING.md](../CONTRIBUTING.md).

## One-call cloud bootstrap

Install the cloud-memory extra, then configure one registered agent, its bound
context, a dedicated agent-bound member key, and the host-side MCP server command:

```bash
pip install 'kagura-agent[memory]'
export KAGURA_AGENT_ID='agent-uuid'
export KAGURA_AGENT_MEMORY_MCP_CONTEXT='context-uuid'
export KAGURA_AGENT_MEMORY_API_KEY='agent-bound-member-key'
export KAGURA_AGENT_MEMORY_MCP_SERVER='kagura-memory-mcp'
# Optional for self-hosted deployments; the SDK otherwise uses production:
# export KAGURA_AGENT_MEMORY_MCP_URL='https://memory.example.com/mcp'
```

The member key must be bound to the configured agent/context. `run`, `repl`, and
the cockpit then call the SDK's `AgentsClient.bootstrap()` once over REST per task
to obtain the context guide, trusted pinned/recall/upcoming memories, and
agent-state. Ordinary recall/write operations remain on the existing MCP memory
transport. The model never receives the key or correlation/audit metadata. A total
identity/contract failure aborts the task; component failures stay fail-soft and
are logged from the response's `degraded` status.

Bootstrap agent-state is advisory cross-session context. Session resume continues
to use the existing `CheckpointStore`; bootstrap never overwrites a checkpoint.
Local and SQLite backends expose the same method and return explicit empty
upcoming/state/policy lanes, so callers use one grounding path in every tier.
