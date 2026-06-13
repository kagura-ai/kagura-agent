# Extending — giving the agent new API hands (self-host v1)

> Companion to the canonical design doc (`../README.md`) and the ops runbook
> (`operations.md`). Read the **Security membrane** section of the README first —
> everything here operates *inside* that membrane, not around it.

"Let the agent operate an API for me" has three supported shapes, in increasing
order of privilege. Pick the **least** one that does the job.

| Shape | Use when | Secret handling |
|---|---|---|
| **A. Custom MCP server** (`--mcp-config`) | the API has (or you can write) an MCP server; the agent calls typed tools | server reads a token from *its own* env, host-side |
| **B. Shell in the container** | a one-off `curl`/CLI call against an allowlisted host | a leased, short-lived token — never a baked key |
| **C. Cloud cred lease** (AWS/GCP/GitHub/Cloudflare) | the agent runs `aws`/`gcloud`/`gh` against a cloud account | per-task scoped, auto-expiring creds minted by the cockpit |

The one rule that spans all three:

> **Never bake a long-lived API key into the image or the container's ambient
> env.** The membrane exists so that a hijacked agent has nothing worth stealing.
> A standing secret in the container defeats it. Use short-lived, scoped tokens
> (shape C) or keep the secret host-side in an MCP server (shape A).

---

## A. Custom MCP server (`--mcp-config`)

The agent's brain is Claude Code via the Agent SDK, so any **MCP server** you
point it at becomes a set of tools it can call. Memory is reached CLI-first and
is *not* configured here — `--mcp-config` is strictly for **other** MCP servers
(it mirrors Claude Code's own flag).

```bash
kagura-agent run "summarize today's PagerDuty incidents" --mcp-config ./mcp.json
```

`mcp.json` accepts the Claude Code convention `{"mcpServers": {...}}` **or** a
bare `{name: config}` map (`cli/main.py:load_mcp_config`):

```jsonc
{
  "mcpServers": {
    "pagerduty": {
      "command": "node",
      "args": ["./pagerduty-mcp.js"],
      // The token lives in the SERVER's env (host-side), not in the agent
      // container. Prefer a short-lived/scoped token where the API supports it.
      "env": { "PAGERDUTY_TOKEN": "${PAGERDUTY_TOKEN}" }
    }
  }
}
```

- Add `--strict-mcp-config` to **reject any MCP server not listed** in the file —
  no silent ambient passthrough. Use it in production so an unexpected server
  cannot slip in.
- A missing config path fails loud; a non-object `mcpServers` value is rejected
  with a clear error rather than crashing later.

Why this is the safest shape: the secret stays in the MCP server process, which
you run; the agent only ever sees the *tools*, never the credential.

---

## B. Shell + egress allowlist

Inside the container the agent has a shell, so it *can* `curl` an API directly.
Two constraints make this safe — neither is optional:

1. **Egress is default-deny.** A container with no granted egress runs
   `--network none`; it cannot reach the API at all. To allow a destination, the
   run must carry it in the launch spec's egress allowlist, and the host must be
   on the proxy's allowlist. The allowlist is **exact-host** — no wildcards, no
   subdomains (`membrane/egress.py`); `api.example.com` does **not** cover
   `eu.api.example.com`.

   Deploy-level allowlist (`deploy/compose.yml`, comma-separated):

   ```yaml
   egress-proxy:
     environment:
       EGRESS_ALLOWLIST: "api.anthropic.com,memory.kagura-ai.com,api.example.com"
   ```

   The per-run allowlist and the network the launcher attaches are derived from
   the **same** launch 4-tuple (`EgressPolicy.from_spec`), so the policy the
   proxy enforces and the network wiring cannot drift apart.

2. **The token must be leased, not baked.** Pass a short-lived token in at
   launch (shape C, or an MCP-minted one) rather than embedding a long-lived key
   in the image.

> ⚠️ An allowlisted destination can still be an exfil channel (e.g. pushing to an
> attacker repo via an over-broad `gh` token). Pair the allowlist with
> **minimum-scope leases** — see shape C and the launcher's `CredentialBroker`
> in the README.

---

## C. Cloud cred lease (AWS / GCP / GitHub / Cloudflare)

For the four first-class clouds, the cockpit (trusted host) holds the root
credential and **mints a short-lived, scoped credential per task**, which the
launcher injects into the container's env. The container only ever holds the
short-lived cred; the root credential never leaves the host.

The provider implementations live in `membrane/providers.py`:

| Provider | `scope` is… | What lands in the container |
|---|---|---|
| `AwsStsProvider` | the role ARN | `AWS_ACCESS_KEY_ID` / `_SECRET_ACCESS_KEY` / `_SESSION_TOKEN` |
| `GcpImpersonationProvider` | the target SA email | an access token |
| `GitHubAppProvider` | `installation:<id>` | an installation token (~1h) |
| `CloudflareTokenProvider` | a permission scope | a scoped API token (revoked on release) |

Flow: `CredentialBroker.acquire(scope)` → provider `.mint(scope, ttl)` →
launcher injects the env → the agent runs `aws`/`gcloud`/`gh` against it →
the lease expires (stateless) or is revoked on release (Cloudflare).

Two guardrails that hold regardless of what the agent asks for:

- **Memory writes are read-only by default.** Any `memory:*` scope other than
  `memory:read` requires explicit write approval; a read-locked provider refuses
  to mint a write token *even if a caller hands `memory:write` straight to
  `acquire`* (`MemoryCloudProvider`, hardened in #20). Widening to write needs a
  device-flow HITL re-approval.
- **Privileged scopes go through HITL.** A hijacked or unattended agent cannot
  silently widen its own powers.

> **Status:** the provider *logic* is implemented and unit-tested with injected
> transports (the core stays dependency-free). The real transport adapters
> (boto3 / google-auth / httpx) and the launcher's lease→env injection are the
> deployment edge tracked in **#39** — wire those to make shape C live.

---

## Choosing

- The API has an MCP server, or you can write one → **A** (safest; secret stays
  host-side).
- It's AWS/GCP/GitHub/Cloudflare → **C** (per-task scoped, auto-expiring).
- It's a quick call to one host and you have a short-lived token → **B**, with
  the destination on the egress allowlist.

In every case: scope down, keep the secret off the container's standing env, and
let the membrane bound the blast radius.
