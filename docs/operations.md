# Operations — incident runbook: hijack & key rotation (self-host v1)

> Ops companion to the canonical [design document](design.md). Read the
> [Security membrane](design.md#security-membrane-self-host-v1) section first —
> this runbook is the residual detection + rotation layer that sits on top of
> the membrane.

This runbook is short **because the membrane already bounds blast radius** (no
standing creds, project-root FS, egress allowlist). Detection + rotation is the
residual layer, not the primary defense.

## Detection tripwires

The egress log is the primary tripwire; these are the signals worth alerting on:

- **egress** to a non-allowlisted destination (highest signal)
- **cred-use without a matching HITL approval**, or a run requesting powers
  inconsistent with its stated task
- **shell fingerprints**: pipe-to-shell (`curl … | sh`), `base64 -d | sh`,
  writes to `~/.ssh`, wholesale env dump
- **injection fingerprint**: a `recall` result containing instruction-like text
  immediately preceding a behavior change
- **volume anomaly**: a sudden burst of tool calls or egress

**Tiered response — a self-host operator has no on-call, so the highest-signal
tripwires must act _without_ waiting for a human.** MTTR is dominated by "time
until the operator reads the DM," so the design minimizes blast radius during
that window rather than relying on fast human response:

| Tier | Tripwire | Automatic action |
|---|---|---|
| **block** | non-allowlisted egress | denied at the proxy by default (never reaches the network) |
| **contain** | cred-use without HITL approval · `curl \| sh` / `base64 \| sh` / `~/.ssh` write | SIGKILL container + freeze session + suspend new cred grants |
| **notify** | volume anomaly · injection fingerprint | alert to cockpit DM; human decides |

### Egress: one enforcing chokepoint

"Logging egress" is not enough — over TLS the destination is invisible without
SNI inspection, and logging alone does not stop exfiltration. v1 routes **all**
container egress through a **single proxy**: the container has no direct internet
route; only the proxy is reachable; the proxy does **default-deny + allowlist +
structured logging** in one place (SNI-based for HTTPS; DNS via a controlled
resolver). This proxy is simultaneously the membrane's egress control point and
the primary detection tripwire. No SIEM needed for v1 — local structured logs +
a rule that alerts on a non-allowlisted SNI is enough.

> ⚠️ Even an allowlisted destination can be an exfil channel (push to an
> attacker repo via an over-broad `gh` token, write to an allowed bucket). Pair
> the allowlist with **minimum-scope leases** (e.g. `contents:write` on the
> target repo only, no gist/repo-create) — see the
> [launcher's `CredentialBroker`](design.md#the-launcher-per-run-capability-binding).

## Response — contain → rotate → investigate → eradicate → recover

1. **Contain** — cockpit SIGKILLs the container, freezes the session, suspends
   the launcher from granting new creds.
2. **Rotate** — revoke + reissue the scoped creds live during the suspect
   window (per provider: aws / gcloud / cloudflare / gh). Because creds are
   short-lived and per-task, the window is already bounded.
3. **Investigate** — pull the run's event log + egress log + the exact `recall`
   set that fed the run, to locate the injection source.
4. **Eradicate** — `forget` / quarantine the poisoned memory; add a `prevents`
   edge so it won't resurface; tighten the egress allowlist. A `forget` must
   **cascade**: memory-cloud's own `forget` erases the primary memory +
   embeddings/edges server-side, but the agent also derives **checkpoints** and
   **outcome-summaries** from recalled memories that a server-side forget never
   reaches. Run the host-side `forget_cascade` (`patterns.erasure`) to erase those
   derived artifacts too — it is host-side only (the agent surface has no erasure
   verb) and driven off the provenance trail grounding records. This is also the
   technical half of the GDPR erasure obligation (CSO finding C1 / `legal.md` §3):
   erasure must reach derived artifacts, not just the primary memory.
5. **Recover** — resume the category only after review; **demote graduation**
   if a graduated capability was abused.

## Key rotation procedure

The scoped-cred model makes routine rotation cheap and isolates the one thing
that actually matters:

- **Task creds** are minted per-run by the launcher and expire on their own —
  "rotation" of a task cred is usually just letting it lapse / revoking a token,
  no human action.
- **The root credential** the cockpit holds to _mint_ task creds is the crown
  jewel. It lives on the **host with the cockpit, never in a container**.
  Rotating it is the real incident action — document the per-provider steps
  (AWS role/key, GCP SA, Cloudflare token, GitHub app).
- **Cadence**: routine root-cred rotation on a fixed schedule (e.g. quarterly)
  **plus** immediate rotation on any suspected hijack.

## Per-provider short-lived credential feasibility

The [launcher's per-task mint-and-expire model](design.md#the-launcher-per-run-capability-binding)
depends on each provider issuing short-lived, scoped creds. Verified feasibility:

| Provider | Mechanism | TTL | Scoping | Verdict |
|---|---|---|---|---|
| **AWS** | STS `AssumeRole` (+ inline session policy) | 15 min – 12 h | IAM role + session policy | ✅ native, **stateless** signed call |
| **GCP** | IAM Credentials `generateAccessToken` via SA impersonation | ≤ 1 h default (≤ 12 h org policy) | per-SA IAM roles | ✅ native |
| **GitHub** | App installation access token (App JWT → token) | 1 h fixed | repos + permission set per installation | ✅ native |
| **Cloudflare** | Tokens API: mint child token with `not_before` / `expires_on`, scoped permission groups | arbitrary (you set it) | per zone / account / user permission groups | ⚠️ workable, **stateful** |

**Cloudflare is the rough edge, not a blocker.** Unlike AWS STS (a fast signed
call with no server-side state), Cloudflare needs an API round-trip to *create* a
scoped token and another to *delete/revoke* it — a mint→use→revoke lifecycle per
task, with latency and a token to clean up. The parent token holding "Create
Additional Tokens" is itself a root credential (treat as crown jewel, see Key
rotation above). R2 specifically offers proper STS-like temporary credentials.

> **Implication for the launcher design:** AWS / GCP / GitHub fit a stateless
> "mint on the signed path" shape; Cloudflare needs an explicit
> create→use→revoke lifecycle the launcher must track and **clean up on crash**.
> Build the launcher's cred interface to allow both shapes — do not assume
> STS-style statelessness everywhere.

Sources: [Cloudflare — create tokens via API](https://developers.cloudflare.com/fundamentals/api/how-to/create-via-api/),
[Cloudflare — restrict tokens (TTL)](https://developers.cloudflare.com/fundamentals/api/how-to/restrict-tokens/),
[Cloudflare R2 — temporary credentials](https://developers.cloudflare.com/r2/api/tokens/).

## Cockpit availability & recovery

The cockpit is a single long-lived host process — a **single point of failure**.
If it dies, running agent containers are orphaned: still executing, still holding
leases, with no one routing their output or able to kill them.

- **Supervise it.** Run under `systemd` (`Restart=always`) or
  `docker run --restart=always`, plus an **external liveness ping** — if the
  cockpit is down, alerting is down too ("who watches the watcher").
- **Make the registry reconstructable from Docker alone.** Stamp each container
  with labels `{thread_id, task_id, lease_ref}`. On restart the cockpit
  `reconcile`s against `docker ps`: adopt survivors from their labels, mark
  vanished ones `done/killed`. This removes the dependency on the in-memory
  registry's freshness (memory-cloud checkpoints become a convenience, not the
  source of truth).
- **Fail closed on pending approvals.** A crash mid-HITL leaves a container
  awaiting a decision that will never arrive — on restart, **time out and deny**
  any pending `CapabilityRequest`.

## Credential lifecycle operations

The [`CredentialBroker` / `Lease` model](design.md#the-launcher-per-run-capability-binding)
needs a durable backing store so a crash can't leak live cloud credentials:

- **Lease ledger** — an append-only durable record
  `{lease_id, provider, container_id, expires_at, revoke_handle}`. Written on
  `acquire`, marked closed on `release`. Must survive a cockpit crash (not
  in-memory).
- **Sweeper** — on startup **and** periodically, list open leases and **revoke
  any whose owning container is gone**. STS-style leases self-expire (sweep is a
  no-op); Cloudflare leases need an explicit revoke call — this sweeper is the
  cleanup the feasibility note above flags as required.
- **Renewer death.** The budget-renew loop runs in the cockpit/launcher. If it
  dies while a task keeps running, the task's lease lapses mid-flight — the agent
  must **pause/retry gracefully on cred expiry, not crash-loop**. Define this
  behavior in the brain↔launcher contract.

## Minimal v1 ops tooling checklist

1. `systemd` unit (`Restart=always`) for the cockpit + external liveness ping
2. Egress proxy (default-deny + allowlist + structured log) as the single network chokepoint
3. Durable lease ledger + startup/periodic sweeper
4. Container labels `{thread_id, task_id, lease_ref}` → registry reconciles from Docker alone
5. Tiered tripwire policy (block / contain / notify) wired to automatic actions
6. Fail-closed timeout for pending HITL approvals on restart

CI/CD is out of scope while the repo is a placeholder; at first code the image
build ties to the design document's
[digest/lockfile pinning](design.md#image-composition-bake-tools-inject-secrets)
and the "ship Dockerfiles, not prebuilt images" decision in [legal.md](legal.md).

## Credential onboarding (v0.6 / v0.7)

Providers are declared in `kagura-agent.toml` (gitignored, defense-in-depth). The
registry stores **references only** — a pointer to where the secret lives on the
host, never the secret value. Diagnose with `kagura-agent doctor` (add `--probe`
to dry-mint), and register providers with the operator-gated `kagura-agent setup`
wizard (it refuses to write a secret value).

### Secret-reference backends

A secret field is `<name><suffix>`; the **suffix** selects the host-side backend
the reference resolves through. Exactly one suffix per logical secret (two is an
ambiguous-config error). All resolution happens on the trusted host — the value
is leased into the container as an env var, never read inside it.

| Suffix | Reference value | Resolved from | Extra |
|---|---|---|---|
| `*_env` | an environment-variable **name** (e.g. `CF_TOKEN`) | the host env | — (stdlib) |
| `*_file` | a host **file path** (e.g. `/run/secrets/cf`) | the file's contents (trailing newline stripped) | — (stdlib) |
| `*_keyring` | a keychain key `"service/username"` | the host OS keychain | `kagura-agent[keyring]` |

```toml
# Resolve a Cloudflare parent token from the host OS keychain instead of an env
# var or file. The reference is "service/username"; the secret never appears here.
[providers.cf]
kind = "cloudflare"
account_id = "acct1"
parent_token_keyring = "kagura-cf/agent"
```

`*_keyring` needs the optional extra (`pip install 'kagura-agent[keyring]'`). If a
registry uses a `*_keyring` reference but the extra is absent, `kagura-agent
doctor` **WARNs** (it does *not* fail the gate — keyring availability is
host-dependent, so doctor may run on a different host than the agent): both the
`secret-backends` check and the per-provider check report the install hint as a
heads-up. If this host is also where the agent runs, the run then fail-closes
with the same hint — never a silent miss.

### `run --grant` (enforced in v0.7)

```
kagura-agent run "do the task" --grant aws:arn:aws:iam::123:role/agent --grant slack:chat:write
```

`--grant PROVIDER:SCOPE` (repeatable) is parsed into a default-deny, exact-match
`GrantSet` that is **enforced** at the credential chokepoint (`GrantedBroker`):
only the granted `(provider, scope)` pairs are reachable, and the run builds a
broker and acquires leases for **only** the granted providers. With **no
`--grant`, the run acquires no credentials at all** (default-deny — nothing is
minted, the registry is not even read). Leases are short-lived and released when
the run ends.

### Static long-lived tokens (Slack / Discord / Resend)

Some APIs only issue a long-lived static token — there is no short-lived mint.
Register these as a `static_env` provider. Because a standing secret violates the
membrane's no-standing-secret default, `static_env` is **fail-closed**: the
provider refuses to construct unless you explicitly accept the risk with
`standing_secret = true`. Use `value_env` (not `value_file`) for this kind — the
container env-var name is taken from `value_env`. **Always pair it with a tight
egress allowlist (see the note below the examples).**

```toml
[providers.slack]
kind = "static_env"
value_env = "SLACK_BOT_TOKEN"   # host env var holding the token; the container gets the same var
standing_secret = true          # explicit operator consent — required, else refused
# Contain it: allow egress only to slack.com / api.slack.com.

[providers.discord]
kind = "static_env"
value_env = "DISCORD_BOT_TOKEN"
standing_secret = true

[providers.resend]
kind = "static_env"
value_env = "RESEND_API_KEY"
standing_secret = true
```

**Contain the standing secret with egress.** A static token sits in the
container's environment for the task's lifetime — a hijacked agent holds it with
no expiry or revocation. Always pair `static_env` with a tight egress allowlist so
the token can only be used against the intended API (e.g. Resend → allow only
`api.resend.com`). The egress proxy (default-deny + allowlist) is the chokepoint
that makes a leaked standing token unexfiltratable.

## Memory reachability gate (startup)

Every `run` / `repl` is fail-closed on memory reachability: the host must be able
to mint a token via `kagura auth token` or the agent refuses to start (no silent
degrade). The access token is short-lived (~1h), so the first run after expiry
forces a refresh; to keep a transient hiccup at that hourly boundary from
hard-refusing the run, the gate **retries** the probe.

| Env var | Default | Effect |
|---|---|---|
| `KAGURA_MEMORY_PROBE_TIMEOUT` | `60` (s) | Per-attempt subprocess timeout for `kagura auth token`. |
| `KAGURA_MEMORY_PROBE_ATTEMPTS` | `3` | Probe attempts before refusing (clamped to ≥ 1). |
| `KAGURA_MEMORY_PROBE_BACKOFF` | `1.5` (s) | Wait between attempts. |

**Worst-case latency.** On a *hung* outage each attempt can cost the full timeout,
so the run path can block up to roughly `attempts × timeout + (attempts−1) ×
backoff` (~183 s with defaults) before refusing. To restore fast-fail on a known
outage, set `KAGURA_MEMORY_PROBE_ATTEMPTS=1` (and/or a lower
`KAGURA_MEMORY_PROBE_TIMEOUT`). `doctor` already probes **one-shot** so the command
you run to diagnose a memory outage stays fast. Non-finite values (`inf`/`nan`) for
the timeout/backoff are rejected and fall back to the default — they would
otherwise hang the gate.
