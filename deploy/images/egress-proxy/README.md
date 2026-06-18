# Egress proxy (auditable reference) — #94

The egress chokepoint that brokers agent egress. Replaces the opaque
`ghcr.io/kagura-ai/egress-proxy:pinned-by-digest` reference with reviewable,
pinned, in-repo source.

## Why it's auditable

Every security decision is delegated to
[`kagura_agent.membrane.egress_proxy`](../../../src/kagura_agent/membrane/egress_proxy.py),
which is a thin layer over the **same `EgressPolicy`** the membrane uses to
validate launch specs. So the policy the proxy enforces at runtime is provably the
policy the launcher derived from the `LaunchSpec` — there is no second,
divergent implementation to trust. The decision core is unit-tested
(`tests/test_egress_proxy.py`); `proxy.py` is the I/O shell only.

## Enforcement contract

Mirrors `EgressPolicy` exactly:

| Property | Behaviour |
|---|---|
| **default-deny** | a host not on the allowlist → `403 Forbidden`, no tunnel |
| **exact-host** | no wildcard / subdomain matching; a `:port` variant is normalized |
| **fail-closed** | unparseable request, unresolvable source allowlist, or any error → deny |
| **log every decision** | one line per CONNECT: `egress <allow\|deny\|error> host=<h> source=<ip>` |
| **HTTPS only** | only `CONNECT` is brokered; cleartext HTTP forwarding is refused |

The log is the cockpit's primary egress tripwire (see `docs/operations.md`).

## Per-run allowlist (the #92 → #94 handoff)

The launcher stamps each egress-granted container with a `kagura.egress-allow`
label carrying that run's validated allowlist
(`membrane.egress.EGRESS_ALLOW_LABEL`). The **decision core**
(`policy_from_label`) already consumes that label and is unit-tested — so per-run
scoping is supported and audited.

What this reference shell wires today: it resolves the allowlist from the static
`EGRESS_ALLOWLIST` env (the compose bootstrap). Mapping a *source container* to
its per-run label needs a Docker-API lookup by source IP, which requires the proxy
to reach the Docker API — a deployment integration that is intentionally **not**
enabled by default. `_allowlist_for_source` in `proxy.py` is that seam: implement
the per-source lookup there and per-run least-privilege is live, with no change to
the audited decision core. Until then the proxy enforces the static allowlist
(empty → deny).

## Network placement

The proxy sits on **two** networks (see `deploy/compose.yml`): the internal
`agent-egress` (agent-facing, no upstream) and the external `egress-upstream`
(the proxy's own route to allowed hosts). Agents are attached to `agent-egress`
only, so they cannot reach the internet except through the proxy.

## Building & pinning

```sh
docker build -t kagura-agent-egress-proxy:local deploy/images/egress-proxy
```

Before deployment, pin the base digest (replace the all-zero placeholder in the
`Dockerfile`) — `kagura-agent doctor` reports `egress-proxy not pinned` while the
placeholder remains, the same discipline as `Dockerfile.base` / `Dockerfile.python`.
