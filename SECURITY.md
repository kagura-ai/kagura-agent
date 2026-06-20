# Security Policy

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues, discussions, or pull requests.**

Report them privately through GitHub's **private vulnerability reporting**:

- Go to the repository's **Security** tab → **Report a vulnerability**
  ([open a report](https://github.com/kagura-ai/kagura-agent/security/advisories/new)).

Please include, as far as you can:

- the affected component (e.g. the credential membrane, the egress proxy, the cockpit HITL gate, a brain backend);
- a description of the issue and its impact;
- steps to reproduce or a proof of concept;
- any suggested remediation.

We will acknowledge your report, investigate, and keep you informed of the resolution. Please give us a reasonable window to fix the issue before any public disclosure.

## Scope

kagura-agent's threat model centers on **agent hijack via prompt injection** and the **credential membrane** that contains it (see the *Security membrane* section of the [README](README.md)). Reports that are especially in scope:

- bypasses of the credential default-deny / lease scoping / release-on-exit guarantees;
- egress-allowlist or egress-proxy bypasses;
- ways a hijacked run can escalate beyond its granted `{image, creds, mount, egress}` capability tuple;
- memory-provenance / trust-tier bypasses (treating untrusted, externally-ingested memory as trusted);
- secret leakage (a credential reaching a log, error message, checkpoint, or process output).

Out of scope: the documented **self-host single-user** trust model where Docker is a convenience boundary (a kernel 0-day defeating container isolation is an accepted residual risk for that mode — see the README's *Container hardening* note). A shared/SaaS lane would require microVM-class isolation and is a separate threat model.

## Supported versions

This project is a pre-1.0 implemented skeleton; security fixes land on `main`. Pin a commit if you need stability.
