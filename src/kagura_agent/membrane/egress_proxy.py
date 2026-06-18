"""Reference egress-proxy enforcement logic (#94) — the auditable core.

The egress proxy that brokers agent egress (``deploy/images/egress-proxy``) is a
thin socket server around THIS logic, which is itself a thin layer over the same
:class:`~kagura_agent.membrane.egress.EgressPolicy` the membrane validates launch
specs with. So the proxy's *runtime* decision provably matches the policy the
launcher derives from a ``LaunchSpec`` — the same four properties: default-deny,
exact-host, fail-closed, log-every-decision.

Keeping the decision here (in-package, unit-tested) instead of only in the deploy
image means the security-relevant logic is reviewable and covered. The image's
``proxy.py`` is the I/O shell only — socket accept, the CONNECT tunnel, and the
Docker-API lookup that maps a source container to its per-run
:data:`~kagura_agent.membrane.egress.EGRESS_ALLOW_LABEL` — and is the one
un-unit-tested part (a deployment edge, like ``DockerRuntime``).
"""

from __future__ import annotations

from kagura_agent.membrane.egress import EgressDecision, EgressPolicy


def policy_from_label(label_value: str | None) -> EgressPolicy:
    """Build the per-run policy from a container's ``kagura.egress-allow`` label.

    The launcher stamps that label with the run's validated allowlist (#92). A
    missing / blank label yields a default-deny (no-host) policy — **fail-closed**:
    a container we cannot attribute a per-run allowlist to reaches nothing. A
    malformed label (an entry ``EgressPolicy`` rejects, e.g. a wildcard) likewise
    collapses to default-deny rather than crashing the proxy — a parse failure must
    never fail *open*.
    """
    if not label_value or not label_value.strip():
        return EgressPolicy(allow=())
    hosts = tuple(h for h in (part.strip() for part in label_value.split(",")) if h)
    try:
        return EgressPolicy(allow=hosts)
    except ValueError:
        return EgressPolicy(allow=())  # malformed → deny all (fail-closed)


def parse_connect_host(request_line: str) -> str | None:
    """Extract the target authority from an HTTP ``CONNECT`` request line, else None.

    ``CONNECT host:443 HTTP/1.1`` → ``host:443`` (the port is stripped later by the
    policy's own normalization at decide-time). Returns ``None`` for a non-CONNECT
    method or a malformed line; the proxy treats ``None`` as deny (fail-closed) — an
    unparseable request is never tunnelled. CONNECT is the path that matters for an
    allowlist proxy: HTTPS traffic announces its target host in the clear here.
    """
    parts = request_line.split()
    if len(parts) < 2 or parts[0].upper() != "CONNECT":
        return None
    return parts[1] or None


def is_allowed(policy: EgressPolicy, request_line: str) -> bool:
    """Whether the proxy should tunnel this CONNECT request, fail-closed.

    An unparseable request line (``parse_connect_host`` → ``None``) is denied; a
    parseable one is decided by ``policy`` (which also logs the decision)."""
    host = parse_connect_host(request_line)
    if host is None:
        return False
    return policy.decide(host) is EgressDecision.ALLOW
