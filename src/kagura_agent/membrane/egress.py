"""Egress policy: default-deny + allowlist + log.

Unlimited egress is the exfiltration path for a hijacked agent. The membrane
runs a single egress chokepoint that denies by default, allows only listed
hosts, and logs every decision. Non-allowlist egress is the strongest hijack
tripwire (consumed by the cockpit's tiered response in v0.3).
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kagura_agent.membrane.launcher import LaunchSpec

# The proxy sidecar's network. A container that is granted egress joins this
# network (instead of `--network none`) so its only reachable peer is the proxy,
# which enforces the allowlist. The name is shared with the launcher so the
# `docker run --network` flag and the policy stay in lockstep, and MUST match the
# `networks:` key the proxy sidecar is attached to in deploy/compose.yml.
EGRESS_NETWORK = "agent-egress"

#: Docker label carrying a container's PER-RUN egress allowlist. The launcher
#: stamps it on every egress-granted container so the proxy CAN enforce *that
#: run's* hosts (resolved by source container) instead of a single static
#: compose-wide list — the per-run least-privilege the membrane validates but the
#: static `EGRESS_ALLOWLIST` env could not deliver. The reference proxy's decision
#: core consumes it (membrane.egress_proxy.policy_from_label); wiring the
#: source→label lookup is the deploy integration seam (deploy/images/egress-proxy).
#: Sealed runs carry no label (they reach nothing).
EGRESS_ALLOW_LABEL = "kagura.egress-allow"


class EgressDecision(Enum):
    ALLOW = "allow"
    DENY = "deny"


def _normalize_host(host: str) -> str:
    """Canonicalize a host for exact allowlist matching.

    Lower-cases (hostnames are case-insensitive) and strips a trailing
    ``:port``. IPv6 literals are handled explicitly: bracketed forms keep only
    the address inside ``[...]`` (dropping any ``]:port`` suffix), while a bare
    IPv6 literal — which has several colons and no brackets — is left intact so
    it is never mistaken for ``host:port``. Only a single-colon ``host:port``
    (hostname or IPv4) has its port stripped.
    """
    host = host.strip().lower()
    if host.startswith("["):
        end = host.find("]")
        if end != -1:
            return host[1:end]
        return host
    if host.count(":") == 1:
        host = host.rsplit(":", 1)[0]
    return host


class EgressPolicy:
    """Default-deny egress allowlist with **exact host** matching.

    The allowlist matches a single canonical hostname exactly — subdomain and
    wildcard patterns are intentionally **not** supported. Both allow entries
    and the host passed to :meth:`decide` are run through :func:`_normalize_host`
    (lower-case + port strip) so a port or case variant of an allowed host is
    not silently denied. Wildcard-looking entries (containing ``*`` or a leading
    ``.``) are rejected at construction time — fail-closed, so a deployer cannot
    silently mis-set the allowlist and believe subdomain matching is in effect.
    """

    def __init__(self, allow: tuple[str, ...] = ()) -> None:
        normalized: list[str] = []
        for entry in allow:
            raw = entry.strip()
            if not raw:
                raise ValueError("egress allowlist entry is empty")
            # Validate the *normalized* value, not the raw one: a wildcard
            # hidden behind brackets (e.g. "[*.github.com]" → "*.github.com")
            # or a malformed/unclosed bracket ("[::1" → "[::1") must still be
            # rejected fail-closed, so the guard matches the documented contract.
            host = _normalize_host(raw)
            if (
                not host
                or "*" in host
                or host.startswith(".")
                or "[" in host
                or "]" in host
                # A comma is the label delimiter: `as_label` joins entries with
                # "," and the proxy's `policy_from_label` splits on "," (#119). A
                # comma in a host would be stored as ONE junk entry the gate denies,
                # yet the label round-trip would re-expand it into multiple ALLOWED
                # hosts — a fail-open allowlist bypass. Whitespace (incl. newlines)
                # never appears in a real host and would pollute the --label value.
                # Reject both at the same fail-closed gate so the launcher and the
                # proxy can never disagree about what this run may reach.
                or "," in host
                or any(c.isspace() for c in host)
            ):
                raise ValueError(
                    f"egress allowlist entry {entry!r} is not a plain exact "
                    "hostname; wildcard/subdomain patterns and malformed values "
                    "are not supported"
                )
            normalized.append(host)
        self._allow = frozenset(normalized)
        self.log: list[tuple[str, EgressDecision]] = []

    @classmethod
    def from_spec(cls, spec: LaunchSpec) -> EgressPolicy:
        """Derive the proxy's allowlist from a launch spec's egress element.

        Keeps the policy the proxy enforces and the network the launcher attaches
        sourced from the *same* 4-tuple, so they cannot drift apart.
        """
        return cls(allow=spec.egress_allow)

    def decide(self, host: str) -> EgressDecision:
        normalized = _normalize_host(host)
        decision = (
            EgressDecision.ALLOW if normalized in self._allow else EgressDecision.DENY
        )
        self.log.append((normalized, decision))
        return decision

    def as_label(self) -> str:
        """The allowlist as a deterministic, comma-joined string for the per-run
        ``EGRESS_ALLOW_LABEL``. Sorted so the same allowlist always renders the same
        label (stable across runs / reproducible in tests). Empty for a default-deny
        (no-host) policy."""
        return ",".join(sorted(self._allow))
