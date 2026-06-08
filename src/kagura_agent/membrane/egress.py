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


class EgressDecision(Enum):
    ALLOW = "allow"
    DENY = "deny"


class EgressPolicy:
    def __init__(self, allow: tuple[str, ...] = ()) -> None:
        self._allow = frozenset(allow)
        self.log: list[tuple[str, EgressDecision]] = []

    @classmethod
    def from_spec(cls, spec: LaunchSpec) -> EgressPolicy:
        """Derive the proxy's allowlist from a launch spec's egress element.

        Keeps the policy the proxy enforces and the network the launcher attaches
        sourced from the *same* 4-tuple, so they cannot drift apart.
        """
        return cls(allow=spec.egress_allow)

    def decide(self, host: str) -> EgressDecision:
        decision = EgressDecision.ALLOW if host in self._allow else EgressDecision.DENY
        self.log.append((host, decision))
        return decision
