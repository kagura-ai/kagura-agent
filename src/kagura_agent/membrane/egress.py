"""Egress policy: default-deny + allowlist + log.

Unlimited egress is the exfiltration path for a hijacked agent. The membrane
runs a single egress chokepoint that denies by default, allows only listed
hosts, and logs every decision. Non-allowlist egress is the strongest hijack
tripwire (consumed by the cockpit's tiered response in v0.3).
"""

from __future__ import annotations

from enum import Enum


class EgressDecision(Enum):
    ALLOW = "allow"
    DENY = "deny"


class EgressPolicy:
    def __init__(self, allow: tuple[str, ...] = ()) -> None:
        self._allow = frozenset(allow)
        self.log: list[tuple[str, EgressDecision]] = []

    def decide(self, host: str) -> EgressDecision:
        decision = EgressDecision.ALLOW if host in self._allow else EgressDecision.DENY
        self.log.append((host, decision))
        return decision
