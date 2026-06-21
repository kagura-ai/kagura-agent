"""#165 S1: the forge-resistant verified-outcome value object.

`verified=True` may originate ONLY from a signal the agent cannot emit — the exit
code of a host-run check, or an explicit operator approval — **never** from the
agent's own DoneEvent/SessionResult self-report. This encodes, at the type level,
the same rail graduation.py enforces ("verified, never self-reported", CSO M1) and
the empirical reason for it: a signal the agent can reach is forgeable
(ImpossibleBench, arXiv:2510.20270). The default is fail-closed (UNVERIFIED).

`input_trust` is the host-side classification of whether the memories that grounded
a run were all trusted-tier; it fails closed on an empty/unknown provenance set so a
run with no recorded grounding can never earn trust (the input-trust rail, CSO C1).
This object only *carries* that classification — gating graduation/reinforcement on
it is the consumer's job (#165 S2); ``GraduationEngine.should_propose`` is where the
input-trust rail is actually enforced.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from kagura_agent.mcp.memory_cloud import TRUSTED_TIER

#: The host-side input-trust value for a run that drew on any non-trusted memory.
UNTRUSTED_INPUT = "untrusted"

#: Sources that represent an independent, host-arbitrated verdict the agent cannot
#: emit. Only these may carry ``verified=True``.
_INDEPENDENT_SOURCES = ("exit_code", "hitl_approval")
#: ``"unverified"`` is the fail-closed default (abstain); never carries verified.
_VALID_SOURCES = (*_INDEPENDENT_SOURCES, "unverified")
_VALID_INPUT_TRUST = (TRUSTED_TIER, UNTRUSTED_INPUT)


@dataclass(frozen=True)
class VerifiedOutcome:
    """A host-derived run outcome. Construct via the classmethods, not directly.

    The invariants are enforced in ``__post_init__`` so they hold no matter how the
    object is built: an unknown ``source`` or ``input_trust`` is rejected, and
    ``verified=True`` is impossible unless ``source`` is an independent verdict.
    """

    verified: bool
    category: str
    input_trust: str
    source: str

    def __post_init__(self) -> None:
        if self.source not in _VALID_SOURCES:
            raise ValueError(f"unknown verified-outcome source: {self.source!r}")
        if self.input_trust not in _VALID_INPUT_TRUST:
            raise ValueError(
                f"input_trust must be one of {_VALID_INPUT_TRUST}, got {self.input_trust!r}"
            )
        if self.verified and self.source not in _INDEPENDENT_SOURCES:
            raise ValueError(
                "verified=True requires an independent source "
                f"({_INDEPENDENT_SOURCES}); a self-reported / unverified run is never verified"
            )

    @classmethod
    def unverified(cls, category: str, *, input_trust: str) -> VerifiedOutcome:
        """The fail-closed default: no independent verdict exists for this run."""
        return cls(verified=False, category=category, input_trust=input_trust, source="unverified")

    @classmethod
    def from_exit_code(cls, code: int, category: str, *, input_trust: str) -> VerifiedOutcome:
        """Verified iff a host-run check exited zero (the agent cannot emit this)."""
        return cls(
            verified=code == 0, category=category, input_trust=input_trust, source="exit_code"
        )

    @classmethod
    def from_hitl_approval(
        cls, approved: bool, category: str, *, input_trust: str
    ) -> VerifiedOutcome:
        """Verified iff a human operator approved the run."""
        return cls(
            verified=approved, category=category, input_trust=input_trust, source="hitl_approval"
        )


def derive_input_trust(tiers: Sequence[str]) -> str:
    """Classify a run's input trust from the trust tiers of its grounding memories.

    Returns ``TRUSTED_TIER`` only if the provenance set is non-empty AND every
    grounding memory was trusted-tier; an empty/unknown set, or any non-trusted
    source, yields ``UNTRUSTED_INPUT`` (fail-closed).
    """
    if not tiers:
        return UNTRUSTED_INPUT
    if all(tier == TRUSTED_TIER for tier in tiers):
        return TRUSTED_TIER
    return UNTRUSTED_INPUT
