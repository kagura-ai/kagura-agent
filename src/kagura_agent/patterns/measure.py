"""#165 S1: the host-side MEASURE producer — a run + its verdict -> a VerifiedOutcome.

The composition point of the safety skeleton, run by the trusted host (CLI/cockpit),
never inside the agent container:

- ``input_trust`` is derived from the **actual** trust tiers the ``ProvenanceLog``
  captured for the session, so it reflects the real grounding rather than a
  trusted-only read path's vacuous "all trusted" (the design's Δ2 finding). It fails
  closed: an un-grounded session has no tiers -> ``derive_input_trust`` -> untrusted.
- ``verified`` comes ONLY from an independent signal the agent cannot emit — a
  host-run check's exit code, or an operator approval — and abstains to ``False``
  (UNVERIFIED) when neither is supplied. "The agent finished" is never a pass
  (ImpossibleBench, arXiv:2510.20270; the CSO-M1 rail in ``graduation.py``).
"""

from __future__ import annotations

from kagura_agent.membrane.verified_outcome import VerifiedOutcome, derive_input_trust
from kagura_agent.patterns.erasure import ProvenanceLog


def measure_outcome(
    category: str,
    *,
    session_id: str,
    provenance: ProvenanceLog,
    exit_code: int | None = None,
    approved: bool | None = None,
) -> VerifiedOutcome:
    """Build the host-arbitrated :class:`VerifiedOutcome` for a finished run.

    Pass the run's single independent verdict — ``exit_code`` (a host-run check; zero
    passes) OR ``approved`` (an operator decision) — or neither to abstain. Supplying
    both is a caller error: a run has exactly one verdict.
    """
    if exit_code is not None and approved is not None:
        raise ValueError(
            "a run has one independent verdict: pass exit_code OR approved, not both"
        )
    input_trust = derive_input_trust(provenance.tiers_for(session_id))
    if exit_code is not None:
        return VerifiedOutcome.from_exit_code(exit_code, category, input_trust=input_trust)
    if approved is not None:
        return VerifiedOutcome.from_hitl_approval(approved, category, input_trust=input_trust)
    return VerifiedOutcome.unverified(category, input_trust=input_trust)
