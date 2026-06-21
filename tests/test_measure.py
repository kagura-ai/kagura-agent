"""#165 S1: the host-side MEASURE producer.

Composes the two merged primitives into a run's outcome: ``input_trust`` from the
ProvenanceLog's ACTUAL captured tiers (fail-closed on an un-grounded session), and
``verified`` from an independent signal the agent cannot emit — a host-run check's
exit code or an operator approval — abstaining to unverified when neither exists.
"""

import pytest

from kagura_agent.patterns.erasure import ProvenanceLog
from kagura_agent.patterns.measure import measure_outcome


def _grounded(*tiers: str) -> ProvenanceLog:
    log = ProvenanceLog()
    log.record_grounding("s", [(f"m{i}", tier) for i, tier in enumerate(tiers)])
    return log


def test_exit_code_zero_with_trusted_grounding_is_verified_and_trusted() -> None:
    out = measure_outcome("tests", session_id="s", provenance=_grounded("trusted"), exit_code=0)
    assert out.verified is True
    assert out.input_trust == "trusted"
    assert out.source == "exit_code"
    assert out.category == "tests"


def test_quarantine_grounding_is_untrusted_even_when_the_check_passes() -> None:
    # The positive-assertion input-trust test (Δ2): a passing check does NOT launder
    # untrusted grounding into a trusted-input run.
    out = measure_outcome(
        "tests", session_id="s", provenance=_grounded("trusted", "quarantine"), exit_code=0
    )
    assert out.verified is True
    assert out.input_trust == "untrusted"


def test_ungrounded_session_input_trust_is_fail_closed() -> None:
    out = measure_outcome("tests", session_id="s", provenance=ProvenanceLog(), exit_code=0)
    assert out.input_trust == "untrusted"  # no recorded grounding -> not trusted


def test_nonzero_exit_code_is_not_verified() -> None:
    out = measure_outcome("tests", session_id="s", provenance=_grounded("trusted"), exit_code=1)
    assert out.verified is False
    assert out.source == "exit_code"


def test_no_verdict_abstains_to_unverified() -> None:
    # Fail-closed: with no independent verdict (e.g. a non-test task), the run is
    # UNVERIFIED — never assumed verified from the agent merely finishing.
    out = measure_outcome("research", session_id="s", provenance=_grounded("trusted"))
    assert out.verified is False
    assert out.source == "unverified"


def test_hitl_approval_is_verified() -> None:
    out = measure_outcome(
        "research", session_id="s", provenance=_grounded("trusted"), approved=True
    )
    assert out.verified is True
    assert out.source == "hitl_approval"


def test_hitl_denial_is_not_verified() -> None:
    out = measure_outcome(
        "research", session_id="s", provenance=_grounded("trusted"), approved=False
    )
    assert out.verified is False
    assert out.source == "hitl_approval"


def test_hitl_path_also_derives_input_trust_from_grounding() -> None:
    # The Δ2 invariant holds on the approval path too: an operator approval must not
    # launder untrusted grounding into a trusted-input outcome.
    out = measure_outcome(
        "research", session_id="s", provenance=_grounded("trusted", "quarantine"), approved=True
    )
    assert out.verified is True
    assert out.input_trust == "untrusted"


def test_supplying_both_verdicts_is_a_caller_error() -> None:
    with pytest.raises(ValueError, match="one independent verdict"):
        measure_outcome(
            "tests", session_id="s", provenance=_grounded("trusted"), exit_code=0, approved=True
        )


def test_exit_code_with_explicit_denial_is_a_caller_error() -> None:
    # Falsy boundary: exit_code=0 and approved=False are BOTH verdicts (`is not None`),
    # so this must still raise — a passing check must never silently override an
    # explicit operator denial (a `is not None` -> truthiness regression would).
    with pytest.raises(ValueError, match="one independent verdict"):
        measure_outcome(
            "tests", session_id="s", provenance=_grounded("trusted"), exit_code=0, approved=False
        )
