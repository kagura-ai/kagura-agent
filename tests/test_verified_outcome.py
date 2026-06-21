"""#165 S1: the forge-resistant `VerifiedOutcome` value object.

`verified=True` may originate ONLY from an independent signal the agent cannot
emit (a host-run check exit code, or an operator approval) — never from a
self-reported / unverified run. The default is fail-closed (UNVERIFIED), and
`input_trust` is derived from the actual trust tiers of the grounding memories,
failing closed on an empty/unknown provenance set.
"""

import dataclasses

import pytest

from kagura_agent.mcp.memory_cloud import QUARANTINE_TIER, TRUSTED_TIER
from kagura_agent.membrane.verified_outcome import VerifiedOutcome, derive_input_trust


def test_unverified_is_fail_closed_default() -> None:
    outcome = VerifiedOutcome.unverified("deploy", input_trust=TRUSTED_TIER)
    assert outcome.verified is False
    assert outcome.source == "unverified"
    assert outcome.category == "deploy"


def test_exit_code_zero_is_verified() -> None:
    outcome = VerifiedOutcome.from_exit_code(0, "tests", input_trust=TRUSTED_TIER)
    assert outcome.verified is True
    assert outcome.source == "exit_code"


def test_nonzero_exit_code_is_not_verified() -> None:
    outcome = VerifiedOutcome.from_exit_code(1, "tests", input_trust=TRUSTED_TIER)
    assert outcome.verified is False
    # a failed check is distinct from "no check ran": it still records its source.
    assert outcome.source == "exit_code"
    assert VerifiedOutcome.from_exit_code(137, "tests", input_trust=TRUSTED_TIER).verified is False


def test_negative_exit_code_is_not_verified() -> None:
    # A signal-terminated check (a raw negative code, or POSIX 128+N) is a failure,
    # never a pass — only exit 0 is verified.
    assert VerifiedOutcome.from_exit_code(-1, "tests", input_trust=TRUSTED_TIER).verified is False
    assert VerifiedOutcome.from_exit_code(-9, "tests", input_trust=TRUSTED_TIER).verified is False


def test_hitl_approval_is_verified() -> None:
    outcome = VerifiedOutcome.from_hitl_approval(True, "research", input_trust=TRUSTED_TIER)
    assert outcome.verified is True
    assert outcome.source == "hitl_approval"


def test_hitl_denial_is_not_verified() -> None:
    outcome = VerifiedOutcome.from_hitl_approval(False, "research", input_trust=TRUSTED_TIER)
    assert outcome.verified is False
    assert outcome.source == "hitl_approval"


def test_cannot_claim_verified_without_an_independent_source() -> None:
    # The core forge-resistance invariant: a run can never be marked verified
    # while its source is "unverified" (i.e. an agent self-report in disguise).
    with pytest.raises(ValueError, match="independent source"):
        VerifiedOutcome(verified=True, category="x", input_trust=TRUSTED_TIER, source="unverified")


def test_llm_judge_source_is_rejected() -> None:
    # An LLM judge is never an admissible live verdict source (#165 Δ2).
    with pytest.raises(ValueError, match="source"):
        VerifiedOutcome(verified=True, category="x", input_trust=TRUSTED_TIER, source="llm_judge")


def test_invalid_input_trust_is_rejected() -> None:
    with pytest.raises(ValueError, match="input_trust"):
        VerifiedOutcome(verified=False, category="x", input_trust="external", source="unverified")


def test_outcome_is_frozen() -> None:
    outcome = VerifiedOutcome.unverified("x", input_trust=TRUSTED_TIER)
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.verified = True  # type: ignore[misc]


def test_input_trust_trusted_only_when_all_grounding_trusted() -> None:
    assert derive_input_trust([TRUSTED_TIER, TRUSTED_TIER]) == TRUSTED_TIER


def test_input_trust_untrusted_if_any_grounding_untrusted() -> None:
    assert derive_input_trust([TRUSTED_TIER, QUARANTINE_TIER]) == "untrusted"


def test_input_trust_empty_provenance_is_untrusted() -> None:
    # Fail-closed: a run with no recorded grounding provenance cannot earn trust.
    assert derive_input_trust([]) == "untrusted"


def test_derived_input_trust_is_a_valid_constructor_input() -> None:
    # The two halves agree: derive_input_trust's outputs are always accepted by the
    # value object, and an untrusted-grounded run can still be verified by the host.
    untrusted = VerifiedOutcome.from_exit_code(
        0, "x", input_trust=derive_input_trust([QUARANTINE_TIER])
    )
    assert untrusted.verified is True
    assert untrusted.input_trust == "untrusted"

    trusted = VerifiedOutcome.from_hitl_approval(
        True, "x", input_trust=derive_input_trust([TRUSTED_TIER])
    )
    assert trusted.input_trust == TRUSTED_TIER
