"""The auditable egress-proxy core (#94): its decision IS EgressPolicy.

These lock the proxy's enforcement semantics (default-deny, exact-host,
fail-closed) so the vendored proxy image can stay a thin I/O shell over tested
logic instead of an un-reviewable external binary.
"""

from kagura_agent.membrane.egress import EgressDecision, EgressPolicy
from kagura_agent.membrane.egress_proxy import (
    is_allowed,
    parse_connect_host,
    policy_from_label,
)

# --- policy_from_label: the per-run allowlist from the #92 label --------------


def test_policy_from_label_builds_allowlist() -> None:
    policy = policy_from_label("api.anthropic.com,memory.kagura-ai.com")
    assert policy.decide("api.anthropic.com") is EgressDecision.ALLOW
    assert policy.decide("memory.kagura-ai.com") is EgressDecision.ALLOW
    assert policy.decide("evil.example.com") is EgressDecision.DENY


def test_policy_from_label_blank_or_missing_is_default_deny() -> None:
    # Fail-closed: a container with no per-run allowlist label reaches nothing.
    for blank in (None, "", "   "):
        assert policy_from_label(blank).decide("api.anthropic.com") is EgressDecision.DENY


def test_policy_from_label_strips_and_ignores_empty_entries() -> None:
    policy = policy_from_label(" a.example.com , , b.example.com ")
    assert policy.decide("a.example.com") is EgressDecision.ALLOW
    assert policy.decide("b.example.com") is EgressDecision.ALLOW


def test_policy_from_label_malformed_entry_fails_closed() -> None:
    # A wildcard entry (which EgressPolicy rejects) must not crash the proxy — it
    # collapses to default-deny, never fail-open.
    policy = policy_from_label("*.evil.com,api.anthropic.com")
    assert policy.decide("api.anthropic.com") is EgressDecision.DENY  # whole label rejected
    assert policy.decide("anything.evil.com") is EgressDecision.DENY


# --- parse_connect_host -------------------------------------------------------


def test_parse_connect_host_extracts_authority() -> None:
    assert parse_connect_host("CONNECT api.anthropic.com:443 HTTP/1.1") == "api.anthropic.com:443"
    assert parse_connect_host("connect host:443 HTTP/1.1") == "host:443"  # method case-insensitive


def test_parse_connect_host_rejects_non_connect_and_malformed() -> None:
    assert parse_connect_host("GET http://evil.com/ HTTP/1.1") is None  # not a CONNECT
    assert parse_connect_host("CONNECT") is None  # no target
    assert parse_connect_host("") is None


# --- is_allowed: parse + decide, fail-closed ----------------------------------


def test_is_allowed_tunnels_only_allowlisted_hosts() -> None:
    policy = EgressPolicy(allow=("api.anthropic.com",))
    assert is_allowed(policy, "CONNECT api.anthropic.com:443 HTTP/1.1") is True
    assert is_allowed(policy, "CONNECT evil.example.com:443 HTTP/1.1") is False


def test_is_allowed_denies_unparseable_request_fail_closed() -> None:
    policy = EgressPolicy(allow=("api.anthropic.com",))
    assert is_allowed(policy, "GET / HTTP/1.1") is False  # not CONNECT → deny
    assert is_allowed(policy, "garbage") is False


def test_is_allowed_logs_the_decision_via_policy() -> None:
    # The proxy's decisions ride EgressPolicy's log — the audit trail the runbook
    # treats as the primary tripwire.
    policy = EgressPolicy(allow=("api.anthropic.com",))
    is_allowed(policy, "CONNECT api.anthropic.com:443 HTTP/1.1")
    is_allowed(policy, "CONNECT evil.example.com:443 HTTP/1.1")
    assert policy.log == [
        ("api.anthropic.com", EgressDecision.ALLOW),
        ("evil.example.com", EgressDecision.DENY),
    ]
