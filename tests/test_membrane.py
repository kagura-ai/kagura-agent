"""v0.2: the membrane. What crosses the container boundary, and what cannot.

The membrane guards *reach*, not in-container freedom. Hard invariants:
- `docker.sock` is never mounted (mounting it = host root).
- host FS is never exposed outside the project root.
- the produced `docker run` args bake in hardening (non-root, cap-drop ALL,
  no-new-privileges, read-only rootfs, no host net/pid/ipc).
- egress is default-deny + allowlist.
"""

import pytest

from kagura_agent.membrane.egress import EGRESS_NETWORK, EgressDecision, EgressPolicy
from kagura_agent.membrane.launcher import (
    LaunchSpec,
    MembraneViolation,
    Mount,
    docker_run_args,
    validate_spec,
)

# --- mount guards ---------------------------------------------------------

def test_docker_sock_mount_is_rejected() -> None:
    spec = LaunchSpec(
        image="kagura-agent:python",
        mounts=(Mount(source="/var/run/docker.sock", target="/var/run/docker.sock"),),
    )
    with pytest.raises(MembraneViolation, match="docker.sock"):
        validate_spec(spec, project_root="/work/project")


def test_host_fs_outside_project_root_is_rejected() -> None:
    spec = LaunchSpec(
        image="kagura-agent:python",
        mounts=(Mount(source="/home/user/.ssh", target="/root/.ssh"),),
    )
    with pytest.raises(MembraneViolation, match="project root"):
        validate_spec(spec, project_root="/work/project")


def test_project_root_mount_is_allowed() -> None:
    spec = LaunchSpec(
        image="kagura-agent:python",
        mounts=(Mount(source="/work/project/src", target="/work/src"),),
    )
    validate_spec(spec, project_root="/work/project")  # must not raise


def test_validate_spec_fails_closed_on_unexpected_resolve_error(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # A guard must convert ANY error while resolving a mount path into a
    # MembraneViolation (fail-closed) — never leak a raw OSError the caller might
    # misread as a transient/system error and retry into a launch.
    def boom(_path: str) -> str:
        raise OSError("kernel says no")

    monkeypatch.setattr("os.path.realpath", boom)
    spec = LaunchSpec(image="x", mounts=(Mount(source="/work/project/src", target="/w"),))
    with pytest.raises(MembraneViolation):
        validate_spec(spec, project_root="/work/project")


def test_docker_run_args_injects_leased_creds_as_env() -> None:
    # Leased, time-boxed creds (e.g. the short-lived memory-cloud access token)
    # reach the container only via -e env injection — never baked into the image.
    spec = LaunchSpec(image="x", env={"KAGURA_TOKEN": "kmc-leased-1"})
    args = docker_run_args(spec)
    assert "-e" in args
    assert "KAGURA_TOKEN=kmc-leased-1" in args


def test_docker_run_args_mounts_source_verbatim(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Resolution happens once in validate_spec; docker_run_args must NOT re-resolve
    # (a second, independent resolution is exactly what a symlink swap between
    # validate and run could redirect). It mounts the source it is handed.
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    spec = LaunchSpec(image="x", mounts=(Mount(source=str(link), target="/w"),))

    args = docker_run_args(spec)

    assert f"{link}:/w:ro" in args  # verbatim, NOT resolved to `real`


# --- hardening flags in the run args --------------------------------------

def test_docker_run_args_bake_in_hardening() -> None:
    spec = LaunchSpec(image="kagura-agent:python", mounts=())
    args = docker_run_args(spec)
    joined = " ".join(args)

    assert "--user" in args  # non-root
    assert "--cap-drop" in args and "ALL" in args
    assert "--security-opt no-new-privileges" in joined
    assert "--read-only" in args
    assert "--network none" in joined or "--network" in args  # no host net by default
    assert "--pids-limit" in args
    # never the dangerous ones
    assert "docker.sock" not in joined
    assert "--privileged" not in args


# --- egress default-deny + allowlist --------------------------------------

def test_egress_denies_by_default() -> None:
    policy = EgressPolicy(allow=())
    assert policy.decide("evil.example.com") is EgressDecision.DENY


def test_egress_allows_listed_host() -> None:
    policy = EgressPolicy(allow=("api.anthropic.com",))
    assert policy.decide("api.anthropic.com") is EgressDecision.ALLOW
    assert policy.decide("evil.example.com") is EgressDecision.DENY


def test_egress_logs_every_decision() -> None:
    policy = EgressPolicy(allow=("api.anthropic.com",))
    policy.decide("api.anthropic.com")
    policy.decide("evil.example.com")
    assert policy.log == [
        ("api.anthropic.com", EgressDecision.ALLOW),
        ("evil.example.com", EgressDecision.DENY),
    ]


# --- egress wiring: the 4-tuple's `egress` element must be load-bearing -------
# Without an allowlist the network is fully sealed (`--network none`). With one,
# the container joins the egress proxy network so the sidecar brokers it against
# the allowlist; it must NOT fall back to `none` (that would silently drop the
# requested egress) nor to host networking (that would bypass the proxy).

def test_no_egress_seals_the_network() -> None:
    spec = LaunchSpec(image="kagura-agent:python", egress_allow=())
    joined = " ".join(docker_run_args(spec))
    assert "--network none" in joined


def test_egress_allowlist_joins_proxy_network_not_none() -> None:
    spec = LaunchSpec(image="kagura-agent:python", egress_allow=("api.anthropic.com",))
    joined = " ".join(docker_run_args(spec))
    assert f"--network {EGRESS_NETWORK}" in joined
    assert "--network none" not in joined
    assert "--network host" not in joined


def test_egress_policy_derived_from_spec_matches_allowlist() -> None:
    spec = LaunchSpec(image="x", egress_allow=("api.anthropic.com",))
    policy = EgressPolicy.from_spec(spec)
    assert policy.decide("api.anthropic.com") is EgressDecision.ALLOW
    assert policy.decide("evil.example.com") is EgressDecision.DENY


# --- egress host normalization (#19): exact match, but port/case-robust -------
# The allowlist is exact-host match by contract. Before matching we normalize
# both sides (strip port, lower-case) so a port variant or case variant of an
# allowed host is not silently denied (which would be a silent misconfiguration
# in the availability direction, and masks the allow intent in the hijack log).


def test_egress_strips_port_before_match() -> None:
    policy = EgressPolicy(allow=("api.github.com",))
    assert policy.decide("api.github.com:443") is EgressDecision.ALLOW


def test_egress_is_case_insensitive_on_both_sides() -> None:
    policy = EgressPolicy(allow=("API.GitHub.com",))
    assert policy.decide("api.github.COM") is EgressDecision.ALLOW


def test_egress_strips_port_from_allow_entry_too() -> None:
    # An operator who writes the port into the allowlist still matches the
    # bare host — both sides go through the same normalization.
    policy = EgressPolicy(allow=("api.github.com:443",))
    assert policy.decide("api.github.com") is EgressDecision.ALLOW


def test_egress_ipv6_bracket_port_is_stripped() -> None:
    policy = EgressPolicy(allow=("[::1]",))
    assert policy.decide("[::1]:443") is EgressDecision.ALLOW


def test_egress_bare_ipv6_has_no_port_to_strip() -> None:
    # A bare IPv6 literal (no brackets) has multiple colons and must NOT be
    # mistaken for host:port — the whole thing is the host.
    policy = EgressPolicy(allow=("2001:db8::1",))
    assert policy.decide("2001:db8::1") is EgressDecision.ALLOW


def test_egress_logs_the_normalized_host() -> None:
    # The hijack tripwire log (consumed by the v0.3 cockpit) records the
    # normalized host so port/case variants don't split into separate entries.
    policy = EgressPolicy(allow=("api.github.com",))
    policy.decide("API.github.com:443")
    assert policy.log == [("api.github.com", EgressDecision.ALLOW)]


def test_egress_rejects_wildcard_entry_fail_closed() -> None:
    with pytest.raises(ValueError, match="wildcard"):
        EgressPolicy(allow=("*.github.com",))


def test_egress_rejects_leading_dot_subdomain_entry() -> None:
    with pytest.raises(ValueError, match="wildcard"):
        EgressPolicy(allow=(".github.com",))


def test_egress_rejects_empty_allow_entry() -> None:
    with pytest.raises(ValueError):
        EgressPolicy(allow=("",))


def test_egress_rejects_bracketed_wildcard_entry() -> None:
    # A wildcard hidden behind brackets must not slip past the guard and
    # become a literal allowlist entry — the check runs post-normalization.
    with pytest.raises(ValueError, match="wildcard"):
        EgressPolicy(allow=("[*.github.com]",))


def test_egress_rejects_bracketed_leading_dot_entry() -> None:
    with pytest.raises(ValueError, match="wildcard"):
        EgressPolicy(allow=("[.github.com]",))


def test_egress_rejects_malformed_unclosed_bracket_entry() -> None:
    # An unclosed IPv6 bracket normalizes to itself; reject it rather than
    # store a bracket-bearing literal that can never match a real host.
    with pytest.raises(ValueError):
        EgressPolicy(allow=("[::1",))


# Security-critical direction: normalization must NOT widen the allow set.
# These DENY assertions pin a near-miss against each normalization step so a
# future "make it a suffix match" regression is caught.


def test_egress_port_strip_does_not_allow_a_non_allowed_host() -> None:
    policy = EgressPolicy(allow=("api.github.com",))
    assert policy.decide("evil.com:443") is EgressDecision.DENY


def test_egress_is_exact_not_suffix_or_superstring_match() -> None:
    policy = EgressPolicy(allow=("api.github.com",))
    assert policy.decide("api.github.com.evil.com") is EgressDecision.DENY
    assert policy.decide("notapi.github.com") is EgressDecision.DENY


def test_egress_trailing_dot_fqdn_is_denied() -> None:
    # Contract: exact match. A trailing-dot FQDN is a distinct string and is
    # denied (fail-safe over-deny), not silently treated as the bare host.
    policy = EgressPolicy(allow=("api.github.com",))
    assert policy.decide("api.github.com.") is EgressDecision.DENY


def test_egress_port_only_normalizes_to_empty_and_is_denied() -> None:
    policy = EgressPolicy(allow=("api.github.com",))
    assert policy.decide(":443") is EgressDecision.DENY
