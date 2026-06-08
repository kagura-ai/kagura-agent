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
