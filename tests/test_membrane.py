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
