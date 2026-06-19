"""v0.2: the membrane. What crosses the container boundary, and what cannot.

The membrane guards *reach*, not in-container freedom. Hard invariants:
- `docker.sock` is never mounted (mounting it = host root).
- host FS is never exposed outside the project root.
- the produced `docker run` args bake in hardening (non-root, cap-drop ALL,
  no-new-privileges, read-only rootfs, no host net/pid/ipc).
- egress is default-deny + allowlist.
"""

import importlib.resources
import json
import socket

import pytest

from kagura_agent.membrane.egress import (
    EGRESS_ALLOW_LABEL,
    EGRESS_NETWORK,
    EgressDecision,
    EgressPolicy,
)
from kagura_agent.membrane.launcher import (
    LaunchSpec,
    MembraneViolation,
    Mount,
    _dangerous_mount_reason,
    _seccomp_profile,
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


def test_containerd_sock_mount_is_rejected() -> None:
    # The old substring guard only knew the literal "docker.sock"; other host
    # control sockets (containerd, podman, crio) were missed.
    # project_root contains the socket so ONLY the socket guard (not the
    # containment check) can reject it — isolating the behavior under test.
    spec = LaunchSpec(
        image="kagura-agent:python",
        mounts=(
            Mount(
                source="/run/containerd/containerd.sock",
                target="/run/containerd/containerd.sock",
            ),
        ),
    )
    with pytest.raises(MembraneViolation):
        validate_spec(spec, project_root="/run/containerd")


def test_uppercase_docker_sock_mount_is_rejected() -> None:
    # Case-insensitive mounts (e.g. /var/run/DOCKER.SOCK) defeated the old
    # case-sensitive substring check. project_root contains it so only the
    # socket guard can reject it.
    spec = LaunchSpec(
        image="kagura-agent:python",
        mounts=(Mount(source="/var/run/DOCKER.SOCK", target="/var/run/docker.sock"),),
    )
    with pytest.raises(MembraneViolation):
        validate_spec(spec, project_root="/var/run")


def test_sock_suffix_inside_project_root_is_rejected() -> None:
    # A *.sock even inside the project root is refused: the socket guard runs
    # before the containment check, so a symlink/hardlink to a host socket
    # placed inside the root cannot smuggle one in.
    spec = LaunchSpec(
        image="kagura-agent:python",
        mounts=(Mount(source="/work/project/app.sock", target="/w/app.sock"),),
    )
    with pytest.raises(MembraneViolation):
        validate_spec(spec, project_root="/work/project")


def test_unix_socket_without_sock_extension_is_rejected_by_type(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # The real fix is type-based, not name-based: an actual unix socket whose
    # name does NOT end in .sock (D-Bus/systemd-style) is still refused via
    # os.stat + S_ISSOCK — closing the same name-matching weakness the issue
    # is about. Lives inside project_root so only the socket-type guard can
    # reject it.
    if not hasattr(socket, "AF_UNIX"):  # pragma: no cover - e.g. Windows
        pytest.skip("AF_UNIX sockets not supported on this platform")
    sock_path = tmp_path / "bus"  # deliberately no .sock extension
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        srv.bind(str(sock_path))
        spec = LaunchSpec(image="x", mounts=(Mount(source=str(sock_path), target="/w"),))
        with pytest.raises(MembraneViolation):
            validate_spec(spec, project_root=str(tmp_path))
    finally:
        srv.close()


# --- subtree scan: a mounted *directory* must not hide a socket/device node ---
# Defense-in-depth (deferred from #17, coordinated with #18): the per-path guard
# above only inspects the mount source itself. A directory inside the project
# root that *contains* a unix socket or device node in its subtree is still a
# host-reach vector (the container can traverse to the inode), so validate_spec
# walks the subtree and refuses it. The walk is BOUNDED and FAILS CLOSED on a cap.


def _bind_unix_socket(path):  # type: ignore[no-untyped-def]
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(path))
    return srv


def test_workspace_dir_containing_a_sock_file_is_rejected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # A *.sock unix socket planted inside an otherwise-legitimate workspace dir
    # must sink the whole mount, not just be caught when mounted directly.
    if not hasattr(socket, "AF_UNIX"):  # pragma: no cover - e.g. Windows
        pytest.skip("AF_UNIX sockets not supported on this platform")
    workspace = tmp_path / "ws"
    (workspace / "sub").mkdir(parents=True)
    (workspace / "sub" / "code.py").write_text("print(1)\n", encoding="utf-8")
    srv = _bind_unix_socket(workspace / "sub" / "app.sock")
    try:
        spec = LaunchSpec(image="x", mounts=(Mount(source=str(workspace), target="/w"),))
        with pytest.raises(MembraneViolation, match="socket"):
            validate_spec(spec, project_root=str(tmp_path))
    finally:
        srv.close()


def test_workspace_dir_containing_an_extensionless_socket_is_rejected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # The detection is type-based, not name-based: an AF_UNIX socket whose name
    # does NOT end in .sock (D-Bus/systemd style) hidden in the subtree is still
    # refused via S_ISSOCK.
    if not hasattr(socket, "AF_UNIX"):  # pragma: no cover - e.g. Windows
        pytest.skip("AF_UNIX sockets not supported on this platform")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    srv = _bind_unix_socket(workspace / "bus")  # deliberately no .sock extension
    try:
        spec = LaunchSpec(image="x", mounts=(Mount(source=str(workspace), target="/w"),))
        with pytest.raises(MembraneViolation, match="socket"):
            validate_spec(spec, project_root=str(tmp_path))
    finally:
        srv.close()


def test_clean_workspace_dir_passes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # A directory whose subtree holds only regular files and subdirectories
    # validates without raising.
    workspace = tmp_path / "ws"
    (workspace / "a" / "b").mkdir(parents=True)
    (workspace / "a" / "main.py").write_text("x = 1\n", encoding="utf-8")
    (workspace / "a" / "b" / "data.json").write_text("{}\n", encoding="utf-8")
    spec = LaunchSpec(image="x", mounts=(Mount(source=str(workspace), target="/w"),))
    validate_spec(spec, project_root=str(tmp_path))  # must not raise


def test_subtree_scan_fails_closed_on_entry_cap(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    # A subtree too large to fully verify is REFUSED, not waved through: when the
    # bounded walk hits its entry cap it raises rather than returning "clean".
    from kagura_agent.membrane import launcher as L

    monkeypatch.setattr(L, "_MAX_SUBTREE_ENTRIES", 3)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    for i in range(10):
        (workspace / f"f{i}.py").write_text("\n", encoding="utf-8")
    spec = LaunchSpec(image="x", mounts=(Mount(source=str(workspace), target="/w"),))
    with pytest.raises(MembraneViolation, match="cap"):
        validate_spec(spec, project_root=str(tmp_path))


def test_subtree_scan_fails_closed_on_depth_cap(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    # A subtree deeper than the depth cap is likewise refused (fail-closed),
    # bounding the walk so a pathological tree cannot make validation hang.
    from kagura_agent.membrane import launcher as L

    monkeypatch.setattr(L, "_MAX_SUBTREE_DEPTH", 1)
    workspace = tmp_path / "ws"
    (workspace / "a" / "b" / "c").mkdir(parents=True)
    spec = LaunchSpec(image="x", mounts=(Mount(source=str(workspace), target="/w"),))
    with pytest.raises(MembraneViolation, match="cap"):
        validate_spec(spec, project_root=str(tmp_path))


def test_subtree_scan_flags_a_device_node_by_type(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Coordinate with #18 (device-node hardening): a char/block device node in a
    # mounted subtree is a host-reach vector just like a socket. Detection is by
    # file *type* (mknod needs root), so we fake a char-device entry and assert
    # the helper flags it.
    import os
    import stat as stat_mod

    from kagura_agent.membrane import launcher as L

    class _FakeStat:
        def __init__(self, mode: int) -> None:
            self.st_mode = mode

    class _FakeEntry:
        def __init__(self, path: str, mode: int) -> None:
            self.path = path
            self.name = os.path.basename(path)
            self._mode = mode

        def is_symlink(self) -> bool:
            return False

        def is_dir(self, *, follow_symlinks: bool = True) -> bool:
            return stat_mod.S_ISDIR(self._mode)

        def stat(self, *, follow_symlinks: bool = True) -> "_FakeStat":
            return _FakeStat(self._mode)

    class _FakeScandir:
        def __init__(self, entries):  # type: ignore[no-untyped-def]
            self._entries = entries

        def __enter__(self):  # type: ignore[no-untyped-def]
            return iter(self._entries)

        def __exit__(self, *exc) -> bool:  # type: ignore[no-untyped-def]
            return False

    def fake_scandir(path):  # type: ignore[no-untyped-def]
        return _FakeScandir([_FakeEntry(f"{path}/ttyS0", stat_mod.S_IFCHR | 0o600)])

    monkeypatch.setattr(os, "scandir", fake_scandir)
    assert L._subtree_special_file_reason("/ws") is not None


# --- the socket guard itself (cross-platform: tests the resolved-path check
# directly, independent of realpath / project-root containment) ---------------


def test_dangerous_mount_reason_flags_known_control_sockets() -> None:
    for p in (
        "/var/run/docker.sock",
        "/run/docker.sock",
        "/run/containerd/containerd.sock",
    ):
        assert _dangerous_mount_reason(p) is not None, p


def test_dangerous_mount_reason_is_case_insensitive() -> None:
    assert _dangerous_mount_reason("/var/run/DOCKER.SOCK") is not None


def test_dangerous_mount_reason_flags_any_sock_suffix() -> None:
    # Any *.sock is refused, even one that isn't a known control socket.
    assert _dangerous_mount_reason("/work/project/app.sock") is not None


def test_dangerous_mount_reason_allows_a_plain_directory() -> None:
    assert _dangerous_mount_reason("/work/project/src") is None


def test_dangerous_mount_reason_flags_socket_by_type_not_name(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    # The real fix is type-based: an extension-less unix socket (D-Bus/systemd
    # style) is caught via os.stat S_ISSOCK, not its name.
    import os
    import stat as stat_mod

    target = str(tmp_path / "bus")  # no .sock extension

    class _SockStat:
        st_mode = stat_mod.S_IFSOCK | 0o600

    real_stat = os.stat

    def fake_stat(path, *a, **k):  # type: ignore[no-untyped-def]
        if str(path) == target:
            return _SockStat()
        return real_stat(path, *a, **k)

    monkeypatch.setattr(os, "stat", fake_stat)
    assert _dangerous_mount_reason(target) is not None


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


# --- seccomp + host-reach escape-flag invariants (#18) -----------------------


def test_docker_run_args_sets_an_explicit_seccomp_profile() -> None:
    # Explicitly setting seccomp guarantees the profile applies even on a daemon
    # defaulting to seccomp=unconfined.
    args = docker_run_args(LaunchSpec(image="x"))
    joined = " ".join(args)
    assert "--security-opt" in args
    assert "seccomp=" in joined
    assert "seccomp=unconfined" not in joined
    # exactly one seccomp= flag (no accidental duplicate)
    assert sum(1 for a in args if a.startswith("seccomp=")) == 1


def test_empty_seccomp_env_falls_back_to_bundled_default(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # A set-but-empty override must not emit a bare `seccomp=` (docker errors);
    # it falls back to the bundled default.
    monkeypatch.setenv("KAGURA_SECCOMP_PROFILE", "   ")
    joined = " ".join(docker_run_args(LaunchSpec(image="x")))
    assert "seccomp=" in joined
    assert "seccomp= " not in joined + " "  # not an empty value
    assert "seccomp-agent.json" in joined


def test_seccomp_profile_is_deploy_configurable(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("KAGURA_SECCOMP_PROFILE", "/host/custom/seccomp.json")
    args = docker_run_args(LaunchSpec(image="x"))
    assert "seccomp=/host/custom/seccomp.json" in " ".join(args)


def test_seccomp_default_points_at_the_bundled_profile(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("KAGURA_SECCOMP_PROFILE", raising=False)
    # Package data → resolves the same way editable or wheel-installed.
    assert _seccomp_profile().replace("\\", "/").endswith(
        "kagura_agent/membrane/seccomp-agent.json"
    )


def test_seccomp_unconfined_is_refused(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # The env override must not be allowed to disable seccomp entirely.
    monkeypatch.setenv("KAGURA_SECCOMP_PROFILE", "unconfined")
    with pytest.raises(MembraneViolation, match="unconfined"):
        docker_run_args(LaunchSpec(image="x"))


def test_docker_run_args_never_emits_host_reach_escape_flags() -> None:
    # --device / --add-host / --privileged would punch through the membrane even
    # on --network none; the launcher must never produce them. LaunchSpec carries
    # no field that could; this locks that invariant.
    args = docker_run_args(
        LaunchSpec(image="x", env={"K": "v"}, egress_allow=("api.anthropic.com",))
    )
    joined = " ".join(args)
    assert "--device" not in args
    assert "--add-host" not in args
    assert "--privileged" not in args
    assert "seccomp=unconfined" not in joined


def test_bundled_seccomp_profile_denies_high_risk_syscalls() -> None:
    # Read the profile as package data (its canonical home), not a repo path.
    profile_text = (
        importlib.resources.files("kagura_agent.membrane")
        .joinpath("seccomp-agent.json")
        .read_text(encoding="utf-8")
    )
    profile = json.loads(profile_text)
    assert profile["defaultAction"] == "SCMP_ACT_ALLOW"  # defense-in-depth deny-list
    denied = {
        name
        for rule in profile["syscalls"]
        if rule["action"] == "SCMP_ACT_ERRNO"
        for name in rule["names"]
    }
    for syscall in ("perf_event_open", "ptrace", "bpf", "userfaultfd", "init_module", "mount"):
        assert syscall in denied, syscall


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


# --- egress IN-PATH enforcement (#92): per-run label + app-layer proxy ---------


def _env_pairs(args: list[str]) -> dict[str, str]:
    """Collect `-e KEY=VALUE` env pairs emitted into a docker run argv."""
    out: dict[str, str] = {}
    for flag, val in zip(args, args[1:], strict=False):
        if flag == "-e" and "=" in val:
            k, v = val.split("=", 1)
            out[k] = v
    return out


def _label_value(args: list[str], key: str) -> str | None:
    for flag, val in zip(args, args[1:], strict=False):
        if flag == "--label" and val.startswith(f"{key}="):
            return val.split("=", 1)[1]
    return None


def test_egress_granted_injects_proxy_env_and_per_run_label() -> None:
    spec = LaunchSpec(image="kagura-agent:python", egress_allow=("api.anthropic.com",))
    args = docker_run_args(spec)
    env = _env_pairs(args)
    # App-layer proxy routing (defense in depth atop the internal network seal).
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        assert env.get(var) == "http://egress-proxy:3128"
    assert env.get("NO_PROXY") == "localhost,127.0.0.1"
    # Per-run allowlist propagated as a label the proxy can enforce per source.
    assert _label_value(args, EGRESS_ALLOW_LABEL) == "api.anthropic.com"


def test_sealed_run_injects_no_proxy_env_or_egress_label() -> None:
    # A `--network none` run reaches nothing — no proxy env, no allowlist label.
    args = docker_run_args(LaunchSpec(image="kagura-agent:python", egress_allow=()))
    env = _env_pairs(args)
    assert "HTTP_PROXY" not in env and "http_proxy" not in env
    assert _label_value(args, EGRESS_ALLOW_LABEL) is None


def test_per_run_label_scopes_to_this_runs_allowlist_host_a_not_b() -> None:
    # The AC's regression: a run granted egress to host A must not be able to reach
    # host B. The per-run label carries ONLY A (sorted, deterministic), and the
    # policy the proxy derives from it allows A and denies B — so B is unreachable
    # in-path even though egress is granted.
    spec = LaunchSpec(image="x", egress_allow=("a-host.example.com",))
    args = docker_run_args(spec)
    assert _label_value(args, EGRESS_ALLOW_LABEL) == "a-host.example.com"  # B absent

    policy = EgressPolicy.from_spec(spec)
    assert policy.decide("a-host.example.com") is EgressDecision.ALLOW
    assert policy.decide("b-host.example.com") is EgressDecision.DENY


def test_per_run_label_is_sorted_and_deterministic() -> None:
    spec = LaunchSpec(image="x", egress_allow=("z.example.com", "a.example.com"))
    assert _label_value(docker_run_args(spec), EGRESS_ALLOW_LABEL) == (
        "a.example.com,z.example.com"
    )


def test_injected_proxy_wins_over_a_spec_env_proxy_var() -> None:
    # Fail-OPEN guard (code-review): a caller-supplied HTTP_PROXY in spec.env must
    # NOT override the membrane's injected routing. Enforcement env is emitted AFTER
    # spec.env, and docker keeps the LAST -e for a key, so the membrane value wins.
    spec = LaunchSpec(
        image="x",
        egress_allow=("api.anthropic.com",),
        env={"HTTP_PROXY": "http://attacker-controlled:9999"},
    )
    args = docker_run_args(spec)
    # both -e pairs are present; the membrane's is last → effective.
    pairs = [
        v
        for f, v in zip(args, args[1:], strict=False)
        if f == "-e" and v.startswith("HTTP_PROXY=")
    ]
    assert pairs[-1] == "HTTP_PROXY=http://egress-proxy:3128"


def test_egress_proxy_endpoint_is_env_overridable(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("KAGURA_EGRESS_PROXY", "http://proxy.internal:8080")
    spec = LaunchSpec(image="x", egress_allow=("api.anthropic.com",))
    assert _env_pairs(docker_run_args(spec))["HTTP_PROXY"] == "http://proxy.internal:8080"
    # blank override falls back to the default (an empty proxy would disable routing)
    monkeypatch.setenv("KAGURA_EGRESS_PROXY", "   ")
    assert _env_pairs(docker_run_args(spec))["HTTP_PROXY"] == "http://egress-proxy:3128"


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


def test_egress_rejects_comma_bearing_host_fail_closed() -> None:
    # #119: a comma is the label delimiter (as_label joins on ",", the proxy's
    # policy_from_label splits on ","). A host containing a comma must be rejected
    # at construction — otherwise it is stored as ONE junk entry the gate denies,
    # yet the label round-trip splits it into multiple ALLOWED hosts (fail-open).
    with pytest.raises(ValueError, match="not a plain exact"):
        EgressPolicy(allow=("api.anthropic.com,attacker.example.com",))


def test_egress_rejects_whitespace_in_host() -> None:
    # #119: whitespace (incl. newline) never appears in a real hostname and would
    # pollute the --label value (label injection). Reject at construction.
    for bad in ("a.com evil.com", "api.anthropic.com\nattacker.com", "a\tb.com"):
        with pytest.raises(ValueError, match="not a plain exact"):
            EgressPolicy(allow=(bad,))


def test_comma_host_cannot_smuggle_an_allowed_host_through_the_label() -> None:
    # #119 regression: the launcher gate and the proxy must agree. A comma-bearing
    # entry is rejected, so it can never be smuggled into the label and re-expanded
    # into an extra allowed host by the proxy's policy_from_label.
    from kagura_agent.membrane.egress_proxy import policy_from_label

    # A legitimate multi-host policy round-trips losslessly...
    legit = EgressPolicy(allow=("api.anthropic.com", "api.github.com"))
    rebuilt = policy_from_label(legit.as_label())
    assert rebuilt.decide("api.anthropic.com") is EgressDecision.ALLOW
    assert rebuilt.decide("api.github.com") is EgressDecision.ALLOW
    assert rebuilt.decide("attacker.example.com") is EgressDecision.DENY
    # ...but a comma-smuggled entry never even constructs, so the gate's DENY for
    # the smuggled host can no longer disagree with the proxy.
    with pytest.raises(ValueError):
        EgressPolicy(allow=("api.anthropic.com,attacker.example.com",))


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
