"""The launcher: per-run `{image, creds, mount, egress}` → hardened `docker run`.

This module owns two things:
1. `validate_spec` — the membrane's hard gate. Rejects docker.sock mounts and
   any host path outside the project root *before* anything runs (fail-closed).
2. `docker_run_args` — bakes the container-hardening flags into every run, so a
   caller cannot forget them.

Credential leasing lives in `membrane.lease`; the launcher only carries the
already-leased, time-boxed env into the spec.
"""

from __future__ import annotations

import importlib.resources
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import PurePath

from kagura_agent.membrane.egress import EGRESS_ALLOW_LABEL, EGRESS_NETWORK, EgressPolicy

# Stamped on every launched container so the cockpit can reconstruct the session
# registry from Docker alone after a restart (docs/operations.md). MUST match the
# label `DockerRuntime.list()` filters on, or reconcile()/kill miss live agents.
AGENT_LABEL = "kagura-agent"


class MembraneViolation(RuntimeError):
    """A launch spec would breach the membrane; refuse to run it."""


@dataclass(frozen=True)
class Mount:
    source: str
    target: str
    read_only: bool = True


@dataclass(frozen=True)
class LaunchSpec:
    image: str
    mounts: tuple[Mount, ...] = ()
    egress_allow: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)


# Known container/host control sockets — reaching any of these from a container
# is escape to host root. Kept lower-cased for case-insensitive comparison; this
# list is for clear, specific errors, NOT the primary guard (see below).
_DANGEROUS_SOCKETS = frozenset(
    {
        "/var/run/docker.sock",
        "/run/docker.sock",
        "/run/containerd/containerd.sock",
        "/var/run/containerd/containerd.sock",
        "/run/podman/podman.sock",
        "/var/run/podman/podman.sock",
        "/var/run/crio/crio.sock",
        "/run/crio/crio.sock",
    }
)


def _dangerous_mount_reason(resolved: str) -> str | None:
    """Why this already-realpath-resolved source is too dangerous to mount, else None.

    The original guard matched the literal substring ``docker.sock`` case-
    sensitively, which missed ``containerd.sock``, ``DOCKER.SOCK``, and any other
    privileged socket. The fix does NOT lean on name matching either (the same
    fragility): the primary check is by **file type** — ``os.stat`` + ``S_ISSOCK``
    refuses any unix socket regardless of name (incl. extension-less D-Bus/systemd
    sockets). The canonical denylist and the ``*.sock`` suffix are belt-and-
    suspenders for clearer errors and for paths that don't exist / can't be
    stat'd (where type can't be checked).

    Scope: this validates the *resolved mount source itself*. Mounting a parent
    *directory* whose subtree merely contains a socket/device node is handled
    separately by `_subtree_special_file_reason` (defense-in-depth), in addition
    to the project-root containment check (`_is_within`) that keeps host control
    sockets — which live outside the workspace — out in the first place.
    """
    lowered = resolved.lower()
    if lowered in _DANGEROUS_SOCKETS:
        return "a known container control socket (= host root)"
    try:
        if stat.S_ISSOCK(os.stat(resolved).st_mode):
            return "a unix socket (host-reach vector)"
    except OSError:
        pass  # path missing / unstattable — fall through to the name-based guard
    if PurePath(lowered).name.endswith(".sock"):
        return "a unix socket (*.sock)"
    return None


# Bounds for the subtree scan below. A real workspace is well under these; a tree
# that exceeds either is refused (fail-closed) rather than scanned unbounded or
# silently passed. Sized to verify normal projects without letting a pathological
# (deep/huge) tree stall validation.
_MAX_SUBTREE_ENTRIES = 50_000
_MAX_SUBTREE_DEPTH = 64


def _subtree_special_file_reason(directory: str) -> str | None:
    """Why this directory's subtree is unsafe to bind-mount, else None.

    Defense-in-depth beyond `_dangerous_mount_reason` (which inspects only the
    mount *source* itself): a directory inside the project root could still hide
    a unix socket or device node (char/block) somewhere in its subtree, and a
    bind mount exposes those real inodes to the container — a host-reach / escape
    vector. We walk the subtree and refuse it if any such special file is found.

    Symlinks are NOT followed: a bind mount carries the actual inodes, and an
    absolute symlink inside it re-roots into the *container's* namespace at run
    time (so it is not itself a host-reach vector here); following them would also
    risk walking out of the tree or looping. Only real socket/device inodes in
    the tree matter.

    The walk is BOUNDED on both total entries and depth and FAILS CLOSED on a
    cap: a subtree too large/deep to fully verify is refused, never waved through
    as "clean", and the bound stops a pathological tree from stalling validation.
    """
    entries = 0
    stack: list[tuple[str, int]] = [(directory, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > _MAX_SUBTREE_DEPTH:
            return (
                f"subtree exceeds the {_MAX_SUBTREE_DEPTH}-level depth cap "
                "(cannot verify it is free of sockets/device nodes)"
            )
        try:
            with os.scandir(current) as it:
                for entry in it:
                    entries += 1
                    if entries > _MAX_SUBTREE_ENTRIES:
                        return (
                            f"subtree exceeds the {_MAX_SUBTREE_ENTRIES}-entry scan cap "
                            "(cannot verify it is free of sockets/device nodes)"
                        )
                    if entry.is_symlink():
                        continue  # see docstring: not a host-reach vector via bind mount
                    try:
                        mode = entry.stat(follow_symlinks=False).st_mode
                    except OSError:
                        # An entry we cannot stat is one we cannot clear → fail closed.
                        return f"contains an entry that could not be inspected: {entry.path!r}"
                    if stat.S_ISSOCK(mode):
                        return f"subtree contains a unix socket: {entry.path!r} (host-reach vector)"
                    if stat.S_ISCHR(mode) or stat.S_ISBLK(mode):
                        return f"subtree contains a device node: {entry.path!r} (host-reach vector)"
                    if entry.is_dir(follow_symlinks=False):
                        stack.append((entry.path, depth + 1))
        except OSError as e:
            # A directory we must verify but cannot list → fail closed.
            return f"subtree could not be fully scanned ({e})"
    return None


def _is_within(child: str, parent: str) -> bool:
    # Both paths are already realpath-resolved by validate_spec. Use the native
    # path flavor (PurePath) so containment is compared the same way realpath
    # produced the strings: PurePosixPath on the Linux host target (no change),
    # PureWindowsPath for cross-platform dev/test where realpath emits `C:\…`.
    child_p = PurePath(child)
    parent_p = PurePath(parent)
    return child_p == parent_p or parent_p in child_p.parents


def validate_spec(spec: LaunchSpec, *, project_root: str) -> LaunchSpec:
    """Validate a launch spec and return it with mounts resolved, fail-closed.

    Each mount source is resolved to its canonical realpath exactly ONCE; that
    resolved path is what gets validated (docker.sock guard + project-root
    containment) AND what is returned for the bind mount. Resolving once means the
    path validated is exactly the path mounted — there is no second, independent
    resolution that a symlink swap between validate and run could redirect (closes
    the validate->run TOCTOU). A symlink inside the project root pointing at /etc
    or docker.sock is caught here because the check runs on the resolved target.

    Any error while resolving/validating a path is converted to a
    MembraneViolation (fail-closed): the caller must never receive a raw OSError
    it could misread as a transient failure and retry into a launch.
    """
    resolved_mounts: list[Mount] = []
    for mount in spec.mounts:
        # Validate the target too (not only the source): it is concatenated raw
        # into `-v {source}:{target}{ro}`, so a colon opens a volume-options
        # section docker parses (e.g. "/workspace:rw" sneaks `rw` past the trusted
        # `:ro`), and a non-absolute target is malformed. Fail-closed before any
        # OS call, since this is a pure string check.
        if ":" in mount.target or not mount.target.startswith("/"):
            raise MembraneViolation(
                f"refusing mount target {mount.target!r}: must be a plain absolute "
                "container path (no ':' — it would inject volume options — and must "
                "start with '/')"
            )
        try:
            source = os.path.realpath(mount.source)
            root = os.path.realpath(project_root)
            reason = _dangerous_mount_reason(source)
            if reason is not None:
                raise MembraneViolation(
                    f"refusing to mount {mount.source!r} (-> {source!r}): {reason}"
                )
            if not _is_within(source, root):
                raise MembraneViolation(
                    f"refusing to mount {mount.source!r} (-> {source!r}): "
                    f"outside project root {project_root!r}"
                )
            # Defense-in-depth: a within-root *directory* could still hide a unix
            # socket or device node in its subtree (the per-path guard above only
            # checked the source itself). Walk it (bounded, fail-closed) and refuse.
            if os.path.isdir(source):
                subtree_reason = _subtree_special_file_reason(source)
                if subtree_reason is not None:
                    raise MembraneViolation(
                        f"refusing to mount {mount.source!r} (-> {source!r}): {subtree_reason}"
                    )
        except MembraneViolation:
            raise
        except Exception as e:
            raise MembraneViolation(
                f"refusing to mount {mount.source!r}: could not resolve/validate path ({e})"
            ) from e
        resolved_mounts.append(replace(mount, source=source))
    # Validate the egress allowlist at the membrane gate (fail-closed): reject
    # wildcard/subdomain or malformed entries HERE, not only if a proxy later
    # happens to build the policy. Use from_spec (not EgressPolicy(allow=...)) so
    # the launcher and the proxy derive the policy from the SAME spec — its whole
    # reason to exist is to keep them from drifting apart.
    try:
        EgressPolicy.from_spec(spec)
    except ValueError as e:
        raise MembraneViolation(f"refusing launch: invalid egress allowlist ({e})") from e
    return replace(spec, mounts=tuple(resolved_mounts))


# Hardening, baked in so no caller can forget it (README "Container hardening").
# Network mode is intentionally NOT here: it is egress-dependent and added
# per-spec in `docker_run_args` (sealed by default, proxy network when allowed).
_HARDENING = [
    "--user", "1000:1000",          # non-root
    "--cap-drop", "ALL",
    "--security-opt", "no-new-privileges",
    "--read-only",                  # read-only rootfs (+ tmpfs for scratch)
    "--tmpfs", "/tmp",
    "--pids-limit", "512",
    "--memory", "2g",
    "--cpus", "2",
]

# Defense-in-depth seccomp profile, layered on --cap-drop ALL above. Explicitly
# setting --security-opt seccomp=<profile> guarantees the high-risk syscalls
# (perf_event_open, bpf, …) are denied even on a daemon defaulting to
# seccomp=unconfined. Docker reads this path on the HOST it runs on (not inside
# the container). The profile ships as package data, so it resolves the same way
# whether kagura-agent is installed editable or as a wheel; deploy-configurable
# via KAGURA_SECCOMP_PROFILE. (importlib.resources.files returns a real fs path
# for a normally-installed/unzipped package, which docker needs; a zipapp/zip-
# import deploy would need KAGURA_SECCOMP_PROFILE set to an extracted path.)
_DEFAULT_SECCOMP_PROFILE = str(
    importlib.resources.files("kagura_agent.membrane") / "seccomp-agent.json"
)


def _seccomp_profile() -> str:
    # A set-but-empty/whitespace override is treated as unset → bundled default
    # (an empty seccomp= would make docker error out, not fail safe).
    profile = os.environ.get("KAGURA_SECCOMP_PROFILE", "").strip() or _DEFAULT_SECCOMP_PROFILE
    # Refuse the unconfined footgun: disabling seccomp would punch the very hole
    # this hardening exists to close, and is worse than the daemon default.
    if profile.strip().casefold() == "unconfined":
        raise MembraneViolation(
            "KAGURA_SECCOMP_PROFILE=unconfined would disable seccomp entirely; the "
            "membrane refuses to launch without a seccomp profile. Point it at a "
            "profile file instead (or unset it to use the bundled default)."
        )
    return profile


def _network_args(spec: LaunchSpec) -> list[str]:
    # Sealed by default. Only a non-empty egress allowlist attaches the container
    # to the proxy network; the proxy (EgressPolicy.from_spec) then enforces the
    # allowlist. Never host networking — that would bypass the proxy entirely.
    if spec.egress_allow:
        return ["--network", EGRESS_NETWORK]
    return ["--network", "none"]


#: Default endpoint the container is pointed at for HTTP(S)_PROXY when egress is
#: granted: the compose `egress-proxy` service on the agent-egress network, at the
#: conventional proxy port. Deploy-overridable via KAGURA_EGRESS_PROXY for a
#: different sidecar name/port. The proxy host/port is a deployment detail, so it
#: is configurable rather than hard-coded.
_DEFAULT_EGRESS_PROXY = "http://egress-proxy:3128"


def _egress_proxy_endpoint() -> str:
    # A set-but-blank override falls back to the default (an empty proxy URL would
    # disable proxying — the opposite of what egress enforcement wants).
    return os.environ.get("KAGURA_EGRESS_PROXY", "").strip() or _DEFAULT_EGRESS_PROXY


def _egress_enforcement_args(spec: LaunchSpec) -> list[str]:
    """Per-run egress enforcement, layered on the network seal (#92).

    Only for an egress-GRANTED spec (a sealed `--network none` run reaches nothing,
    so there is nothing to broker). Two mechanisms, both fail-closed by omission on
    a sealed run:

    1. **Per-run allowlist propagation** — stamp the container with its own
       ``EGRESS_ALLOW_LABEL`` (the validated `from_spec` allowlist), so the proxy
       enforces *this run's* hosts looked up per source container, instead of a
       single static compose-wide `EGRESS_ALLOWLIST`. This is the per-run
       least-privilege the membrane validates but the static env could not deliver.
    2. **App-layer proxy routing** — inject ``HTTP(S)_PROXY`` so even an app that
       ignores routing is funnelled through the proxy, defense-in-depth atop the
       network seal (`internal: true` on the proxy network). ``NO_PROXY`` keeps
       loopback direct.
    """
    if not spec.egress_allow:
        return []
    policy = EgressPolicy.from_spec(spec)
    endpoint = _egress_proxy_endpoint()
    args = ["--label", f"{EGRESS_ALLOW_LABEL}={policy.as_label()}"]
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        args += ["-e", f"{var}={endpoint}"]
    for var in ("NO_PROXY", "no_proxy"):
        args += ["-e", f"{var}=localhost,127.0.0.1"]
    return args


def docker_run_args(spec: LaunchSpec) -> list[str]:
    args = ["docker", "run", "--rm"]
    args += ["--label", AGENT_LABEL]  # so reconcile()/list() can find this container
    args += list(_HARDENING)
    args += ["--security-opt", f"seccomp={_seccomp_profile()}"]
    # Invariant: the produced args NEVER include host-reach escape flags
    # (--device, --add-host, --privileged, seccomp=unconfined). LaunchSpec
    # intentionally carries no `device`/`add_host` field; if one is ever added
    # it MUST be denied here rather than forwarded. Locked by a regression test.
    args += _network_args(spec)
    for mount in spec.mounts:
        ro = ":ro" if mount.read_only else ""
        # `mount.source` is already the canonical realpath (resolved once by
        # validate_spec). Do NOT re-resolve here: a second resolution is exactly
        # what a symlink swap between validate and run could redirect.
        args += ["-v", f"{mount.source}:{mount.target}{ro}"]
    for key, value in spec.env.items():
        args += ["-e", f"{key}={value}"]
    # Per-run egress enforcement (allowlist label + app-layer proxy env) — only when
    # egress is granted; sealed runs get nothing, by omission (fail-closed). Emitted
    # AFTER spec.env so the membrane's HTTP(S)_PROXY value wins on a duplicate key
    # (docker keeps the last -e): a caller-supplied proxy var in spec.env must never
    # be able to override and defeat the injected routing.
    args += _egress_enforcement_args(spec)
    # `--` terminates option parsing so the image (the trailing positional) is
    # always treated as the image, never as a docker flag — even if the value
    # begins with `-`. Defense-in-depth alongside the spec's no-escape-flags shape.
    args += ["--", spec.image]
    return args
