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
from pathlib import PurePosixPath

from kagura_agent.membrane.egress import EGRESS_NETWORK


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
    *directory* that merely contains a socket is governed by the project-root
    containment check (`_is_within`) — host control sockets live outside the
    workspace, so their parent dirs are rejected there. Recursively scanning a
    mounted subtree for sockets is a separate, broader hardening (tracked apart
    from this name/type-matching fix).
    """
    lowered = resolved.lower()
    if lowered in _DANGEROUS_SOCKETS:
        return "a known container control socket (= host root)"
    try:
        if stat.S_ISSOCK(os.stat(resolved).st_mode):
            return "a unix socket (host-reach vector)"
    except OSError:
        pass  # path missing / unstattable — fall through to the name-based guard
    if PurePosixPath(lowered).name.endswith(".sock"):
        return "a unix socket (*.sock)"
    return None


def _is_within(child: str, parent: str) -> bool:
    # Both paths are already realpath-resolved by validate_spec.
    child_p = PurePosixPath(child)
    parent_p = PurePosixPath(parent)
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
        except MembraneViolation:
            raise
        except Exception as e:
            raise MembraneViolation(
                f"refusing to mount {mount.source!r}: could not resolve/validate path ({e})"
            ) from e
        resolved_mounts.append(replace(mount, source=source))
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


def docker_run_args(spec: LaunchSpec) -> list[str]:
    args = ["docker", "run", "--rm"]
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
    args.append(spec.image)
    return args
