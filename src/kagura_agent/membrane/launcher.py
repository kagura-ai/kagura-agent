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

import os
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


_DANGEROUS_SOURCES = ("docker.sock",)


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
            if any(token in source for token in _DANGEROUS_SOURCES):
                raise MembraneViolation(
                    f"refusing to mount {mount.source!r} (-> {source!r}): docker.sock = host root"
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
