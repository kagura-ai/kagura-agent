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
from dataclasses import dataclass, field
from pathlib import PurePosixPath


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
    child_p = PurePosixPath(os.path.normpath(child))
    parent_p = PurePosixPath(os.path.normpath(parent))
    return child_p == parent_p or parent_p in child_p.parents


def validate_spec(spec: LaunchSpec, *, project_root: str) -> None:
    for mount in spec.mounts:
        if any(token in mount.source for token in _DANGEROUS_SOURCES):
            raise MembraneViolation(
                f"refusing to mount {mount.source!r}: docker.sock = host root"
            )
        if not _is_within(mount.source, project_root):
            raise MembraneViolation(
                f"refusing to mount {mount.source!r}: outside project root {project_root!r}"
            )


# Hardening, baked in so no caller can forget it (README "Container hardening").
_HARDENING = [
    "--user", "1000:1000",          # non-root
    "--cap-drop", "ALL",
    "--security-opt", "no-new-privileges",
    "--read-only",                  # read-only rootfs (+ tmpfs for scratch)
    "--tmpfs", "/tmp",
    "--pids-limit", "512",
    "--memory", "2g",
    "--cpus", "2",
    "--network", "none",            # egress only via the proxy sidecar, never host net
]


def docker_run_args(spec: LaunchSpec) -> list[str]:
    args = ["docker", "run", "--rm"]
    args += list(_HARDENING)
    for mount in spec.mounts:
        ro = ":ro" if mount.read_only else ""
        args += ["-v", f"{mount.source}:{mount.target}{ro}"]
    for key, value in spec.env.items():
        args += ["-e", f"{key}={value}"]
    args.append(spec.image)
    return args
