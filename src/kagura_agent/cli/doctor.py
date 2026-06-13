"""`kagura-agent doctor` — preflight environment check (#36).

A single place to answer "is this host ready to `run`?" before the cockpit loop
starts. The startup gate (`ensure_memory_reachable`) already fails closed on
memory, but everything else (the Claude SDK + its auth, docker, the egress
proxy) was only discoverable by *trying* to run. `doctor` surfaces all of it at
once with actionable hints.

Design mirrors the rest of the codebase: the check *logic* is pure — each
``check_*`` takes the already-probed value and never probes itself — so it is
unit-tested by passing booleans. The live probes (`run_doctor`'s defaults) are
thin shells exercised at deployment, like `memory_reachable()`.

Severity model:
- **FAIL** — a required capability is definitively missing (memory unreachable,
  SDK absent, docker down). Fails the gate (non-zero exit).
- **WARN** — present-but-unverifiable, or a deploy-time concern (auth cache not
  confirmable from env, egress proxy not provisioned locally). Does not fail.

So the gate is fail-closed on *definitely broken* and lenient on *can't verify
from here*, matching kagura-engineer's doctor.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from kagura_agent.core.brain.sdk_engine import claude_sdk_available
from kagura_agent.mcp.memory_cloud import memory_reachable

OK = "ok"
WARN = "warn"
FAIL = "fail"

#: doctor's failure exit code — distinct from `run`'s BrainUnavailable (3) and
#: argparse's usage error (2), so a wrapping script can tell them apart.
DOCTOR_FAIL_EXIT = 4

#: Default location of the egress-proxy compose file (relative to repo root).
_COMPOSE_PATH = Path("deploy/compose.yml")


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str  # OK | WARN | FAIL
    detail: str
    hint: str | None = None


def check_memory(*, reachable: bool) -> CheckResult:
    if reachable:
        return CheckResult("memory", OK, "reachable + authenticated via the kagura CLI")
    return CheckResult(
        "memory",
        FAIL,
        "memory-cloud not reachable/authenticated via the kagura CLI",
        hint="run `kagura auth login` on the host",
    )


def check_brain(*, sdk_available: bool, env: Mapping[str, str]) -> CheckResult:
    """SDK presence + which auth mode the env would resolve to.

    The SDK being absent is a hard FAIL (the brain cannot run). Auth is softer:
    ``ANTHROPIC_API_KEY`` *overrides* subscription auth (README L127), which is a
    surprise worth a WARN; a bare ``CLAUDE_CODE_SUBSCRIPTION`` is the happy path;
    and neither-signal is "can't verify" (a CLI subscription cache may still
    exist) → WARN, not FAIL.
    """
    if not sdk_available:
        return CheckResult(
            "brain",
            FAIL,
            "claude-agent-sdk (the 'claude' extra) is not installed",
            hint="install it: `uv run --extra claude ...` or `pip install 'kagura-agent[claude]'`",
        )
    if env.get("ANTHROPIC_API_KEY"):
        return CheckResult(
            "brain",
            WARN,
            "ANTHROPIC_API_KEY is set — it overrides subscription auth (key mode)",
            hint="unset ANTHROPIC_API_KEY to inherit the Pro/Max subscription instead",
        )
    if env.get("CLAUDE_CODE_SUBSCRIPTION"):
        return CheckResult("brain", OK, "claude SDK present; subscription auth")
    return CheckResult(
        "brain",
        WARN,
        "claude SDK present, but auth could not be verified from the environment",
        hint="run `claude` once interactively to confirm subscription login",
    )


def check_docker(*, available: bool) -> CheckResult:
    if available:
        return CheckResult("docker", OK, "docker daemon reachable")
    return CheckResult(
        "docker",
        FAIL,
        "docker daemon not reachable",
        hint="start Docker and ensure the cockpit user can reach the daemon",
    )


def check_egress(*, configured: bool) -> CheckResult:
    if configured:
        return CheckResult("egress", OK, "egress proxy compose file present")
    return CheckResult(
        "egress",
        WARN,
        f"egress proxy not provisioned locally ({_COMPOSE_PATH} not found)",
        hint="provision the default-deny egress proxy from deploy/compose.yml before deploying",
    )


def overall_status(results: list[CheckResult]) -> str:
    """FAIL if any check failed, else WARN if any warned, else OK."""
    statuses = {r.status for r in results}
    if FAIL in statuses:
        return FAIL
    if WARN in statuses:
        return WARN
    return OK


# --- live probes (thin; exercised at deployment) ----------------------------


def _docker_available() -> bool:  # pragma: no cover - shells out to docker
    """Whether the docker daemon answers. `docker info` exits non-zero if not."""
    try:
        proc = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _egress_configured(*, path: Path = _COMPOSE_PATH) -> bool:  # pragma: no cover - fs stat
    return path.is_file()


def run_doctor(
    *,
    memory_probe: Callable[[], bool] = memory_reachable,
    sdk_probe: Callable[[], bool] = claude_sdk_available,
    docker_probe: Callable[[], bool] = _docker_available,
    egress_probe: Callable[[], bool] = _egress_configured,
    env: Mapping[str, str] | None = None,
) -> list[CheckResult]:
    """Run every check, threading injected probes in. Probes default to live ones."""
    import os

    resolved_env = os.environ if env is None else env
    return [
        check_memory(reachable=memory_probe()),
        check_brain(sdk_available=sdk_probe(), env=resolved_env),
        check_docker(available=docker_probe()),
        check_egress(configured=egress_probe()),
    ]


_GLYPH = {OK: "OK ", WARN: "WARN", FAIL: "FAIL"}


def format_report(results: list[CheckResult]) -> str:
    lines = ["kagura-agent doctor"]
    for r in results:
        lines.append(f"  [{_GLYPH[r.status]}] {r.name}: {r.detail}")
        if r.hint is not None:
            lines.append(f"         ↳ {r.hint}")
    lines.append(f"overall: {overall_status(results).upper()}")
    return "\n".join(lines)
