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

import os
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from kagura_agent.core.brain.sdk_engine import claude_sdk_available
from kagura_agent.mcp.memory_cloud import memory_reachable
from kagura_agent.membrane.lease import Budget
from kagura_agent.membrane.registry import ProviderSpec, kind_schema
from kagura_agent.membrane.registry_io import (
    EnvResolver,
    FileResolver,
    SecretRefError,
    _read_file,
    resolve_secret_ref,
)

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
    registry: Iterable[ProviderSpec] | None = None,
    resolve_env: EnvResolver | None = None,
    resolve_file: FileResolver = _read_file,
) -> list[CheckResult]:
    """Run every check, threading injected probes in. Probes default to live ones.

    When ``registry`` is given, a per-provider reference-resolution check is
    appended for each spec (answers "why can't the agent touch <service>?"
    before runtime). Backward-compatible: with no registry, the result set is
    unchanged.
    """
    resolved_env = os.environ if env is None else env
    results = [
        check_memory(reachable=memory_probe()),
        check_brain(sdk_available=sdk_probe(), env=resolved_env),
        check_docker(available=docker_probe()),
        check_egress(configured=egress_probe()),
    ]
    if registry is not None:
        renv = resolve_env if resolve_env is not None else resolved_env.get
        results.extend(check_providers(registry, resolve_env=renv, resolve_file=resolve_file))
    return results


# --------------------------------------------------------------------------
# Credential-aware checks (#59) — registry reference resolution + dry-mint
# --------------------------------------------------------------------------


def check_provider(
    spec: ProviderSpec,
    *,
    resolve_env: EnvResolver = os.environ.get,
    resolve_file: FileResolver = _read_file,
) -> CheckResult:
    """Diagnose whether a provider's secret references resolve on this host.

    A **required** reference that is absent, ambiguous (both ``*_env`` and
    ``*_file``), or unresolvable is a FAIL with an actionable hint naming the env
    var / file path — never the resolved secret value. An absent **optional**
    reference is fine (the provider falls back to ambient/default creds), so it
    stays OK. This is the pre-flight answer to "why can't the agent touch X?".
    """
    name = f"provider:{spec.name}"
    fails: list[str] = []
    optional_absent: list[str] = []
    for ref in kind_schema(spec.kind).secrets:
        env_field, file_field = f"{ref.name}_env", f"{ref.name}_file"
        has_env, has_file = env_field in spec.fields, file_field in spec.fields
        if has_env and has_file:
            fails.append(f"{ref.name}: both {env_field} and {file_field} set (ambiguous)")
            continue
        field = env_field if has_env else file_field if has_file else None
        if field is None:
            if ref.required:
                fails.append(
                    f"{ref.name}: required but neither {env_field} nor {file_field} is set"
                )
            else:
                optional_absent.append(ref.name)
            continue
        try:
            resolve_secret_ref(
                field, str(spec.fields[field]), get_env=resolve_env, read_file=resolve_file
            )
        except SecretRefError as exc:
            fails.append(str(exc))  # names the env var / path, never the value

    if fails:
        return CheckResult(
            name,
            FAIL,
            f"{spec.kind}: {len(fails)} unresolved reference(s)",
            hint="; ".join(fails),
        )
    detail = f"{spec.kind}: all required references resolve"
    if optional_absent:
        detail += f" (optional not set: {', '.join(optional_absent)} — using ambient/default)"
    return CheckResult(name, OK, detail)


def check_providers(
    registry: Iterable[ProviderSpec],
    *,
    resolve_env: EnvResolver = os.environ.get,
    resolve_file: FileResolver = _read_file,
) -> list[CheckResult]:
    """A reference-resolution :class:`CheckResult` per provider in the registry."""
    return [
        check_provider(spec, resolve_env=resolve_env, resolve_file=resolve_file)
        for spec in registry
    ]


def _probe_scope(spec: ProviderSpec) -> str | None:
    """A safe, minimal scope to dry-mint for a provider, or ``None`` to skip.

    ``memory_cloud`` is probed read-only (never ``memory:write`` — that trips the
    device-flow HITL / fail-closed lock). ``aws_sts`` probes its role ARN. Other
    kinds need a deployment-specific scope not derivable from the registry alone,
    so they are skipped.
    """
    if spec.kind == "memory_cloud":
        return "memory:read"
    if spec.kind == "aws_sts":
        return str(spec.fields["role_arn"])
    return None


async def probe_provider(broker: object, name: str, *, scope: str, ttl: int = 60) -> CheckResult:
    """Opt-in dry-mint: acquire a short-lived scoped token, then revoke it.

    A mint failure is a FAIL (the provider cannot actually issue a cred). A mint
    that succeeds but **cannot be revoked** is a loud FAIL that surfaces the
    lease handle for manual cleanup — a live token must never leak silently. The
    resolved token value is never placed in the output.
    """
    cname = f"probe:{name}"
    try:
        lease = await broker.acquire(name, scope=scope, ttl=ttl, budget=Budget(ttl))  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 — any mint failure is a diagnostic FAIL
        # Mint failed → no token was issued, so the provider's error (e.g. STS
        # AccessDenied) is the actionable answer to "why can't the agent touch X".
        return CheckResult(cname, FAIL, f"dry-mint failed for scope {scope!r}", hint=str(exc))
    handle = getattr(lease, "handle", None)
    try:
        await broker.release(lease)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        # {exc} is omitted on purpose: a revoke error could echo the just-minted
        # token. The handle (a revoke id, not the secret) is what the operator
        # needs for manual cleanup.
        return CheckResult(
            cname,
            FAIL,
            f"minted OK for scope {scope!r} but REVOKE FAILED — token may be live until it expires",
            hint=f"revoke manually with handle={handle!r}",
        )
    except BaseException:
        # On interrupt (Ctrl-C / SystemExit) a minted token must not vanish
        # silently: surface the handle to stderr for manual cleanup, then let the
        # interrupt propagate — we do not swallow control-flow exceptions.
        sys.stderr.write(
            f"{cname}: minted but NOT revoked (interrupted) — revoke manually "
            f"with handle={handle!r}\n"
        )
        raise
    return CheckResult(cname, OK, f"dry-mint + revoke OK for scope {scope!r}")


_GLYPH = {OK: "OK ", WARN: "WARN", FAIL: "FAIL"}


def format_report(results: list[CheckResult]) -> str:
    lines = ["kagura-agent doctor"]
    for r in results:
        lines.append(f"  [{_GLYPH[r.status]}] {r.name}: {r.detail}")
        if r.hint is not None:
            lines.append(f"         ↳ {r.hint}")
    lines.append(f"overall: {overall_status(results).upper()}")
    return "\n".join(lines)
