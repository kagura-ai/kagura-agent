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
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from kagura_agent.core.brain.kagura_brain_engine import (
    kagura_brain_available,
    resolve_kagura_brain_backend,
)
from kagura_agent.core.brain.sdk_engine import claude_sdk_available
from kagura_agent.core.brain.select import resolve_brain_backend
from kagura_agent.mcp.memory_cloud import memory_reachable
from kagura_agent.membrane.lease import Budget
from kagura_agent.membrane.registry import ProviderSpec, kind_schema, present_suffix_field
from kagura_agent.membrane.secret_source import (
    SECRET_SUFFIXES,
    EnvResolver,
    FileResolver,
    KeyringResolver,
    _read_file,
    _real_keyring_get_password,
    resolve_secret_field,
)
from kagura_agent.membrane.secret_source import (
    SecretSourceError as SecretRefError,
)

# Aliased on import: the `keyring_available` *parameter* of check_provider /
# check_secret_backends shadows the function name inside those bodies, so the
# live probe is referenced under a private alias (doctor's single keyring probe,
# delegating to the resolver's source of truth — no bare `import keyring` here).
from kagura_agent.membrane.secret_source import (
    keyring_available as _keyring_available,
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


def check_brain(
    *,
    sdk_available: bool,
    env: Mapping[str, str],
    backend: str = "sdk",
    kagura_brain_available: bool = False,
    kagura_backend: str = "claude",
    kagura_cli_present: bool = False,
) -> CheckResult:
    """The selected brain backend's dependency + (for the SDK) its auth mode.

    Backend-aware so doctor predicts the *actual* run: when ``KAGURA_AGENT_BRAIN``
    selects the kagura-brain backend, the dependency that matters is the ``brain``
    extra, not the claude SDK. ``backend`` defaults to ``sdk`` so the existing
    SDK-path behaviour (and its callers/tests) are unchanged.

    SDK path: the SDK being absent is a hard FAIL. Auth is softer:
    ``ANTHROPIC_API_KEY`` *overrides* subscription auth (README L127), a surprise
    worth a WARN; a bare ``CLAUDE_CODE_SUBSCRIPTION`` is the happy path; neither
    signal is "can't verify" (a CLI subscription cache may still exist) → WARN.
    """
    if backend == "kagura-brain":
        if not kagura_brain_available:
            return CheckResult(
                "brain",
                FAIL,
                "kagura-brain backend selected (KAGURA_AGENT_BRAIN) but the 'brain' "
                "extra is not installed",
                hint="install it: `uv run --extra brain ...` or "
                "`pip install 'kagura-agent[brain]'`",
            )
        if not kagura_cli_present:
            # The backend shells out to the claude/codex CLI; a missing binary is a
            # definitely-broken, run-blocking capability → FAIL, not WARN (doctor
            # predicts the run, matching kagura-engineer's check_claude/check_codex).
            return CheckResult(
                "brain",
                FAIL,
                f"kagura-brain backend selected but the {kagura_backend!r} CLI it "
                "shells out to is not on PATH",
                hint=f"install + log in the `{kagura_backend}` CLI",
            )
        return CheckResult(
            "brain",
            WARN,
            f"kagura-brain backend ({kagura_backend}) present; auth is via the "
            "underlying CLI (subscription/BYOK) and is not verifiable here",
            hint="confirm `claude` is logged in (claude backend) or KAGURA_BRAIN_API_KEY "
            "+ endpoint are set (codex/BYOK)",
        )
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


def _cli_on_path(name: str) -> bool:  # pragma: no cover - PATH lookup
    """Whether a CLI binary (`claude` / `codex`) the kagura-brain backend shells
    out to is resolvable on PATH."""
    return shutil.which(name) is not None


def _memory_reachable_once() -> bool:  # pragma: no cover - shells out to the kagura CLI
    """One-shot reachability probe for the diagnostic path.

    doctor is a fast preflight: the run path's bounded retry (which can multiply the
    60s probe timeout on a hung outage) would make the very command you run to
    *diagnose* a memory outage stall for minutes. ``attempts=1`` keeps doctor a
    single, fast probe — still fail-closed, just not retried."""
    return memory_reachable(attempts=1)


def run_doctor(
    *,
    memory_probe: Callable[[], bool] = _memory_reachable_once,
    sdk_probe: Callable[[], bool] = claude_sdk_available,
    kagura_brain_probe: Callable[[], bool] = kagura_brain_available,
    kagura_cli_probe: Callable[[str], bool] = _cli_on_path,
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
    # Materialize once: the registry is consumed by two checks below
    # (check_secret_backends + check_providers), so a one-shot generator would
    # silently make the second consumer see nothing.
    if registry is not None:
        registry = tuple(registry)
    # Probe the optional keyring extra once and share it, so the secret-backends
    # check and the per-provider checks render the SAME severity for a *_keyring
    # reference on a host without the extra (both WARN, never one WARN + one FAIL).
    keyring_present = _keyring_available()
    # Backend-aware brain check (doctor predicts the run). An invalid
    # KAGURA_AGENT_BRAIN / KAGURA_AGENT_BRAIN_BACKEND is reported as a brain FAIL
    # here, not a doctor crash. For the kagura-brain backend the underlying
    # claude/codex CLI binary is probed too (it is the real run-blocking dep).
    try:
        backend = resolve_brain_backend(resolved_env)
        if backend == "kagura-brain":
            kbackend = resolve_kagura_brain_backend(resolved_env)
            brain_result = check_brain(
                sdk_available=sdk_probe(),
                env=resolved_env,
                backend=backend,
                kagura_brain_available=kagura_brain_probe(),
                kagura_backend=kbackend,
                kagura_cli_present=kagura_cli_probe(kbackend),
            )
        else:
            brain_result = check_brain(
                sdk_available=sdk_probe(), env=resolved_env, backend=backend
            )
    except ValueError as exc:
        brain_result = CheckResult(
            "brain",
            FAIL,
            f"invalid brain backend config: {exc}",
            hint="set KAGURA_AGENT_BRAIN to 'sdk'/'kagura-brain' and "
            "KAGURA_AGENT_BRAIN_BACKEND to 'claude'/'codex' (or unset them)",
        )
    results = [
        check_memory(reachable=memory_probe()),
        brain_result,
        check_docker(available=docker_probe()),
        check_egress(configured=egress_probe()),
        check_secret_backends(registry, keyring_available=keyring_present),
    ]
    if registry is not None:
        renv = resolve_env if resolve_env is not None else resolved_env.get
        results.extend(
            check_providers(
                registry,
                resolve_env=renv,
                resolve_file=resolve_file,
                keyring_available=keyring_present,
            )
        )
    return results


# --------------------------------------------------------------------------
# Credential-aware checks (#59) — registry reference resolution + dry-mint
# --------------------------------------------------------------------------


def check_provider(
    spec: ProviderSpec,
    *,
    resolve_env: EnvResolver = os.environ.get,
    resolve_file: FileResolver = _read_file,
    resolve_keyring: KeyringResolver = _real_keyring_get_password,
    keyring_available: bool | None = None,
) -> CheckResult:
    """Diagnose whether a provider's secret references resolve on this host.

    A **required** reference that is absent, ambiguous (two or more of the
    ``*_env`` / ``*_file`` / ``*_keyring`` variants), or unresolvable is a FAIL
    with an actionable hint naming the env var / file path / keychain key — never
    the resolved secret value. An absent **optional** reference is fine (the
    provider falls back to ambient/default creds), so it stays OK. This mirrors
    the suffix-agnostic resolution the run path uses (#65), so doctor predicts the
    run. This is the pre-flight answer to "why can't the agent touch X?".

    A ``*_keyring`` reference on a host where the optional ``keyring`` extra is not
    installed is a **WARN**, not a FAIL: keyring availability is host-dependent (a
    deploy-time concern — doctor may run on a different host than the agent), and
    ``check_secret_backends`` carries the install hint. Escalating it to FAIL here
    would contradict that advisory and hard-fail the gate (#66).
    """
    if keyring_available is None:
        keyring_available = _keyring_available()  # pragma: no cover - real probe
    name = f"provider:{spec.name}"
    fails: list[str] = []
    warns: list[str] = []
    optional_absent: list[str] = []
    for ref in kind_schema(spec.kind).secrets:
        present, suffix_fields = present_suffix_field(ref, spec.fields)
        if len(present) > 1:
            keys = ", ".join(suffix_fields[suf] for suf in present)
            fails.append(f"{ref.name}: more than one of {keys} set (ambiguous)")
            continue
        if not present:
            if ref.required:
                all_fields = " / ".join(suffix_fields[suf] for suf in ref.suffixes)
                fails.append(f"{ref.name}: required but none of {all_fields} is set")
            else:
                optional_absent.append(ref.name)
            continue
        field = suffix_fields[present[0]]
        if present[0] == _KEYRING_SUFFIX and not keyring_available:
            # Can't verify a keyring ref without the extra — a deploy-time concern,
            # not a definitive failure. WARN (the run still fail-closes with the
            # same install hint if this host is also where the agent runs).
            warns.append(
                f"{ref.name}: {field} present but the 'keyring' extra is not installed "
                "on this host — install: pip install 'kagura-agent[keyring]'"
            )
            continue
        try:
            resolve_secret_field(
                field,
                str(spec.fields[field]),
                get_env=resolve_env,
                read_file=resolve_file,
                get_password=resolve_keyring,
            )
        except SecretRefError as exc:
            fails.append(str(exc))  # names the env var / path / keychain key, never the value

    if fails:
        return CheckResult(
            name,
            FAIL,
            f"{spec.kind}: {len(fails)} unresolved reference(s)",
            hint="; ".join(fails),
        )
    if warns:
        return CheckResult(
            name,
            WARN,
            f"{spec.kind}: {len(warns)} reference(s) not verifiable on this host",
            hint="; ".join(warns),
        )
    detail = f"{spec.kind}: all required references resolve"
    if optional_absent:
        detail += f" (optional not set: {', '.join(optional_absent)} — using ambient/default)"
    return CheckResult(name, OK, detail)


#: The keyring suffix, derived from SECRET_SUFFIXES so it stays in sync with the
#: resolver rather than being a hand-typed magic string.
_KEYRING_SUFFIX = SECRET_SUFFIXES[-1]


def check_secret_backends(
    registry: Iterable[ProviderSpec] | None = None,
    *,
    keyring_available: bool | None = None,
) -> CheckResult:
    """Report which secret-reference backends can resolve on this host.

    ``*_env`` and ``*_file`` are always available (stdlib). ``*_keyring`` needs
    the optional ``keyring`` extra. When the extra is **absent** AND the registry
    declares a ``*_keyring`` reference, that reference would fail at resolve time
    — so doctor WARNs ahead with the install hint rather than letting the run
    fail opaquely. A missing optional backend you are **not** using is not a
    problem (stays OK — no noise). WARN (not FAIL) is deliberate: whether keyring
    is needed is host-dependent (a deploy-time concern), and the real resolve
    still fail-closes with the same hint.
    """
    if keyring_available is None:
        keyring_available = _keyring_available()  # pragma: no cover - real probe
    if keyring_available:
        return CheckResult(
            "secret-backends", OK, "env + file + keyring references all resolvable"
        )
    uses_keyring = registry is not None and any(
        field.endswith(_KEYRING_SUFFIX) for spec in registry for field in spec.fields
    )
    if uses_keyring:
        return CheckResult(
            "secret-backends",
            WARN,
            "registry uses a *_keyring reference but the optional 'keyring' extra is not installed",
            hint="install: pip install 'kagura-agent[keyring]' "
            "(else *_keyring references fail to resolve at run time)",
        )
    return CheckResult(
        "secret-backends",
        OK,
        "env + file references resolvable "
        "(keyring extra not installed; no *_keyring reference in use)",
    )


def check_providers(
    registry: Iterable[ProviderSpec],
    *,
    resolve_env: EnvResolver = os.environ.get,
    resolve_file: FileResolver = _read_file,
    resolve_keyring: KeyringResolver = _real_keyring_get_password,
    keyring_available: bool | None = None,
) -> list[CheckResult]:
    """A reference-resolution :class:`CheckResult` per provider in the registry."""
    return [
        check_provider(
            spec,
            resolve_env=resolve_env,
            resolve_file=resolve_file,
            resolve_keyring=resolve_keyring,
            keyring_available=keyring_available,
        )
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
