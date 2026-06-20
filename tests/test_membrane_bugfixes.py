"""Regression tests for the sweeper fail-open token leak.

`sweep()` used to `forget()` any lease whose `release()` raised — including the
case where the lease's provider is no longer registered (config drift across a
restart). That forgot a still-valid stateful cloud token without ever revoking
it: a fail-open leak. The fix keeps an unrevokable lease tracked (so a later
sweep can revoke it) while preserving the original unwedge behavior for a
provider that IS present but whose revoke fails (e.g. a 404 poison handle).
"""

from __future__ import annotations

from typing import Any

import pytest

from kagura_agent.membrane.launcher import (
    AGENT_LABEL,
    LaunchSpec,
    MembraneViolation,
    docker_run_args,
    validate_spec,
)
from kagura_agent.membrane.lease import Budget, CredentialBroker, LeaseLedger
from kagura_agent.membrane.providers import CloudflareTokenProvider


class _HttpErr(Exception):
    """A minimal httpx.HTTPStatusError look-alike carrying ``.response.status_code``."""

    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.response = type("_R", (), {"status_code": status})()


def _cf(revoked: list[str], *, fail_status: int | None = None) -> CloudflareTokenProvider:
    def delete(handle: str) -> None:
        if fail_status is not None:
            raise _HttpErr(fail_status)
        revoked.append(handle)

    return CloudflareTokenProvider(
        create=lambda req: {"result": {"value": "tok", "id": "ID1"}},
        delete=delete,
    )


async def _acquire_one(ledger: LeaseLedger, provider: Any) -> None:
    broker = CredentialBroker({"cf": provider}, clock=lambda: 0.0, ledger=ledger)
    await broker.acquire("cf", scope="z", ttl=300, budget=Budget(3600))


async def test_sweep_keeps_lease_when_provider_missing() -> None:
    revoked: list[str] = []
    ledger = LeaseLedger()
    await _acquire_one(ledger, _cf(revoked))

    # Restart WITHOUT the cf provider registered.
    await CredentialBroker({}, clock=lambda: 0.0, ledger=ledger).sweep()

    assert revoked == []  # could not revoke (no provider)
    # ...so the token MUST remain tracked, not silently forgotten (leak).
    open_after = ledger.open_leases()
    assert len(open_after) == 1
    assert open_after[0].handle == "ID1"


async def test_sweep_revokes_once_provider_restored() -> None:
    revoked: list[str] = []
    ledger = LeaseLedger()
    cf = _cf(revoked)
    await _acquire_one(ledger, cf)

    # First sweep without the provider keeps it tracked...
    await CredentialBroker({}, clock=lambda: 0.0, ledger=ledger).sweep()
    # ...then a sweep with the provider restored revokes and forgets it.
    await CredentialBroker({"cf": cf}, clock=lambda: 0.0, ledger=ledger).sweep()

    assert revoked == ["ID1"]
    assert ledger.open_leases() == []


async def test_sweep_keeps_lease_on_a_transient_revoke_failure() -> None:
    # #124/#131: a TRANSIENT revoke failure (5xx/timeout) on a PRESENT provider must
    # KEEP the lease tracked — forgetting would drop a STILL-VALID credential (an
    # un-revocable leak). Consistent with renew() and the unknown-provider branch.
    revoked: list[str] = []
    ledger = LeaseLedger()
    cf = _cf(revoked, fail_status=503)  # transient
    await _acquire_one(ledger, cf)

    await CredentialBroker({"cf": cf}, clock=lambda: 0.0, ledger=ledger).sweep()

    open_after = ledger.open_leases()
    assert len(open_after) == 1 and open_after[0].handle == "ID1"  # kept, not dropped


async def test_sweep_forgets_a_handle_already_gone_at_the_provider() -> None:
    # #131: a PERMANENT failure (the provider returns 404/410 — the handle no longer
    # exists) is safe to forget: there is no live credential to leak, and forgetting
    # stops the sweep re-attempting a confirmed-dead handle every restart. This is the
    # poison-vs-transient distinction the #124 stopgap deferred.
    revoked: list[str] = []
    ledger = LeaseLedger()
    cf = _cf(revoked, fail_status=404)  # permanent (already gone)
    await _acquire_one(ledger, cf)

    await CredentialBroker({"cf": cf}, clock=lambda: 0.0, ledger=ledger).sweep()

    assert ledger.open_leases() == []  # forgotten — the handle is confirmed gone


# --- launcher: stamp the agent label so reconcile()/list() find containers ---

def test_docker_run_args_stamps_agent_label() -> None:
    args = docker_run_args(LaunchSpec(image="img:tag"))
    # The label DockerRuntime.list() filters on MUST be present, or reconcile()
    # and the hijack-containment kill path silently miss live agent containers.
    assert "--label" in args
    assert AGENT_LABEL in args
    i = args.index("--label")
    assert args[i + 1] == AGENT_LABEL


# --- launcher: egress allowlist is validated fail-closed at the gate ---------

def test_validate_spec_rejects_wildcard_egress() -> None:
    spec = LaunchSpec(image="img", egress_allow=("*.evil.com",))
    with pytest.raises(MembraneViolation, match="egress allowlist"):
        validate_spec(spec, project_root="/tmp")


def test_validate_spec_accepts_plain_host_egress() -> None:
    spec = LaunchSpec(image="img", egress_allow=("api.example.com",))
    assert validate_spec(spec, project_root="/tmp").egress_allow == ("api.example.com",)


def test_validate_spec_accepts_empty_egress() -> None:
    # Sealed (no egress) is the default and must remain valid.
    assert validate_spec(LaunchSpec(image="img"), project_root="/tmp").egress_allow == ()
