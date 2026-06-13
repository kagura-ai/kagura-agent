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


def _cf(revoked: list[str], *, fail: bool = False) -> CloudflareTokenProvider:
    def delete(handle: str) -> None:
        if fail:
            raise RuntimeError("404 token already gone")
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


async def test_sweep_forgets_when_present_provider_revoke_fails() -> None:
    # Preserved unwedge behavior: a provider that IS present but whose revoke
    # raises (poison handle) is forgotten so the sweep doesn't re-hit it forever.
    revoked: list[str] = []
    ledger = LeaseLedger()
    cf = _cf(revoked, fail=True)
    await _acquire_one(ledger, cf)

    await CredentialBroker({"cf": cf}, clock=lambda: 0.0, ledger=ledger).sweep()

    assert ledger.open_leases() == []


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
