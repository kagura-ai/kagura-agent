"""v0.7 (Tasks 3-4): GrantedBroker — default-deny at the credential chokepoint.

`GrantedBroker` wraps a `CredentialBroker` and enforces a task-scoped `GrantSet`:
an ungranted ``(provider, scope)`` is rejected with `GrantDenied` **before** the
inner broker is reached, so an unauthorized pair never triggers a mint. The two
mint entries — ``acquire`` and ``renew`` — are both gated; ``renew`` re-checks
because it trusts a caller-supplied `Lease` and would otherwise let a fabricated
lease mint an ungranted scope. ``release`` / ``container_env`` / ``open_leases``
/ ``sweep`` do not mint and delegate straight through.

`lease_requests` turns a `GrantSet` into a pure, deterministically-ordered tuple
of `LeaseRequest`s (empty GrantSet → empty tuple: default-deny gives the empty
plan for free).
"""

import dataclasses

import pytest

from kagura_agent.membrane.granted_broker import (
    GrantDenied,
    GrantedBroker,
    LeaseRequest,
    lease_requests,
)
from kagura_agent.membrane.lease import Budget, CredentialBroker, Lease
from kagura_agent.membrane.providers import MemoryCloudProvider, MemoryWriteLocked
from kagura_agent.membrane.registry import Grant, GrantSet, parse_grants


def _memory_exchange(req):  # type: ignore[no-untyped-def]
    return {"access_token": f"tok-{req['scope']}"}


class FakeStatelessProvider:
    stateful = False

    def __init__(self) -> None:
        self.minted = 0

    async def mint(self, scope: str, ttl: int) -> tuple[str, str | None]:
        self.minted += 1
        return f"tok-{self.minted}", None

    async def revoke(self, handle: str | None) -> None:  # pragma: no cover
        raise AssertionError("stateless provider must never be revoked")


class FakeStatefulProvider:
    stateful = True

    def __init__(self) -> None:
        self.minted = 0
        self.revoked: list[str] = []

    async def mint(self, scope: str, ttl: int) -> tuple[str, str | None]:
        self.minted += 1
        return f"cf-{self.minted}", f"h-{self.minted}"

    async def revoke(self, handle: str | None) -> None:
        assert handle is not None
        self.revoked.append(handle)


def _clock() -> float:
    return 1000.0


def _granted(*specs: str) -> tuple[CredentialBroker, FakeStatelessProvider, GrantedBroker]:
    provider = FakeStatelessProvider()
    inner = CredentialBroker({"aws": provider}, clock=_clock)
    broker = GrantedBroker(inner, parse_grants(specs))
    return inner, provider, broker


# --------------------------------------------------------------------------
# acquire — the gated mint entry
# --------------------------------------------------------------------------


async def test_acquire_granted_scope_passes_through():
    _inner, provider, broker = _granted("aws:s3:read")
    lease = await broker.acquire("aws", scope="s3:read", ttl=300, budget=Budget(3600))
    assert lease.cred == "tok-1"
    assert provider.minted == 1


async def test_acquire_ungranted_scope_denied_before_inner_mints():
    # The core invariant: an ungranted pair never reaches the inner broker, so
    # the provider is never asked to mint.
    _inner, provider, broker = _granted("aws:s3:read")
    with pytest.raises(GrantDenied):
        await broker.acquire("aws", scope="s3:write", ttl=300, budget=Budget(3600))
    assert provider.minted == 0  # mint NOT reached


async def test_empty_grantset_denies_everything():
    _inner, provider, broker = _granted()  # default-deny
    with pytest.raises(GrantDenied):
        await broker.acquire("aws", scope="s3:read", ttl=300, budget=Budget(3600))
    assert provider.minted == 0


async def test_grant_is_exact_match_provider_and_scope():
    _inner, provider, broker = _granted("aws:s3:read")
    # Wrong scope and wrong provider both denied (exact-match, no prefix/wildcard).
    with pytest.raises(GrantDenied):
        await broker.acquire("aws", scope="s3:read:extra", ttl=300, budget=Budget(3600))
    with pytest.raises(GrantDenied):
        await broker.acquire("gcp", scope="s3:read", ttl=300, budget=Budget(3600))
    assert provider.minted == 0


# --------------------------------------------------------------------------
# renew — must re-check (fabricated-Lease bypass)
# --------------------------------------------------------------------------


async def test_renew_granted_lease_passes_through():
    _inner, provider, broker = _granted("aws:s3:read")
    lease = await broker.acquire("aws", scope="s3:read", ttl=300, budget=Budget(3600))
    renewed = await broker.renew(lease, ttl=300)
    assert renewed.cred == "tok-2"
    assert provider.minted == 2


async def test_renew_fabricated_ungranted_lease_is_denied_without_minting():
    # SECURITY: renew trusts the caller-supplied Lease. A fabricated lease for an
    # UNGRANTED (provider, scope) must be rejected BEFORE inner.renew mints —
    # otherwise a compromised caller widens its reach via renew.
    _inner, provider, broker = _granted("aws:s3:read")
    fabricated = Lease(
        provider="aws",
        scope="s3:write",  # NOT granted
        budget=Budget(3600),
        cred="forged",
        expires_at=2000.0,
        handle=None,
        stateful=False,
    )
    with pytest.raises(GrantDenied):
        await broker.renew(fabricated, ttl=300)
    assert provider.minted == 0  # inner.renew (and its mint) never reached


async def test_renew_fabricated_ungranted_provider_is_denied_without_minting():
    # The renew re-check must gate on the PROVIDER axis too, not just scope: a
    # fabricated lease naming an ungranted provider with an otherwise-granted
    # scope must be denied. A 'gcp' provider IS registered on the inner broker
    # (but not granted), so a dropped provider-check would reach inner.renew and
    # mint, rather than KeyError — making this test discriminating.
    aws = FakeStatelessProvider()
    gcp = FakeStatelessProvider()
    inner = CredentialBroker({"aws": aws, "gcp": gcp}, clock=_clock)
    broker = GrantedBroker(inner, parse_grants(["aws:s3:read"]))  # only aws:s3:read
    fabricated = Lease(
        provider="gcp",  # NOT granted (the scope is, but under a different provider)
        scope="s3:read",
        budget=Budget(3600),
        cred="forged",
        expires_at=2000.0,
        handle=None,
        stateful=False,
    )
    with pytest.raises(GrantDenied):
        await broker.renew(fabricated, ttl=300)
    assert gcp.minted == 0  # inner.renew (and its mint) never reached


# --------------------------------------------------------------------------
# release / container_env / open_leases / sweep — delegate (no mint)
# --------------------------------------------------------------------------


async def test_release_delegates_and_is_not_grant_gated():
    # release is de-escalation (revoke); it must delegate straight through —
    # grant-gating release would only cause leaks.
    provider = FakeStatefulProvider()
    inner = CredentialBroker({"cf": provider}, clock=_clock)
    broker = GrantedBroker(inner, parse_grants(["cf:zone:purge"]))
    lease = await broker.acquire("cf", scope="zone:purge", ttl=300, budget=Budget(3600))
    await broker.release(lease)
    assert provider.revoked == ["h-1"]  # inner.release reached


async def test_open_leases_delegates():
    provider = FakeStatefulProvider()
    inner = CredentialBroker({"cf": provider}, clock=_clock)
    broker = GrantedBroker(inner, parse_grants(["cf:zone:purge"]))
    await broker.acquire("cf", scope="zone:purge", ttl=300, budget=Budget(3600))
    assert len(broker.open_leases()) == 1


async def test_sweep_delegates_and_revokes_open_stateful_leases():
    provider = FakeStatefulProvider()
    inner = CredentialBroker({"cf": provider}, clock=_clock)
    broker = GrantedBroker(inner, parse_grants(["cf:zone:purge"]))
    await broker.acquire("cf", scope="zone:purge", ttl=300, budget=Budget(3600))
    await broker.sweep()
    assert provider.revoked == ["h-1"]


async def test_container_env_delegates():
    # container_env maps already-minted leases to env; it does not mint, so it
    # delegates straight (the cred already exists — no escalation possible).
    _inner, _provider, broker = _granted("aws:s3:read")
    lease = await broker.acquire("aws", scope="s3:read", ttl=300, budget=Budget(3600))
    # FakeStatelessProvider is not an EnvCredProvider, so env is empty but the
    # call must delegate without error.
    assert broker.container_env([lease]) == {}


# --------------------------------------------------------------------------
# GrantSet is agent-external / immutable — no widening after construction
# --------------------------------------------------------------------------


def test_grantedbroker_grantset_is_immutable_and_has_no_mutator():
    # A compromised agent must not widen its reach. The GrantSet is fixed at
    # construction — no setter/rebind — and the underlying grants are an
    # immutable frozenset on a frozen dataclass, so even a leaked reference
    # cannot be widened.
    _inner, _provider, broker = _granted("aws:s3:read")
    assert not any(hasattr(broker, m) for m in ("add_grant", "grant", "set_grants"))
    grants = broker._grants
    assert isinstance(grants.grants, frozenset)
    with pytest.raises(AttributeError):
        grants.grants.add(Grant("aws", "s3:write"))  # frozenset: no add
    with pytest.raises(dataclasses.FrozenInstanceError):
        grants.grants = frozenset()  # type: ignore[misc]  # GrantSet is frozen


async def test_granted_memory_write_still_hits_inner_write_lock():
    # Wrap-not-bypass: a GRANTED memory:write on a read-locked MemoryCloudProvider
    # must still raise the inner MemoryWriteLocked — the grant layer only ADDS a
    # deny, it never masks or weakens the inner _assert_scope_allowed write-lock.
    provider = MemoryCloudProvider(exchange=_memory_exchange)  # read-locked (default)
    inner = CredentialBroker({"memory": provider}, clock=_clock)
    broker = GrantedBroker(inner, parse_grants(["memory:memory:write"]))
    # grant passes (memory:write IS granted), so the failure must be the inner
    # write-lock — NOT GrantDenied.
    with pytest.raises(MemoryWriteLocked):
        await broker.acquire("memory", scope="memory:write", ttl=300, budget=Budget(3600))


# --------------------------------------------------------------------------
# lease_requests — pure, deterministic, default-deny
# --------------------------------------------------------------------------


def test_lease_requests_empty_grantset_is_empty():
    assert lease_requests(GrantSet(), ttl=300, budget_seconds=3600) == ()


def test_lease_requests_one_per_grant_with_ttl_and_budget():
    grants = parse_grants(["aws:s3:read", "cf:zone:purge"])
    reqs = lease_requests(grants, ttl=300, budget_seconds=3600)
    assert len(reqs) == 2
    assert all(isinstance(r, LeaseRequest) for r in reqs)
    assert all(r.ttl == 300 and r.budget_seconds == 3600 for r in reqs)


def test_lease_requests_deterministic_order_sorted_by_provider_then_scope():
    # frozenset iteration is non-deterministic, so the output MUST be sorted.
    grants = GrantSet(
        frozenset(
            {
                Grant("cf", "zone:purge"),
                Grant("aws", "s3:write"),
                Grant("aws", "s3:read"),
            }
        )
    )
    reqs = lease_requests(grants, ttl=300, budget_seconds=3600)
    assert [(r.provider, r.scope) for r in reqs] == [
        ("aws", "s3:read"),
        ("aws", "s3:write"),
        ("cf", "zone:purge"),
    ]


def test_lease_requests_is_pure_same_input_same_output():
    grants = parse_grants(["aws:s3:read", "cf:zone:purge"])
    assert lease_requests(grants, ttl=300, budget_seconds=3600) == lease_requests(
        grants, ttl=300, budget_seconds=3600
    )


def test_lease_request_is_frozen():
    req = LeaseRequest(provider="aws", scope="s3:read", ttl=300, budget_seconds=3600)
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.scope = "s3:write"  # type: ignore[misc]
