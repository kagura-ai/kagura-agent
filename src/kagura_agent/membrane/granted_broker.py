"""v0.7 (Tasks 3-4): GrantedBroker — enforce default-deny at the credential chokepoint.

The :class:`~kagura_agent.membrane.lease.CredentialBroker` is the single place a
credential is ever minted. :class:`GrantedBroker` wraps one and adds a
**task-scoped authorization layer**: it holds an immutable
:class:`~kagura_agent.membrane.registry.GrantSet` and rejects any
``(provider, scope)`` that is not granted **before** the inner broker is reached,
so an unauthorized pair never triggers a mint.

Why a wrapper, not a subclass or a broker edit: authorization (per-run, fixed)
and minting + the broker write-lock (always-on) are distinct concerns. Composing
keeps them in separate layers — ``GrantedBroker`` only ever *adds* a deny, so it
can never weaken the inner broker's ``_assert_scope_allowed`` write-lock. A
granted ``memory:write`` still has to pass that inner lock; the grant layer is an
**additional** gate, never a bypass.

The two mint entries are both gated:

- :meth:`acquire` checks the grant before delegating.
- :meth:`renew` **re-checks** — it is handed a caller-supplied :class:`Lease` and
  triggers a fresh mint, so without the re-check a compromised caller could
  fabricate a lease for an ungranted scope and widen its reach. This re-check is
  load-bearing, not merely defense-in-depth.
- :meth:`container_env` **re-checks** for the same reason — it materialises a
  caller-supplied lease's cred into the container env, so default-deny must gate
  cred *materialisation*, not only minting.

:meth:`release`, :meth:`open_leases`, and :meth:`sweep` do not mint or materialise
a cred (release is de-escalation; the others read/revoke), so they delegate straight
through.

The :class:`GrantSet` is supplied at construction and never re-bound — there is
no setter — so a compromised agent that somehow obtained a reference still could
not widen its own reach (the set is a ``frozenset`` and the broker exposes no
mutator). In the run path (#65) the broker lives host-side and the agent never
holds a handle at all.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from kagura_agent.membrane.lease import Budget, CredentialBroker, Lease
from kagura_agent.membrane.registry import GrantSet


class GrantDenied(RuntimeError):
    """A ``(provider, scope)`` was requested that the task's GrantSet does not allow.

    Raised before the inner broker is reached, so the denied pair never triggers
    a mint. Default-deny: an empty GrantSet denies everything.
    """


class GrantedBroker:
    """Wrap a :class:`CredentialBroker`, enforcing a task-scoped GrantSet.

    The ``grants`` are fixed at construction and never mutated — the broker
    exposes no method to widen them.
    """

    def __init__(self, inner: CredentialBroker, grants: GrantSet) -> None:
        self._inner = inner
        self._grants = grants

    def _require_grant(self, provider: str, scope: str) -> None:
        if not self._grants.allows(provider, scope):
            raise GrantDenied(
                f"grant denied: ({provider!r}, {scope!r}) is not in the task's "
                "GrantSet (default-deny — only explicitly granted scopes are reachable)"
            )

    async def acquire(self, provider: str, *, scope: str, ttl: int, budget: Budget) -> Lease:
        self._require_grant(provider, scope)
        return await self._inner.acquire(provider, scope=scope, ttl=ttl, budget=budget)

    async def renew(self, lease: Lease, *, ttl: int) -> Lease:
        # Re-check: renew trusts the caller-supplied lease and mints anew, so a
        # fabricated lease for an ungranted scope must be denied here.
        self._require_grant(lease.provider, lease.scope)
        return await self._inner.renew(lease, ttl=ttl)

    async def release(self, lease: Lease) -> None:
        # De-escalation (revoke) — never grant-gated; denying it would only leak.
        await self._inner.release(lease)

    def container_env(self, leases: Iterable[Lease]) -> dict[str, str]:
        # Re-check, like renew: container_env MATERIALIZES a lease's cred into the
        # container env, so a fabricated or de-scoped lease for an ungranted scope
        # must be denied here too. Default-deny must cover cred materialization, not
        # only minting — otherwise the one method that turns a lease into a live
        # credential would be the one gap in the invariant.
        leases = list(leases)
        for lease in leases:
            self._require_grant(lease.provider, lease.scope)
        return self._inner.container_env(leases)

    def open_leases(self) -> list[Lease]:
        return self._inner.open_leases()

    async def sweep(self) -> None:
        await self._inner.sweep()


@dataclass(frozen=True)
class LeaseRequest:
    """A planned (not yet minted) lease for one granted ``(provider, scope)``."""

    provider: str
    scope: str
    ttl: int
    budget_seconds: int


def lease_requests(
    grants: GrantSet, *, ttl: int, budget_seconds: int
) -> tuple[LeaseRequest, ...]:
    """Turn a GrantSet into a pure, deterministically-ordered tuple of LeaseRequests.

    One request per grant, sorted by ``(provider, scope)`` — the sort is required
    because ``frozenset`` iteration order is non-deterministic. An empty GrantSet
    yields an empty tuple: default-deny gives the empty plan for free. Pure — no
    I/O, no clock, no global state.
    """
    return tuple(
        LeaseRequest(provider=g.provider, scope=g.scope, ttl=ttl, budget_seconds=budget_seconds)
        for g in sorted(grants.grants, key=lambda g: (g.provider, g.scope))
    )
