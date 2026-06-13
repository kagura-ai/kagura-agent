"""Pending HITL approvals — the async half of the cockpit's approval loop (#32).

`HitlGate.review()` is synchronous (ask → wait → decide in one call). That fits
an inline button prompt, but not the device-flow #14 needs: surface a request
now, the operator approves in a *separate later* `intent=approve` message, then
the pending request resolves. This registry bridges that gap.

It is the **producer-seam primitive**: a caller registers a `CapabilityRequest`
and gets back an `asyncio.Future[Decision]` it can await *outside* the cockpit's
single serve loop; a later approve/deny event resolves the future. Fail-closed:
a pending that times out denies. v0.3 allows one pending per thread.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass

from kagura_agent.cockpit.hitl import CapabilityRequest, Decision


class PendingApprovalExists(RuntimeError):
    """The thread already has a pending approval (v0.3: one per thread)."""


@dataclass
class _Pending:
    request: CapabilityRequest
    future: asyncio.Future[Decision]
    created_at: float


class PendingApprovalRegistry:
    """Tracks capability requests awaiting an operator decision, keyed by thread.

    Expiry is **lazy**: a timed-out pending is observed (and denied) on the next
    `pending()`/`resolve()` touch, so no background timer is needed in the
    cockpit's single-threaded serve loop. A caller that needs a hard bound should
    still `asyncio.wait_for(future, timeout)` on its side.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        ttl_seconds: float = 300.0,
    ) -> None:
        self._clock = clock
        self._ttl = ttl_seconds
        self._pending: dict[str, _Pending] = {}

    def register(self, request: CapabilityRequest) -> asyncio.Future[Decision]:
        """Register a pending request and return the future its decision resolves.

        Non-blocking: the caller awaits the future *outside* the serve loop, so
        the loop stays free to process the later approve/deny event. Raises
        `PendingApprovalExists` if the thread already has a live pending request.
        """
        self._purge_if_expired(request.thread_id)
        if request.thread_id in self._pending:
            raise PendingApprovalExists(
                f"thread {request.thread_id!r} already has a pending approval"
            )
        future: asyncio.Future[Decision] = asyncio.get_running_loop().create_future()
        self._pending[request.thread_id] = _Pending(
            request=request, future=future, created_at=self._clock()
        )
        return future

    def pending(self, thread_id: str) -> bool:
        self._purge_if_expired(thread_id)
        return thread_id in self._pending

    def resolve(self, thread_id: str, *, approved: bool) -> CapabilityRequest | None:
        """Resolve the thread's pending request, returning it (or None if none).

        Returns None when there is no live pending request — including the case
        where it had already expired (fail-closed: a late approval grants nothing).
        """
        self._purge_if_expired(thread_id)
        entry = self._pending.pop(thread_id, None)
        if entry is None:
            return None
        if not entry.future.done():
            entry.future.set_result(Decision(approved=approved))
        return entry.request

    def _purge_if_expired(self, thread_id: str) -> None:
        entry = self._pending.get(thread_id)
        if entry is None:
            return
        if self._clock() - entry.created_at > self._ttl:
            # fail-closed: a pending that timed out denies and is removed.
            if not entry.future.done():
                entry.future.set_result(Decision(approved=False))
            del self._pending[thread_id]
