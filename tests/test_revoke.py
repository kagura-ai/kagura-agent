"""#131: typed revoke-error taxonomy.

A provider's revoke failure is classified as PERMANENT (the handle is already gone
at the provider — safe to forget) or TRANSIENT (a 5xx/timeout/network error — keep
tracked and retry). The sweeper forgets the former and keeps the latter, so a
confirmed-dead handle is not re-attempted forever while a still-valid credential is
never dropped on an unprovable error. Classification is pure (duck-types
``exc.response.status_code``) so the membrane core needs no httpx/SDK import.
"""

from __future__ import annotations

import pytest

from kagura_agent.membrane.revoke import (
    RevokeError,
    RevokePermanent,
    RevokeTransient,
    classify_revoke_error,
)


class _HttpErr(Exception):
    """A minimal httpx.HTTPStatusError look-alike carrying ``.response.status_code``."""

    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.response = type("_R", (), {"status_code": status})()


@pytest.mark.parametrize("status", [404, 410])
def test_gone_status_is_permanent(status: int) -> None:
    # 404/410 == the handle no longer exists at the provider → permanent (forget).
    assert isinstance(classify_revoke_error(_HttpErr(status)), RevokePermanent)


@pytest.mark.parametrize("status", [500, 502, 503, 429, 400, 401])
def test_other_status_is_transient(status: int) -> None:
    # Anything that is not a definitive "gone" is transient (keep + retry) —
    # fail-safe: a 5xx/429/400 might be temporary, so never forget on it.
    assert isinstance(classify_revoke_error(_HttpErr(status)), RevokeTransient)


def test_unclassifiable_error_is_transient_fail_safe() -> None:
    # No ``.response`` (a timeout/connection error, or any plain exception) → we
    # cannot prove the handle is gone, so treat it as transient and KEEP it.
    assert isinstance(classify_revoke_error(RuntimeError("connection reset")), RevokeTransient)
    assert isinstance(classify_revoke_error(TimeoutError()), RevokeTransient)


def test_taxonomy_subclasses_revoke_error() -> None:
    assert issubclass(RevokePermanent, RevokeError)
    assert issubclass(RevokeTransient, RevokeError)
