"""#131: typed revoke-error taxonomy for the credential ledger sweeper.

A stateful provider's ``revoke`` can fail two ways that the sweeper must treat
oppositely:

- **permanent** — the handle is already gone at the provider (e.g. an HTTP 404/410
  for a token revoked out-of-band). There is no live credential to leak, so the
  sweeper should **forget** the lease rather than re-attempt a confirmed-dead handle
  on every restart.
- **transient** — a 5xx, a timeout, or a connection error. The handle may still be
  valid, so the sweeper must **keep** the lease tracked and retry it next sweep,
  never dropping a possibly-still-valid credential on an unprovable failure.

The classification is a pure function that *duck-types* ``exc.response.status_code``
(the shape an ``httpx.HTTPStatusError`` carries) so the membrane core stays free of
any SDK / httpx import. Anything not provably "gone" — including an exception with no
recognizable status — is classified **transient** (fail-safe).

The #124 stopgap kept-on-every-failure (never leaking, but re-attempting a dead
handle each restart); this taxonomy lets the sweeper forget the confirmed-dead ones
while preserving the never-drop-a-live-credential guarantee.
"""

from __future__ import annotations

#: HTTP statuses that prove the handle no longer exists at the provider.
_GONE_STATUSES = frozenset({404, 410})


class RevokeError(RuntimeError):
    """Base: a provider's ``revoke`` failed. Subclasses say whether it is safe to
    forget the lease (permanent) or it must be kept and retried (transient)."""


class RevokePermanent(RevokeError):
    """The handle is already gone at the provider (e.g. 404/410) — safe to forget."""


class RevokeTransient(RevokeError):
    """A transient failure (5xx / timeout / network) — keep tracked and retry later."""


def classify_revoke_error(exc: BaseException) -> RevokeError:
    """Map a raw provider revoke exception to a typed :class:`RevokePermanent` /
    :class:`RevokeTransient`.

    Reads ``exc.response.status_code`` by duck-typing (so no httpx import here): a
    404/410 is permanent; everything else — any other status, or an exception with
    no recognizable status (timeout/connection/plain error) — is transient
    (fail-safe: never forget a possibly-live credential on an unprovable error).
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in _GONE_STATUSES:
        return RevokePermanent(str(exc) or f"handle already gone (HTTP {status})")
    return RevokeTransient(str(exc) or "transient revoke failure")
