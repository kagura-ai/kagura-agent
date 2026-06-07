"""Per-provider auth resolution (subscription | BYOK | key).

v1's only brain is subscription-backed (a self-hosted single user inherits their
Pro/Max plan through the Claude Code CLI subprocess). The security win:
**subscription mode carries no secret into the container**, so there is no
long-lived key for a hijacked agent to exfiltrate. BYOK/key modes exist as the
*shape* for a future SaaS / Codex brain — not wired into v1's default path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


class AuthError(RuntimeError):
    """No supported auth mode could be satisfied from the environment."""


@dataclass(frozen=True)
class AuthResolution:
    mode: str
    secret: str | None = None


# Preference order: subscription first precisely because it injects no secret.
_PRECEDENCE = ("subscription", "byok", "key")


def resolve_auth(
    supported_modes: tuple[str, ...], *, env: Mapping[str, str]
) -> AuthResolution:
    for mode in _PRECEDENCE:
        if mode not in supported_modes:
            continue
        if mode == "subscription" and env.get("CLAUDE_CODE_SUBSCRIPTION"):
            return AuthResolution(mode="subscription", secret=None)
        if mode in ("byok", "key") and env.get("ANTHROPIC_API_KEY"):
            return AuthResolution(mode="key", secret=env["ANTHROPIC_API_KEY"])
    raise AuthError(
        f"no satisfiable auth mode among {supported_modes!r} "
        "(set CLAUDE_CODE_SUBSCRIPTION or ANTHROPIC_API_KEY)"
    )
