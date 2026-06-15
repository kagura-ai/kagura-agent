"""v0.6 (Task 4): host-side registry loader + secret-reference resolution.

`load_registry` reads a TOML registry file and hands its `[providers]` table to
`parse_registry` (#56). `resolve_secret_ref` turns a single `*_env` / `*_file`
reference from a `ProviderSpec` into its secret value.

**Membrane invariant — host-side only.** Resolution reads host environment
variables and host files; it runs in the trusted cockpit/host and the resolved
value is injected downstream as a leased container env var. This module must
**never** be imported or executed inside the agent container — doing so would
hand the container the ability to read the host's secrets directly, collapsing
the boundary.

Fail-closed: a missing/empty env var, an unreadable/empty file, or any
underlying error is normalized to :class:`SecretRefError` so a caller can rely
on "either a non-empty secret, or an exception" — never a silently-empty secret.
Error messages name the reference (env var name / file path), never the resolved
secret value.

The env getter and file reader are injectable so #58 (`build_broker`) can pass
its own resolvers and tests need not touch the real environment or filesystem.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Callable
from pathlib import Path

from kagura_agent.membrane.registry import ProviderSpec, parse_registry

#: Resolve an environment variable name to its value (or None if unset).
EnvResolver = Callable[[str], str | None]
#: Read a host file path and return its text contents.
FileResolver = Callable[[str], str]


class SecretRefError(RuntimeError):
    """A secret reference could not be resolved on the host (missing or empty).

    Carries an actionable message that names the reference (env var / file path)
    but never the resolved secret value.
    """


def load_registry(path: str | Path) -> tuple[ProviderSpec, ...]:
    """Read a TOML registry file and validate its ``[providers]`` table.

    Returns the parsed :class:`ProviderSpec` tuple (empty if there is no
    ``[providers]`` table). Fail-closed ``ValueError`` on a missing file, invalid
    TOML, or any shape violation surfaced by :func:`parse_registry` (unknown
    kind, inline secret, missing required field, ...).
    """
    p = Path(path)
    try:
        with p.open("rb") as fh:
            config = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ValueError(f"registry file not found: {p}") from exc
    except OSError as exc:
        raise ValueError(f"cannot read registry file {p}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid TOML in registry {p}: {exc}") from exc
    except UnicodeDecodeError as exc:
        # TOML must be UTF-8; a non-UTF-8 file makes tomllib raise UnicodeDecodeError
        # (a ValueError subclass, but not caught above and with no file path in it).
        raise ValueError(
            f"registry file {p} is not valid UTF-8 (TOML must be UTF-8): {exc}"
        ) from exc

    providers = config.get("providers", {})
    return parse_registry(providers)


def _read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def resolve_secret_ref(
    field_name: str,
    ref: str,
    *,
    get_env: EnvResolver = os.environ.get,
    read_file: FileResolver = _read_file,
) -> str:
    """Resolve one ``*_env`` / ``*_file`` reference to its secret value (host-side).

    - ``field_name`` ending ``_env``: ``ref`` is an environment variable name;
      its value is returned verbatim. Unset or whitespace-only → SecretRefError.
    - ``field_name`` ending ``_file``: ``ref`` is a host file path; its contents
      are returned with the trailing newline stripped. Unreadable or
      whitespace-only → SecretRefError.
    - Any other ``field_name`` is not a reference → SecretRefError.

    Every underlying failure (KeyError, OSError, decode error, a custom
    resolver raising) is normalized to SecretRefError so the caller's fail-closed
    contract holds. The resolved value is never included in an error message.
    """
    if field_name.endswith("_env"):
        try:
            value = get_env(ref)
        except Exception as exc:  # noqa: BLE001 — fail-closed: any failure → SecretRefError
            # Do NOT interpolate {exc} into the message: a custom resolver may
            # raise with the secret value embedded in its text. The original is
            # preserved as __cause__ (via `from exc`) for host-side debug logs.
            raise SecretRefError(
                f"could not read environment variable {ref!r} for {field_name}"
            ) from exc
        if value is None or not value.strip():
            raise SecretRefError(
                f"environment variable {ref!r} for {field_name} is unset or empty"
            )
        return value

    if field_name.endswith("_file"):
        try:
            contents = read_file(ref)
        except Exception as exc:  # noqa: BLE001 — fail-closed: any failure → SecretRefError
            # {exc} omitted on purpose — a custom reader could embed file
            # contents in its message. __cause__ keeps the detail for debug.
            raise SecretRefError(
                f"could not read secret file {ref!r} for {field_name}"
            ) from exc
        # Secret files conventionally end with a trailing newline (LF or CRLF);
        # strip trailing line endings but preserve internal structure (e.g.
        # multi-line PEM keys) and any meaningful trailing spaces.
        secret = contents.rstrip("\r\n")
        if not secret.strip():
            raise SecretRefError(f"secret file {ref!r} for {field_name} is empty")
        return secret

    raise SecretRefError(
        f"{field_name!r} is not a secret reference (expected a *_env or *_file field)"
    )
