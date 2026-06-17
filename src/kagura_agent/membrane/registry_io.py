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
from kagura_agent.membrane.secret_source import SecretSourceError, resolve_secret_field

#: Resolve an environment variable name to its value (or None if unset).
EnvResolver = Callable[[str], str | None]
#: Read a host file path and return its text contents.
FileResolver = Callable[[str], str]


# v0.7 (#65): the secret-resolution logic is unified in ``secret_source``. This
# name stays as a backward-compatible alias of the canonical
# :class:`~kagura_agent.membrane.secret_source.SecretSourceError` so every
# existing ``except SecretRefError`` (doctor probe, _run_probes) keeps catching,
# and the folded resolver's error is caught unchanged.
SecretRefError = SecretSourceError


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

    v0.7 (#65): this delegates to the unified suffix resolver
    :func:`~kagura_agent.membrane.secret_source.resolve_secret_field`, so the
    ``*_env`` / ``*_file`` logic lives in exactly one place and ``*_keyring`` (and
    any future backend) resolves here too without changes. The default keychain
    backend applies for ``*_keyring`` refs.
    """
    return resolve_secret_field(field_name, ref, get_env=get_env, read_file=read_file)
