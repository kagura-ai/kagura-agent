"""v0.6 (Task 4): host-side registry loader.

`load_registry` reads a TOML registry file and hands its `[providers]` table to
`parse_registry` (#56). Secret-*reference* resolution lived here in v0.6; #65
folded it into the single suffix-dispatch resolver in
:mod:`kagura_agent.membrane.secret_source`. This module now re-exports that
resolver surface (so ``from registry_io import resolve_secret_field`` /
``EnvResolver`` / ``SecretRefError`` keep working) and the byte-for-byte
duplicate resolver code is gone (#82).

**Membrane invariant — host-side only.** `load_registry` and the re-exported
resolvers read host files / env / keychain in the trusted cockpit/host; resolved
values are injected downstream as leased container env vars. This module must
**never** be imported or executed inside the agent container — doing so would
hand the container the ability to read the host's secrets directly, collapsing
the boundary.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from kagura_agent.membrane.registry import ProviderSpec, parse_registry

# v0.6's resolver surface now lives in secret_source (#65 folded it in). Re-export
# the canonical names so existing importers (cli/main.py) keep a stable path —
# there is no second copy of the logic to drift. ``SecretRefError`` is the
# historical name for the canonical SecretSourceError, kept so every existing
# ``except SecretRefError`` catches unchanged. ``__all__`` makes these explicit
# re-exports (mypy --strict otherwise treats an aliased import as private).
from kagura_agent.membrane.secret_source import (
    EnvResolver,
    FileResolver,
    _read_file,
    resolve_secret_field,
)
from kagura_agent.membrane.secret_source import (
    SecretSourceError as SecretRefError,
)

__all__ = [
    "EnvResolver",
    "FileResolver",
    "SecretRefError",
    "_read_file",
    "load_registry",
    "resolve_secret_field",
]


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
