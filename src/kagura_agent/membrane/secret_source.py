"""v0.7 (Task 1): pluggable host-side secret sources by field suffix.

A secret reference in a :class:`~kagura_agent.membrane.registry.ProviderSpec` is
a ``<name><suffix>`` field whose value points at *where* the secret lives, never
the secret itself. The suffix selects the backend:

  - ``<name>_env``     → a host environment variable name
  - ``<name>_file``    → a host file path
  - ``<name>_keyring`` → a host OS-keychain entry, given as ``"service/username"``

:data:`SECRET_SUFFIXES` is the registry of recognized suffixes; each maps to a
:class:`SecretSource`. :func:`resolve_ref` dispatches one reference to the right
source; :func:`resolve_secret_field` is the host-facing convenience that wires
the default sources. Adding a backend (e.g. Vault) is **one new SecretSource +
one SECRET_SUFFIXES entry** — no per-kind schema change. This is the single
suffix-dispatch resolver the validator (#63) and the run path (#65) build on;
v0.6's ``registry_io.resolve_secret_ref`` folds into it there.

**Membrane invariant — host-side only.** Every source reads host state (env
vars, files, the OS keychain) in the trusted cockpit/host; the resolved value is
injected downstream as a leased container env var. This module must **never** be
imported or executed inside the agent container — doing so would hand the
container the ability to read the host's secrets directly, collapsing the
boundary.

Fail-closed: an unknown suffix, an empty/whitespace value, or any underlying
error is normalized to :class:`SecretSourceError` so a caller can rely on
"either a non-empty secret, or an exception" — never a silently-empty secret.
Error messages name the reference (env var / file path / keychain key), never
the resolved secret value (a custom backend could embed the value in its own
exception text, so ``{exc}`` is deliberately omitted; the original is preserved
as ``__cause__`` for host-side debug logs).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

#: Resolve an environment variable name to its value (or None if unset).
EnvResolver = Callable[[str], str | None]
#: Read a host file path and return its text contents.
FileResolver = Callable[[str], str]
#: Look up a keychain password by (service, username); None if absent.
KeyringResolver = Callable[[str, str], str | None]

#: Recognized secret-reference suffixes, in resolution-precedence order. The
#: env/file pair leads (v0.6 behavior) with keyring appended. The registry
#: validator (#63) and the run path (#65) import this rather than hardcoding the
#: suffix set, so a new backend is added in exactly one place plus its source.
SECRET_SUFFIXES: tuple[str, ...] = ("_env", "_file", "_keyring")


class SecretSourceError(RuntimeError):
    """A secret reference could not be resolved on the host (missing or empty).

    Carries an actionable message that names the reference (env var / file path /
    keychain key) but never the resolved secret value.
    """


@runtime_checkable
class SecretSource(Protocol):
    """A host-side backend that resolves one secret reference to its value.

    ``suffix`` is the field-name suffix this source handles (one of
    :data:`SECRET_SUFFIXES`). ``resolve`` returns a non-empty secret or raises
    :class:`SecretSourceError` — never a silently-empty value, never a message
    containing the resolved secret.
    """

    suffix: ClassVar[str]

    def resolve(self, field_name: str, ref: str) -> str: ...


def _require_nonblank(value: str | None, *, what: str) -> str:
    """Return ``value`` if it is non-empty after stripping, else fail closed."""
    if value is None or not value.strip():
        raise SecretSourceError(f"{what} is unset or empty")
    return value


@dataclass(frozen=True)
class EnvSource:
    """Resolve ``*_env`` references against host environment variables."""

    suffix: ClassVar[str] = "_env"
    get_env: EnvResolver = os.environ.get

    def resolve(self, field_name: str, ref: str) -> str:
        try:
            value = self.get_env(ref)
        except Exception as exc:  # noqa: BLE001 — fail-closed: any failure → SecretSourceError
            raise SecretSourceError(
                f"could not read environment variable {ref!r} for {field_name}"
            ) from exc
        return _require_nonblank(value, what=f"environment variable {ref!r} for {field_name}")


def _read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


@dataclass(frozen=True)
class FileSource:
    """Resolve ``*_file`` references against host files (trailing newline stripped)."""

    suffix: ClassVar[str] = "_file"
    read_file: FileResolver = _read_file

    def resolve(self, field_name: str, ref: str) -> str:
        try:
            contents = self.read_file(ref)
        except Exception as exc:  # noqa: BLE001 — fail-closed: any failure → SecretSourceError
            raise SecretSourceError(
                f"could not read secret file {ref!r} for {field_name}"
            ) from exc
        # Secret files conventionally end with a trailing newline (LF or CRLF);
        # strip trailing line endings but preserve internal structure (e.g.
        # multi-line PEM keys) and any meaningful trailing spaces.
        secret = contents.rstrip("\r\n")
        if not secret.strip():
            raise SecretSourceError(f"secret file {ref!r} for {field_name} is empty")
        return secret


def _import_keyring() -> Any:
    """Import the optional ``keyring`` backend, or fail closed with an install hint.

    The fail-closed branch (extra missing → :class:`SecretSourceError` naming the
    install command) is the user-facing behavior #62 requires, so it is unit-
    tested independently of the real keychain. Only the success line — reached
    solely when the optional extra is installed — carries ``# pragma: no cover``.
    """
    try:
        import keyring as _kr
    except ImportError as exc:
        raise SecretSourceError(
            "the optional 'keyring' extra is required for *_keyring references "
            "(install: pip install 'kagura-agent[keyring]')"
        ) from exc
    return _kr  # pragma: no cover - reached only when the optional 'keyring' extra is installed


def _real_keyring_get_password(  # pragma: no cover - real OS keychain backend
    service: str, username: str
) -> str | None:
    """Default keyring backend — the real OS keychain (optional ``keyring`` extra).

    Not unit-covered: it touches the host keychain. Tests inject their own
    ``get_password``; only this glue calls the real SDK.
    """
    password: str | None = _import_keyring().get_password(service, username)
    return password


@dataclass(frozen=True)
class KeyringSource:
    """Resolve ``*_keyring`` references against the host OS keychain.

    The reference encodes both halves of a keychain lookup as
    ``"service/username"`` (split on the first ``/`` only — a username may itself
    contain slashes). A reference missing either half fails closed.
    """

    suffix: ClassVar[str] = "_keyring"
    get_password: KeyringResolver = _real_keyring_get_password

    def resolve(self, field_name: str, ref: str) -> str:
        service, sep, username = ref.partition("/")
        # Strip before validating so a whitespace-only half (e.g. "  /agent" or
        # "svc/   ") fails closed here instead of reaching the keychain with a
        # blank service/username — consistent with registry_io's .strip() guard.
        service, username = service.strip(), username.strip()
        if not sep or not service or not username:
            raise SecretSourceError(
                f"keyring reference {ref!r} for {field_name} must be 'service/username'"
            )
        try:
            value = self.get_password(service, username)
        except SecretSourceError:
            raise  # already fail-closed + value-free (e.g. missing extra)
        except Exception as exc:  # noqa: BLE001 — fail-closed: any failure → SecretSourceError
            raise SecretSourceError(
                f"could not read keyring entry for {field_name}"
            ) from exc
        return _require_nonblank(value, what=f"keyring entry {ref!r} for {field_name}")


def default_sources(
    *,
    get_env: EnvResolver = os.environ.get,
    read_file: FileResolver = _read_file,
    get_password: KeyringResolver = _real_keyring_get_password,
) -> dict[str, SecretSource]:
    """Build the suffix→source map wiring the built-in backends.

    Each resolver is injectable so callers (#65 ``build_broker``) and tests can
    substitute their own without touching the real environment, filesystem, or
    keychain. The keyring backend is lazy — its real SDK is imported only when a
    ``*_keyring`` reference is actually resolved.
    """
    return {
        "_env": EnvSource(get_env=get_env),
        "_file": FileSource(read_file=read_file),
        "_keyring": KeyringSource(get_password=get_password),
    }


def _suffix_of(field_name: str) -> str | None:
    # Real secret fields end in exactly one suffix (the schema names a logical
    # secret + a single backend suffix), so first-match over SECRET_SUFFIXES is
    # unambiguous in practice.
    for suffix in SECRET_SUFFIXES:
        if field_name.endswith(suffix):
            return suffix
    return None


def resolve_ref(field_name: str, ref: str, *, sources: Mapping[str, SecretSource]) -> str:
    """Dispatch one ``*_<suffix>`` reference to the matching source in ``sources``.

    Fail-closed: a ``field_name`` with no recognized suffix, or a recognized
    suffix whose source is absent from ``sources``, raises
    :class:`SecretSourceError` rather than returning an empty value.
    """
    suffix = _suffix_of(field_name)
    source = sources.get(suffix) if suffix is not None else None
    if source is None:
        raise SecretSourceError(
            f"{field_name!r} is not a secret reference resolvable by the configured "
            f"sources (expected a field ending in one of {SECRET_SUFFIXES})"
        )
    return source.resolve(field_name, ref)


def resolve_secret_field(
    field_name: str,
    ref: str,
    *,
    get_env: EnvResolver = os.environ.get,
    read_file: FileResolver = _read_file,
    get_password: KeyringResolver = _real_keyring_get_password,
) -> str:
    """Resolve one secret reference host-side via the default suffix dispatch.

    The host-facing single resolver: wires :func:`default_sources` and delegates
    to :func:`resolve_ref`. v0.6's ``registry_io.resolve_secret_ref`` folds into
    this in #65 (the run-path wiring) so every provider gets keyring/vault for
    free without a per-kind schema change.
    """
    sources = default_sources(get_env=get_env, read_file=read_file, get_password=get_password)
    return resolve_ref(field_name, ref, sources=sources)
