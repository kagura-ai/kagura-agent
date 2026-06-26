"""v0.6 (Tasks 1-3): declarative provider registry core.

This is the operator-facing entry point for registering credential providers.
The CredentialBroker / Lease / providers machinery already exists (v0.2); what
was missing was a *declarative* way for an operator to say "here are my
providers" without ever handing a secret to the agent.

Security invariant — **the registry stores references only, never secrets.**
A provider table may carry plain config (``role_arn``, ``account_id`` …) and
*references* to secrets (``parent_token_env`` → a host env var,
``parent_token_file`` → a host file path, or ``parent_token_keyring`` → a host
OS-keychain entry), but never a bare secret value. The
inline-secret guard enforces this fail-closed: a bare ``parent_token`` /
``private_key`` (or any other obviously-secret key) is a ``ValueError``, not a
silently-stored secret.

Three fail-closed gates, all ``ValueError``:
  - unknown ``kind``
  - inline (bare) secret
  - missing required field / required secret reference

The reference *resolution* (env var → value, file → contents) happens later and
**only on the host** (Task 4, ``registry_io``) — never inside the agent
container. This module is pure: it validates shapes, it does not read env or
files.

Public surface consumed by #57-61:
  ``ProviderSpec``, ``KNOWN_KINDS``, ``kind_schema()``, ``FieldSchema``,
  ``SecretRef``, ``Grant``, ``GrantSet``, ``parse_registry()``, ``parse_grants()``.
``_KIND_FIELDS`` is private — read it through ``kind_schema()``.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from kagura_agent.membrane.secret_source import SECRET_SUFFIXES

# --------------------------------------------------------------------------
# Typed per-kind schema (the stable seam #57-61 read through kind_schema())
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SecretRef:
    """A secret a kind may reference. Given as one of the backend-suffix forms
    in :data:`~kagura_agent.membrane.secret_source.SECRET_SUFFIXES`
    (``<name>_env`` / ``<name>_file`` / ``<name>_keyring``); the bare ``<name>``
    form is always rejected (inline-secret guard).

    ``suffixes`` restricts which backend variants are valid for *this* secret
    (default: all of ``SECRET_SUFFIXES``). ``static_env`` pins its ``value`` to
    ``_env`` only — the container env-var NAME is taken from ``value_env``, so a
    ``*_file`` / ``*_keyring`` reference has no container var to name and the
    factory cannot honor it. Restricting here keeps doctor and the run in
    agreement (both reject it at parse time) instead of doctor passing a config
    the run aborts on."""

    name: str
    required: bool
    suffixes: tuple[str, ...] = SECRET_SUFFIXES


@dataclass(frozen=True)
class FieldSchema:
    """The fields a provider kind accepts.

    - ``required`` / ``optional``: plain (non-secret) config field names.
    - ``secrets``: secret references; each accepted as any ``SECRET_SUFFIXES``
      variant (``*_env`` / ``*_file`` / ``*_keyring``), exactly one per secret.
    """

    required: frozenset[str]
    optional: frozenset[str]
    secrets: tuple[SecretRef, ...]


# Per-kind field schemas. Reference-only by construction: secrets never appear
# as bare keys here, only as SecretRef logical names expanded to *_env / *_file.
_KIND_FIELDS: dict[str, FieldSchema] = {
    "aws_sts": FieldSchema(
        required=frozenset({"role_arn"}),
        optional=frozenset({"region", "session_name"}),
        # STS AssumeRole normally uses the host's ambient credential chain, so an
        # explicit parent token is optional.
        secrets=(SecretRef("parent_token", required=False),),
    ),
    "gcp_impersonation": FieldSchema(
        required=frozenset({"service_account"}),
        optional=frozenset({"token_lifetime", "delegates"}),
        secrets=(SecretRef("parent_token", required=False),),
    ),
    "github_app": FieldSchema(
        required=frozenset({"app_id", "installation_id"}),
        optional=frozenset(),
        secrets=(SecretRef("private_key", required=True),),
    ),
    "cloudflare": FieldSchema(
        required=frozenset({"account_id"}),
        optional=frozenset({"zone_id"}),
        secrets=(SecretRef("parent_token", required=True),),
    ),
    "memory_cloud": FieldSchema(
        required=frozenset(),
        optional=frozenset({"base_url", "context_id"}),
        secrets=(SecretRef("parent_token", required=True),),
    ),
    "static_env": FieldSchema(
        # Long-lived static keys (Slack / Discord / Resend). The provider (#61)
        # refuses to mint unless standing_secret is explicitly True.
        required=frozenset(),
        optional=frozenset({"standing_secret"}),
        # value is _env-only: the container env-var name IS value_env; a
        # *_file / *_keyring reference has no container var to name.
        secrets=(SecretRef("value", required=True, suffixes=("_env",)),),
    ),
}

#: Public set of recognized provider kinds. Downstream (#58/#59) enumerate this.
KNOWN_KINDS: frozenset[str] = frozenset(_KIND_FIELDS)

# Defense-in-depth: a curated set of common bare secret-key names that are
# rejected with the *inline-secret* signal even when the kind does not declare
# them, so a typo'd secret (e.g. private_key under aws_sts) fails with a clear
# "use *_env/*_file" message rather than a vague "unknown field". This list is
# NOT exhaustive — any undeclared key is still rejected by the unknown-field
# gate; the denylist only upgrades the *error message* for well-known secret
# names. Exact-key match (not substring), so "standing_secret" stays safe.
_BARE_SECRET_DENYLIST: frozenset[str] = frozenset(
    {
        "parent_token",
        "private_key",
        "api_key",
        "api_secret",
        "api_token",
        "auth_token",
        "refresh_token",
        "oauth_token",
        "access_token",
        "bearer_token",
        "bot_token",
        "token",
        "secret",
        "secret_key",
        "service_key",
        "signing_secret",
        "webhook_secret",
        "client_secret",
        "password",
        "passwd",
    }
)

_PROVIDER_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# An *_env reference must name a host environment variable, not carry a value.
# Requiring a valid identifier shape (no spaces, dashes, '=', newlines) catches
# the common mistake of pasting a raw secret ("sk-live-...") into the reference
# field — a value-level guard on top of the key-level inline-secret guard.
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def kind_schema(kind: str) -> FieldSchema:
    """Return the typed :class:`FieldSchema` for ``kind`` (fail-closed on unknown).

    The public accessor for the otherwise-private ``_KIND_FIELDS`` — the setup
    wizard (#60) and doctor (#59) read per-kind fields through this, never the
    private mapping.
    """
    try:
        return _KIND_FIELDS[kind]
    except KeyError:
        raise ValueError(f"unknown provider kind {kind!r} (known: {sorted(KNOWN_KINDS)})") from None


def present_suffix_field(
    ref: SecretRef, fields: Mapping[str, Any]
) -> tuple[list[str], dict[str, str]]:
    """Map one secret ref over a spec's ``fields``, honoring the ref's suffix set.

    Returns ``(present, suffix_fields)`` where ``suffix_fields`` is
    ``{suffix: "<name><suffix>"}`` for every backend variant *this* secret allows
    (``ref.suffixes`` — narrower than :data:`SECRET_SUFFIXES` for a pinned secret
    like ``static_env``'s ``_env``-only ``value``), and ``present`` is the subset
    of those suffixes whose field is actually set in ``fields``. 0 = absent,
    1 = the chosen backend, 2+ = ambiguous. The single suffix-dispatch kernel the
    validator, ``build_broker``, and ``doctor`` share, so the present-field rule
    (and the allowed-suffix narrowing) lives in exactly one place.
    """
    suffix_fields = {suf: f"{ref.name}{suf}" for suf in ref.suffixes}
    present = [suf for suf, fld in suffix_fields.items() if fld in fields]
    return present, suffix_fields


# --------------------------------------------------------------------------
# ProviderSpec + parse_registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderSpec:
    """A validated, reference-only provider declaration.

    ``fields`` is a read-only view of the operator's declared config — plain
    values and ``*_env`` / ``*_file`` / ``*_keyring`` references only, never a
    bare secret.
    """

    name: str
    kind: str
    fields: Mapping[str, Any]


def parse_registry(providers: Mapping[str, Any]) -> tuple[ProviderSpec, ...]:
    """Validate a ``[providers]`` table into a tuple of :class:`ProviderSpec`.

    ``providers`` maps an operator-chosen provider name to its config table
    (already parsed from TOML by the caller — this module does no I/O).

    Fail-closed (``ValueError``) on: a non-mapping provider table, an empty /
    malformed provider name, a missing or unknown ``kind``, an inline (bare)
    secret, an unknown field, a missing required field, a missing required
    secret reference, or an ambiguous secret (two or more of the
    ``*_env`` / ``*_file`` / ``*_keyring`` variants for one logical secret).
    """
    if not isinstance(providers, Mapping):
        raise ValueError(f"providers must be a table/mapping, got {type(providers).__name__}")

    specs: list[ProviderSpec] = []
    for name, table in providers.items():
        specs.append(_parse_one(name, table))
    return tuple(specs)


def _parse_one(name: Any, table: Any) -> ProviderSpec:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"provider name must be a non-empty string, got {name!r}")
    if not _PROVIDER_NAME_RE.match(name):
        raise ValueError(
            f"invalid provider name {name!r}: only letters, digits, '.', '_', '-' are allowed"
        )
    if not isinstance(table, Mapping):
        raise ValueError(f"provider {name!r} must be a table/mapping, got {type(table).__name__}")

    kind = table.get("kind")
    if kind is None:
        raise ValueError(f"provider {name!r} is missing required 'kind'")
    if not isinstance(kind, str) or kind not in KNOWN_KINDS:
        raise ValueError(
            f"provider {name!r} has unknown kind {kind!r} (known: {sorted(KNOWN_KINDS)})"
        )

    schema = _KIND_FIELDS[kind]
    secret_names = {s.name for s in schema.secrets}
    # Suffix-agnostic (#63): every backend variant a secret allows is accepted,
    # so a new backend (e.g. *_keyring) needs no per-kind edit. A secret may pin
    # a narrower set via SecretRef.suffixes (static_env's value is _env-only).
    ref_keys = {f"{s.name}{suf}" for s in schema.secrets for suf in s.suffixes}
    allowed = schema.required | schema.optional | ref_keys | {"kind"}

    fields: dict[str, Any] = {}
    for key, value in table.items():
        if key == "kind":
            continue
        if key in secret_names:
            raise ValueError(
                f"inline secret {key!r} not allowed for provider {name!r}: "
                f"the registry stores references only — use {key}_env or {key}_file"
            )
        if key in _BARE_SECRET_DENYLIST:
            # A denylisted secret name this kind does NOT declare: suggesting
            # {key}_env/{key}_file would be wrong (not a valid field here), so
            # give the generic reference-only message instead of a dead-end hint.
            raise ValueError(
                f"inline secret {key!r} not allowed for provider {name!r}: "
                f"the registry stores references only, never secret values"
            )
        if key not in allowed:
            raise ValueError(
                f"unknown field {key!r} for provider {name!r} (kind={kind}); "
                f"allowed: {sorted(allowed - {'kind'})}"
            )
        fields[key] = value

    missing = schema.required - fields.keys()
    if missing:
        raise ValueError(
            f"provider {name!r} (kind={kind}) is missing required field(s): {sorted(missing)}"
        )

    # A present-but-empty required field is treated as missing — fail-closed, so a
    # downstream consumer never receives an empty/placeholder ARN/account_id where
    # it expects a real value. "Empty" covers None, a blank/whitespace string, and
    # any *falsy* non-string (False / 0 / [] / {}) — each would otherwise reach a
    # `str(...)` downstream as a misleading "False"/"0"/"[]" instead of failing
    # here. A truthy int (e.g. github_app's numeric app_id) is allowed through.
    for req in schema.required:
        val = fields[req]
        blank_str = isinstance(val, str) and not val.strip()
        falsy_nonstr = not isinstance(val, str) and not val
        if blank_str or falsy_nonstr:
            raise ValueError(
                f"provider {name!r} (kind={kind}) has an empty required field {req!r}"
            )

    # standing_secret is the SECURITY CONSENT gate for the membrane's one sanctioned
    # long-lived standing secret (StaticEnvProvider mints only when it is the bare
    # bool True). Validate its TYPE here at the parse boundary — like every other
    # field — so a mis-set consent flag (the string "true", or 1) is rejected by
    # doctor/dry-run up front, not silently passed and only refused two layers down
    # at mint. `isinstance(_, bool)` accepts True/False and rejects 1/0/"true"
    # (a real bool False is well-typed declined consent; the mint-time `is True`
    # gate enforces the VALUE — type and value are separate concerns).
    consent = fields.get("standing_secret")
    if consent is not None and not isinstance(consent, bool):
        raise ValueError(
            f"provider {name!r} (kind={kind}) field 'standing_secret' must be a bare "
            f"boolean (true/false), not {type(consent).__name__} {consent!r}"
        )

    for ref in schema.secrets:
        # Suffix-agnostic (#63): a logical secret may be satisfied by exactly one
        # backend suffix. Count the present variants — 0+required = missing
        # (fail-closed), exactly 1 = validate its value, 2+ = ambiguous (reject).
        present, suffix_keys = present_suffix_field(ref, fields)

        # Ambiguity is checked first, regardless of required/optional: two
        # references for one logical secret is always a misconfiguration.
        if len(present) > 1:
            keys = ", ".join(suffix_keys[suf] for suf in present)
            raise ValueError(
                f"ambiguous secret {ref.name!r} for provider {name!r}: set only one of {keys}"
            )

        if present:
            suf = present[0]
            key = suffix_keys[suf]
            val = fields[key]
            if suf == "_env":
                # An *_env reference must name a host env var, not carry a value —
                # the NAME-shape check catches a raw secret pasted into the field.
                if not isinstance(val, str) or not _ENV_NAME_RE.fullmatch(val):
                    # Deliberately value-FREE: the rejected text is very often the raw
                    # secret the operator pasted by mistake (that is what this guard
                    # catches), and this message is printed to stderr/journald/CI by
                    # doctor and the run path. Echoing it would leak the secret to logs
                    # (CWE-532). The field name alone is already actionable — matching
                    # the bare-inline-secret guard above and the _file/_keyring branches.
                    raise ValueError(
                        f"{key} for provider {name!r} must be an environment variable "
                        f"NAME (e.g. CF_TOKEN), not a value"
                    )
            elif suf == "_file":
                if not isinstance(val, str) or not val.strip():
                    raise ValueError(
                        f"{key} for provider {name!r} must be a non-empty file path"
                    )
            else:
                # _keyring today (and any future non-env/file suffix): require a
                # non-empty reference string. The backend-specific shape — keyring's
                # "service/username", a future Vault path — is enforced fail-closed
                # at resolve time by secret_source, so we don't duplicate it here.
                if not isinstance(val, str) or not val.strip():
                    raise ValueError(
                        f"{key} for provider {name!r} must be a non-empty keyring "
                        f"reference ('service/username')"
                    )
        elif ref.required:
            all_keys = " / ".join(suffix_keys[suf] for suf in ref.suffixes)
            raise ValueError(
                f"provider {name!r} (kind={kind}) is missing required secret "
                f"{ref.name!r}: set one of {all_keys}"
            )

    # Deep-copy so ProviderSpec is a true immutable snapshot: MappingProxyType
    # blocks key reassignment but not in-place mutation of mutable leaf values
    # (e.g. a `delegates` list); deep-copying severs that shared reference.
    return ProviderSpec(name=name, kind=kind, fields=MappingProxyType(copy.deepcopy(dict(fields))))


# --------------------------------------------------------------------------
# Task-scoped grants — default-deny, exact-match
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Grant:
    """A single (provider, scope) permission. Both matched exactly."""

    provider: str
    scope: str


@dataclass(frozen=True)
class GrantSet:
    """An immutable set of grants. **Default-deny**: an empty set allows nothing,
    and :meth:`allows` is exact-match only (no prefix / wildcard)."""

    grants: frozenset[Grant] = field(default_factory=frozenset)

    def allows(self, provider: str, scope: str) -> bool:
        return Grant(provider, scope) in self.grants


def parse_grants(specs: Iterable[str]) -> GrantSet:
    """Parse ``provider:scope`` strings into a :class:`GrantSet`.

    Split on the **first** colon so scopes that contain colons (AWS ARNs) keep
    them. An empty iterable yields an empty (deny-all) GrantSet; a malformed
    entry (no colon, empty provider, or empty scope) is fail-closed ``ValueError``.
    """
    if isinstance(specs, str) or not isinstance(specs, Iterable):
        raise ValueError(
            f"grants must be an iterable of 'provider:scope' strings, got {type(specs).__name__}"
        )
    grants: set[Grant] = set()
    for raw in specs:
        if not isinstance(raw, str):
            raise ValueError(f"grant must be a string, got {type(raw).__name__}")
        s = raw.strip()
        idx = s.find(":")
        if idx == -1:
            raise ValueError(f"malformed grant {raw!r}: expected 'provider:scope'")
        provider, scope = s[:idx].strip(), s[idx + 1 :].strip()
        if not provider or not scope:
            raise ValueError(f"malformed grant {raw!r}: both provider and scope are required")
        grants.add(Grant(provider, scope))
    return GrantSet(frozenset(grants))
