"""v0.6 (Tasks 8-9): setup wizard — reference-only TOML helpers + operator gate.

The wizard helps an operator register a provider in ``kagura-agent.toml`` without
ever putting a secret in the file. Its invariant: **references only**. Every
rendered ``[providers.<name>]`` block is guarded three ways before it is returned:

  1. ``parse_registry`` — rejects a bare-secret *key*, a secret pasted into a
     ``*_env`` field (env-name shape), an unknown kind, a missing required field.
  2. a secret-*value* scan over every field value — rejects a recognizable secret
     (AWS key, ``sk-``/``ghp_``/Slack token, PEM private key) pasted into *any*
     field, including a plain field like ``session_name`` whose value the registry
     does not otherwise constrain.
  3. a ``tomllib`` round-trip — the rendered text is re-parsed and compared field
     for field, so an escaping bug never silently corrupts a value.

Writing config is **operator-gated**: :func:`apply_provider` refuses unless
``setup_authorized=True``. The agent cannot authorize itself — this keeps config
mutation a human decision and blocks a hijacked agent from writing its own
provider config (a confused-deputy escalation).

Secrets themselves are never handled here: ``setup memory`` points the operator
at the CLI-first ``kagura auth login`` flow (the token lives in the host's
credential store, not the config).
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from typing import Any

from kagura_agent.membrane.registry import parse_registry


class SetupNotAuthorized(RuntimeError):
    """Raised when a config-writing setup action runs without operator authorization."""


# Recognizable secret shapes — a value matching any of these must never be
# written to config, no matter which field it lands in (defense-in-depth on top
# of the registry's key-level inline-secret guard).
_SECRET_VALUE_RE = re.compile(
    r"AKIA[A-Z0-9]{16}"  # AWS access key id
    r"|sk-[A-Za-z0-9]{16,}"  # OpenAI / Anthropic-style key
    r"|ghp_[A-Za-z0-9]{20,}"  # GitHub personal access token
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"  # Slack token
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"  # PEM private key
)

# A TOML bare key (usable unquoted in a header). A provider name with anything
# else (notably ``.``, which TOML reads as a table separator) must be quoted.
_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# A whole-line TOML table / array-of-tables header, optional trailing comment.
# Deliberately does NOT match a multi-line array element like ``["a", "b"],``.
_SECTION_HEADER_RE = re.compile(r"^\s*\[\[?[^\]]+\]\]?\s*(#.*)?$")

# Control characters TOML basic strings forbid raw (tab/newline/cr are handled
# separately as \t/\n/\r); the rest must be \u-escaped.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _toml_value(value: Any) -> str:
    """Render a Python value as a TOML scalar/array (basic strings, fully escaped)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        escaped = (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )
        escaped = _CONTROL_CHARS_RE.sub(lambda m: f"\\u{ord(m.group()):04x}", escaped)
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise ValueError(f"unsupported TOML value type {type(value).__name__} for setup")


def _header_key(name: str) -> str:
    """The provider name as a TOML header key — quoted if it isn't a bare key."""
    return name if _BARE_KEY_RE.match(name) else _toml_value(name)


def _header_path(line: str) -> str | None:
    """The dotted path of a TOML section header line, or None if not a header.

    ``[providers.aws] # note`` → ``providers.aws``; a multi-line array element
    like ``["a@x"],`` → None (so it is never mistaken for a section boundary).
    """
    if not _SECTION_HEADER_RE.match(line):
        return None
    inner = re.sub(r"\s*#.*$", "", line).strip()  # drop a trailing comment
    return inner.strip("[").rstrip("]").strip()


def render_provider_block(name: str, kind: str, fields: Mapping[str, Any]) -> str:
    """Render a validated, reference-only ``[providers.<name>]`` TOML block."""
    table = {"kind": kind, **dict(fields)}

    # Guard 1: reference-only / kind / shape — reuse the registry's validation.
    parse_registry({name: table})

    # Guard 2: no recognizable secret value in ANY field (plain or reference).
    for key, value in table.items():
        if isinstance(value, str) and _SECRET_VALUE_RE.search(value):
            raise ValueError(
                f"field {key!r} of provider {name!r} looks like a secret value; the "
                "registry stores references only — put the secret in an env var / file "
                "and reference it with *_env / *_file"
            )

    lines = [f"[providers.{_header_key(name)}]", f"kind = {_toml_value(kind)}"]
    for key, value in fields.items():
        lines.append(f"{key} = {_toml_value(value)}")
    block = "\n".join(lines) + "\n"

    # Guard 3: round-trip fidelity (escaping correctness).
    reparsed = tomllib.loads(block)["providers"][name]
    if reparsed != table:  # pragma: no cover - defensive invariant
        # Unreachable for any value _toml_value can render (escaping is exact); a
        # mismatch would mean a renderer bug, so fail closed rather than write it.
        raise ValueError(
            f"rendered TOML for provider {name!r} did not round-trip — refusing to write"
        )
    return block


def upsert_provider(existing_text: str, name: str, block: str) -> str:
    """Idempotently insert or replace the ``[providers.<name>]`` section.

    The section is located by its **header path** (comment-tolerant, so
    ``[providers.aws] # note`` is found) and runs to the next section header
    (a multi-line array's ``[`` lines are not mistaken for one). Other sections
    and comments are preserved; re-running with the same ``block`` is idempotent.
    """
    target = f"providers.{_header_key(name)}"
    lines = existing_text.splitlines()
    block_lines = block.rstrip("\n").split("\n")

    start = next((i for i, ln in enumerate(lines) if _header_path(ln) == target), None)
    if start is None:
        base = existing_text.rstrip("\n")
        return (base + "\n\n" if base else "") + "\n".join(block_lines) + "\n"

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if _header_path(lines[j]) is not None:  # the next real section header
            end = j
            break
    tail = lines[end:]
    # One reproducible blank line before a following section (keeps it idempotent).
    sep = [""] if tail and any(t.strip() for t in tail) else []
    new_lines = lines[:start] + block_lines + sep + tail
    return "\n".join(new_lines).rstrip("\n") + "\n"


def apply_provider(
    existing_text: str,
    name: str,
    kind: str,
    fields: Mapping[str, Any],
    *,
    setup_authorized: bool,
) -> str:
    """Render + upsert a provider block — **operator-gated**.

    Refuses unless ``setup_authorized`` is True (the agent cannot set this for
    itself). Authorization does not relax the reference-only guards in
    :func:`render_provider_block`.
    """
    if not setup_authorized:
        raise SetupNotAuthorized(
            "setup writes kagura-agent.toml and must be operator-authorized "
            "(setup_authorized=True); the agent cannot authorize itself"
        )
    block = render_provider_block(name, kind, fields)
    return upsert_provider(existing_text, name, block)


def setup_memory_guidance() -> str:
    """CLI-first guidance for memory auth — no token is ever written to config."""
    return (
        "Memory access is authenticated via the kagura CLI, not a token in config:\n"
        "  1. Run `kagura auth login`  (OAuth device flow; the refresh token is\n"
        "     stored host-side at ~/.kagura/credentials.json — never in kagura-agent.toml).\n"
        "  2. The agent leases a short-lived, scoped access token at run time.\n"
        "Read-only by default; widening to memory:write requires a device-flow re-approval."
    )


def setup_transport_guidance() -> str:
    """Guidance for wiring a Slack / Discord transport (references only)."""
    return (
        "Transports (Slack / Discord) are long-lived tokens — register them as a\n"
        "static_env provider so only a reference (the env var name) lands in config:\n"
        '  setup adds e.g. [providers.slack] kind="static_env" value_env="SLACK_BOT_TOKEN"\n'
        "  standing_secret=true. Put the actual token in the host environment, not here."
    )
