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
    r"|sk-[A-Za-z0-9_-]{16,}"  # OpenAI / Anthropic key (incl. sk-ant-api03-*, sk-proj-*)
    r"|ghp_[A-Za-z0-9]{20,}"  # GitHub personal access token
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"  # Slack token
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"  # PEM private key
)

# A TOML bare key (usable unquoted in a header). A provider name with anything
# else (notably ``.``, which TOML reads as a table separator) must be quoted.
_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")

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


def _assert_no_secret_value(key: str, value: Any, provider: str) -> None:
    """Refuse a recognizable secret in ``value`` — recursing into list/tuple
    elements so a secret hidden in a list field (e.g. ``delegates``) is caught too."""
    if isinstance(value, str):
        if _SECRET_VALUE_RE.search(value):
            raise ValueError(
                f"field {key!r} of provider {provider!r} looks like a secret value; the "
                "registry stores references only — put the secret in an env var / file "
                "and reference it with *_env / *_file"
            )
    elif isinstance(value, (list, tuple)):
        for element in value:
            _assert_no_secret_value(key, element, provider)


def _header_key(name: str) -> str:
    """The provider name as a TOML header key — quoted if it isn't a bare key."""
    return name if _BARE_KEY_RE.match(name) else _toml_value(name)


def _header_path(line: str) -> str | None:
    """The dotted path of a TOML table / array-of-tables header line, or None.

    Quote-AWARE: a quoted table name may legally contain ``]`` (e.g. ``["b]c"]``);
    a regex that stopped at the first ``]`` misread such a header as a non-header, so
    upsert_provider absorbed and silently DROPPED that section. Scanning with
    string-awareness fixes it. Examples: ``[providers.aws] # note`` → ``providers.aws``;
    ``["b]c"]`` → ``"b]c"``; a multi-line array element like ``["a@x"],`` → None (so it
    is never mistaken for a section boundary).
    """
    s = line.strip()
    if not s.startswith("["):
        return None
    double = s.startswith("[[")  # [[array-of-tables]]
    open_len = 2 if double else 1
    i = open_len
    quote: str | None = None  # active string delimiter inside the name, else None
    while i < len(s):
        ch = s[i]
        if quote is not None:
            if ch == "\\" and quote == '"':
                i += 2  # skip an escaped char inside a basic string
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "]":
            break  # first top-level (unquoted) ']' closes the name
        i += 1
    else:
        # Ran off the end with no top-level ']' — not a header. Also covers an
        # unterminated quoted name: its ']' (if any) is swallowed inside the still-open
        # string, so the loop never breaks. (After a break, quote is always None,
        # because ']' only closes the name in the quote-None branch above.)
        return None
    inner = s[open_len:i].strip()
    rest = s[i:]
    if double:
        if not rest.startswith("]]"):
            return None  # '[[' must close with ']]'
        rest = rest[2:]
    else:
        rest = rest[1:]  # consume the single ']'
    rest = rest.strip()
    if rest and not rest.startswith("#"):
        return None  # trailing non-comment content → an array row, not a header
    return inner or None


def _bracket_delta(line: str) -> int:
    """Net unclosed ``[`` count on a line, ignoring brackets inside strings/comments.

    Used to track whether a following line is a multi-line-array continuation
    (depth > 0) rather than a real section header. Handles single-line basic and
    literal strings (with escapes in basic strings) and ``#`` comments, so a value
    like ``role_arn = "a[b]"`` contributes 0 and a ``# [note]`` comment is ignored.
    Multi-line (triple-quoted) string *values* are not expected in a generated
    provider block and are out of scope.
    """
    depth = 0
    quote: str | None = None  # active string delimiter ('"' or "'"), else None
    i, n = 0, len(line)
    while i < n:
        ch = line[i]
        if quote is None:
            if ch == "#":
                break  # comment runs to end of line
            elif ch in "\"'":
                quote = ch
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
        elif ch == "\\" and quote == '"':
            i += 2  # skip an escaped char inside a basic string
            continue
        elif ch == quote:
            quote = None
        i += 1
    return depth


def render_provider_block(name: str, kind: str, fields: Mapping[str, Any]) -> str:
    """Render a validated, reference-only ``[providers.<name>]`` TOML block."""
    table = {"kind": kind, **dict(fields)}

    # Guard 1: reference-only / kind / shape — reuse the registry's validation.
    parse_registry({name: table})

    # Guard 2: no recognizable secret value in ANY field (plain or reference,
    # recursing into list elements).
    for key, value in table.items():
        _assert_no_secret_value(key, value, name)

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
    # Split on '\n' only (NOT str.splitlines()): splitlines() also breaks on U+0085 /
    # U+2028 / U+2029, which are LEGAL raw characters inside a TOML basic string —
    # rejoining with '\n' would turn one into a real newline and corrupt the value
    # into invalid TOML. split('\n') preserves them (and any '\r' rides along on the
    # line, so CRLF round-trips faithfully).
    lines = existing_text.split("\n")
    block_lines = block.rstrip("\n").split("\n")

    start = next((i for i, ln in enumerate(lines) if _header_path(ln) == target), None)
    if start is None:
        base = existing_text.rstrip("\n")
        return (base + "\n\n" if base else "") + "\n".join(block_lines) + "\n"

    # Walk the section body. Two things make this non-trivial:
    #  - A multi-line array's continuation row can start with "[" (e.g. a comma-free
    #    last row `[3, 4]`); track bracket depth so it is never read as a header.
    #  - Trailing blank lines and comments after the section's last key belong to
    #    the *next* section (they often document it); they must be preserved, so
    #    `body_end` advances only on real content (or open-array continuation).
    depth = 0
    body_end = start + 1  # one past the last line that belongs to this section
    for j in range(start + 1, len(lines)):
        line = lines[j]
        if depth == 0 and _header_path(line) is not None:
            break  # the next real section header
        stripped = line.strip()
        if depth > 0 or (stripped and not stripped.startswith("#")):
            body_end = j + 1  # a key/value line, or a continuation inside an array
        depth = max(0, depth + _bracket_delta(line))

    tail = lines[body_end:]  # trailing blanks/comments + the next section, preserved
    # One reproducible blank line before following content (keeps it idempotent),
    # but don't double-space when the tail already begins with a blank line.
    sep = [""] if tail and tail[0].strip() else []
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
    if setup_authorized is not True:
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
