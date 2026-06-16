"""v0.6 (Tasks 8-9): setup wizard — reference-only TOML helpers + operator gate.

The wizard writes a `[providers.<name>]` block to kagura-agent.toml. Its single
security invariant: it writes **references only** — never a secret value. That is
enforced structurally by round-tripping every rendered block through
`parse_registry` (#56's inline-secret + env-name guards), so a bare secret or a
secret pasted into a `*_env` field is refused before anything is written.

Writing config is **operator-gated**: `apply_provider` refuses unless
`setup_authorized=True`, so a hijacked agent cannot silently write its own
provider config.
"""

import tomllib

import pytest

from kagura_agent.cli.setup import (
    SetupNotAuthorized,
    apply_provider,
    render_provider_block,
    setup_memory_guidance,
    setup_transport_guidance,
    upsert_provider,
)

# --------------------------------------------------------------------------
# render_provider_block — reference-only, round-trips
# --------------------------------------------------------------------------


def test_render_plain_field_and_round_trips():
    block = render_provider_block("aws", "aws_sts", {"role_arn": "arn:aws:iam::1:role/a"})
    assert block.startswith("[providers.aws]")
    assert 'kind = "aws_sts"' in block
    parsed = tomllib.loads(block)["providers"]["aws"]
    assert parsed == {"kind": "aws_sts", "role_arn": "arn:aws:iam::1:role/a"}


def test_render_reference_field():
    block = render_provider_block(
        "cf", "cloudflare", {"account_id": "a", "parent_token_env": "CF_TOKEN"}
    )
    assert 'parent_token_env = "CF_TOKEN"' in block
    assert tomllib.loads(block)["providers"]["cf"]["parent_token_env"] == "CF_TOKEN"


def test_render_rejects_bare_secret():
    with pytest.raises(ValueError, match="inline secret|parent_token"):
        render_provider_block(
            "cf", "cloudflare", {"account_id": "a", "parent_token": "sk-live-xyz"}
        )


def test_render_rejects_secret_pasted_into_env_field():
    # A raw secret in a *_env field (dashes → not an env-var name) is refused by
    # the registry round-trip — the wizard never writes a secret value.
    with pytest.raises(ValueError):
        render_provider_block(
            "cf", "cloudflare", {"account_id": "a", "parent_token_env": "sk-live-xyz"}
        )


def test_render_rejects_unknown_kind():
    with pytest.raises(ValueError):
        render_provider_block("x", "azure", {"role_arn": "r"})


def test_render_escapes_windows_path_and_round_trips():
    block = render_provider_block(
        "gh",
        "github_app",
        {"app_id": "1", "installation_id": "2", "private_key_file": "C:\\secrets\\gh.pem"},
    )
    # The backslashes must survive a TOML round-trip intact (escaping correctness).
    assert tomllib.loads(block)["providers"]["gh"]["private_key_file"] == "C:\\secrets\\gh.pem"


def test_render_static_env_bool_flag_round_trips():
    block = render_provider_block(
        "slack", "static_env", {"value_env": "SLACK_TOKEN", "standing_secret": True}
    )
    assert tomllib.loads(block)["providers"]["slack"]["standing_secret"] is True


def test_render_int_field_round_trips():
    block = render_provider_block(
        "gh", "github_app", {"app_id": 12345, "installation_id": "2", "private_key_env": "K"}
    )
    assert tomllib.loads(block)["providers"]["gh"]["app_id"] == 12345


def test_render_list_field_round_trips():
    block = render_provider_block(
        "gcp", "gcp_impersonation", {"service_account": "sa@x", "delegates": ["a@x", "b@x"]}
    )
    assert tomllib.loads(block)["providers"]["gcp"]["delegates"] == ["a@x", "b@x"]


def test_render_unsupported_value_type_is_rejected():
    # A non-scalar/array value (e.g. a nested table) has no TOML scalar rendering.
    with pytest.raises(ValueError, match="unsupported"):
        render_provider_block("aws", "aws_sts", {"role_arn": "r", "region": {"nested": 1}})


def test_render_rejects_secret_value_in_a_plain_field():
    # parse_registry only guards key NAMES; a secret pasted into a plain field
    # (session_name) must still be caught by the value scan, not written.
    with pytest.raises(ValueError, match="looks like a secret"):
        render_provider_block(
            "aws", "aws_sts", {"role_arn": "r", "session_name": "sk-" + "A" * 24}
        )


def test_render_rejects_pattern_secret_in_env_field():
    # An identifier-shaped secret (ghp_...) passes the env-name check but is a
    # recognizable secret value — the value scan refuses it.
    with pytest.raises(ValueError, match="looks like a secret"):
        render_provider_block(
            "cf", "cloudflare", {"account_id": "a", "parent_token_env": "ghp_" + "a" * 24}
        )


def test_render_dotted_name_is_quoted_and_round_trips():
    # A provider name with a dot would create nested tables if rendered bare;
    # it must be quoted so it stays a single key.
    block = render_provider_block("aws.prod", "aws_sts", {"role_arn": "r"})
    assert '[providers."aws.prod"]' in block
    assert tomllib.loads(block)["providers"]["aws.prod"]["role_arn"] == "r"


def test_render_escapes_control_char_and_round_trips():
    block = render_provider_block("aws", "aws_sts", {"role_arn": "a\x00b"})
    assert tomllib.loads(block)["providers"]["aws"]["role_arn"] == "a\x00b"


# --------------------------------------------------------------------------
# upsert_provider — idempotent insert / replace
# --------------------------------------------------------------------------


def test_upsert_into_empty():
    block = render_provider_block("aws", "aws_sts", {"role_arn": "r"})
    out = upsert_provider("", "aws", block)
    assert "[providers.aws]" in out
    assert tomllib.loads(out)["providers"]["aws"]["role_arn"] == "r"


def test_upsert_preserves_other_providers():
    a = render_provider_block("aws", "aws_sts", {"role_arn": "ra"})
    c = render_provider_block("cf", "cloudflare", {"account_id": "a", "parent_token_env": "CF"})
    out = upsert_provider(a, "cf", c)
    parsed = tomllib.loads(out)["providers"]
    assert set(parsed) == {"aws", "cf"}
    assert parsed["aws"]["role_arn"] == "ra"


def test_upsert_replaces_same_name():
    old = render_provider_block("aws", "aws_sts", {"role_arn": "old"})
    new = render_provider_block("aws", "aws_sts", {"role_arn": "new"})
    out = upsert_provider(old, "aws", new)
    parsed = tomllib.loads(out)["providers"]
    assert parsed["aws"]["role_arn"] == "new"
    assert len([ln for ln in out.splitlines() if ln.strip() == "[providers.aws]"]) == 1


def test_upsert_is_idempotent():
    a = render_provider_block("aws", "aws_sts", {"role_arn": "r"})
    c = render_provider_block("cf", "cloudflare", {"account_id": "a", "parent_token_env": "CF"})
    once = upsert_provider(a, "cf", c)
    twice = upsert_provider(once, "cf", c)
    assert once == twice


def test_upsert_replaces_provider_followed_by_another_section():
    # Replace a provider that has a following [providers.*] section — exercises
    # the section-boundary detection (stop at the next "[").
    aws_old = render_provider_block("aws", "aws_sts", {"role_arn": "old"})
    cf = render_provider_block("cf", "cloudflare", {"account_id": "x", "parent_token_env": "CF"})
    two = upsert_provider(aws_old, "cf", cf)  # aws then cf
    out = upsert_provider(two, "aws", render_provider_block("aws", "aws_sts", {"role_arn": "new"}))
    parsed = tomllib.loads(out)["providers"]
    assert parsed["aws"]["role_arn"] == "new"
    assert parsed["cf"]["account_id"] == "x"  # following section preserved


def test_upsert_exact_header_match_not_prefix():
    prod = render_provider_block("aws-prod", "aws_sts", {"role_arn": "prod"})
    out = upsert_provider(prod, "aws", render_provider_block("aws", "aws_sts", {"role_arn": "dev"}))
    parsed = tomllib.loads(out)["providers"]
    assert parsed["aws-prod"]["role_arn"] == "prod"  # not clobbered by the "aws" upsert
    assert parsed["aws"]["role_arn"] == "dev"


# --------------------------------------------------------------------------
# operator gate
# --------------------------------------------------------------------------


def test_upsert_does_not_corrupt_multiline_array_in_another_section():
    # A multi-line array element line that starts with "[" must NOT be mistaken
    # for a section boundary (else the replaced section's body is orphaned).
    existing = (
        "[providers.first]\n"
        'kind = "aws_sts"\n'
        'role_arn = "r1"\n'
        "extra = [\n"
        "  [1, 2],\n"
        "  [3, 4],\n"
        "]\n"
        "\n"
        "[providers.second]\n"
        'kind = "aws_sts"\n'
        'role_arn = "r2"\n'
    )
    new = render_provider_block("first", "aws_sts", {"role_arn": "NEW"})
    out = upsert_provider(existing, "first", new)
    parsed = tomllib.loads(out)["providers"]  # must remain valid TOML
    assert parsed["first"]["role_arn"] == "NEW"
    assert "extra" not in parsed["first"]  # the old multi-line array was fully replaced
    assert parsed["second"]["role_arn"] == "r2"  # following section intact


def test_upsert_matches_header_with_trailing_comment():
    # A header with a trailing comment must be found (not appended as a duplicate).
    existing = '[providers.aws] # legacy import\nkind = "aws_sts"\nrole_arn = "old"\n'
    new = render_provider_block("aws", "aws_sts", {"role_arn": "new"})
    out = upsert_provider(existing, "aws", new)
    parsed = tomllib.loads(out)["providers"]  # raises if a duplicate section was appended
    assert parsed["aws"]["role_arn"] == "new"


def test_apply_provider_refuses_without_authorization():
    with pytest.raises(SetupNotAuthorized):
        apply_provider("", "aws", "aws_sts", {"role_arn": "r"}, setup_authorized=False)


def test_apply_provider_writes_when_authorized():
    out = apply_provider("", "aws", "aws_sts", {"role_arn": "r"}, setup_authorized=True)
    assert tomllib.loads(out)["providers"]["aws"]["role_arn"] == "r"


def test_apply_provider_still_reference_only_when_authorized():
    # Authorization does not bypass the reference-only guard.
    with pytest.raises(ValueError):
        apply_provider(
            "", "cf", "cloudflare", {"account_id": "a", "parent_token": "sk"}, setup_authorized=True
        )


# --------------------------------------------------------------------------
# guidance (CLI-first, no secret written)
# --------------------------------------------------------------------------


def test_setup_memory_guidance_is_cli_first():
    g = setup_memory_guidance()
    assert "kagura auth login" in g


def test_setup_transport_guidance_nonempty():
    assert setup_transport_guidance().strip()


def test_parse_setup_command():
    from kagura_agent.cli.main import parse_args

    assert parse_args(["setup"]).topic is None
    assert parse_args(["setup", "memory"]).topic == "memory"
    assert parse_args(["setup", "transport"]).topic == "transport"
