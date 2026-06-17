"""v0.6 (Tasks 1-3): provider registry core.

The registry is the *declarative* entry point an operator uses to register
credential providers. Its single security invariant: **it stores references
only, never secret values**. Bare secrets (`parent_token`, `private_key`, ...)
are rejected fail-closed; only reference forms (`*_env` / `*_file`) are allowed.

Three fail-closed gates, all raising ValueError:
  - unknown `kind`
  - inline (bare) secret
  - missing required field / required secret

Plus task-scoped grants: `GrantSet` is default-deny, exact-match — an empty set
denies everything.
"""

import dataclasses

import pytest

from kagura_agent.membrane.registry import (
    _BARE_SECRET_DENYLIST,
    KNOWN_KINDS,
    FieldSchema,
    Grant,
    GrantSet,
    SecretRef,
    kind_schema,
    parse_grants,
    parse_registry,
)

# --------------------------------------------------------------------------
# parse_registry — happy paths
# --------------------------------------------------------------------------


def test_parses_aws_sts_with_plain_fields():
    specs = parse_registry(
        {"prod-aws": {"kind": "aws_sts", "role_arn": "arn:aws:iam::123:role/agent"}}
    )
    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "prod-aws"
    assert spec.kind == "aws_sts"
    assert spec.fields["role_arn"] == "arn:aws:iam::123:role/agent"


def test_parses_multiple_providers_as_ordered_tuple():
    specs = parse_registry(
        {
            "aws": {"kind": "aws_sts", "role_arn": "arn:aws:iam::1:role/a"},
            "cf": {"kind": "cloudflare", "account_id": "acct1", "parent_token_env": "CF_TOKEN"},
        }
    )
    assert isinstance(specs, tuple)
    assert {s.name for s in specs} == {"aws", "cf"}


def test_accepts_secret_reference_env_form():
    specs = parse_registry(
        {"cf": {"kind": "cloudflare", "account_id": "a", "parent_token_env": "CF_PARENT_TOKEN"}}
    )
    assert specs[0].fields["parent_token_env"] == "CF_PARENT_TOKEN"


def test_accepts_secret_reference_file_form():
    specs = parse_registry(
        {
            "gh": {
                "kind": "github_app",
                "app_id": "111",
                "installation_id": "222",
                "private_key_file": "/run/secrets/gh.pem",
            }
        }
    )
    assert specs[0].fields["private_key_file"] == "/run/secrets/gh.pem"


def test_static_env_with_standing_secret_flag():
    specs = parse_registry(
        {"slack": {"kind": "static_env", "value_env": "SLACK_BOT_TOKEN", "standing_secret": True}}
    )
    assert specs[0].kind == "static_env"
    assert specs[0].fields["standing_secret"] is True


# --------------------------------------------------------------------------
# parse_registry — fail-closed gates (all ValueError)
# --------------------------------------------------------------------------


def test_unknown_kind_is_fail_closed():
    with pytest.raises(ValueError, match="unknown.*kind|kind.*unknown"):
        parse_registry({"x": {"kind": "azure_managed_identity", "role_arn": "r"}})


def test_missing_kind_is_fail_closed():
    with pytest.raises(ValueError):
        parse_registry({"x": {"role_arn": "r"}})


def test_inline_bare_secret_is_rejected():
    # Bare parent_token (not *_env/*_file) must be refused — registry holds no secrets.
    with pytest.raises(ValueError, match="inline secret|parent_token"):
        parse_registry(
            {"cf": {"kind": "cloudflare", "account_id": "a", "parent_token": "sk-live-xyz"}}
        )


def test_inline_private_key_rejected_even_on_unrelated_kind():
    # Global bare-secret denylist: a secret-looking bare key is rejected with the
    # inline-secret signal even when the kind does not declare it.
    with pytest.raises(ValueError, match="inline secret|private_key"):
        parse_registry(
            {"aws": {"kind": "aws_sts", "role_arn": "r", "private_key": "-----BEGIN..."}}
        )


def test_missing_required_plain_field_is_fail_closed():
    with pytest.raises(ValueError, match="required|missing"):
        parse_registry({"aws": {"kind": "aws_sts"}})  # role_arn missing


@pytest.mark.parametrize("empty", ["", "   ", None])
def test_empty_required_field_is_fail_closed(empty):
    # A present-but-empty required field must be treated as missing (fail-closed),
    # so a downstream consumer never receives an empty ARN.
    with pytest.raises(ValueError, match="empty required field|role_arn"):
        parse_registry({"aws": {"kind": "aws_sts", "role_arn": empty}})


@pytest.mark.parametrize("falsy", [[], False, 0, {}])
def test_falsy_non_string_required_field_is_fail_closed(falsy):
    # #82: a *falsy non-string* required field ([], False, 0, {}) used to slip
    # through the None/blank-string guard and reach a downstream str(...) as a
    # misleading "False"/"0"/"[]". It must fail closed too.
    with pytest.raises(ValueError, match="empty required field"):
        parse_registry({"aws": {"kind": "aws_sts", "role_arn": falsy}})


def test_truthy_int_required_field_is_accepted():
    # A truthy int required field (github_app's numeric app_id) must still pass —
    # the falsy guard rejects 0/False, not every non-string.
    specs = parse_registry(
        {
            "gh": {
                "kind": "github_app",
                "app_id": 12345,
                "installation_id": "2",
                "private_key_env": "K",
            }
        }
    )
    assert specs[0].fields["app_id"] == 12345


def test_denylist_only_key_gives_generic_message_not_dead_end_hint():
    # 'token' is denylisted but not a declared secret of aws_sts, so the error
    # must NOT suggest token_env/token_file (which aren't valid fields here).
    with pytest.raises(ValueError) as exc:
        parse_registry({"aws": {"kind": "aws_sts", "role_arn": "r", "token": "sk-live"}})
    msg = str(exc.value)
    assert "inline secret" in msg
    assert "token_env" not in msg and "token_file" not in msg


def test_missing_required_secret_is_fail_closed():
    # cloudflare requires a parent_token reference; absent → fail-closed.
    with pytest.raises(ValueError, match="required|parent_token|missing"):
        parse_registry({"cf": {"kind": "cloudflare", "account_id": "a"}})


def test_ambiguous_secret_both_env_and_file_is_fail_closed():
    with pytest.raises(ValueError, match="ambiguous|both|parent_token"):
        parse_registry(
            {
                "cf": {
                    "kind": "cloudflare",
                    "account_id": "a",
                    "parent_token_env": "CF_TOKEN",
                    "parent_token_file": "/run/secrets/cf",
                }
            }
        )


# --- #63: suffix-agnostic validator (env/file/keyring), exactly-one-suffix ---


def test_accepts_secret_reference_keyring_form():
    # #63: a *_keyring variant is accepted with NO per-kind schema edit — the
    # validator allows every SECRET_SUFFIXES variant of a declared secret name.
    specs = parse_registry(
        {"cf": {"kind": "cloudflare", "account_id": "a", "parent_token_keyring": "cf-svc/agent"}}
    )
    assert specs[0].fields["parent_token_keyring"] == "cf-svc/agent"


def test_required_secret_satisfied_by_keyring_alone():
    # cloudflare's parent_token is required; a lone *_keyring reference satisfies it.
    specs = parse_registry(
        {"cf": {"kind": "cloudflare", "account_id": "a", "parent_token_keyring": "svc/agent"}}
    )
    assert specs[0].kind == "cloudflare"


def test_static_env_value_is_env_only():
    # static_env's value is _env-only: the container env-var NAME is value_env, so
    # value_file / value_keyring have no var to name and the factory can't honor
    # them. Reject at parse time (a SecretRef.suffixes restriction) so doctor and
    # the run agree, rather than doctor passing a config the run aborts on.
    specs = parse_registry({"s": {"kind": "static_env", "value_env": "SLACK_TOKEN"}})
    assert specs[0].fields["value_env"] == "SLACK_TOKEN"
    for bad in ("value_file", "value_keyring"):
        with pytest.raises(ValueError, match="unknown field"):
            parse_registry({"s": {"kind": "static_env", bad: "x"}})


def test_static_env_missing_value_lists_only_env_variant():
    # The missing-required message reflects the restricted suffix set (value_env),
    # not all three variants.
    with pytest.raises(ValueError, match="value_env") as exc:
        parse_registry({"s": {"kind": "static_env", "standing_secret": True}})
    assert "value_keyring" not in str(exc.value) and "value_file" not in str(exc.value)


def test_env_reference_with_trailing_newline_is_rejected():
    # _ENV_NAME_RE must reject a trailing newline (it documents "no newlines").
    # `$` matched before a final \n; fullmatch closes that so the operator gets the
    # clear "must be an environment variable NAME" error at parse time, not a
    # confusing deferred "unset var" at resolve time.
    with pytest.raises(ValueError, match="environment variable|NAME"):
        parse_registry(
            {"cf": {"kind": "cloudflare", "account_id": "a", "parent_token_env": "CF_TOKEN\n"}}
        )


def test_keyring_reference_must_be_non_empty():
    with pytest.raises(ValueError, match="keyring|non-empty"):
        parse_registry(
            {"cf": {"kind": "cloudflare", "account_id": "a", "parent_token_keyring": "   "}}
        )


def test_ambiguous_secret_env_and_keyring_is_fail_closed():
    with pytest.raises(ValueError, match="ambiguous"):
        parse_registry(
            {
                "cf": {
                    "kind": "cloudflare",
                    "account_id": "a",
                    "parent_token_env": "CF_TOKEN",
                    "parent_token_keyring": "svc/agent",
                }
            }
        )


def test_ambiguous_secret_three_suffixes_is_fail_closed():
    with pytest.raises(ValueError, match="ambiguous"):
        parse_registry(
            {
                "cf": {
                    "kind": "cloudflare",
                    "account_id": "a",
                    "parent_token_env": "CF_TOKEN",
                    "parent_token_file": "/run/secrets/cf",
                    "parent_token_keyring": "svc/agent",
                }
            }
        )


def test_keyring_reference_must_be_a_string():
    # A non-string keyring value (e.g. a TOML integer) must fail closed at the
    # registry, not slip through to secret_source at resolve time.
    with pytest.raises(ValueError, match="keyring|non-empty"):
        parse_registry(
            {"cf": {"kind": "cloudflare", "account_id": "a", "parent_token_keyring": 42}}
        )


def test_missing_required_secret_message_lists_all_suffixes():
    # The missing-required-secret message must offer every backend variant,
    # including keyring — not just env/file.
    with pytest.raises(ValueError) as exc:
        parse_registry({"cf": {"kind": "cloudflare", "account_id": "a"}})
    msg = str(exc.value)
    assert "parent_token_keyring" in msg
    assert "parent_token_env" in msg and "parent_token_file" in msg


def test_optional_secret_with_two_suffixes_still_ambiguous():
    # aws_sts.parent_token is OPTIONAL, but two suffixes are still ambiguous —
    # the ambiguity check must fire regardless of required/optional.
    with pytest.raises(ValueError, match="ambiguous"):
        parse_registry(
            {
                "aws": {
                    "kind": "aws_sts",
                    "role_arn": "arn:aws:iam::1:role/x",
                    "parent_token_env": "AWS_TOK",
                    "parent_token_keyring": "svc/agent",
                }
            }
        )


def test_unknown_field_is_fail_closed():
    with pytest.raises(ValueError, match="unknown field|unexpected"):
        parse_registry({"aws": {"kind": "aws_sts", "role_arn": "r", "bogus_field": "x"}})


def test_empty_provider_name_is_fail_closed():
    with pytest.raises(ValueError, match="name"):
        parse_registry({"": {"kind": "aws_sts", "role_arn": "r"}})


def test_provider_table_must_be_mapping():
    with pytest.raises(ValueError):
        parse_registry({"aws": "not-a-table"})


@pytest.mark.parametrize("bad", [["a", "list"], 42, "string", None])
def test_top_level_providers_must_be_mapping(bad):
    with pytest.raises(ValueError, match="mapping|table"):
        parse_registry(bad)


def test_env_reference_must_be_a_variable_name_not_a_value():
    # Pasting a raw secret into the *_env reference field is a value-level mistake
    # the key-level guard cannot see; the env-name shape check catches it.
    with pytest.raises(ValueError, match="environment variable|NAME"):
        parse_registry(
            {"cf": {"kind": "cloudflare", "account_id": "a", "parent_token_env": "sk-live-xyz"}}
        )


def test_file_reference_must_be_non_empty():
    with pytest.raises(ValueError, match="file path"):
        parse_registry(
            {"cf": {"kind": "cloudflare", "account_id": "a", "parent_token_file": "   "}}
        )


def test_registry_holds_no_secret_values():
    # The acceptance invariant: every stored field value is a reference or plain
    # config — never a bare secret. Assert no stored key is a known bare secret.
    specs = parse_registry(
        {"cf": {"kind": "cloudflare", "account_id": "a", "parent_token_env": "CF_TOKEN"}}
    )
    for spec in specs:
        assert "parent_token" not in spec.fields  # only parent_token_env is stored
        assert not (set(spec.fields) & _BARE_SECRET_DENYLIST)


def test_provider_spec_is_an_immutable_snapshot():
    # gcp delegates is a list — mutating the operator's original dict (or the
    # stored view) must not bleed into the stored spec.
    table = {"gcp": {"kind": "gcp_impersonation", "service_account": "sa@x", "delegates": ["a@x"]}}
    spec = parse_registry(table)[0]
    table["gcp"]["delegates"].append("attacker@evil")  # mutate the source after parse
    assert spec.fields["delegates"] == ["a@x"]


# --------------------------------------------------------------------------
# KNOWN_KINDS / kind_schema public API (consumed by #57-61)
# --------------------------------------------------------------------------


def test_known_kinds_membership():
    assert KNOWN_KINDS >= {
        "aws_sts",
        "gcp_impersonation",
        "github_app",
        "cloudflare",
        "memory_cloud",
        "static_env",
    }


def test_kind_schema_returns_typed_schema():
    schema = kind_schema("github_app")
    assert isinstance(schema, FieldSchema)
    assert "app_id" in schema.required
    assert any(s.name == "private_key" and s.required for s in schema.secrets)


def test_kind_schema_unknown_is_fail_closed():
    with pytest.raises(ValueError):
        kind_schema("nope")


def test_field_schema_is_frozen():
    schema = kind_schema("aws_sts")
    with pytest.raises(dataclasses.FrozenInstanceError):
        schema.required = frozenset()  # type: ignore[misc]


def test_provider_spec_is_frozen():
    spec = parse_registry({"aws": {"kind": "aws_sts", "role_arn": "r"}})[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.kind = "cloudflare"  # type: ignore[misc]


# --------------------------------------------------------------------------
# Grant / GrantSet / parse_grants — default-deny, exact-match
# --------------------------------------------------------------------------


def test_grantset_exact_match_allows():
    gs = parse_grants(["aws:arn:aws:iam::123:role/agent"])
    assert gs.allows("aws", "arn:aws:iam::123:role/agent")


def test_grantset_scope_is_exact_not_prefix():
    gs = parse_grants(["aws:arn:aws:iam::123:role/agent"])
    assert not gs.allows("aws", "arn:aws:iam::123:role/agent-readonly")
    assert not gs.allows("aws", "arn:aws:iam::123:role")


def test_grantset_provider_is_exact():
    gs = parse_grants(["aws:s3-read"])
    assert not gs.allows("aws-prod", "s3-read")


def test_empty_grantset_denies_everything():
    gs = parse_grants([])
    assert not gs.allows("aws", "anything")
    assert gs == GrantSet(frozenset())


def test_parse_grants_splits_on_first_colon_for_arns():
    # ARNs contain colons; scope must keep them all.
    gs = parse_grants(["aws:arn:aws:iam::123:role/x"])
    assert gs.allows("aws", "arn:aws:iam::123:role/x")


def test_parse_grants_strips_whitespace():
    gs = parse_grants(["  aws : s3-read  "])
    assert gs.allows("aws", "s3-read")


@pytest.mark.parametrize("bad", ["", "   ", "no-colon", "aws:", ":scope", "  :  "])
def test_parse_grants_malformed_is_fail_closed(bad):
    with pytest.raises(ValueError):
        parse_grants([bad])


@pytest.mark.parametrize("bad", [42, None, "aws:s3-read"])
def test_parse_grants_non_iterable_or_bare_string_is_fail_closed(bad):
    # A non-iterable, or a bare 'provider:scope' string passed instead of a list
    # (a common mistake — iterating a str yields characters), is fail-closed
    # ValueError, not a TypeError or silently-wrong per-character parse.
    with pytest.raises(ValueError, match="iterable|provider:scope"):
        parse_grants(bad)


def test_grant_and_grantset_are_frozen():
    g = Grant("aws", "s3-read")
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.provider = "gcp"  # type: ignore[misc]
    gs = GrantSet(frozenset({g}))
    with pytest.raises(dataclasses.FrozenInstanceError):
        gs.grants = frozenset()  # type: ignore[misc]


def test_secret_ref_is_frozen():
    ref = SecretRef("parent_token", required=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.required = False  # type: ignore[misc]
