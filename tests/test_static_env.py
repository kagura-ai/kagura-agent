"""v0.6 (Tasks 10-12): StaticEnvProvider + run --grant (parse-only).

`StaticEnvProvider` is the *documented exception* to the membrane's
no-standing-secret rule: some APIs (Slack / Discord / Resend) only issue a
long-lived static key. The provider carries one, but **refuses to construct**
unless the operator explicitly accepts the risk with ``standing_secret=True`` —
so a standing secret is never used by accident.

`run --grant PROVIDER:SCOPE` is **parse-only** in this release (enforcement lands
in v0.7). It validates grant syntax into a GrantSet and emits a loud
"not enforced yet" warning so an operator is never misled into thinking the
grants are active.
"""

import pytest

from kagura_agent.cli.main import GRANT_NOT_ENFORCED_WARNING, parse_args, resolve_grants
from kagura_agent.membrane.providers import StandingSecretRefused, StaticEnvProvider
from kagura_agent.membrane.registry import GrantSet

# --------------------------------------------------------------------------
# StaticEnvProvider — standing-secret gate
# --------------------------------------------------------------------------


def test_static_env_refuses_without_standing_secret():
    with pytest.raises(StandingSecretRefused, match="standing_secret"):
        StaticEnvProvider(value="xoxb-tok", env_var="SLACK_BOT_TOKEN")


def test_static_env_refuses_with_explicit_false():
    with pytest.raises(StandingSecretRefused):
        StaticEnvProvider(value="xoxb-tok", env_var="SLACK_BOT_TOKEN", standing_secret=False)


@pytest.mark.parametrize("truthy_non_bool", ["false", "true", "1", 1])
def test_static_env_refuses_truthy_non_bool_standing_secret(truthy_non_bool):
    # A non-bool consent value (e.g. a quoted TOML "false", which is truthy) must
    # NOT open the gate — only the literal bool True does (fail-closed).
    with pytest.raises(StandingSecretRefused):
        StaticEnvProvider(value="t", env_var="X", standing_secret=truthy_non_bool)


def test_static_env_constructs_with_standing_secret():
    p = StaticEnvProvider(value="xoxb-tok", env_var="SLACK_BOT_TOKEN", standing_secret=True)
    assert p.stateful is False


async def test_static_env_mint_returns_value_and_nothing_to_revoke():
    p = StaticEnvProvider(value="xoxb-tok", env_var="SLACK_BOT_TOKEN", standing_secret=True)
    cred, handle = await p.mint("any-scope", 60)
    assert cred == "xoxb-tok"
    assert handle is None  # static key — nothing to revoke
    assert await p.revoke(None) is None


async def test_static_env_mint_ignores_scope_and_ttl():
    p = StaticEnvProvider(value="V", env_var="X", standing_secret=True)
    assert (await p.mint("s1", 1))[0] == (await p.mint("s2", 99999))[0] == "V"


def test_static_env_cred_to_env_maps_to_the_named_var():
    p = StaticEnvProvider(value="V", env_var="SLACK_BOT_TOKEN", standing_secret=True)
    assert p.cred_to_env("xoxb-tok") == {"SLACK_BOT_TOKEN": "xoxb-tok"}


def test_static_env_is_an_env_cred_provider():
    # Structural conformance so CredentialBroker.container_env maps it (if
    # cred_to_env were ever dropped, container_env would silently skip it).
    from kagura_agent.membrane.cred_env import EnvCredProvider

    p = StaticEnvProvider(value="V", env_var="X", standing_secret=True)
    assert isinstance(p, EnvCredProvider)


# --------------------------------------------------------------------------
# run --grant — parse-only, loud "not enforced" warning
# --------------------------------------------------------------------------


def test_grant_argparse_appends():
    ns = parse_args(["run", "do a thing", "--grant", "aws:s3-read", "--grant", "gcp:x"])
    assert ns.grants == ["aws:s3-read", "gcp:x"]


def test_grant_argparse_default_is_none():
    ns = parse_args(["run", "do a thing"])
    assert ns.grants is None


def test_resolve_grants_none_yields_empty_set_no_warning():
    gs, warning = resolve_grants(None)
    assert gs == GrantSet(frozenset())
    assert warning is None


def test_resolve_grants_parses_to_grantset_with_warning():
    gs, warning = resolve_grants(["aws:arn:aws:iam::1:role/x"])
    assert gs.allows("aws", "arn:aws:iam::1:role/x")
    assert warning is not None
    assert warning == GRANT_NOT_ENFORCED_WARNING


def test_grant_not_enforced_warning_is_loud_and_honest():
    w = GRANT_NOT_ENFORCED_WARNING.lower()
    assert "not" in w and "enforc" in w  # tells the operator grants aren't active yet


def test_resolve_grants_malformed_is_fail_closed():
    with pytest.raises(ValueError):
        resolve_grants(["no-colon"])
