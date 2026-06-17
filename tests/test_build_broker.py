"""v0.6 (Task 5): build_broker — registry → live CredentialBroker.

The orchestration (dedup, host-side secret resolution, dispatch to a provider
factory, broker assembly) is pure and unit-tested here with a STUB factory and
STUB resolvers — no cloud SDKs. The real provider construction lives behind the
injectable `_factory` seam (the default is the `# pragma: no cover` deployment
edge that needs boto3/google-auth/httpx + deployment-supplied callables).
"""

import pytest

from kagura_agent.membrane.cloud_transports import _resolve_spec_secrets, build_broker
from kagura_agent.membrane.lease import Budget, CredentialBroker
from kagura_agent.membrane.providers import (
    MemoryCloudProvider,
    MemoryWriteLocked,
    StandingSecretRefused,
)
from kagura_agent.membrane.registry import ProviderSpec, kind_schema, parse_registry


def _clock() -> float:
    return 1000.0


class StubProvider:
    """A minimal stateless CredProvider, so the assembled broker is exercisable
    without any cloud SDK."""

    stateful = False

    def __init__(self, tag: str = "x") -> None:
        self.tag = tag

    async def mint(self, scope: str, ttl: int) -> tuple[str, str | None]:
        return f"tok-{self.tag}", None

    async def revoke(self, handle: str | None) -> None:  # pragma: no cover
        return None


def _specs(table):
    return parse_registry(table)


# --------------------------------------------------------------------------
# build_broker — orchestration (stub factory)
# --------------------------------------------------------------------------


async def test_build_broker_assembles_broker_and_wires_each_provider():
    specs = _specs(
        {
            "aws": {"kind": "aws_sts", "role_arn": "arn:aws:iam::1:role/a"},
            "gh": {
                "kind": "github_app",
                "app_id": "111",
                "installation_id": "222",
                "private_key_env": "GH_KEY",
            },
        }
    )
    calls: list[tuple[str, dict]] = []

    def factory(spec, secrets):
        calls.append((spec.name, dict(secrets)))
        return StubProvider(spec.name)

    broker = build_broker(
        specs, clock=_clock, resolve_env={"GH_KEY": "pem-body"}.get, _factory=factory
    )
    assert isinstance(broker, CredentialBroker)

    # The provider really got wired under its name (acquire reaches the stub).
    lease = await broker.acquire("aws", scope="arn:aws:iam::1:role/a", ttl=60, budget=Budget(3600))
    assert lease.cred == "tok-aws"

    # Both specs were dispatched; the github_app secret was resolved host-side,
    # and aws_sts (no secret reference) got an empty secrets dict (not a spurious
    # optional parent_token).
    assert {name for name, _ in calls} == {"aws", "gh"}
    gh_secrets = next(s for name, s in calls if name == "gh")
    aws_secrets = next(s for name, s in calls if name == "aws")
    assert gh_secrets == {"private_key": "pem-body"}
    assert aws_secrets == {}


def test_resolve_spec_secrets_ambiguous_env_and_file_is_fail_closed():
    # parse_registry guards this, but a hand-built spec must not slip both through.
    spec = ProviderSpec(
        name="cf",
        kind="cloudflare",
        fields={"account_id": "a", "parent_token_env": "CF", "parent_token_file": "/run/s"},
    )
    with pytest.raises(ValueError, match="ambiguous"):
        _resolve_spec_secrets(spec, resolve_env={"CF": "v"}.get, resolve_file=lambda p: "f")


def test_resolve_spec_secrets_keyring_form():
    # #65: a *_keyring secret resolves host-side via the injected keyring reader —
    # suffix-agnostic, no per-kind change.
    spec = ProviderSpec(
        name="cf",
        kind="cloudflare",
        fields={"account_id": "a", "parent_token_keyring": "svc/agent"},
    )
    out = _resolve_spec_secrets(
        spec,
        resolve_env={}.get,
        resolve_file=lambda _p: "",
        resolve_keyring=lambda s, u: f"kr[{s}/{u}]",
    )
    assert out == {"parent_token": "kr[svc/agent]"}


def test_resolve_spec_secrets_ambiguous_env_and_keyring_is_fail_closed():
    spec = ProviderSpec(
        name="cf",
        kind="cloudflare",
        fields={"account_id": "a", "parent_token_env": "CF", "parent_token_keyring": "svc/agent"},
    )
    with pytest.raises(ValueError, match="ambiguous"):
        _resolve_spec_secrets(
            spec,
            resolve_env={"CF": "v"}.get,
            resolve_file=lambda _p: "",
            resolve_keyring=lambda s, u: "x",
        )


async def test_build_broker_keyring_resolution_reaches_factory():
    # End-to-end through build_broker: an injected keyring reader resolves a
    # *_keyring secret and the resolved value reaches the provider factory.
    specs = _specs(
        {"cf": {"kind": "cloudflare", "account_id": "a", "parent_token_keyring": "cf-svc/agent"}}
    )
    seen: dict[str, str] = {}

    def factory(spec, secrets):
        seen.update(secrets)
        return StubProvider(spec.name)

    build_broker(
        specs,
        clock=_clock,
        resolve_keyring=lambda s, u: f"kr-{s}-{u}",
        _factory=factory,
    )
    assert seen == {"parent_token": "kr-cf-svc-agent"}


async def test_memory_cloud_built_from_registry_is_read_locked_end_to_end():
    # A registry-derived memory_cloud carries no write signal, so a deployer
    # factory builds it read-locked; the broker then refuses memory:write.
    specs = _specs({"mem": {"kind": "memory_cloud", "parent_token_env": "MEM"}})

    def factory(spec, secrets):
        return MemoryCloudProvider(
            exchange=lambda req: {"access_token": "mem-tok"}, write_approved=False
        )

    broker = build_broker(specs, clock=_clock, resolve_env={"MEM": "tok"}.get, _factory=factory)
    with pytest.raises(MemoryWriteLocked):
        await broker.acquire("mem", scope="memory:write", ttl=60, budget=Budget(3600))
    lease = await broker.acquire("mem", scope="memory:read", ttl=60, budget=Budget(3600))
    assert lease.cred == "mem-tok"


def test_build_broker_duplicate_provider_name_is_fail_closed():
    specs = _specs({"x": {"kind": "aws_sts", "role_arn": "r"}})
    with pytest.raises(ValueError, match="duplicate"):
        build_broker(specs + specs, clock=_clock, _factory=lambda s, sec: StubProvider())


def test_build_broker_empty_registry_yields_empty_broker():
    broker = build_broker((), clock=_clock, _factory=lambda s, sec: StubProvider())
    assert isinstance(broker, CredentialBroker)
    assert broker.open_leases() == []


def test_build_broker_resolution_is_host_side_via_injected_resolvers():
    specs = _specs({"cf": {"kind": "cloudflare", "account_id": "a", "parent_token_file": "/run/s"}})
    seen_files: list[str] = []

    def read_file(path):
        seen_files.append(path)
        return "cf-token\n"

    captured = {}

    def factory(spec, secrets):
        captured.update(secrets)
        return StubProvider()

    build_broker(specs, clock=_clock, resolve_file=read_file, _factory=factory)
    assert seen_files == ["/run/s"]  # resolved on the host, via the injected reader
    assert captured == {"parent_token": "cf-token"}


# --------------------------------------------------------------------------
# _resolve_spec_secrets
# --------------------------------------------------------------------------


def test_resolve_spec_secrets_no_secret_fields_is_empty():
    spec = _specs({"aws": {"kind": "aws_sts", "role_arn": "r"}})[0]
    assert _resolve_spec_secrets(spec, resolve_env={}.get, resolve_file=lambda p: "") == {}


def test_resolve_spec_secrets_env_form():
    spec = _specs({"cf": {"kind": "cloudflare", "account_id": "a", "parent_token_env": "CF"}})[0]
    out = _resolve_spec_secrets(spec, resolve_env={"CF": "v"}.get, resolve_file=lambda p: "")
    assert out == {"parent_token": "v"}


def test_resolve_spec_secrets_file_form():
    spec = _specs(
        {
            "gh": {
                "kind": "github_app",
                "app_id": "1",
                "installation_id": "2",
                "private_key_file": "/k",
            }
        }
    )[0]
    out = _resolve_spec_secrets(spec, resolve_env={}.get, resolve_file=lambda p: "pem\n")
    assert out == {"private_key": "pem"}


# --------------------------------------------------------------------------
# memory_cloud read-locked — STRUCTURAL guarantee (no SDK needed)
# --------------------------------------------------------------------------


def test_memory_cloud_schema_has_no_write_approval_field():
    # The registry cannot express write approval for memory_cloud, so a broker
    # built from a registry is read-locked BY CONSTRUCTION. Write approval only
    # ever comes via the #14/#15 device-flow HITL graduation path, never config.
    schema = kind_schema("memory_cloud")
    all_field_names = set(schema.required) | set(schema.optional) | {s.name for s in schema.secrets}
    assert not any("write" in f.lower() or "approv" in f.lower() for f in all_field_names)


# --------------------------------------------------------------------------
# default factory — clear errors for kinds needing deployment callables
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "table,needle",
    [
        (
            {"cf": {"kind": "cloudflare", "account_id": "a", "parent_token_env": "CF"}},
            "token_spec",
        ),
        (
            {"mem": {"kind": "memory_cloud", "parent_token_env": "MEM"}},
            "exchange",
        ),
    ],
)
def test_default_factory_errors_for_kinds_needing_deployment_wiring(table, needle):
    specs = _specs(table)
    env = {"CF": "x", "MEM": "x"}
    with pytest.raises(ValueError, match=needle):
        build_broker(specs, clock=_clock, resolve_env=env.get)  # default factory


async def test_default_factory_builds_static_env_with_standing_secret():
    # #61: static_env is now buildable by the default factory (no SDK needed) and
    # the token must round-trip all the way to the container env var.
    specs = _specs(
        {"slack": {"kind": "static_env", "value_env": "SLACK_BOT_TOKEN", "standing_secret": True}}
    )
    broker = build_broker(specs, clock=_clock, resolve_env={"SLACK_BOT_TOKEN": "xoxb-tok"}.get)
    lease = await broker.acquire("slack", scope="static", ttl=60, budget=Budget(60))
    assert lease.cred == "xoxb-tok"
    # The whole point of static_env: the container gets the named env var.
    assert broker.container_env([lease]) == {"SLACK_BOT_TOKEN": "xoxb-tok"}


def test_default_factory_refuses_static_env_without_standing_secret():
    specs = _specs({"slack": {"kind": "static_env", "value_env": "SLACK_BOT_TOKEN"}})
    with pytest.raises(StandingSecretRefused):
        build_broker(specs, clock=_clock, resolve_env={"SLACK_BOT_TOKEN": "x"}.get)


def test_default_factory_refuses_static_env_with_string_standing_secret():
    # A quoted string standing_secret="false" must not be bool()-coerced to True
    # and bypass the gate — parse_registry stores it verbatim, the factory passes
    # it raw, and StaticEnvProvider's identity check refuses it.
    specs = _specs(
        {
            "slack": {
                "kind": "static_env",
                "value_env": "SLACK_BOT_TOKEN",
                "standing_secret": "false",
            }
        }
    )
    with pytest.raises(StandingSecretRefused):
        build_broker(specs, clock=_clock, resolve_env={"SLACK_BOT_TOKEN": "x"}.get)


def test_default_factory_github_app_missing_private_key_is_valueerror():
    # #82: a hand-built spec missing its required secret must raise the module's
    # consistent ValueError, not a bare KeyError deep in the factory.
    spec = ProviderSpec(
        name="gh", kind="github_app", fields={"app_id": "1", "installation_id": "2"}
    )
    with pytest.raises(ValueError, match="private_key"):
        build_broker([spec], clock=_clock)


def test_required_secret_helper_returns_value_and_raises_when_absent():
    from kagura_agent.membrane.cloud_transports import _required_secret

    spec = ProviderSpec(name="gh", kind="github_app", fields={})
    assert _required_secret(spec, {"private_key": "pem"}, "private_key") == "pem"
    with pytest.raises(ValueError, match="missing its required 'private_key'"):
        _required_secret(spec, {}, "private_key")


def test_default_factory_aws_sts_rejects_explicit_parent_token():
    # #82: aws_sts mints with the host's ambient credentials and cannot honor an
    # explicit parent_token — a present one must fail loudly, not be silently
    # dropped (which would mislead the operator into thinking they pinned a cred).
    spec = ProviderSpec(
        name="aws",
        kind="aws_sts",
        fields={"role_arn": "arn:aws:iam::1:role/a", "parent_token_env": "AWS_TOK"},
    )
    with pytest.raises(ValueError, match="parent_token"):
        build_broker([spec], clock=_clock, resolve_env={"AWS_TOK": "x"}.get)


def test_reject_unhonored_parent_token_allows_absent():
    # The absent case (the common one) is fine: no parent_token → ambient creds.
    from kagura_agent.membrane.cloud_transports import _reject_unhonored_parent_token

    spec = ProviderSpec(name="aws", kind="aws_sts", fields={"role_arn": "r"})
    assert _reject_unhonored_parent_token(spec) is None


def test_default_factory_static_env_requires_value_env_not_value_file():
    # parse_registry now rejects value_file/value_keyring for static_env (value is
    # _env-only), but build_broker accepts any hand-built ProviderSpec, so the
    # factory keeps its own actionable guard as defense-in-depth for that path.
    spec = ProviderSpec(
        name="slack",
        kind="static_env",
        fields={"value_file": "/run/s", "standing_secret": True},
    )
    with pytest.raises(ValueError, match="value_env"):
        build_broker([spec], clock=_clock, resolve_file=lambda p: "tok\n")
