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
from kagura_agent.membrane.providers import MemoryCloudProvider, MemoryWriteLocked
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
# default factory — clear errors for kinds needing deployment callables / #61
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "table,needle",
    [
        (
            {"slack": {"kind": "static_env", "value_env": "SLACK", "standing_secret": True}},
            "#61",
        ),
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
    env = {"SLACK": "x", "CF": "x", "MEM": "x"}
    with pytest.raises(ValueError, match=needle):
        build_broker(specs, clock=_clock, resolve_env=env.get)  # default factory
