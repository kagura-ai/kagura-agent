"""#39: leased cred -> container env translation + broker.container_env wiring.

The genuinely missing link that lets "the agent operate a cloud API": a minted
`Lease.cred` carries a provider-specific cred shape (AWS = a JSON blob of 3 keys;
GCP/GitHub/Cloudflare = a single token). Translating that into the container's
env (which `docker_run_args` already injects as `-e KEY=VALUE`) is what was
stubbed for deployment. This logic is pure and fully testable.
"""

from __future__ import annotations

import json

import pytest

from kagura_agent.membrane.cred_env import (
    EnvCredProvider,
    aws_cred_env,
    cloudflare_cred_env,
    gcp_cred_env,
    github_cred_env,
)
from kagura_agent.membrane.launcher import LaunchSpec, docker_run_args
from kagura_agent.membrane.lease import Budget, CredentialBroker, Lease
from kagura_agent.membrane.providers import (
    AwsStsProvider,
    CloudflareTokenProvider,
    GcpImpersonationProvider,
    GitHubAppProvider,
    MemoryCloudProvider,
)

_AWS_BLOB = json.dumps(
    {
        "AWS_ACCESS_KEY_ID": "AKIA",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_SESSION_TOKEN": "session",
    }
)


# --- pure translators -------------------------------------------------------

def test_aws_cred_env_parses_blob() -> None:
    assert aws_cred_env(_AWS_BLOB) == {
        "AWS_ACCESS_KEY_ID": "AKIA",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_SESSION_TOKEN": "session",
    }


def test_aws_cred_env_rejects_incomplete_blob() -> None:
    with pytest.raises(ValueError, match="AWS"):
        aws_cred_env(json.dumps({"AWS_ACCESS_KEY_ID": "AKIA"}))


def test_aws_cred_env_rejects_non_json() -> None:
    with pytest.raises(ValueError):
        aws_cred_env("not json at all")


def test_gcp_cred_env_uses_gcloud_token_var() -> None:
    assert gcp_cred_env("ya29.tok") == {"CLOUDSDK_AUTH_ACCESS_TOKEN": "ya29.tok"}


def test_github_cred_env_sets_both_gh_vars() -> None:
    # `gh` prefers GH_TOKEN; many actions/tools read GITHUB_TOKEN. Set both.
    assert github_cred_env("ghs_x") == {"GH_TOKEN": "ghs_x", "GITHUB_TOKEN": "ghs_x"}


def test_cloudflare_cred_env() -> None:
    assert cloudflare_cred_env("cf_tok") == {"CLOUDFLARE_API_TOKEN": "cf_tok"}


# --- provider cred_to_env methods + EnvCredProvider membership --------------

def test_cloud_providers_implement_env_cred_provider() -> None:
    aws = AwsStsProvider(assume_role=lambda req: {})
    gcp = GcpImpersonationProvider(generate_token=lambda req: {})
    gh = GitHubAppProvider(create_token=lambda req: {})
    cf = CloudflareTokenProvider(create=lambda req: {}, delete=lambda h: None)
    for p in (aws, gcp, gh, cf):
        assert isinstance(p, EnvCredProvider)
    assert aws.cred_to_env(_AWS_BLOB)["AWS_SESSION_TOKEN"] == "session"
    assert gcp.cred_to_env("t") == {"CLOUDSDK_AUTH_ACCESS_TOKEN": "t"}
    assert gh.cred_to_env("t")["GH_TOKEN"] == "t"
    assert cf.cred_to_env("t") == {"CLOUDFLARE_API_TOKEN": "t"}


def test_memory_provider_is_not_an_env_cred_provider() -> None:
    # Memory's in-container token is handled by the kagura CLI, not generic env
    # injection — so it must NOT be swept into the container env blindly.
    mem = MemoryCloudProvider(exchange=lambda req: {})
    assert not isinstance(mem, EnvCredProvider)


# --- broker.container_env ---------------------------------------------------

async def test_container_env_from_acquired_aws_lease() -> None:
    aws = AwsStsProvider(
        assume_role=lambda req: {
            "Credentials": {
                "AccessKeyId": "AKIA",
                "SecretAccessKey": "secret",
                "SessionToken": "session",
            }
        }
    )
    broker = CredentialBroker({"aws": aws}, clock=lambda: 0.0)
    lease = await broker.acquire(
        "aws", scope="arn:aws:iam::1:role/r", ttl=900, budget=Budget(3600)
    )
    env = broker.container_env([lease])
    assert env == {
        "AWS_ACCESS_KEY_ID": "AKIA",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_SESSION_TOKEN": "session",
    }


def test_container_env_skips_checkpointed_lease_without_cred() -> None:
    aws = AwsStsProvider(assume_role=lambda req: {})
    broker = CredentialBroker({"aws": aws}, clock=lambda: 0.0)
    checkpointed = Lease(
        provider="aws", scope="r", budget=Budget(3600), cred=None,
        expires_at=0.0, handle=None, stateful=False,
    )
    assert broker.container_env([checkpointed]) == {}


def test_container_env_skips_provider_without_env_mapping() -> None:
    mem = MemoryCloudProvider(exchange=lambda req: {"access_token": "AT"})
    broker = CredentialBroker({"memory": mem}, clock=lambda: 0.0)
    lease = Lease(
        provider="memory", scope="memory:read", budget=Budget(3600),
        cred="AT", expires_at=0.0, handle=None, stateful=False,
    )
    assert broker.container_env([lease]) == {}


def test_container_env_detects_key_conflict() -> None:
    aws = AwsStsProvider(assume_role=lambda req: {})
    broker = CredentialBroker({"aws": aws}, clock=lambda: 0.0)
    other = json.dumps(
        {
            "AWS_ACCESS_KEY_ID": "DIFFERENT",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "AWS_SESSION_TOKEN": "session",
        }
    )
    l1 = Lease(provider="aws", scope="r1", budget=Budget(3600), cred=_AWS_BLOB,
               expires_at=0.0, handle=None, stateful=False)
    l2 = Lease(provider="aws", scope="r2", budget=Budget(3600), cred=other,
               expires_at=0.0, handle=None, stateful=False)
    with pytest.raises(ValueError, match="conflict"):
        broker.container_env([l1, l2])


def test_container_env_unknown_provider_raises() -> None:
    broker = CredentialBroker({}, clock=lambda: 0.0)
    ghost = Lease(provider="ghost", scope="r", budget=Budget(3600), cred="x",
                  expires_at=0.0, handle=None, stateful=False)
    with pytest.raises(KeyError):
        broker.container_env([ghost])


def test_container_env_flows_into_docker_run_args() -> None:
    # The whole point: env produced here is what docker_run_args injects as -e.
    aws = AwsStsProvider(
        assume_role=lambda req: {
            "Credentials": {
                "AccessKeyId": "AKIA",
                "SecretAccessKey": "secret",
                "SessionToken": "session",
            }
        }
    )
    broker = CredentialBroker({"aws": aws}, clock=lambda: 0.0)
    lease = Lease(provider="aws", scope="r", budget=Budget(3600), cred=_AWS_BLOB,
                  expires_at=0.0, handle=None, stateful=False)
    spec = LaunchSpec(image="img:tag", env=broker.container_env([lease]))
    args = docker_run_args(spec)
    assert "AWS_ACCESS_KEY_ID=AKIA" in args
    assert "AWS_SESSION_TOKEN=session" in args
