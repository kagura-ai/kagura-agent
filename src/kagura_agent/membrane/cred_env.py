"""Leased cred -> container env translation (#39).

A `Lease.cred` carries a provider-specific cred shape. To let the agent run
`aws`/`gcloud`/`gh`/wrangler against it, that shape has to land in the
container's environment under the variable names each tool reads. These
translators are pure (string in, env dict out) so the convention is unit-tested
without any cloud SDK; the broker stitches them together in `container_env`.

`EnvCredProvider` marks a provider whose cred maps to container env. It is
deliberately a *separate* capability from `CredProvider`: `MemoryCloudProvider`
mints a token too, but memory is reached by the kagura CLI inside the container,
not by a generic env var — so it must NOT implement this, and `container_env`
skips it rather than guessing an env var for it.
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

_AWS_KEYS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")


def aws_cred_env(cred: str) -> dict[str, str]:
    """Parse the AWS STS JSON blob (`AwsStsProvider.mint`) into the 3 AWS_* vars.

    Fail-closed: a non-JSON cred or a blob missing any of the three keys raises,
    rather than silently injecting a partial/garbage AWS env that would fail
    deep inside an `aws` call with an opaque error.
    """
    try:
        data = json.loads(cred)
    except (ValueError, TypeError) as e:
        raise ValueError(f"AWS cred is not valid JSON: {e}") from e
    if not isinstance(data, dict) or any(k not in data for k in _AWS_KEYS):
        raise ValueError(
            f"AWS cred JSON must contain {_AWS_KEYS!r}; got keys "
            f"{sorted(data) if isinstance(data, dict) else type(data).__name__}"
        )
    return {k: str(data[k]) for k in _AWS_KEYS}


def gcp_cred_env(token: str) -> dict[str, str]:
    """The access token the `gcloud` CLI uses when this var is set."""
    return {"CLOUDSDK_AUTH_ACCESS_TOKEN": token}


def github_cred_env(token: str) -> dict[str, str]:
    """`gh` prefers GH_TOKEN; GITHUB_TOKEN is what many tools/actions read. Both."""
    return {"GH_TOKEN": token, "GITHUB_TOKEN": token}


def cloudflare_cred_env(token: str) -> dict[str, str]:
    """The API token wrangler / the Cloudflare CLI read."""
    return {"CLOUDFLARE_API_TOKEN": token}


@runtime_checkable
class EnvCredProvider(Protocol):
    """A `CredProvider` whose minted cred maps to container environment vars."""

    def cred_to_env(self, cred: str) -> dict[str, str]: ...
