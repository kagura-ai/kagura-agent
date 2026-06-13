"""Real transport adapters for the cloud cred providers (#39).

The providers in `providers.py` take their network call as an injected callable
so the core stays dependency-free and unit-testable. This module supplies those
callables from the real SDKs (boto3 / google-auth / httpx / PyJWT), behind the
optional extras `aws` / `gcp` / `github` / `cloudflare`.

Like `core/brain/sdk_engine.py`, this is the deployment edge: it needs live
clouds and the optional SDKs, so it is exercised at deployment, not in unit
tests (whole module is `# pragma: no cover`, lazy-imports its SDK, and is listed
in the coverage `omit`). The cockpit (trusted host) constructs these and hands
the resulting providers to the `CredentialBroker`; the agent container never
sees the SDKs or the root credential — only the short-lived minted cred.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from kagura_agent.membrane.providers import (
    AwsStsProvider,
    CloudflareTokenProvider,
    GcpImpersonationProvider,
    GitHubAppProvider,
)


def aws_provider(  # pragma: no cover - needs boto3 + live STS
    *, session_name: str = "kagura-agent", sts_client: Any | None = None
) -> AwsStsProvider:
    """STS AssumeRole via boto3. `scope` (the role ARN) and `ttl` come per-mint."""
    import boto3

    sts = sts_client if sts_client is not None else boto3.client("sts")
    return AwsStsProvider(
        assume_role=lambda req: sts.assume_role(**req), session_name=session_name
    )


def gcp_provider() -> GcpImpersonationProvider:  # pragma: no cover - needs google-auth + live IAM
    """SA impersonation via the IAM Credentials `generateAccessToken` REST API,
    authorized with the host's application-default credentials."""
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    session = AuthorizedSession(creds)

    def generate_token(req: Mapping[str, Any]) -> dict[str, Any]:
        name = req["name"]  # projects/-/serviceAccounts/<email>
        resp = session.post(
            f"https://iamcredentials.googleapis.com/v1/{name}:generateAccessToken",
            json={"scope": req["scope"], "lifetime": req["lifetime"]},
        )
        resp.raise_for_status()
        return dict(resp.json())  # {"accessToken": ..., "expireTime": ...}

    return GcpImpersonationProvider(generate_token=generate_token)


def github_app_provider(  # pragma: no cover - needs PyJWT + httpx + live GitHub
    *, app_id: str, private_key: str
) -> GitHubAppProvider:
    """Installation token via App JWT → POST .../access_tokens.

    `private_key` is the App's PEM. The short-lived App JWT is signed per mint
    (10-min cap, clock-skew backdated); GitHub returns a ~1h installation token.
    """
    import time

    import httpx
    import jwt

    def create_token(req: Mapping[str, Any]) -> dict[str, Any]:
        installation_id = req["installation_id"]
        now = int(time.time())
        assertion = jwt.encode(
            {"iat": now - 60, "exp": now + 540, "iss": app_id},
            private_key,
            algorithm="RS256",
        )
        resp = httpx.post(
            f"https://api.github.com/app/installations/{installation_id}/access_tokens",
            headers={
                "Authorization": f"Bearer {assertion}",
                "Accept": "application/vnd.github+json",
            },
        )
        resp.raise_for_status()
        return dict(resp.json())  # {"token": ..., "expires_at": ...}

    return GitHubAppProvider(create_token=create_token)


def cloudflare_provider(  # pragma: no cover - needs httpx + live Cloudflare
    *,
    parent_token: str,
    token_spec: Callable[[str, int], dict[str, Any]],
) -> CloudflareTokenProvider:
    """Scoped child token via the Cloudflare Tokens API (mint → use → revoke).

    `token_spec(scope, ttl)` is supplied by the deployer: it maps our `(scope,
    ttl)` onto Cloudflare's native token body (name + `policies` + `expires_on`),
    because the permission-group schema is account-specific and must not be
    guessed here. `parent_token` (the "Create Additional Tokens" token) is a root
    credential — host-only, treat as a crown jewel (see docs/operations.md).
    """
    import httpx

    base = "https://api.cloudflare.com/client/v4/user/tokens"
    headers = {"Authorization": f"Bearer {parent_token}"}

    def create(req: Mapping[str, Any]) -> dict[str, Any]:
        body = token_spec(req["scope"], req["ttl"])
        resp = httpx.post(base, headers=headers, json=body)
        resp.raise_for_status()
        return dict(resp.json())  # {"result": {"value": ..., "id": ...}, ...}

    def delete(token_id: str) -> None:
        resp = httpx.delete(f"{base}/{token_id}", headers=headers)
        resp.raise_for_status()

    return CloudflareTokenProvider(create=create, delete=delete)
