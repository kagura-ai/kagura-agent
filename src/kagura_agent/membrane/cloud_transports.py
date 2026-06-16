"""Real transport adapters for the cloud cred providers (#39).

The providers in `providers.py` take their network call as an injected callable
so the core stays dependency-free and unit-testable. This module supplies those
callables from the real SDKs (boto3 / google-auth / httpx / PyJWT), behind the
optional extras `aws` / `gcp` / `github` / `cloudflare`.

Like `core/brain/sdk_engine.py`, the SDK-backed factory functions are the
deployment edge: each needs live clouds and the optional SDKs, so it is marked
`# pragma: no cover` and lazy-imports its SDK. The cockpit (trusted host)
constructs these and hands the resulting providers to the `CredentialBroker`;
the agent container never sees the SDKs or the root credential — only the
short-lived minted cred.

`build_broker` (#58) is the opposite: the *pure* orchestration that turns a
validated registry into a live `CredentialBroker` (dedup, host-side secret
resolution, dispatch to a provider factory, assembly). It is unit-tested with a
stub factory, so this module is no longer wholesale coverage-omitted — only the
`# pragma: no cover` SDK factory bodies are excluded.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from kagura_agent.membrane.lease import CredentialBroker, CredProvider
from kagura_agent.membrane.providers import (
    AwsStsProvider,
    CloudflareTokenProvider,
    GcpImpersonationProvider,
    GitHubAppProvider,
    StaticEnvProvider,
)
from kagura_agent.membrane.registry import ProviderSpec, kind_schema
from kagura_agent.membrane.registry_io import (
    EnvResolver,
    FileResolver,
    _read_file,
    resolve_secret_ref,
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


# --------------------------------------------------------------------------
# build_broker — the registry → live CredentialBroker deployment edge (#58)
# --------------------------------------------------------------------------

#: Construct a live provider from a validated spec and its host-resolved secrets.
ProviderFactory = Callable[[ProviderSpec, Mapping[str, str]], CredProvider]


def _resolve_spec_secrets(
    spec: ProviderSpec,
    *,
    resolve_env: EnvResolver,
    resolve_file: FileResolver,
) -> dict[str, str]:
    """Resolve every ``*_env`` / ``*_file`` secret a spec's kind declares into
    ``{logical_name: value}`` — host-side, via #57's resolver.

    Required secrets are guaranteed present by ``parse_registry``; optional ones
    are resolved only when their reference field is present.
    """
    resolved: dict[str, str] = {}
    for ref in kind_schema(spec.kind).secrets:
        env_field, file_field = f"{ref.name}_env", f"{ref.name}_file"
        has_env, has_file = env_field in spec.fields, file_field in spec.fields
        if has_env and has_file:
            # parse_registry already rejects this, but build_broker accepts any
            # Iterable[ProviderSpec] (e.g. a hand-built spec from #59/#60), so
            # guard here too rather than silently letting _env win and dropping
            # the _file reference (wrong credential source, no diagnostic).
            raise ValueError(
                f"provider {spec.name!r}: ambiguous secret {ref.name!r} — both "
                f"{env_field} and {file_field} are set; provide exactly one"
            )
        field = env_field if has_env else file_field if has_file else None
        if field is not None:
            resolved[ref.name] = resolve_secret_ref(
                field, spec.fields[field], get_env=resolve_env, read_file=resolve_file
            )
    return resolved


def build_broker(
    registry: Iterable[ProviderSpec],
    *,
    clock: Callable[[], float],
    resolve_env: EnvResolver = os.environ.get,
    resolve_file: FileResolver = _read_file,
    _factory: ProviderFactory | None = None,
) -> CredentialBroker:
    """Build a live ``CredentialBroker`` from a validated provider registry.

    Pure orchestration: for each spec, refuse a duplicate provider name
    fail-closed, resolve its secret references **host-side**, and dispatch to
    ``_factory(spec, resolved_secrets)`` to construct the live provider; then
    assemble the broker. The SDK-backed construction lives behind ``_factory``
    (default: :func:`_default_factory`, the deployment edge).

    Secrets are resolved here, on the trusted host — never inside the agent
    container (membrane invariant). ``memory_cloud`` is read-locked by
    construction: the registry cannot express write approval, which only ever
    comes via the #14/#15 device-flow HITL graduation path.
    """
    factory = _factory if _factory is not None else _default_factory
    providers: dict[str, CredProvider] = {}
    for spec in registry:
        if spec.name in providers:
            raise ValueError(
                f"duplicate provider name {spec.name!r} in registry — refusing to "
                "silently override (two specs would map one name to different accounts)"
            )
        secrets = _resolve_spec_secrets(spec, resolve_env=resolve_env, resolve_file=resolve_file)
        providers[spec.name] = factory(spec, secrets)
    return CredentialBroker(providers, clock=clock)


def _default_factory(  # pragma: no cover - deployment edge (needs cloud SDKs)
    spec: ProviderSpec, secrets: Mapping[str, str]
) -> CredProvider:
    """Map a spec + host-resolved secrets onto a live provider via the SDK factories.

    Handles the kinds buildable from the registry plus the host's ambient
    credentials (``aws_sts`` / ``gcp_impersonation`` / ``github_app``). Kinds that
    additionally need a deployment-supplied transport callable (``cloudflare`` →
    ``token_spec``, ``memory_cloud`` → ``exchange``) or a provider not yet
    implemented (``static_env`` → #61) raise an actionable error directing the
    deployer to pass a custom ``_factory``.
    """
    kind = spec.kind
    if kind == "aws_sts":
        return aws_provider(session_name=str(spec.fields.get("session_name", "kagura-agent")))
    if kind == "gcp_impersonation":
        return gcp_provider()
    if kind == "github_app":
        return github_app_provider(
            app_id=str(spec.fields["app_id"]), private_key=secrets["private_key"]
        )
    if kind == "cloudflare":
        raise ValueError(
            f"provider {spec.name!r} (cloudflare) needs a deployment-supplied token_spec "
            "callable (the permission-group schema is account-specific) — pass a custom "
            "_factory to build_broker"
        )
    if kind == "memory_cloud":
        raise ValueError(
            f"provider {spec.name!r} (memory_cloud) needs a deployment-supplied exchange "
            "callable (the kagura auth session) — pass a custom _factory to build_broker"
        )
    if kind == "static_env":
        value_env = spec.fields.get("value_env")
        if value_env is None:
            raise ValueError(
                f"provider {spec.name!r} (static_env) needs value_env to name the container "
                "env var; value_file is not supported for static_env"
            )
        return StaticEnvProvider(
            value=secrets["value"],
            env_var=str(value_env),
            # Pass the raw value (no bool() coercion — that would make a quoted
            # "false" truthy); StaticEnvProvider's identity gate is the guard.
            standing_secret=spec.fields.get("standing_secret", False),
        )
    raise ValueError(f"provider {spec.name!r}: unsupported kind {kind!r}")
