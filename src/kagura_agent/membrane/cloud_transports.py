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
from kagura_agent.membrane.registry import ProviderSpec, kind_schema, present_suffix_field
from kagura_agent.membrane.secret_source import (
    EnvResolver,
    FileResolver,
    KeyringResolver,
    _read_file,
    _real_keyring_get_password,
    resolve_secret_field,
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
    resolve_keyring: KeyringResolver = _real_keyring_get_password,
) -> dict[str, str]:
    """Resolve every secret a spec's kind declares into ``{logical_name: value}``
    — host-side, via the unified suffix resolver (#62/#65).

    Suffix-agnostic: whichever ``SECRET_SUFFIXES`` variant (``*_env`` / ``*_file``
    / ``*_keyring``) the operator declared is resolved through the same
    :func:`resolve_secret_field`, so a new backend works for every provider with
    no per-kind change. Required secrets are guaranteed present by
    ``parse_registry``; optional ones are resolved only when present.
    """
    resolved: dict[str, str] = {}
    for ref in kind_schema(spec.kind).secrets:
        present, suffix_fields = present_suffix_field(ref, spec.fields)
        if len(present) > 1:
            # parse_registry already rejects this, but build_broker accepts any
            # Iterable[ProviderSpec] (e.g. a hand-built spec from #59/#60), so
            # guard here too rather than silently letting one suffix win and
            # dropping the others (wrong credential source, no diagnostic).
            keys = ", ".join(suffix_fields[suf] for suf in present)
            raise ValueError(
                f"provider {spec.name!r}: ambiguous secret {ref.name!r} — set only one of {keys}"
            )
        if present:
            field = suffix_fields[present[0]]
            resolved[ref.name] = resolve_secret_field(
                field,
                spec.fields[field],
                get_env=resolve_env,
                read_file=resolve_file,
                get_password=resolve_keyring,
            )
    return resolved


def build_broker(
    registry: Iterable[ProviderSpec],
    *,
    clock: Callable[[], float],
    resolve_env: EnvResolver = os.environ.get,
    resolve_file: FileResolver = _read_file,
    resolve_keyring: KeyringResolver = _real_keyring_get_password,
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
        secrets = _resolve_spec_secrets(
            spec,
            resolve_env=resolve_env,
            resolve_file=resolve_file,
            resolve_keyring=resolve_keyring,
        )
        providers[spec.name] = factory(spec, secrets)
    return CredentialBroker(providers, clock=clock)


def _required_secret(spec: ProviderSpec, secrets: Mapping[str, str], logical: str) -> str:
    """Look up a required resolved secret, failing with the module's consistent
    ``ValueError`` (not a bare ``KeyError``) when it is absent.

    ``parse_registry`` guarantees a required secret is present, but ``build_broker``
    accepts any hand-built ``ProviderSpec`` (the #59/#60 paths), so a spec missing
    its required reference must surface the same actionable error as every other
    misconfiguration here rather than a raw ``KeyError`` deep in the factory.
    """
    try:
        return secrets[logical]
    except KeyError:
        raise ValueError(
            f"provider {spec.name!r} ({spec.kind}) is missing its required {logical!r} "
            f"secret — set {logical}_env / {logical}_file / {logical}_keyring"
        ) from None


def _reject_unhonored_parent_token(spec: ProviderSpec) -> None:
    """Refuse an explicit ``parent_token`` on a kind that cannot consume it.

    ``aws_sts`` / ``gcp_impersonation`` mint via the host's *ambient* credential
    chain (boto3 default chain / application-default credentials); there is no
    plumbing to feed them an explicit ``parent_token`` (and a single token cannot
    even satisfy AWS's key+secret auth). The schema keeps the field optional for
    its absent-means-ambient semantics, but a *present* one would be silently
    ignored — so fail loudly instead of misleading the operator (#82).
    """
    if "parent_token" in spec.fields or any(
        f"parent_token{suf}" in spec.fields for suf in ("_env", "_file", "_keyring")
    ):
        raise ValueError(
            f"provider {spec.name!r} ({spec.kind}) sets parent_token, but this kind mints "
            "with the host's ambient credentials and cannot honor an explicit parent_token "
            "— remove parent_token_* (or pass a custom _factory that consumes it)"
        )


def _default_factory(  # pragma: no cover - deployment edge (needs cloud SDKs)
    spec: ProviderSpec, secrets: Mapping[str, str]
) -> CredProvider:
    """Map a spec + host-resolved secrets onto a live provider via the SDK factories.

    Handles the kinds buildable from the registry plus the host's ambient
    credentials (``aws_sts`` / ``gcp_impersonation`` / ``github_app``). Kinds that
    additionally need a deployment-supplied transport callable (``cloudflare`` →
    ``token_spec``, ``memory_cloud`` → ``exchange``) raise an actionable error
    directing the deployer to pass a custom ``_factory``.
    """
    kind = spec.kind
    if kind == "aws_sts":
        _reject_unhonored_parent_token(spec)
        return aws_provider(session_name=str(spec.fields.get("session_name", "kagura-agent")))
    if kind == "gcp_impersonation":
        _reject_unhonored_parent_token(spec)
        return gcp_provider()
    if kind == "github_app":
        return github_app_provider(
            app_id=str(spec.fields["app_id"]),
            private_key=_required_secret(spec, secrets, "private_key"),
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
            value=_required_secret(spec, secrets, "value"),
            env_var=str(value_env),
            # Pass the raw value (no bool() coercion — that would make a quoted
            # "false" truthy); StaticEnvProvider's identity gate is the guard.
            standing_secret=spec.fields.get("standing_secret", False),
        )
    raise ValueError(f"provider {spec.name!r}: unsupported kind {kind!r}")
