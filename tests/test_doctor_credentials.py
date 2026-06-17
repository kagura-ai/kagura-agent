"""v0.6 (Tasks 6-7): credential-aware doctor.

`check_provider` answers "why can't the agent touch <service>?" before runtime:
it resolves each provider's secret references host-side and surfaces an
unresolved reference as a FAIL with an actionable hint (the env var / file path,
never the secret value). A required reference that is absent/unresolved FAILs; an
absent OPTIONAL reference does NOT fail (e.g. aws_sts uses ambient host creds).

`probe_provider` is the opt-in `--probe` dry-mint: acquire a short-lived scoped
token then immediately release it; a mint that succeeds but cannot be revoked is
a loud FAIL that surfaces the handle for manual cleanup.
"""

import pytest

from kagura_agent.cli.doctor import (
    FAIL,
    OK,
    WARN,
    _probe_scope,
    check_provider,
    check_providers,
    check_secret_backends,
    probe_provider,
    run_doctor,
)
from kagura_agent.membrane.registry import ProviderSpec, parse_registry


def _spec(table):
    return parse_registry(table)[0]


# --------------------------------------------------------------------------
# check_secret_backends (#66) — keyring-extra awareness
# --------------------------------------------------------------------------


def test_secret_backends_ok_when_keyring_available():
    r = check_secret_backends(keyring_available=True)
    assert r.status == OK
    assert "keyring" in r.detail


def test_secret_backends_warns_when_keyring_used_but_extra_absent():
    # The acceptance: a *_keyring reference + no extra → pre-run WARN with the
    # install hint, never an opaque run-time failure.
    registry = parse_registry(
        {"cf": {"kind": "cloudflare", "account_id": "a", "parent_token_keyring": "svc/agent"}}
    )
    r = check_secret_backends(registry, keyring_available=False)
    assert r.status == WARN
    assert "keyring" in (r.hint or "") and "install" in (r.hint or "").lower()


def test_secret_backends_ok_when_keyring_absent_but_unused():
    # env/file-only registry on a host without the keyring extra is fine — a
    # missing optional backend you are not using must NOT warn (no noise).
    registry = parse_registry(
        {"cf": {"kind": "cloudflare", "account_id": "a", "parent_token_env": "CF"}}
    )
    r = check_secret_backends(registry, keyring_available=False)
    assert r.status == OK


def test_secret_backends_ok_when_no_registry_and_keyring_absent():
    r = check_secret_backends(None, keyring_available=False)
    assert r.status == OK


def test_run_doctor_includes_secret_backends_check():
    results = run_doctor(
        memory_probe=lambda: True,
        sdk_probe=lambda: True,
        docker_probe=lambda: True,
        egress_probe=lambda: True,
        env={"CLAUDE_CODE_SUBSCRIPTION": "1"},
    )
    assert any(r.name == "secret-backends" for r in results)


def test_run_doctor_keyring_registry_without_extra_is_warn_not_fail(monkeypatch):
    # #66 acceptance, end-to-end: a *_keyring registry on a host WITHOUT the extra
    # makes doctor WARN (pre-run heads-up) — it must NOT hard-fail the gate. The
    # secret-backends check and the provider check render the SAME WARN (never
    # WARN + FAIL), so overall_status is WARN, not FAIL. Monkeypatched for
    # determinism regardless of whether the test host has the keyring extra.
    from kagura_agent.cli import doctor as doc

    monkeypatch.setattr(doc, "_keyring_importable", lambda: False)
    registry = parse_registry(
        {"cf": {"kind": "cloudflare", "account_id": "a", "parent_token_keyring": "svc/agent"}}
    )
    results = run_doctor(
        memory_probe=lambda: True,
        sdk_probe=lambda: True,
        docker_probe=lambda: True,
        egress_probe=lambda: True,
        env={"CLAUDE_CODE_SUBSCRIPTION": "1"},
        registry=registry,
    )
    assert doc.overall_status(results) == WARN
    assert all(r.status != FAIL for r in results)  # keyring-extra-absent is never a FAIL


# --------------------------------------------------------------------------
# check_provider — required / optional / resolution
# --------------------------------------------------------------------------


def test_check_provider_ok_when_required_reference_resolves():
    spec = _spec({"cf": {"kind": "cloudflare", "account_id": "a", "parent_token_env": "CF"}})
    r = check_provider(spec, resolve_env={"CF": "tok"}.get)
    assert r.status == OK
    assert r.name == "provider:cf"


def test_check_provider_fail_when_required_reference_unresolved():
    spec = _spec({"cf": {"kind": "cloudflare", "account_id": "a", "parent_token_env": "CF"}})
    r = check_provider(spec, resolve_env={}.get)  # CF unset
    assert r.status == FAIL
    assert "CF" in (r.hint or "")  # names the env var (no value exists to leak here)


def test_check_provider_ok_when_required_keyring_reference_resolves():
    # #65: doctor is suffix-agnostic — a *_keyring reference the run path can
    # resolve must pass doctor too (doctor predicts the run). keyring_available
    # is forced True (the extra is present), so the reference is actually resolved.
    spec = _spec(
        {"cf": {"kind": "cloudflare", "account_id": "a", "parent_token_keyring": "svc/agent"}}
    )
    r = check_provider(spec, resolve_keyring=lambda s, u: "kr-tok", keyring_available=True)
    assert r.status == OK


def test_check_provider_fail_when_required_keyring_reference_unresolved():
    # Extra present (keyring_available=True) but the keychain entry is absent →
    # a genuine resolution failure is a FAIL (distinct from the missing-extra WARN).
    spec = _spec(
        {"cf": {"kind": "cloudflare", "account_id": "a", "parent_token_keyring": "svc/agent"}}
    )
    r = check_provider(spec, resolve_keyring=lambda s, u: None, keyring_available=True)
    assert r.status == FAIL
    assert "keyring" in (r.hint or "")  # names the keychain ref, not the value


def test_check_provider_warns_keyring_ref_when_extra_absent():
    # #66: a *_keyring reference on a host WITHOUT the keyring extra is a WARN, not
    # a FAIL — keyring availability is host-dependent (deploy-time concern). This
    # is what lets doctor's overall verdict stay non-FAIL, matching the
    # secret-backends WARN rather than contradicting it.
    spec = _spec(
        {"cf": {"kind": "cloudflare", "account_id": "a", "parent_token_keyring": "svc/agent"}}
    )
    r = check_provider(spec, keyring_available=False)
    assert r.status == WARN
    assert "keyring" in (r.hint or "") and "install" in (r.hint or "").lower()


def test_check_provider_optional_reference_absent_is_not_fail():
    # aws_sts.parent_token is OPTIONAL — its absence means "use ambient host
    # creds", so the provider must NOT be reported as broken.
    spec = _spec({"aws": {"kind": "aws_sts", "role_arn": "arn:aws:iam::1:role/a"}})
    r = check_provider(spec, resolve_env={}.get)
    assert r.status == OK
    assert "optional" in r.detail.lower()


def test_check_provider_does_not_leak_resolved_value():
    spec = _spec({"cf": {"kind": "cloudflare", "account_id": "a", "parent_token_env": "CF"}})
    r = check_provider(spec, resolve_env={"CF": "super-secret"}.get)
    assert "super-secret" not in (r.detail + (r.hint or ""))


def test_check_provider_fail_on_ambiguous_hand_built_spec():
    # parse_registry rejects both *_env and *_file, but check_provider accepts any
    # ProviderSpec (e.g. hand-built by #60 wizard) — diagnose the ambiguity.
    spec = ProviderSpec(
        name="cf",
        kind="cloudflare",
        fields={"account_id": "a", "parent_token_env": "CF", "parent_token_file": "/run/s"},
    )
    r = check_provider(spec, resolve_env={"CF": "t"}.get)
    assert r.status == FAIL
    assert "ambiguous" in (r.hint or "")


def test_check_provider_fail_on_required_reference_absent_hand_built_spec():
    spec = ProviderSpec(name="cf", kind="cloudflare", fields={"account_id": "a"})
    r = check_provider(spec, resolve_env={}.get)
    assert r.status == FAIL
    assert "required" in (r.hint or "")


def test_check_providers_one_result_per_provider():
    specs = parse_registry(
        {
            "aws": {"kind": "aws_sts", "role_arn": "r"},
            "cf": {"kind": "cloudflare", "account_id": "a", "parent_token_env": "CF"},
        }
    )
    results = check_providers(specs, resolve_env={"CF": "t"}.get)
    assert {r.name for r in results} == {"provider:aws", "provider:cf"}


# --------------------------------------------------------------------------
# run_doctor — registry-aware
# --------------------------------------------------------------------------


def _probes(memory=True, sdk=True, docker=True, egress=True):
    return dict(
        memory_probe=lambda: memory,
        sdk_probe=lambda: sdk,
        docker_probe=lambda: docker,
        egress_probe=lambda: egress,
    )


def test_run_doctor_without_registry_is_unchanged():
    results = run_doctor(**_probes(), env={"CLAUDE_CODE_SUBSCRIPTION": "1"})
    assert not any(r.name.startswith("provider:") for r in results)


def test_run_doctor_with_registry_appends_provider_checks():
    specs = _spec({"cf": {"kind": "cloudflare", "account_id": "a", "parent_token_env": "CF"}})
    results = run_doctor(
        **_probes(),
        env={"CLAUDE_CODE_SUBSCRIPTION": "1"},
        registry=[specs],
        resolve_env={"CF": "t"}.get,
    )
    provider_checks = [r for r in results if r.name == "provider:cf"]
    assert len(provider_checks) == 1
    assert provider_checks[0].status == OK


# --------------------------------------------------------------------------
# _probe_scope — safe scope derivation
# --------------------------------------------------------------------------


def test_probe_scope_memory_cloud_is_read_never_write():
    spec = _spec({"mem": {"kind": "memory_cloud", "parent_token_env": "M"}})
    assert _probe_scope(spec) == "memory:read"


def test_probe_scope_aws_is_role_arn():
    spec = _spec({"aws": {"kind": "aws_sts", "role_arn": "arn:aws:iam::1:role/a"}})
    assert _probe_scope(spec) == "arn:aws:iam::1:role/a"


def test_probe_scope_undeerivable_kind_is_none():
    spec = _spec(
        {
            "gh": {
                "kind": "github_app",
                "app_id": "1",
                "installation_id": "2",
                "private_key_env": "K",
            }
        }
    )
    assert _probe_scope(spec) is None


# --------------------------------------------------------------------------
# probe_provider — dry-mint + revoke (stub broker)
# --------------------------------------------------------------------------


class StubBroker:
    def __init__(self, *, mint_ok=True, revoke_ok=True):
        self.mint_ok = mint_ok
        self.revoke_ok = revoke_ok
        self.released = False

    async def acquire(self, name, *, scope, ttl, budget):
        if not self.mint_ok:
            raise RuntimeError("STS denied")

        class _Lease:
            handle = "tok-id-123"

        return _Lease()

    async def release(self, lease):
        if not self.revoke_ok:
            raise RuntimeError("delete failed")
        self.released = True


async def test_probe_provider_ok_on_mint_and_revoke():
    broker = StubBroker()
    r = await probe_provider(broker, "aws", scope="arn:...", ttl=30)
    assert r.status == OK
    assert broker.released is True


async def test_probe_provider_fail_on_mint():
    r = await probe_provider(StubBroker(mint_ok=False), "aws", scope="arn:...", ttl=30)
    assert r.status == FAIL
    assert "dry-mint failed" in r.detail


async def test_probe_provider_fail_loudly_on_revoke_failure_surfacing_handle():
    # The riskiest path: token minted but not revoked → live token leaked. Must
    # FAIL loudly and surface the handle for manual cleanup.
    r = await probe_provider(StubBroker(revoke_ok=False), "cf", scope="zone:read", ttl=30)
    assert r.status == FAIL
    assert "REVOKE FAILED" in r.detail
    assert "tok-id-123" in (r.hint or "")


async def test_probe_provider_revoke_failure_does_not_leak_token_in_hint():
    # A revoke error could echo the just-minted token; it must not reach output.
    class LeakyBroker(StubBroker):
        async def release(self, lease):
            raise RuntimeError("CF delete failed: token=SUPER-SECRET-TOKEN")

    r = await probe_provider(LeakyBroker(), "cf", scope="zone:read", ttl=30)
    assert r.status == FAIL
    assert "SUPER-SECRET-TOKEN" not in (r.detail + (r.hint or ""))
    assert "tok-id-123" in (r.hint or "")  # handle still surfaced for manual revoke


async def test_probe_provider_reraises_on_interrupt_and_surfaces_handle(capsys):
    # On Ctrl-C during revoke, a minted token must not vanish silently: surface
    # the handle to stderr, then let the interrupt propagate (do not swallow it).
    class InterruptBroker(StubBroker):
        async def release(self, lease):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        await probe_provider(InterruptBroker(), "cf", scope="zone:read", ttl=30)
    err = capsys.readouterr().err
    assert "tok-id-123" in err
    assert "NOT revoked" in err
