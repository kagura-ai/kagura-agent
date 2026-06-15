"""v0.6 (Task 4): host registry loader + secret-reference resolution.

`load_registry` reads a TOML file on the host and hands its `[providers]` table
to `parse_registry`. `resolve_secret_ref` turns a single `*_env` / `*_file`
reference into its secret value — **host-side only**, fail-closed:

  - `*_env` → host environment variable; unset OR empty → SecretRefError.
  - `*_file` → host file; trailing newline stripped; empty → SecretRefError.
  - any underlying error (missing var, unreadable file, decode error) is
    normalized to SecretRefError so the caller's fail-closed contract holds.

Resolution NEVER happens inside the agent container — these run in the trusted
cockpit/host and the resolved value is injected as a leased env var downstream.
"""

import pytest

from kagura_agent.membrane.registry_io import (
    SecretRefError,
    load_registry,
    resolve_secret_ref,
)

# --------------------------------------------------------------------------
# load_registry
# --------------------------------------------------------------------------


def _write(tmp_path, text):
    p = tmp_path / "kagura-agent.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_registry_parses_providers_table(tmp_path):
    p = _write(
        tmp_path,
        """
        [providers.prod-aws]
        kind = "aws_sts"
        role_arn = "arn:aws:iam::123:role/agent"

        [providers.cf]
        kind = "cloudflare"
        account_id = "acct1"
        parent_token_env = "CF_TOKEN"
        """,
    )
    specs = load_registry(p)
    assert {s.name for s in specs} == {"prod-aws", "cf"}
    assert next(s for s in specs if s.name == "cf").fields["parent_token_env"] == "CF_TOKEN"


def test_load_registry_missing_providers_table_is_empty(tmp_path):
    p = _write(tmp_path, "title = 'no providers here'\n")
    assert load_registry(p) == ()


def test_load_registry_file_not_found_is_actionable(tmp_path):
    with pytest.raises(ValueError, match="not found|no such|registry"):
        load_registry(tmp_path / "does-not-exist.toml")


def test_load_registry_invalid_toml_is_actionable(tmp_path):
    p = _write(tmp_path, "this is = = not valid toml\n")
    with pytest.raises(ValueError, match="TOML|invalid"):
        load_registry(p)


def test_load_registry_non_mapping_providers_is_actionable(tmp_path):
    p = _write(tmp_path, 'providers = "not a table"\n')
    with pytest.raises(ValueError, match="mapping|table"):
        load_registry(p)


def test_load_registry_directory_path_is_actionable(tmp_path):
    # A directory raises IsADirectoryError (an OSError) → actionable ValueError,
    # not a raw traceback.
    with pytest.raises(ValueError):
        load_registry(tmp_path)


def test_load_registry_propagates_inline_secret_fail_closed(tmp_path):
    p = _write(
        tmp_path,
        """
        [providers.cf]
        kind = "cloudflare"
        account_id = "a"
        parent_token = "sk-live-xyz"
        """,
    )
    with pytest.raises(ValueError, match="inline secret|parent_token"):
        load_registry(p)


# --------------------------------------------------------------------------
# resolve_secret_ref — env form
# --------------------------------------------------------------------------


def test_resolve_env_returns_value():
    val = resolve_secret_ref("parent_token_env", "CF_TOKEN", get_env={"CF_TOKEN": "tok-123"}.get)
    assert val == "tok-123"


def test_resolve_env_unset_is_fail_closed():
    with pytest.raises(SecretRefError, match="CF_TOKEN|unset|empty"):
        resolve_secret_ref("parent_token_env", "CF_TOKEN", get_env={}.get)


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_resolve_env_empty_is_fail_closed(blank):
    with pytest.raises(SecretRefError):
        resolve_secret_ref("parent_token_env", "CF_TOKEN", get_env={"CF_TOKEN": blank}.get)


def test_resolve_env_getter_error_normalized_to_secret_ref_error():
    def boom(_name):
        raise RuntimeError("env backend down")

    with pytest.raises(SecretRefError):
        resolve_secret_ref("parent_token_env", "CF_TOKEN", get_env=boom)


def test_resolve_env_does_not_leak_value_in_error():
    # An unset var can't leak; assert the message names the var, not a value.
    with pytest.raises(SecretRefError) as exc:
        resolve_secret_ref("parent_token_env", "CF_TOKEN", get_env={}.get)
    assert "CF_TOKEN" in str(exc.value)


def test_resolve_env_resolver_exception_does_not_leak_value():
    # A custom resolver may raise with the secret embedded in its message; the
    # SecretRefError text must NOT surface it (it stays only as __cause__).
    def leaky(_name):
        raise RuntimeError("cache says CF_TOKEN=SUPER-SECRET-VALUE")

    with pytest.raises(SecretRefError) as exc:
        resolve_secret_ref("parent_token_env", "CF_TOKEN", get_env=leaky)
    assert "SUPER-SECRET-VALUE" not in str(exc.value)
    assert isinstance(exc.value.__cause__, RuntimeError)


# --------------------------------------------------------------------------
# resolve_secret_ref — file form
# --------------------------------------------------------------------------


def test_resolve_file_returns_contents_stripping_trailing_newline():
    val = resolve_secret_ref(
        "private_key_file", "/run/secrets/gh.pem", read_file=lambda _p: "secret-body\n"
    )
    assert val == "secret-body"


def test_resolve_file_keeps_internal_whitespace():
    val = resolve_secret_ref(
        "private_key_file", "/x", read_file=lambda _p: "-----BEGIN-----\nline2\n"
    )
    assert val == "-----BEGIN-----\nline2"


def test_resolve_file_strips_crlf_line_ending():
    # A secret file written on Windows ends with \r\n; the trailing carriage
    # return must not survive (it silently breaks API auth otherwise).
    val = resolve_secret_ref("private_key_file", "/x", read_file=lambda _p: "secret\r\n")
    assert val == "secret"


@pytest.mark.parametrize("blank", ["", "\n", "   \n  ", "   ", "\r\n"])
def test_resolve_file_empty_is_fail_closed(blank):
    with pytest.raises(SecretRefError):
        resolve_secret_ref("private_key_file", "/x", read_file=lambda _p: blank)


def test_resolve_file_read_error_normalized_to_secret_ref_error():
    def boom(_path):
        raise OSError("permission denied")

    with pytest.raises(SecretRefError, match="/run/secrets/x"):
        resolve_secret_ref("private_key_file", "/run/secrets/x", read_file=boom)


def test_resolve_file_read_error_does_not_leak_value():
    def leaky(_path):
        raise OSError("partial read: contents=SUPER-SECRET-PEM")

    with pytest.raises(SecretRefError) as exc:
        resolve_secret_ref("private_key_file", "/x", read_file=leaky)
    assert "SUPER-SECRET-PEM" not in str(exc.value)


# --------------------------------------------------------------------------
# resolve_secret_ref — bad field name & default resolvers
# --------------------------------------------------------------------------


def test_resolve_non_reference_field_is_fail_closed():
    with pytest.raises(SecretRefError, match="_env|_file|reference"):
        resolve_secret_ref("role_arn", "arn:...")


def test_resolve_env_default_resolver_reads_real_environment(monkeypatch):
    monkeypatch.setenv("KAGURA_TEST_SECRET", "from-real-env")
    assert resolve_secret_ref("parent_token_env", "KAGURA_TEST_SECRET") == "from-real-env"


def test_resolve_env_default_resolver_unset_is_fail_closed(monkeypatch):
    monkeypatch.delenv("KAGURA_TEST_ABSENT", raising=False)
    with pytest.raises(SecretRefError):
        resolve_secret_ref("parent_token_env", "KAGURA_TEST_ABSENT")


def test_resolve_file_default_resolver_reads_real_file(tmp_path):
    f = tmp_path / "secret.txt"
    f.write_text("file-secret\n", encoding="utf-8")
    assert resolve_secret_ref("private_key_file", str(f)) == "file-secret"


def test_resolve_file_default_resolver_missing_is_fail_closed(tmp_path):
    with pytest.raises(SecretRefError):
        resolve_secret_ref("private_key_file", str(tmp_path / "nope.txt"))
