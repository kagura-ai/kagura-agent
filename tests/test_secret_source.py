"""v0.7 (Task 1): pluggable host-side secret sources by field suffix.

A secret reference is a ``<name><suffix>`` field; the suffix selects the backend:

  - ``*_env``     → host environment variable name
  - ``*_file``    → host file path
  - ``*_keyring`` → host OS-keychain entry, given as ``"service/username"``

`resolve_ref` dispatches one reference to the right `SecretSource`;
`resolve_secret_field` is the host-facing convenience that wires the default
sources. Both are **host-side only**, fail-closed:

  - unknown suffix / empty value / resolution failure → SecretSourceError
  - the resolved secret value is never included in an error message

Adding a backend (e.g. Vault) is one new `SecretSource` + one `SECRET_SUFFIXES`
entry — no per-kind schema change. The real OS-keychain call is the only line
not exercised here (it carries `# pragma: no cover`); every dispatch and
fail-closed path is tested through injected sources.
"""

import pytest

from kagura_agent.membrane.secret_source import (
    SECRET_SUFFIXES,
    EnvSource,
    FileSource,
    KeyringSource,
    SecretSource,
    SecretSourceError,
    default_sources,
    resolve_ref,
    resolve_secret_field,
)

# --------------------------------------------------------------------------
# SECRET_SUFFIXES + protocol conformance
# --------------------------------------------------------------------------


def test_secret_suffixes_value_and_order():
    # Order is the resolution precedence the registry/validator rely on; keep
    # the env/file pair first (v0.6 behavior) with keyring appended.
    assert SECRET_SUFFIXES == ("_env", "_file", "_keyring")


@pytest.mark.parametrize(
    "source",
    [EnvSource(), FileSource(), KeyringSource()],
)
def test_each_source_conforms_to_protocol(source):
    assert isinstance(source, SecretSource)
    assert source.suffix in SECRET_SUFFIXES
    assert callable(source.resolve)


def test_sources_have_distinct_expected_suffixes():
    assert EnvSource().suffix == "_env"
    assert FileSource().suffix == "_file"
    assert KeyringSource().suffix == "_keyring"


# --------------------------------------------------------------------------
# EnvSource
# --------------------------------------------------------------------------


def test_env_source_returns_value():
    src = EnvSource(get_env={"CF_TOKEN": "tok-123"}.get)
    assert src.resolve("parent_token_env", "CF_TOKEN") == "tok-123"


def test_env_source_unset_is_fail_closed():
    with pytest.raises(SecretSourceError, match="unset or empty"):
        EnvSource(get_env={}.get).resolve("parent_token_env", "CF_TOKEN")


@pytest.mark.parametrize("blank", ["", "   ", "\n", "\t"])
def test_env_source_empty_is_fail_closed(blank):
    with pytest.raises(SecretSourceError):
        EnvSource(get_env={"CF_TOKEN": blank}.get).resolve("parent_token_env", "CF_TOKEN")


def test_env_source_getter_error_normalized():
    def boom(_name):
        raise RuntimeError("backend down")

    with pytest.raises(SecretSourceError, match="CF_TOKEN"):
        EnvSource(get_env=boom).resolve("parent_token_env", "CF_TOKEN")


def test_env_source_does_not_leak_value_in_error():
    def leaky(_name):
        raise RuntimeError("sk-live-DEADBEEF")

    with pytest.raises(SecretSourceError) as exc:
        EnvSource(get_env=leaky).resolve("parent_token_env", "CF_TOKEN")
    assert "sk-live-DEADBEEF" not in str(exc.value)


# --------------------------------------------------------------------------
# FileSource
# --------------------------------------------------------------------------


def test_file_source_returns_contents_stripping_trailing_newline():
    src = FileSource(read_file=lambda _p: "secret-body\n")
    assert src.resolve("private_key_file", "/run/secrets/gh.pem") == "secret-body"


def test_file_source_keeps_internal_whitespace():
    src = FileSource(read_file=lambda _p: "-----BEGIN-----\nline2\n")
    assert src.resolve("private_key_file", "/x") == "-----BEGIN-----\nline2"


def test_file_source_strips_crlf_line_ending():
    src = FileSource(read_file=lambda _p: "secret\r\n")
    assert src.resolve("private_key_file", "/x") == "secret"


def test_file_source_default_reader_reads_real_file(tmp_path):
    # The default FileSource reads a real host file (the production reader).
    p = tmp_path / "gh.pem"
    p.write_text("real-file-secret\n", encoding="utf-8")
    assert FileSource().resolve("private_key_file", str(p)) == "real-file-secret"


@pytest.mark.parametrize("blank", ["", "   ", "\n", "\r\n"])
def test_file_source_empty_is_fail_closed(blank):
    with pytest.raises(SecretSourceError):
        FileSource(read_file=lambda _p: blank).resolve("private_key_file", "/x")


def test_file_source_read_error_normalized():
    def boom(_p):
        raise OSError("no such file")

    with pytest.raises(SecretSourceError, match="private_key_file"):
        FileSource(read_file=boom).resolve("private_key_file", "/run/secrets/x")


def test_file_source_read_error_does_not_leak_value():
    def leaky(_p):
        raise RuntimeError("-----BEGIN RSA PRIVATE KEY-----")

    with pytest.raises(SecretSourceError) as exc:
        FileSource(read_file=leaky).resolve("private_key_file", "/x")
    assert "BEGIN RSA PRIVATE KEY" not in str(exc.value)


# --------------------------------------------------------------------------
# KeyringSource (real OS keychain injected — no extra needed in tests)
# --------------------------------------------------------------------------


def test_keyring_source_returns_value():
    src = KeyringSource(get_password=lambda svc, user: f"pw[{svc}/{user}]")
    assert src.resolve("parent_token_keyring", "gh-app/agent") == "pw[gh-app/agent]"


def test_keyring_source_unset_is_fail_closed():
    src = KeyringSource(get_password=lambda svc, user: None)
    with pytest.raises(SecretSourceError, match="unset or empty"):
        src.resolve("parent_token_keyring", "gh-app/agent")


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_keyring_source_empty_is_fail_closed(blank):
    src = KeyringSource(get_password=lambda svc, user: blank)
    with pytest.raises(SecretSourceError):
        src.resolve("parent_token_keyring", "gh-app/agent")


@pytest.mark.parametrize(
    "bad_ref",
    ["gh-app", "/agent", "gh-app/", "", "  ", "  /agent", "svc/   ", "   /   "],
)
def test_keyring_source_malformed_ref_is_fail_closed(bad_ref):
    # The ref must encode both service and username as "service/username".
    # A whitespace-only half ("  /agent", "svc/   ") must fail closed here, not
    # reach the keychain with a blank service/username.
    src = KeyringSource(get_password=lambda svc, user: "should-not-be-called")
    with pytest.raises(SecretSourceError, match="service/username"):
        src.resolve("parent_token_keyring", bad_ref)


def test_keyring_source_backend_error_normalized():
    def boom(_svc, _user):
        raise RuntimeError("keychain locked")

    with pytest.raises(SecretSourceError, match="parent_token_keyring"):
        KeyringSource(get_password=boom).resolve("parent_token_keyring", "gh-app/agent")


def test_keyring_missing_extra_fails_closed_with_install_hint(monkeypatch):
    # When the optional 'keyring' extra is not installed, the default backend's
    # import guard must fail closed with an actionable install hint — not an
    # opaque ImportError. Forcing `import keyring` to raise simulates the
    # extra-absent host regardless of whether keyring happens to be installed.
    import sys

    from kagura_agent.membrane import secret_source as ss

    monkeypatch.setitem(sys.modules, "keyring", None)
    with pytest.raises(SecretSourceError, match="optional 'keyring' extra is required"):
        ss._import_keyring()


def test_keyring_source_propagates_secret_source_error_unchanged():
    # A SecretSourceError from the backend (e.g. the missing-'keyring'-extra
    # hint raised by the default reader) must propagate verbatim, not be
    # re-wrapped into the generic "could not read keyring entry" message.
    def missing_extra(_svc, _user):
        raise SecretSourceError("the optional 'keyring' extra is required")

    with pytest.raises(SecretSourceError, match="optional 'keyring' extra is required"):
        KeyringSource(get_password=missing_extra).resolve("parent_token_keyring", "gh-app/agent")


def test_keyring_source_backend_error_does_not_leak_value():
    def leaky(_svc, _user):
        raise RuntimeError("super-secret-pw")

    with pytest.raises(SecretSourceError) as exc:
        KeyringSource(get_password=leaky).resolve("parent_token_keyring", "gh-app/agent")
    assert "super-secret-pw" not in str(exc.value)


def test_keyring_source_splits_on_first_slash_only():
    # A username may itself contain a slash; only the first separator splits.
    seen = {}

    def capture(svc, user):
        seen["svc"], seen["user"] = svc, user
        return "ok"

    KeyringSource(get_password=capture).resolve("t_keyring", "svc/user/with/slash")
    assert seen == {"svc": "svc", "user": "user/with/slash"}


# --------------------------------------------------------------------------
# default_sources + resolve_ref dispatch
# --------------------------------------------------------------------------


def test_default_sources_maps_every_suffix():
    sources = default_sources()
    assert set(sources) == set(SECRET_SUFFIXES)
    assert sources["_env"].suffix == "_env"
    assert sources["_file"].suffix == "_file"
    assert sources["_keyring"].suffix == "_keyring"


def test_resolve_ref_dispatches_by_suffix():
    sources = {
        "_env": EnvSource(get_env={"X": "env-val"}.get),
        "_file": FileSource(read_file=lambda _p: "file-val\n"),
        "_keyring": KeyringSource(get_password=lambda s, u: "kr-val"),
    }
    assert resolve_ref("a_env", "X", sources=sources) == "env-val"
    assert resolve_ref("b_file", "/p", sources=sources) == "file-val"
    assert resolve_ref("c_keyring", "svc/user", sources=sources) == "kr-val"


def test_resolve_ref_unknown_suffix_is_fail_closed():
    with pytest.raises(SecretSourceError, match="not a secret reference"):
        resolve_ref("role_arn", "arn:...", sources=default_sources())


def test_resolve_ref_suffix_without_source_is_fail_closed():
    # A recognized suffix whose source was not provided must fail closed, not
    # KeyError or silently return nothing.
    only_env = {"_env": EnvSource(get_env={"X": "v"}.get)}
    with pytest.raises(SecretSourceError):
        resolve_ref("k_keyring", "svc/user", sources=only_env)


# --------------------------------------------------------------------------
# resolve_secret_field — host-facing convenience
# --------------------------------------------------------------------------


def test_resolve_secret_field_env():
    val = resolve_secret_field("parent_token_env", "CF_TOKEN", get_env={"CF_TOKEN": "t"}.get)
    assert val == "t"


def test_resolve_secret_field_file():
    val = resolve_secret_field("private_key_file", "/x", read_file=lambda _p: "body\n")
    assert val == "body"


def test_resolve_secret_field_keyring():
    val = resolve_secret_field(
        "parent_token_keyring", "svc/user", get_password=lambda s, u: "kr"
    )
    assert val == "kr"


def test_resolve_secret_field_unknown_suffix_is_fail_closed():
    with pytest.raises(SecretSourceError, match="not a secret reference"):
        resolve_secret_field("role_arn", "arn:...")


def test_resolve_secret_field_does_not_leak_value_on_keyring_error():
    def leaky(_s, _u):
        raise RuntimeError("leaked-kr-secret")

    with pytest.raises(SecretSourceError) as exc:
        resolve_secret_field("parent_token_keyring", "svc/user", get_password=leaky)
    assert "leaked-kr-secret" not in str(exc.value)


def test_resolve_secret_field_default_env_reads_real_environment(monkeypatch):
    monkeypatch.setenv("KAGURA_TEST_SECRET_SRC", "from-real-env")
    assert resolve_secret_field("parent_token_env", "KAGURA_TEST_SECRET_SRC") == "from-real-env"
