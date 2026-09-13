from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from probabilistic_record_linkage.core.config import Settings, get_settings

pytestmark = pytest.mark.unit


def test_defaults_fail_closed() -> None:
    settings = Settings(_env_file=None)
    assert settings.api_key_sha256 == []
    assert settings.cors_allow_origins == []
    assert settings.enable_docs is False
    assert settings.metrics_require_auth is True
    assert settings.host == "127.0.0.1"
    assert settings.jwt_secret is None
    assert settings.id_hash_key is None
    assert settings.pprl_key_file is None
    assert settings.upload_max_rows == 20_000
    assert settings.job_max_per_client == 1


def test_environment_variables_are_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "not-a-real-secret-" + "x" * 20
    monkeypatch.setenv("APP_API_KEY_SHA256", f"{'a' * 64}, {'b' * 64}")
    monkeypatch.setenv("APP_CORS_ALLOW_ORIGINS", "https://app.example.com,http://localhost:3000")
    monkeypatch.setenv("APP_JWT_SECRET", secret)
    monkeypatch.setenv("APP_UPLOAD_MAX_ROWS", "500")
    monkeypatch.setenv("APP_PPRL_KEY_FILE", "/run/secrets/pprl.key")
    settings = Settings(_env_file=None)
    assert settings.api_key_sha256 == ["a" * 64, "b" * 64]
    assert settings.cors_allow_origins == ["https://app.example.com", "http://localhost:3000"]
    assert settings.jwt_secret is not None
    assert settings.jwt_secret.get_secret_value() == secret
    assert secret not in repr(settings)
    assert settings.upload_max_rows == 500
    assert settings.pprl_key_file == Path("/run/secrets/pprl.key")


@pytest.mark.parametrize("origins", [["*"], ["https://example.com/"], ["example.com"]])
def test_wildcard_or_malformed_origins_are_rejected(origins: list[str]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, cors_allow_origins=origins)


@pytest.mark.parametrize("digest", ["A" * 64, "abc", "g" * 64])
def test_non_sha256_key_hashes_are_rejected(digest: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, api_key_sha256=[digest])


@pytest.mark.parametrize("field", ["jwt_secret", "id_hash_key"])
def test_short_secrets_are_rejected(field: str) -> None:
    with pytest.raises(ValidationError, match="at least 32"):
        Settings(_env_file=None, **{field: "short"})  # type: ignore[arg-type]


def test_limits_are_validated() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, duckdb_memory_limit="lots")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, upload_max_rows=1)


def test_env_example_documents_every_setting(repo_root: Path) -> None:
    lines = (repo_root / ".env.example").read_text(encoding="utf-8").splitlines()
    documented = {
        line.split("=", 1)[0].strip() for line in lines if line.strip() and not line.startswith("#")
    }
    assert documented == {f"APP_{name.upper()}" for name in Settings.model_fields}


def test_env_example_leaves_secrets_empty(repo_root: Path) -> None:
    values = dict(
        line.split("=", 1)
        for line in (repo_root / ".env.example").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )
    assert values["APP_JWT_SECRET"] == ""
    assert values["APP_ID_HASH_KEY"] == ""
    assert values["APP_PPRL_KEY_FILE"] == ""


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
