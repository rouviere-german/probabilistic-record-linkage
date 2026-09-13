"""Typed configuration loaded from environment variables (prefix ``APP_``).

Only ``.env.example`` is committed. Secrets are ``SecretStr`` so they never appear in ``repr``
or logs. API keys are configured as SHA-256 digests, so the process never holds raw keys. The
Bloom-filter key for privacy mode is deliberately *not* a setting value: only the path of a
mounted secret file is configured (``APP_PPRL_KEY_FILE``).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

__all__ = ["Settings", "get_settings"]

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_ORIGIN = re.compile(r"^https?://[A-Za-z0-9.-]+(:\d{1,5})?$")
MIN_SECRET_CHARS = 32


def _split_csv(value: object) -> object:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    """Service settings. Every field can be set with ``APP_<FIELD_NAME>``."""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- service ---
    service_name: str = Field(
        default="probabilistic-record-linkage", pattern=r"^[a-z0-9][a-z0-9-]{0,62}$"
    )
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    enable_docs: bool = False

    # --- HTTP security ---
    api_key_sha256: Annotated[list[str], NoDecode] = Field(default_factory=list)
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    rate_limit_per_second: float = Field(default=5.0, gt=0)
    rate_limit_burst: int = Field(default=20, ge=1)
    rate_limit_max_clients: int = Field(default=10_000, ge=1)
    max_body_bytes: int = Field(default=8_388_608, ge=1)
    metrics_require_auth: bool = True

    # --- scoped JWT authentication ---
    jwt_secret: SecretStr | None = None
    jwt_issuer: str = "probabilistic-record-linkage"
    jwt_audience: str = "linkage-api"
    jwt_max_ttl_seconds: int = Field(default=3600, ge=60, le=86_400)

    # --- observability ---
    tracing_exporter: Literal["none", "console"] = "none"
    id_hash_key: SecretStr | None = None

    # --- linkage ---
    data_dir: Path = Path("var")
    model_name: str = Field(default="person-linkage", pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    nicknames_file: Path | None = None
    pprl_key_file: Path | None = None
    upload_max_rows: int = Field(default=20_000, ge=2, le=100_000)
    job_max_concurrent: int = Field(default=2, ge=1, le=16)
    job_max_per_client: int = Field(default=1, ge=1, le=16)
    cluster_size_cap: int = Field(default=8, ge=2, le=1000)
    duckdb_memory_limit: str = Field(default="512MB", pattern=r"^[1-9][0-9]{0,5}(MB|GB)$")

    @field_validator("api_key_sha256", mode="before")
    @classmethod
    def _parse_key_hashes(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("api_key_sha256")
    @classmethod
    def _check_key_hashes(cls, value: list[str]) -> list[str]:
        for digest in value:
            if not _SHA256_HEX.fullmatch(digest):
                raise ValueError("api_key_sha256 entries must be lowercase hex SHA-256 digests")
        return value

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _parse_origins(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("cors_allow_origins")
    @classmethod
    def _check_origins(cls, value: list[str]) -> list[str]:
        for origin in value:
            if not _ORIGIN.fullmatch(origin):
                raise ValueError(
                    "cors_allow_origins must be explicit scheme://host[:port] origins "
                    "('*' is refused)"
                )
        return value

    @field_validator("jwt_secret", "id_hash_key")
    @classmethod
    def _check_secret_length(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value()) < MIN_SECRET_CHARS:
            raise ValueError(f"secrets must be at least {MIN_SECRET_CHARS} characters")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings (cached). Tests construct ``Settings(...)`` directly instead."""
    return Settings()
