"""Wire settings, the linkage service and the hardened core app together."""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from fastapi import FastAPI
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer
from prometheus_client import CollectorRegistry

from probabilistic_record_linkage.api.routes import build_router
from probabilistic_record_linkage.core.app import create_app
from probabilistic_record_linkage.core.config import Settings, get_settings
from probabilistic_record_linkage.core.tracing import build_tracer_provider
from probabilistic_record_linkage.linkage.audit import AuditLog
from probabilistic_record_linkage.linkage.clustering import ClusterConfig
from probabilistic_record_linkage.linkage.monitoring import LinkageMetrics, create_linkage_metrics
from probabilistic_record_linkage.linkage.nicknames import NicknameTable
from probabilistic_record_linkage.linkage.pipeline import PipelineConfig
from probabilistic_record_linkage.linkage.service import (
    REVIEW_AUDIT_ANCHOR,
    IdHasher,
    LinkageService,
    default_audit_path,
)
from probabilistic_record_linkage.linkage.store import LinkageStore
from probabilistic_record_linkage.ml.registry import ModelRegistry

__all__ = ["build_service", "create_linkage_app", "local_service", "workdir_service"]


def _workdir_key(state_dir: Path) -> bytes:
    """Pseudonymization key for local CLI/demo state (created once, owner-only permissions)."""
    path = state_dir / "secrets" / "id-hash.key"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(secrets.token_hex(32).encode("ascii"))
    return path.read_bytes().strip()


def workdir_service(state_dir: Path) -> LinkageService:
    """Service for a local state directory, with its nickname reference data if present."""
    nick_path = state_dir / "reference" / "nicknames.json"
    nicknames = (
        NicknameTable.from_json(nick_path.read_text(encoding="utf-8"))
        if nick_path.exists()
        else None
    )
    return local_service(state_dir, id_hash_key=_workdir_key(state_dir), nicknames=nicknames)


def local_service(
    state_dir: Path,
    *,
    metrics: LinkageMetrics | None = None,
    id_hash_key: bytes | None = None,
    nicknames: NicknameTable | None = None,
    config: PipelineConfig | None = None,
    tracer: Tracer | None = None,
    max_rows: int = 2_000_000,
    memory_limit: str = "512MB",
    model_name: str = "person-linkage",
    max_concurrent_jobs: int = 2,
    max_jobs_per_client: int = 1,
) -> LinkageService:
    """A service whose store, audit log and registry live under ``state_dir``.

    The audit log's head is anchored in the DuckDB store, so a deleted or truncated log file is
    detected by ``/readyz`` and refused by the next decision.
    """
    store = LinkageStore(state_dir / "linkage.duckdb", memory_limit=memory_limit)
    try:
        audit = AuditLog(
            default_audit_path(state_dir), anchor=store.audit_anchor(REVIEW_AUDIT_ANCHOR)
        )
    except BaseException:
        store.close()
        raise
    return LinkageService(
        store=store,
        audit=audit,
        metrics=metrics or create_linkage_metrics(),
        id_hasher=IdHasher(id_hash_key or secrets.token_bytes(32)),
        registry=ModelRegistry(state_dir / "registry"),
        model_name=model_name,
        nicknames=nicknames,
        config=config,
        max_rows=max_rows,
        max_concurrent_jobs=max_concurrent_jobs,
        max_jobs_per_client=max_jobs_per_client,
        tracer=tracer,
    )


def build_service(
    settings: Settings, *, metrics: LinkageMetrics, tracer: Tracer | None = None
) -> LinkageService:
    nicknames = (
        NicknameTable.from_json(settings.nicknames_file.read_text(encoding="utf-8"))
        if settings.nicknames_file is not None
        else None
    )
    key = settings.id_hash_key.get_secret_value().encode("utf-8") if settings.id_hash_key else None
    return local_service(
        settings.data_dir,
        metrics=metrics,
        id_hash_key=key,
        nicknames=nicknames,
        config=PipelineConfig(clustering=ClusterConfig(size_cap=settings.cluster_size_cap)),
        tracer=tracer,
        max_rows=settings.upload_max_rows,
        memory_limit=settings.duckdb_memory_limit,
        model_name=settings.model_name,
        max_concurrent_jobs=settings.job_max_concurrent,
        max_jobs_per_client=settings.job_max_per_client,
    )


def create_linkage_app(
    settings: Settings | None = None,
    *,
    service: LinkageService | None = None,
    tracer_provider: TracerProvider | None = None,
) -> FastAPI:
    cfg = settings if settings is not None else get_settings()
    provider = tracer_provider or build_tracer_provider(cfg.service_name, cfg.tracing_exporter)
    if service is None:
        registry = CollectorRegistry()
        service = build_service(
            cfg,
            metrics=create_linkage_metrics(registry),
            tracer=provider.get_tracer("probabilistic_record_linkage.linkage"),
        )
    return create_app(
        cfg,
        readiness_checks=service.readiness_checks(),
        registry=service.metrics.registry,
        tracer_provider=provider,
        routers=[build_router(service, cfg)],
        state={"linkage_service": service},
    )
