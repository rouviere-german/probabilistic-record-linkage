"""Application service shared by the HTTP API and the CLI: jobs, entity lookup, review decisions.

Controls implemented here:

- job concurrency caps: a global cap and a per-client cap, both enforced before any work starts;
- personal data is returned only when the caller holds ``pii:read``; otherwise entity lookups
  and the review queue carry record ids and match probabilities only;
- record ids and principals are HMAC-pseudonymized before they reach logs, span attributes and
  stored reviewer columns;
- every reviewer decision is appended to the hash-chained audit log *before* it changes clusters,
  and re-resolution never auto-commits clusters that conflict with a cannot-link or exceed the
  size cap (they stay ``pending_review``).
"""

from __future__ import annotations

import hashlib
import hmac
import threading
import uuid
from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import polars as pl
from opentelemetry import trace
from opentelemetry.trace import Tracer

from probabilistic_record_linkage.core.logging import get_logger
from probabilistic_record_linkage.linkage.audit import AuditLog
from probabilistic_record_linkage.linkage.model import LinkageModel, load_model
from probabilistic_record_linkage.linkage.monitoring import LinkageMetrics
from probabilistic_record_linkage.linkage.nicknames import NicknameTable
from probabilistic_record_linkage.linkage.normalize import RECORD_COLUMNS
from probabilistic_record_linkage.linkage.pipeline import LinkageRun, PipelineConfig, link_sources
from probabilistic_record_linkage.linkage.store import LinkageStore
from probabilistic_record_linkage.ml.contracts import validate_source
from probabilistic_record_linkage.ml.registry import ModelRegistry, RegistryError

__all__ = [
    "IdHasher",
    "JobNotFoundError",
    "LinkageService",
    "TooManyJobsError",
]

_log = get_logger("probabilistic_record_linkage.service")


class TooManyJobsError(RuntimeError):
    """A concurrency cap was reached; retry later."""


class JobNotFoundError(KeyError):
    """Unknown job, record or review item."""


class IdHasher:
    """Keyed pseudonyms for identifiers that must not appear in logs, metrics or traces."""

    def __init__(self, key: bytes) -> None:
        if len(key) < 16:
            raise ValueError("the pseudonymization key must be at least 16 bytes")
        self._key = key

    def __call__(self, value: str) -> str:
        return hmac.new(self._key, value.encode("utf-8"), hashlib.sha256).hexdigest()[:16]


class LinkageService:
    def __init__(
        self,
        *,
        store: LinkageStore,
        audit: AuditLog,
        metrics: LinkageMetrics,
        id_hasher: IdHasher,
        registry: ModelRegistry | None = None,
        model_name: str = "person-linkage",
        nicknames: NicknameTable | None = None,
        config: PipelineConfig | None = None,
        max_concurrent_jobs: int = 2,
        max_jobs_per_client: int = 1,
        max_rows: int = 20_000,
        max_cached_runs: int = 4,
        tracer: Tracer | None = None,
    ) -> None:
        if max_concurrent_jobs < 1 or max_jobs_per_client < 1:
            raise ValueError("job caps must be >= 1")
        self.store = store
        self.audit = audit
        self.metrics = metrics
        self.hash_id = id_hasher
        self.registry = registry
        self.model_name = model_name
        self.nicknames = nicknames
        self.config = config or PipelineConfig()
        self.max_rows = max_rows
        self.tracer = tracer or trace.get_tracer("probabilistic_record_linkage.service")
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent_jobs, thread_name_prefix="link-job"
        )
        self._max_concurrent = max_concurrent_jobs
        self._max_per_client = max_jobs_per_client
        self._active: Counter[str] = Counter()
        self._lock = threading.Lock()
        self._runs: OrderedDict[str, LinkageRun] = OrderedDict()
        self._max_cached = max_cached_runs
        self._futures: dict[str, Future[None]] = {}

    # --- readiness ------------------------------------------------------------------------
    def production_model(self) -> LinkageModel | None:
        if self.registry is None:
            return None
        try:
            model, _ = load_model(self.registry, self.model_name, "production")
        except (RegistryError, ValueError, OSError):
            return None
        return model

    def readiness_checks(self) -> dict[str, Callable[[], bool]]:
        return {
            "model_manifest": lambda: self.production_model() is not None,
            "duckdb_writable": self.store.writable,
        }

    # --- jobs -----------------------------------------------------------------------------
    def submit(
        self,
        client: str,
        sources: Mapping[str, pl.DataFrame],
        *,
        use_registered_model: bool = False,
        wait: bool = False,
    ) -> str:
        total = sum(frame.height for frame in sources.values())
        if total > self.max_rows:
            raise ValueError("too many records for one job")
        for name in sorted(sources):
            validate_source(sources[name], max_rows=self.max_rows)
        client_hash = self.hash_id(client)
        with self._lock:
            if sum(self._active.values()) >= self._max_concurrent:
                raise TooManyJobsError("the service is running its maximum number of jobs")
            if self._active[client_hash] >= self._max_per_client:
                raise TooManyJobsError("this client already has a running job")
            self._active[client_hash] += 1
        job_id = uuid.uuid4().hex
        try:
            self.store.create_job(job_id, client_hash, total)
            future = self._executor.submit(
                self._run_job, job_id, client_hash, dict(sources), use_registered_model
            )
        except BaseException:
            with self._lock:
                self._active[client_hash] -= 1
            raise
        self._futures[job_id] = future
        if wait:
            future.result()
        return job_id

    def wait(self, job_id: str, timeout: float | None = None) -> None:
        future = self._futures.get(job_id)
        if future is not None:
            future.result(timeout=timeout)

    def _link(
        self,
        sources: Mapping[str, pl.DataFrame],
        *,
        use_registered_model: bool,
        must: list[tuple[str, str]] | None = None,
        cannot: list[tuple[str, str]] | None = None,
    ) -> tuple[LinkageRun, LinkageModel | None]:
        model = self.production_model() if use_registered_model else None
        run = link_sources(
            sources,
            nicknames=self.nicknames,
            config=model.pipeline if model is not None else self.config,
            tracer=self.tracer,
            metrics=self.metrics,
            fitted=model.fitted() if model is not None else None,
            thresholds=model.thresholds if model is not None else None,
            must_link=must or (),
            cannot_link=cannot or (),
            validate=False,
            max_rows=self.max_rows,
        )
        return run, model

    def _run_job(
        self,
        job_id: str,
        client_hash: str,
        sources: dict[str, pl.DataFrame],
        use_registered_model: bool,
    ) -> None:
        try:
            run, model = self._link(sources, use_registered_model=use_registered_model)
            summary: dict[str, Any] = dict(run.summary())
            summary["used_registered_model"] = model is not None
            likely = run.drift_patterns()
            if model is not None and likely.size:
                drift = model.drift_monitor().check(likely)
                self.metrics.comparison_pattern_psi.set(drift.psi)
                summary["drift"] = drift.as_dict()
            raw = pl.concat([sources[name].select(RECORD_COLUMNS) for name in sorted(sources)])
            self.store.save_run(job_id, run, raw, run.review_candidates())
            self._cache(job_id, run)
            self.store.finish_job(job_id, status="succeeded", summary=summary)
            self.metrics.review_queue_depth.set(self.store.open_review_depth())
            _log.info(
                "link_job_succeeded",
                job=self.hash_id(job_id),
                records=run.n_records,
                pairs=run.blocking.n_pairs,
            )
        except Exception as exc:  # noqa: BLE001 - a failed job is recorded, never raised into the pool
            _log.warning("link_job_failed", job=self.hash_id(job_id), error_type=type(exc).__name__)
            self.store.finish_job(job_id, status="failed", error=type(exc).__name__)
        finally:
            with self._lock:
                self._active[client_hash] -= 1

    def _cache(self, job_id: str, run: LinkageRun) -> None:
        with self._lock:
            self._runs[job_id] = run
            self._runs.move_to_end(job_id)
            while len(self._runs) > self._max_cached:
                self._runs.popitem(last=False)

    def job(self, job_id: str) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        job.pop("client_hash", None)
        return job

    # --- entities -------------------------------------------------------------------------
    def entity(self, job_id: str, record_id: str, *, include_pii: bool) -> dict[str, Any]:
        with self.tracer.start_as_current_span("linkage.entity_lookup") as span:
            span.set_attribute("linkage.record", self.hash_id(record_id))
            span.set_attribute("linkage.pii", include_pii)
            view = self.store.entity(job_id, record_id)
        if view is None:
            raise JobNotFoundError(record_id)
        self.metrics.entity_lookups_total.labels("yes" if include_pii else "no").inc()
        members = []
        for member in view.members:
            item: dict[str, Any] = {
                "record_id": member.record_id,
                "source": member.source,
                "match_probability": member.match_probability,
            }
            if include_pii:
                item["fields"] = member.fields
            members.append(item)
        return {
            "job_id": view.job_id,
            "entity_id": f"{view.job_id[:12]}-{view.cluster_id}",
            "status": view.status,
            "size": len(members),
            "members": members,
        }

    # --- review ---------------------------------------------------------------------------
    def review_queue(
        self, job_id: str, *, include_pii: bool, limit: int = 100
    ) -> list[dict[str, Any]]:
        if self.store.get_job(job_id) is None:
            raise JobNotFoundError(job_id)
        items = self.store.review_items(job_id, limit=limit)
        fields = (
            self.store.record_fields(
                job_id, [i["left_row"] for i in items] + [i["right_row"] for i in items]
            )
            if include_pii and items
            else {}
        )
        result = []
        for item in items:
            entry = {
                "item_id": item["item_id"],
                "reason": item["reason"],
                "probability": item["probability"],
                "left_record_id": item["left_record_id"],
                "right_record_id": item["right_record_id"],
            }
            if include_pii:
                entry["left_fields"] = fields.get(item["left_row"])
                entry["right_fields"] = fields.get(item["right_row"])
            result.append(entry)
        return result

    def _run_for(self, job_id: str) -> LinkageRun:
        with self._lock:
            cached = self._runs.get(job_id)
        if cached is not None:
            return cached
        raw = self.store.raw_records(job_id)
        if raw.height == 0:
            raise JobNotFoundError(job_id)
        sources = {
            str(name): frame.drop("source")
            for (name,), frame in raw.partition_by(
                "source", as_dict=True, maintain_order=True
            ).items()
        }
        job = self.store.get_job(job_id) or {}
        summary = job.get("summary") or {}
        run, _ = self._link(
            sources, use_registered_model=bool(summary.get("used_registered_model"))
        )
        self._cache(job_id, run)
        return run

    def run(self, job_id: str) -> LinkageRun:
        """The in-memory run of a finished job (rebuilt deterministically after a restart)."""
        job = self.store.get_job(job_id)
        if job is None or job["status"] != "succeeded":
            raise JobNotFoundError(job_id)
        return self._run_for(job_id)

    def decide(self, job_id: str, item_id: str, *, decision: str, reviewer: str) -> dict[str, Any]:
        """Record and audit a decision, then re-resolve the job's clusters with its constraints."""
        reviewer_hash = self.hash_id(reviewer)
        item = self.store.decide(job_id, item_id, decision, reviewer_hash)
        run = self._run_for(job_id)
        left_id, right_id = run.record_ids[item["left_row"]], run.record_ids[item["right_row"]]
        entry = self.audit.append(
            "review_decision",
            reviewer_hash,
            {
                "job_id": job_id,
                "item_id": item_id,
                "left_record_id": left_id,
                "right_record_id": right_id,
                "decision": decision,
                "probability": item["probability"],
                "reason": item["reason"],
            },
        )
        must, cannot = self.store.constraints(job_id)
        resolution = run.resolve(must, cannot)
        self.store.update_clusters(job_id, resolution)
        self.store.add_review_items(job_id, run.review_candidates())
        self.metrics.review_decisions_total.labels(decision).inc()
        self.metrics.review_queue_depth.set(self.store.open_review_depth())
        left_row, right_row = item["left_row"], item["right_row"]
        return {
            "item_id": item_id,
            "decision": decision,
            "audit_head": entry.hash,
            "same_entity": bool(resolution.labels[left_row] == resolution.labels[right_row]),
            "left_status": "pending_review" if resolution.pending[left_row] else "committed",
            "right_status": "pending_review" if resolution.pending[right_row] else "committed",
        }

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


def default_audit_path(data_dir: Path) -> Path:
    return data_dir / "audit" / "review-decisions.jsonl"
