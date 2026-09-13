from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from fastapi.testclient import TestClient

from probabilistic_record_linkage.api.app import create_linkage_app, local_service
from probabilistic_record_linkage.core.config import Settings
from probabilistic_record_linkage.core.security import issue_token
from probabilistic_record_linkage.linkage.audit import verify_file
from probabilistic_record_linkage.linkage.generator import SyntheticDataset
from probabilistic_record_linkage.linkage.model import register_run
from probabilistic_record_linkage.linkage.service import LinkageService, default_audit_path

pytestmark = pytest.mark.integration

SECRET = "api-test-jwt-secret-" + "z" * 20


def _auth(*scopes: str, subject: str = "alice") -> dict[str, str]:
    token = issue_token(
        SECRET.encode(),
        subject=subject,
        scopes=list(scopes),
        issuer="probabilistic-record-linkage",
        audience="linkage-api",
        ttl_seconds=600,
    )
    return {"Authorization": f"Bearer {token}"}


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "jwt_secret": SECRET,
        "environment": "test",
        "upload_max_rows": 5_000,
        "rate_limit_burst": 1_000,
        "rate_limit_per_second": 1_000.0,
        "metrics_require_auth": False,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _records(dataset: SyntheticDataset, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, frame in sorted(dataset.sources.items()):
        for row in frame.iter_rows(named=True):
            rows.append(
                {
                    "source": name,
                    **{k: v.isoformat() if isinstance(v, date) else v for k, v in row.items()},
                }
            )
    return rows[:limit] if limit is not None else rows


def _close(service: LinkageService) -> None:
    service.shutdown()
    service.store.close()


@dataclass
class Env:
    client: TestClient
    service: LinkageService
    job_id: str
    state: Path


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory, small_dataset: SyntheticDataset) -> Iterator[Env]:
    state = tmp_path_factory.mktemp("api-state")
    service = local_service(state, id_hash_key=b"i" * 32, nicknames=small_dataset.world.nicknames)
    app = create_linkage_app(_settings(), service=service)
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/link-jobs",
                json={"records": _records(small_dataset)},
                headers=_auth("link:run"),
            )
            assert response.status_code == 202, response.text
            job_id = response.json()["job_id"]
            service.wait(job_id)
            yield Env(client, service, job_id, state)
    finally:
        _close(service)


def test_link_routes_require_a_token_with_the_scope(env: Env) -> None:
    client = env.client
    assert client.post("/v1/link-jobs", json={"records": []}).status_code == 401
    forbidden = client.post("/v1/link-jobs", json={"records": []}, headers=_auth("entity:read"))
    assert forbidden.status_code == 403
    assert forbidden.json()["error"]["code"] == "forbidden"
    assert (
        client.get(f"/v1/link-jobs/{env.job_id}", headers=_auth("entity:read")).status_code == 403
    )
    assert client.get("/v1/link-jobs/not-a-job-id", headers=_auth("link:run")).status_code == 422
    assert client.get(f"/v1/link-jobs/{'0' * 32}", headers=_auth("link:run")).status_code == 404


def test_entity_lookup_returns_fields_only_with_pii_read(
    env: Env, small_dataset: SyntheticDataset
) -> None:
    client = env.client
    job = client.get(f"/v1/link-jobs/{env.job_id}", headers=_auth("link:run")).json()
    assert job["status"] == "succeeded"
    assert "client_hash" not in job
    assert job["summary"]["records"] == small_dataset.n_records
    record_id = small_dataset.sources["s1"]["record_id"][0]
    url = f"/v1/link-jobs/{env.job_id}/entities/{record_id}"
    plain = client.get(url, headers=_auth("entity:read"))
    assert plain.status_code == 200
    members = plain.json()["members"]
    assert record_id in {member["record_id"] for member in members}
    assert all("fields" not in member for member in members)
    detailed = client.get(url, headers=_auth("entity:read", "pii:read")).json()["members"]
    assert all("fields" in member for member in detailed)
    assert client.get(url, headers=_auth("pii:read")).status_code == 403
    missing = f"/v1/link-jobs/{env.job_id}/entities/s1-9999999"
    assert client.get(missing, headers=_auth("entity:read")).status_code == 404


def test_review_decisions_are_audited_and_final(env: Env) -> None:
    client = env.client
    url = f"/v1/link-jobs/{env.job_id}/review-items"
    items = client.get(url, headers=_auth("review:write")).json()["items"]
    assert items
    assert all("left_fields" not in item for item in items)
    assert (
        "left_fields"
        in client.get(url, headers=_auth("review:write", "pii:read")).json()["items"][0]
    )
    decision_url = f"{url}/{items[0]['item_id']}/decision"
    reviewer = _auth("review:write", subject="reviewer-1")
    decided = client.post(decision_url, json={"decision": "non_match"}, headers=reviewer)
    assert decided.status_code == 200, decided.text
    head = decided.json()["audit_head"]
    assert (
        client.post(decision_url, json={"decision": "non_match"}, headers=reviewer).status_code
        == 409
    )
    assert (
        client.post(decision_url, json={"decision": "maybe"}, headers=reviewer).status_code == 422
    )
    verification = verify_file(default_audit_path(env.state))
    assert verification.ok
    assert verification.head == head
    assert "reviewer-1" not in default_audit_path(env.state).read_text(encoding="utf-8")


def test_readiness_needs_a_verified_model_and_a_writable_store(env: Env) -> None:
    client = env.client
    if env.service.production_model() is None:
        not_ready = client.get("/readyz")
        assert not_ready.status_code == 503
        assert not_ready.json()["checks"] == {
            "model_manifest": "fail",
            "duckdb_writable": "ok",
            "audit_chain": "ok",
        }
        registry = env.service.registry
        assert registry is not None
        version = register_run(registry, env.service.run(env.job_id))
        registry.set_alias("person-linkage", "production", version.digest)
    assert client.get("/readyz").status_code == 200


def test_jobs_can_score_with_the_registered_model(
    env: Env, small_dataset: SyntheticDataset
) -> None:
    registry = env.service.registry
    assert registry is not None
    if env.service.production_model() is None:
        version = register_run(registry, env.service.run(env.job_id))
        registry.set_alias("person-linkage", "production", version.digest)
    response = env.client.post(
        "/v1/link-jobs",
        json={"records": _records(small_dataset), "use_registered_model": True},
        headers=_auth("link:run", subject="scorer"),
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    env.service.wait(job_id)
    summary = env.client.get(f"/v1/link-jobs/{job_id}", headers=_auth("link:run")).json()["summary"]
    assert summary["used_registered_model"] is True
    assert summary["drift"]["drifted"] is False


def test_metrics_expose_linkage_gauges_but_no_record_ids(
    env: Env, small_dataset: SyntheticDataset
) -> None:
    payload = env.client.get("/metrics").text
    for name in (
        "blocking_pairs_total", "reduction_ratio", "match_rate", "review_queue_depth",
        "max_cluster_size", "em_iterations", "comparison_pattern_psi", "entity_lookups_total",
    ):  # fmt: skip
        assert name in payload
    for record_id in small_dataset.sources["s1"]["record_id"].head(50).to_list():
        assert record_id not in payload


def test_uploads_are_limited_by_rows_fields_and_body_size(
    tmp_path: Path, small_dataset: SyntheticDataset
) -> None:
    service = local_service(tmp_path, id_hash_key=b"i" * 32)
    app = create_linkage_app(_settings(upload_max_rows=10, max_body_bytes=200_000), service=service)
    rows = _records(small_dataset, limit=11)
    headers = _auth("link:run")
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            too_many = client.post("/v1/link-jobs", json={"records": rows}, headers=headers)
            assert too_many.status_code == 413
            assert too_many.json()["error"]["code"] == "too_many_rows"
            extra = client.post(
                "/v1/link-jobs",
                json={"records": [dict(rows[0], entity_id=1), rows[1]]},
                headers=headers,
            )
            assert extra.status_code == 422
            long_name = client.post(
                "/v1/link-jobs",
                json={"records": [dict(rows[0], first_name="x" * 65), rows[1]]},
                headers=headers,
            )
            assert long_name.status_code == 422
            assert "xxxxxxxx" not in long_name.text
            duplicate = client.post(
                "/v1/link-jobs", json={"records": [rows[0], rows[0]]}, headers=headers
            )
            assert duplicate.json()["error"]["code"] == "duplicate_record_ids"
            old = client.post(
                "/v1/link-jobs",
                json={"records": [dict(rows[0], dob="1850-01-01"), rows[1]]},
                headers=headers,
            )
            assert old.status_code == 422
            assert old.json()["error"]["code"] == "data_contract_violation"
            assert "1850" not in old.text
            big = client.post(
                "/v1/link-jobs",
                content=b"x" * 300_000,
                headers={**headers, "Content-Type": "application/json"},
            )
            assert big.status_code == 413
    finally:
        _close(service)


def test_job_concurrency_caps_return_429(tmp_path: Path, small_dataset: SyntheticDataset) -> None:
    service = local_service(
        tmp_path, id_hash_key=b"i" * 32, max_concurrent_jobs=2, max_jobs_per_client=1
    )
    release = threading.Event()
    original = service._run_job

    def slow_run(
        job_id: str, client_hash: str, sources: dict[str, pl.DataFrame], use_model: bool
    ) -> None:
        release.wait(timeout=60)
        original(job_id, client_hash, sources, use_model)

    setattr(service, "_run_job", slow_run)  # noqa: B010 - replace the bound method on this instance
    app = create_linkage_app(_settings(), service=service)
    body = {"records": _records(small_dataset, limit=400)}
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            first = client.post(
                "/v1/link-jobs", json=body, headers=_auth("link:run", subject="alice")
            )
            assert first.status_code == 202
            again = client.post(
                "/v1/link-jobs", json=body, headers=_auth("link:run", subject="alice")
            )
            assert again.status_code == 429
            assert again.headers["retry-after"] == "5"
            other = client.post(
                "/v1/link-jobs", json=body, headers=_auth("link:run", subject="bob")
            )
            assert other.status_code == 202
            third = client.post(
                "/v1/link-jobs", json=body, headers=_auth("link:run", subject="carol")
            )
            assert third.status_code == 429
            release.set()
            service.wait(first.json()["job_id"])
            service.wait(other.json()["job_id"])
            later = client.post(
                "/v1/link-jobs", json=body, headers=_auth("link:run", subject="alice")
            )
            assert later.status_code == 202
            service.wait(later.json()["job_id"])
    finally:
        release.set()
        _close(service)


def test_rate_limits_apply_per_token_subject(tmp_path: Path) -> None:
    service = local_service(tmp_path, id_hash_key=b"i" * 32)
    app = create_linkage_app(
        _settings(rate_limit_burst=2, rate_limit_per_second=0.001), service=service
    )
    url = f"/v1/link-jobs/{'0' * 32}"
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            codes = [
                client.get(url, headers=_auth("link:run", subject="alice")).status_code
                for _ in range(3)
            ]
            assert codes == [404, 404, 429]
            assert client.get(url, headers=_auth("link:run", subject="bob")).status_code == 404
    finally:
        _close(service)
