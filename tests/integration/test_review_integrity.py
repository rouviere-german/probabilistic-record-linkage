"""Review decisions only change clusters after they are durably audited."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from probabilistic_record_linkage.api.app import local_service
from probabilistic_record_linkage.linkage.audit import AuditError
from probabilistic_record_linkage.linkage.generator import SyntheticDataset
from probabilistic_record_linkage.linkage.service import LinkageService, default_audit_path

pytestmark = pytest.mark.integration

KEY = b"i" * 32


@dataclass
class Job:
    service: LinkageService
    job_id: str
    state: Path


def _close(service: LinkageService) -> None:
    service.shutdown()
    service.store.close()


@pytest.fixture
def job(tmp_path: Path, small_dataset: SyntheticDataset) -> Iterator[Job]:
    service = local_service(tmp_path, id_hash_key=KEY, nicknames=small_dataset.world.nicknames)
    try:
        job_id = service.submit("alice", small_dataset.sources, wait=True)
        yield Job(service, job_id, tmp_path)
    finally:
        _close(service)


def _items(job: Job, count: int) -> list[str]:
    items = job.service.review_queue(job.job_id, include_pii=False, limit=count)
    assert len(items) == count
    return [str(item["item_id"]) for item in items]


def test_a_failed_audit_append_leaves_the_decision_unapplied(
    job: Job, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = job.service
    (item_id,) = _items(job, 1)

    def failing_append(*args: object, **kwargs: object) -> None:
        raise AuditError("disk full")

    monkeypatch.setattr(service.audit, "append", failing_append)
    with pytest.raises(AuditError):
        service.decide(job.job_id, item_id, decision="non_match", reviewer="rev")
    open_ids = {i["item_id"] for i in service.store.review_items(job.job_id, limit=10_000)}
    assert item_id in open_ids
    assert service.constraints(job.job_id) == ([], [])
    monkeypatch.undo()
    decided = service.decide(job.job_id, item_id, decision="non_match", reviewer="rev")
    assert len(decided["audit_head"]) == 64
    assert len(service.constraints(job.job_id)[1]) == 1


def test_decisions_without_a_matching_audit_entry_never_become_constraints(job: Job) -> None:
    service = job.service
    first, forged, reused = _items(job, 3)
    decided = service.decide(job.job_id, first, decision="non_match", reviewer="rev")
    # A decision written straight into the store with no audit entry, and one that reuses the
    # hash of a real entry recorded for a different item.
    service.store.decide(job.job_id, forged, "non_match", "x", audit_hash="f" * 64)
    service.store.decide(job.job_id, reused, "match", "x", audit_hash=decided["audit_head"])
    must, cannot = service.constraints(job.job_id)
    assert must == []
    assert len(cannot) == 1


def test_a_deleted_audit_log_is_detected_through_the_store_anchor(
    job: Job, small_dataset: SyntheticDataset
) -> None:
    service = job.service
    (item_id, other_id) = _items(job, 2)
    service.decide(job.job_id, item_id, decision="match", reviewer="rev")
    assert service.readiness_checks()["audit_chain"]()
    _close(service)
    default_audit_path(job.state).unlink()

    restarted = local_service(job.state, id_hash_key=KEY, nicknames=small_dataset.world.nicknames)
    job.service = restarted  # closed by the fixture
    assert restarted.readiness_checks()["audit_chain"]() is False
    with pytest.raises(AuditError, match="truncated or replaced"):
        restarted.decide(job.job_id, other_id, decision="match", reviewer="rev")
    open_ids = {i["item_id"] for i in restarted.store.review_items(job.job_id, limit=10_000)}
    assert other_id in open_ids
