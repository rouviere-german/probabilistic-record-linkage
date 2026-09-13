from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from probabilistic_record_linkage.core.canonical import canonical_json, sha256_hex
from probabilistic_record_linkage.linkage.audit import GENESIS, AuditError, AuditLog, verify_file

pytestmark = pytest.mark.unit


def _clock() -> datetime:
    return datetime(2026, 9, 13, tzinfo=UTC)


def _filled(path: Path, entries: int = 3) -> AuditLog:
    log = AuditLog(path, clock=_clock)
    for i in range(entries):
        log.append(
            "review_decision", f"reviewer-{i}", {"item_id": f"item-{i}", "decision": "match"}
        )
    return log


def test_appends_form_a_verifiable_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = _filled(path, 2)
    result = verify_file(path)
    assert (result.ok, result.entries, result.head) == (True, 2, log.head)
    entries = log.entries()
    assert entries[0].prev == GENESIS
    assert entries[1].prev == entries[0].hash
    assert AuditLog(path).head == log.head
    assert verify_file(tmp_path / "missing.jsonl").ok


@settings(max_examples=40, deadline=None)
@given(st.data())
def test_any_single_byte_change_is_detected(data: st.DataObject) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "audit.jsonl"
        _filled(path, 3)
        raw = bytearray(path.read_bytes())
        position = data.draw(st.integers(0, len(raw) - 1))
        raw[position] = data.draw(st.integers(0, 127).filter(lambda b: b != raw[position]))
        path.write_bytes(bytes(raw))
        assert not verify_file(path).ok


def test_deleted_reordered_and_truncated_entries_are_detected(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = _filled(path, 3)
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(lines[0] + lines[2])
    assert not verify_file(path).ok
    path.write_bytes(lines[1] + lines[0] + lines[2])
    assert not verify_file(path).ok
    path.write_bytes(lines[0] + lines[1])
    assert verify_file(path).ok  # a clean prefix: only an anchored head reveals the truncation
    assert not verify_file(path, expect_head=log.head).ok


def test_entries_are_redacted_before_hashing(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    entry = AuditLog(path).append(
        "review_decision",
        "r",
        {"note": "mail jane@example.com", "first_name": "Anna", "item_id": "x"},
    )
    assert entry.data == {
        "note": "mail [REDACTED:EMAIL]",
        "first_name": "[REDACTED]",
        "item_id": "x",
    }
    assert "Anna" not in path.read_text(encoding="utf-8")
    assert verify_file(path).ok


def test_type_coerced_and_non_canonical_entries_fail_verification(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    _filled(path, 1)
    entry = json.loads(path.read_bytes())
    body = {key: value for key, value in entry.items() if key != "hash"}
    body["seq"] = "0"
    forged = {**body, "hash": sha256_hex(canonical_json(body))}
    path.write_bytes((canonical_json(forged) + "\n").encode())
    result = verify_file(path)
    assert not result.ok
    assert "types" in (result.error or "")
    path.write_bytes((json.dumps(entry, indent=1) + "\n").encode())
    assert not verify_file(path).ok
    path.write_bytes(b"[]\n")
    assert not verify_file(path).ok
    path.write_bytes(b"not json\n")
    assert "invalid JSON" in (verify_file(path).error or "")


def test_a_tampered_log_cannot_be_opened_for_appending(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    _filled(path, 2)
    path.write_bytes(path.read_bytes().replace(b"reviewer-0", b"reviewer-9"))
    with pytest.raises(AuditError):
        AuditLog(path)


def test_two_writers_extend_one_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    first, second = AuditLog(path), AuditLog(path)
    first.append("review_decision", "a", {"n": 1})
    second.append("review_decision", "b", {"n": 2})
    last = first.append("review_decision", "a", {"n": 3})
    result = verify_file(path)
    assert (result.ok, result.entries, result.head) == (True, 3, last.hash)
