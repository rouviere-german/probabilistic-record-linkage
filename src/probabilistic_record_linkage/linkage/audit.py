"""Hash-chained, append-only audit log for reviewer decisions.

Every entry stores the SHA-256 of the previous entry, and its own hash covers the canonical JSON of
all its fields (sequence number, time, event, actor, data, previous hash). Editing, deleting,
reordering or re-encoding an entry breaks verification. Data is redacted before hashing, so the log
never becomes a second copy of personal data. Appends take a cross-process file lock, re-read the
current head and fsync, so concurrent writers extend a single chain.

Truncation and replacement are caught three ways:

- an :class:`AuditLog` remembers the last head it wrote or read; an append refuses a file that no
  longer contains that head (another writer's *extension* of it is accepted);
- an optional :class:`HeadAnchor` (the service keeps it in DuckDB, a different file) persists the
  entry count and head after every append; :meth:`AuditLog.verify`, ``/readyz`` and
  ``audit verify --workdir`` check that the log still contains the anchored entry, so deleting or
  cutting the log file between process restarts is detected;
- ``verify_file(path, expect_head=...)`` compares against a head kept outside the host.

Someone who can rewrite both the log and the anchor can still rebuild a consistent history; anchor
the head outside the host for that threat.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from probabilistic_record_linkage.core.canonical import canonical_json, sha256_hex
from probabilistic_record_linkage.core.fileio import file_lock
from probabilistic_record_linkage.core.redaction import redact_value

__all__ = [
    "GENESIS",
    "AuditEntry",
    "AuditError",
    "AuditLog",
    "HeadAnchor",
    "VerifyResult",
    "verify_file",
]

GENESIS = "0" * 64
_FIELDS = ("seq", "ts", "event", "actor", "data", "prev", "hash")


class AuditError(RuntimeError):
    """The audit log is corrupt or was modified."""


class HeadAnchor(Protocol):
    """Durable record of the latest ``(entry count, head hash)``, kept apart from the log file."""

    def load(self) -> tuple[int, str] | None: ...

    def save(self, entries: int, head: str) -> None: ...


@dataclass(frozen=True, slots=True)
class AuditEntry:
    seq: int
    ts: str
    event: str
    actor: str
    data: dict[str, Any]
    prev: str
    hash: str

    def body(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "event": self.event,
            "actor": self.actor,
            "data": self.data,
            "prev": self.prev,
        }


@dataclass(frozen=True, slots=True)
class VerifyResult:
    ok: bool
    entries: int
    head: str
    error: str | None = None


def _parse(line: str, number: int) -> AuditEntry:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise AuditError(f"line {number}: invalid JSON") from exc
    if not isinstance(raw, dict) or sorted(raw) != sorted(_FIELDS):
        raise AuditError(f"line {number}: unexpected fields")
    seq, ts, event, actor, data, prev, digest = (raw[name] for name in _FIELDS)
    if type(seq) is not int or not all(
        isinstance(v, str) for v in (ts, event, actor, prev, digest)
    ):
        raise AuditError(f"line {number}: field types are wrong")
    if not isinstance(data, dict):
        raise AuditError(f"line {number}: data must be an object")
    entry = AuditEntry(seq, ts, event, actor, data, prev, digest)
    if canonical_json({**entry.body(), "hash": entry.hash}) != line:
        raise AuditError(f"line {number}: entry is not canonically encoded")
    return entry


def _check_chain(entries: list[AuditEntry]) -> str:
    previous = GENESIS
    for index, entry in enumerate(entries):
        if entry.seq != index:
            raise AuditError(f"entry {index}: sequence number {entry.seq} is out of order")
        if entry.prev != previous:
            raise AuditError(f"entry {index}: previous-hash link is broken")
        if sha256_hex(canonical_json(entry.body())) != entry.hash:
            raise AuditError(f"entry {index}: hash does not match its content")
        previous = entry.hash
    return previous


def _check_contains(entries: list[AuditEntry], count: int, head: str, what: str) -> None:
    """The verified chain must still contain entry ``count - 1`` with hash ``head``."""
    if count == 0 and head == GENESIS:
        return
    if count < 1 or count > len(entries) or entries[count - 1].hash != head:
        raise AuditError(f"audit log was truncated or replaced: it no longer contains the {what}")


def _read(path: Path) -> list[AuditEntry]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [_parse(line, number + 1) for number, line in enumerate(lines)]


def verify_file(
    path: Path, *, expect_head: str | None = None, anchor: tuple[int, str] | None = None
) -> VerifyResult:
    """Verify the chain; ``expect_head`` must equal the head, ``anchor`` must still be contained."""
    try:
        entries = _read(path)
        head = _check_chain(entries)
        if anchor is not None:
            _check_contains(entries, anchor[0], anchor[1], "anchored head")
    except AuditError as exc:
        return VerifyResult(False, 0, GENESIS, str(exc))
    if expect_head is not None and head != expect_head:
        return VerifyResult(False, len(entries), head, "head does not match the anchored head")
    return VerifyResult(True, len(entries), head)


class AuditLog:
    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        anchor: HeadAnchor | None = None,
    ) -> None:
        self.path = path
        self._clock = clock
        self._anchor = anchor
        path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(path):
            entries = _read(path)
            self._head = _check_chain(entries)
            self._count = len(entries)

    @property
    def head(self) -> str:
        return self._head

    def append(self, event: str, actor: str, data: Mapping[str, Any]) -> AuditEntry:
        with file_lock(self.path):
            entries = _read(self.path)
            head = _check_chain(entries)
            _check_contains(entries, self._count, self._head, "last head this process saw")
            anchored = self._anchor.load() if self._anchor is not None else None
            if anchored is not None:
                _check_contains(entries, anchored[0], anchored[1], "anchored head")
            body = {
                "seq": len(entries),
                "ts": self._clock().isoformat(),
                "event": event,
                "actor": actor,
                "data": redact_value(dict(data)),
                "prev": head,
            }
            digest = sha256_hex(canonical_json(body))
            line = canonical_json({**body, "hash": digest}) + "\n"
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            self._head, self._count = digest, len(entries) + 1
            if self._anchor is not None:
                self._anchor.save(self._count, digest)
            return AuditEntry(body["seq"], body["ts"], event, actor, body["data"], head, digest)

    def entries(self) -> list[AuditEntry]:
        """All entries of a verified chain that still contains the anchored head."""
        with file_lock(self.path):
            entries = _read(self.path)
            _check_chain(entries)
            anchored = self._anchor.load() if self._anchor is not None else None
            if anchored is not None:
                _check_contains(entries, anchored[0], anchored[1], "anchored head")
            return entries

    def verify(self) -> VerifyResult:
        """Chain integrity plus the persisted anchor (used by ``/readyz``)."""
        with file_lock(self.path):
            anchored = self._anchor.load() if self._anchor is not None else None
            return verify_file(self.path, anchor=anchored)
