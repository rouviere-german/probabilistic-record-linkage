"""DuckDB persistence for link jobs, resolved clusters, scored edges and the review queue.

Bulk data (records, edges) is written as a temporary Parquet file by Polars and inserted with
DuckDB's Parquet reader, so no Arrow/pandas bridge is needed. One connection is shared behind a
lock; DuckDB is used as an embedded analytical store, not as a multi-writer OLTP database.

The store holds the delivered source values, i.e. personal data at rest. The HTTP layer returns
them only to callers with the ``pii:read`` scope; protecting the file itself (disk encryption,
backups) is a deployment concern listed in docs/THREAT_MODEL.md.

Reviewer decisions carry the hash of the audit entry that recorded them. :meth:`constraints` turns
only decisions whose audit entry is present in the verified chain (and binds this job, item and
decision) into constraints, so a decision written without an audit entry never changes clusters.
The store also keeps the audit log's anchor (entry count and head), a second copy of the head in a
different file.
"""

from __future__ import annotations

import json
import re
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl

from probabilistic_record_linkage.core.canonical import sha256_hex
from probabilistic_record_linkage.linkage.clustering import (
    STATUS_COMMITTED,
    STATUS_PENDING,
    Resolution,
)
from probabilistic_record_linkage.linkage.pipeline import LinkageRun, ReviewCandidate
from probabilistic_record_linkage.linkage.thresholds import BAND_REVIEW

__all__ = [
    "DECISIONS",
    "EntityMember",
    "EntityView",
    "LinkageStore",
    "ReviewConflictError",
    "StoreAuditAnchor",
    "decision_ref",
]

DECISIONS = frozenset({"match", "non_match"})
_MEMORY = re.compile(r"^[1-9][0-9]{0,5}(MB|GB)$")
_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS jobs (
        job_id VARCHAR PRIMARY KEY,
        client_hash VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        created_at VARCHAR NOT NULL,
        finished_at VARCHAR,
        n_records BIGINT NOT NULL,
        summary VARCHAR,
        error VARCHAR
    )""",
    """CREATE TABLE IF NOT EXISTS records (
        job_id VARCHAR NOT NULL,
        row_id BIGINT NOT NULL,
        record_id VARCHAR NOT NULL,
        source VARCHAR NOT NULL,
        first_name VARCHAR,
        last_name VARCHAR,
        dob DATE,
        sex VARCHAR,
        street VARCHAR,
        city VARCHAR,
        postcode VARCHAR,
        cluster_id BIGINT NOT NULL,
        status VARCHAR NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS edges (
        job_id VARCHAR NOT NULL,
        left_row BIGINT NOT NULL,
        right_row BIGINT NOT NULL,
        probability DOUBLE NOT NULL,
        band TINYINT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS review_items (
        item_id VARCHAR PRIMARY KEY,
        job_id VARCHAR NOT NULL,
        left_row BIGINT NOT NULL,
        right_row BIGINT NOT NULL,
        probability DOUBLE NOT NULL,
        reason VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        created_at VARCHAR NOT NULL,
        decision VARCHAR,
        reviewer_hash VARCHAR,
        decided_at VARCHAR
    )""",
    """CREATE TABLE IF NOT EXISTS audit_anchor (
        name VARCHAR PRIMARY KEY,
        entries BIGINT NOT NULL,
        head VARCHAR NOT NULL,
        updated_at VARCHAR NOT NULL
    )""",
    # Columns added after the first schema; IF NOT EXISTS makes them no-ops on new databases.
    "ALTER TABLE review_items ADD COLUMN IF NOT EXISTS audit_hash VARCHAR",
)
_RECORD_FIELDS = ("first_name", "last_name", "dob", "sex", "street", "city", "postcode")


class ReviewConflictError(RuntimeError):
    """The review item does not exist, belongs to another job or was already decided."""


def decision_ref(job_id: str, item_id: str, decision: str) -> str:
    """Digest binding an audit entry to one job, review item and decision."""
    return sha256_hex(f"review-decision|{job_id}|{item_id}|{decision}")


@dataclass(frozen=True, slots=True)
class EntityMember:
    row_id: int
    record_id: str
    source: str
    match_probability: float | None  # strongest edge to another member, None for singletons
    fields: dict[str, str | None]


@dataclass(frozen=True, slots=True)
class EntityView:
    job_id: str
    cluster_id: int
    status: str
    members: tuple[EntityMember, ...]


def _now() -> str:
    return datetime.now(UTC).isoformat()


class LinkageStore:
    def __init__(
        self, path: Path | None = None, *, threads: int = 2, memory_limit: str = "512MB"
    ) -> None:
        if not _MEMORY.fullmatch(memory_limit):
            raise ValueError("memory_limit must look like 512MB or 2GB")
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._con = duckdb.connect(str(path) if path is not None else ":memory:")
        # Both values are validated above (int and a strict pattern), so the literals are safe.
        self._con.execute(f"SET threads = {int(threads)}")
        self._con.execute(f"SET memory_limit = '{memory_limit}'")
        self._lock = threading.RLock()
        with self._lock:
            for statement in _SCHEMA:
                self._con.execute(statement)

    def close(self) -> None:
        with self._lock:
            self._con.close()

    def writable(self) -> bool:
        try:
            with self._lock:
                self._con.execute("CREATE TEMP TABLE IF NOT EXISTS _probe (x INTEGER)")
                self._con.execute("INSERT INTO _probe VALUES (1)")
                self._con.execute("DELETE FROM _probe")
        except duckdb.Error:
            return False
        return True

    def _insert_frame(self, table: str, frame: pl.DataFrame) -> None:
        if frame.height == 0:
            return
        with tempfile.TemporaryDirectory(prefix="prl-store-") as tmp:
            path = Path(tmp) / f"{table}.parquet"
            frame.write_parquet(path)
            relation = self._con.read_parquet(str(path))
            relation.insert_into(table)
            del relation

    # --- audit anchor -------------------------------------------------------------------------
    def load_audit_anchor(self, name: str) -> tuple[int, str] | None:
        with self._lock:
            row = self._con.execute(
                "SELECT entries, head FROM audit_anchor WHERE name = ?", [name]
            ).fetchone()
        return (int(row[0]), str(row[1])) if row is not None else None

    def save_audit_anchor(self, name: str, entries: int, head: str) -> None:
        """Move the anchor forward (an older entry count never replaces a newer one)."""
        with self._lock:
            current = self.load_audit_anchor(name)
            if current is not None and current[0] >= entries:
                return
            self._con.execute(
                "INSERT OR REPLACE INTO audit_anchor (name, entries, head, updated_at) "
                "VALUES (?, ?, ?, ?)",
                [name, entries, head, _now()],
            )

    def audit_anchor(self, name: str) -> StoreAuditAnchor:
        return StoreAuditAnchor(self, name)

    # --- jobs -------------------------------------------------------------------------------
    def create_job(self, job_id: str, client_hash: str, n_records: int) -> None:
        with self._lock:
            self._con.execute(
                "INSERT INTO jobs (job_id, client_hash, status, created_at, n_records) "
                "VALUES (?, ?, 'running', ?, ?)",
                [job_id, client_hash, _now(), n_records],
            )

    def finish_job(
        self,
        job_id: str,
        *,
        status: str,
        summary: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self._con.execute(
                "UPDATE jobs SET status = ?, finished_at = ?, summary = ?, error = ? "
                "WHERE job_id = ?",
                [
                    status,
                    _now(),
                    json.dumps(summary, sort_keys=True) if summary else None,
                    error,
                    job_id,
                ],
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._con.execute(
                "SELECT job_id, client_hash, status, created_at, finished_at, n_records, "
                "summary, error FROM jobs WHERE job_id = ?",
                [job_id],
            ).fetchone()
        if row is None:
            return None
        keys = (
            "job_id",
            "client_hash",
            "status",
            "created_at",
            "finished_at",
            "n_records",
            "summary",
            "error",
        )
        job = dict(zip(keys, row, strict=True))
        job["summary"] = json.loads(job["summary"]) if job["summary"] else None
        return job

    # --- results ----------------------------------------------------------------------------
    def save_run(
        self, job_id: str, run: LinkageRun, raw: pl.DataFrame, review: Sequence[ReviewCandidate]
    ) -> int:
        """Persist records (in run row order), match/review edges and new review items."""
        if raw.height != run.n_records or raw["record_id"].to_list() != list(run.record_ids):
            raise ValueError("raw records must be in the run's row order")
        labels = run.resolution.labels
        status = np.where(run.resolution.pending, STATUS_PENDING, STATUS_COMMITTED)
        records = raw.select(
            pl.lit(job_id).alias("job_id"),
            pl.int_range(0, raw.height, dtype=pl.Int64).alias("row_id"),
            pl.col("record_id"),
            pl.Series("source", list(run.record_sources), dtype=pl.Utf8),
            *[pl.col(name) for name in _RECORD_FIELDS],
            pl.Series("cluster_id", labels.astype(np.int64)),
            pl.Series("status", status.tolist(), dtype=pl.Utf8),
        )
        bands = run.bands()
        keep = bands >= BAND_REVIEW
        edges = pl.DataFrame(
            {
                "job_id": [job_id] * int(keep.sum()),
                "left_row": run.scored.left[keep],
                "right_row": run.scored.right[keep],
                "probability": run.scored.probability[keep],
                "band": bands[keep].astype(np.int8),
            },
            schema={
                "job_id": pl.Utf8,
                "left_row": pl.Int64,
                "right_row": pl.Int64,
                "probability": pl.Float64,
                "band": pl.Int8,
            },
        )
        with self._lock:
            self._con.execute("BEGIN TRANSACTION")
            try:
                self._con.execute("DELETE FROM records WHERE job_id = ?", [job_id])
                self._con.execute("DELETE FROM edges WHERE job_id = ?", [job_id])
                self._insert_frame("records", records)
                self._insert_frame("edges", edges)
                added = self._add_review_items(job_id, review)
                self._con.execute("COMMIT")
            except BaseException:
                self._con.execute("ROLLBACK")
                raise
        return added

    def _add_review_items(self, job_id: str, review: Sequence[ReviewCandidate]) -> int:
        added = 0
        for item in review:
            item_id = sha256_hex(f"{job_id}|{item.left}|{item.right}")[:24]
            exists = self._con.execute(
                "SELECT 1 FROM review_items WHERE item_id = ?", [item_id]
            ).fetchone()
            if exists:
                continue
            self._con.execute(
                "INSERT INTO review_items (item_id, job_id, left_row, right_row, probability, "
                "reason, status, created_at) VALUES (?, ?, ?, ?, ?, ?, 'open', ?)",
                [item_id, job_id, item.left, item.right, item.probability, item.reason, _now()],
            )
            added += 1
        return added

    def add_review_items(self, job_id: str, review: Sequence[ReviewCandidate]) -> int:
        with self._lock:
            return self._add_review_items(job_id, review)

    def update_clusters(self, job_id: str, resolution: Resolution) -> None:
        frame = pl.DataFrame(
            {
                "row_id": np.arange(resolution.labels.size, dtype=np.int64),
                "cluster_id": resolution.labels.astype(np.int64),
                "status": np.where(resolution.pending, STATUS_PENDING, STATUS_COMMITTED).tolist(),
            },
            schema={"row_id": pl.Int64, "cluster_id": pl.Int64, "status": pl.Utf8},
        )
        with self._lock, tempfile.TemporaryDirectory(prefix="prl-store-") as tmp:
            path = Path(tmp) / "clusters.parquet"
            frame.write_parquet(path)
            self._con.execute("BEGIN TRANSACTION")
            try:
                self._con.execute(
                    "UPDATE records SET cluster_id = c.cluster_id, status = c.status "
                    "FROM read_parquet(?) AS c "
                    "WHERE records.job_id = ? AND records.row_id = c.row_id",
                    [str(path), job_id],
                )
                self._con.execute("COMMIT")
            except BaseException:
                self._con.execute("ROLLBACK")
                raise

    def entity(self, job_id: str, record_id: str) -> EntityView | None:
        with self._lock:
            anchor = self._con.execute(
                "SELECT cluster_id, status FROM records WHERE job_id = ? AND record_id = ?",
                [job_id, record_id],
            ).fetchone()
            if anchor is None:
                return None
            cluster_id, status = int(anchor[0]), str(anchor[1])
            rows = self._con.execute(
                "SELECT row_id, record_id, source, first_name, last_name, dob, sex, street, "
                "city, postcode FROM records WHERE job_id = ? AND cluster_id = ? ORDER BY row_id",
                [job_id, cluster_id],
            ).fetchall()
            member_rows = [int(r[0]) for r in rows]
            edges = self._con.execute(
                "SELECT left_row, right_row, probability FROM edges WHERE job_id = ? AND band = 2 "
                "AND left_row IN (SELECT UNNEST(?)) AND right_row IN (SELECT UNNEST(?))",
                [job_id, member_rows, member_rows],
            ).fetchall()
        best: dict[int, float] = {}
        for left, right, probability in edges:
            for row in (int(left), int(right)):
                best[row] = max(best.get(row, 0.0), float(probability))
        members = []
        for r in rows:
            values: dict[str, str | None] = {}
            for name, value in zip(_RECORD_FIELDS, r[3:], strict=True):
                values[name] = value.isoformat() if isinstance(value, date) else value
            members.append(
                EntityMember(int(r[0]), str(r[1]), str(r[2]), best.get(int(r[0])), values)
            )
        return EntityView(job_id, cluster_id, status, tuple(members))

    # --- review -----------------------------------------------------------------------------
    def review_items(
        self, job_id: str, *, status: str = "open", limit: int = 100
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._con.execute(
                "SELECT i.item_id, i.probability, i.reason, i.status, i.decision, "
                "l.record_id, r.record_id, i.left_row, i.right_row "
                "FROM review_items i "
                "JOIN records l ON l.job_id = i.job_id AND l.row_id = i.left_row "
                "JOIN records r ON r.job_id = i.job_id AND r.row_id = i.right_row "
                "WHERE i.job_id = ? AND i.status = ? "
                "ORDER BY i.probability DESC, i.item_id LIMIT ?",
                [job_id, status, limit],
            ).fetchall()
        keys = (
            "item_id",
            "probability",
            "reason",
            "status",
            "decision",
            "left_record_id",
            "right_record_id",
            "left_row",
            "right_row",
        )
        return [dict(zip(keys, row, strict=True)) for row in rows]

    def record_fields(self, job_id: str, rows: Sequence[int]) -> dict[int, dict[str, str | None]]:
        with self._lock:
            result = self._con.execute(
                "SELECT row_id, first_name, last_name, dob, sex, street, city, postcode "
                "FROM records WHERE job_id = ? AND row_id IN (SELECT UNNEST(?))",
                [job_id, list(rows)],
            ).fetchall()
        fields: dict[int, dict[str, str | None]] = {}
        for row in result:
            fields[int(row[0])] = {
                name: value.isoformat() if isinstance(value, date) else value
                for name, value in zip(_RECORD_FIELDS, row[1:], strict=True)
            }
        return fields

    def open_review_item(self, job_id: str, item_id: str) -> dict[str, Any]:
        """The open item ``item_id`` of ``job_id``; nothing is changed."""
        with self._lock:
            row = self._con.execute(
                "SELECT job_id, left_row, right_row, probability, reason, status "
                "FROM review_items WHERE item_id = ?",
                [item_id],
            ).fetchone()
        if row is None or row[0] != job_id or row[5] != "open":
            raise ReviewConflictError("review item is not open for this job")
        return {
            "item_id": item_id,
            "left_row": int(row[1]),
            "right_row": int(row[2]),
            "probability": float(row[3]),
            "reason": str(row[4]),
        }

    def decide(
        self, job_id: str, item_id: str, decision: str, reviewer_hash: str, *, audit_hash: str
    ) -> dict[str, Any]:
        """Close an open item with a decision already recorded in the audit log."""
        if decision not in DECISIONS:
            raise ValueError("decision must be match or non_match")
        with self._lock:
            item = self.open_review_item(job_id, item_id)
            updated = self._con.execute(
                "UPDATE review_items SET status = 'decided', decision = ?, reviewer_hash = ?, "
                "decided_at = ?, audit_hash = ? "
                "WHERE item_id = ? AND job_id = ? AND status = 'open' RETURNING item_id",
                [decision, reviewer_hash, _now(), audit_hash, item_id, job_id],
            ).fetchall()
        if len(updated) != 1:
            raise ReviewConflictError("review item is not open for this job")
        return {**item, "decision": decision}

    def constraints(
        self, job_id: str, audited: Mapping[str, str]
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
        """``(must-link, cannot-link, ignored)`` from decided items; the latest decision wins.

        ``audited`` maps the hash of every verified ``review_decision`` audit entry to its
        ``decision_ref``. Decisions whose audit entry is missing or bound to another job, item or
        decision are ignored and counted.
        """
        with self._lock:
            rows = self._con.execute(
                "SELECT item_id, left_row, right_row, decision, audit_hash FROM review_items "
                "WHERE job_id = ? AND status = 'decided' ORDER BY decided_at, item_id",
                [job_id],
            ).fetchall()
        latest: dict[tuple[int, int], str] = {}
        ignored = 0
        for item_id, left, right, decision, audit_hash in rows:
            expected = decision_ref(job_id, str(item_id), str(decision))
            if audit_hash is None or audited.get(str(audit_hash)) != expected:
                ignored += 1
                continue
            latest[(int(left), int(right))] = str(decision)
        must = [pair for pair, decision in latest.items() if decision == "match"]
        cannot = [pair for pair, decision in latest.items() if decision == "non_match"]
        return must, cannot, ignored

    def open_review_depth(self) -> int:
        with self._lock:
            row = self._con.execute(
                "SELECT count(*) FROM review_items WHERE status = 'open'"
            ).fetchone()
        return int(row[0]) if row else 0

    def raw_records(self, job_id: str) -> pl.DataFrame:
        """Source records of a job in row order (for re-linking after restarts)."""
        with self._lock:
            rows = self._con.execute(
                "SELECT source, record_id, first_name, last_name, dob, sex, street, city, postcode "
                "FROM records WHERE job_id = ? ORDER BY row_id",
                [job_id],
            ).fetchall()
        columns = ("source", "record_id", *_RECORD_FIELDS)
        data = {name: [row[i] for row in rows] for i, name in enumerate(columns)}
        return pl.DataFrame(
            data,
            schema={
                "source": pl.Utf8,
                "record_id": pl.Utf8,
                "first_name": pl.Utf8,
                "last_name": pl.Utf8,
                "dob": pl.Date,
                "sex": pl.Utf8,
                "street": pl.Utf8,
                "city": pl.Utf8,
                "postcode": pl.Utf8,
            },
        )


@dataclass(frozen=True, slots=True)
class StoreAuditAnchor:
    """:class:`audit.HeadAnchor` backed by the store's ``audit_anchor`` table."""

    store: LinkageStore
    name: str

    def load(self) -> tuple[int, str] | None:
        return self.store.load_audit_anchor(self.name)

    def save(self, entries: int, head: str) -> None:
        self.store.save_audit_anchor(self.name, entries, head)
