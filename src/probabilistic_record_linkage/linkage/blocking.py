"""Candidate generation: MinHash LSH on name shingles unioned with sorted neighbourhood (M2).

- LSH: records whose ``"first last"`` bigram sets collide in any band become candidates. Buckets
  larger than ``max_bucket`` (very common name combinations) are not expanded into all pairs;
  their members are sorted by date of birth and paired with their ``bucket_window`` nearest
  neighbours instead, which bounds the quadratic blow-up without dropping the bucket.
- Sorted neighbourhood (Hernandez and Stolfo 1995): records are sorted by each configured
  composite key (for example date of birth, then Soundex of the surname) and every record is
  paired with the next ``sn_window - 1`` records. It recovers matches whose names changed too much
  for LSH (nicknames, double typos).

Pairs are stored as ``(left, right)`` row indices with ``left < right``, deduplicated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from probabilistic_record_linkage.linkage.minhash import EMPTY, MinHasher, band_keys, shingle_sets
from probabilistic_record_linkage.linkage.normalize import PreparedRecords

__all__ = [
    "BlockingConfig",
    "BlockingResult",
    "bucket_pairs",
    "dedupe_pairs",
    "lsh_pairs",
    "pair_completeness",
    "reduction_ratio",
    "run_blocking",
    "sorted_neighbourhood_pairs",
    "true_pair_count",
]

Int64Array = npt.NDArray[np.int64]


class BlockingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    num_perm: int = Field(default=120, ge=4, le=512)
    bands: int = Field(default=20, ge=1, le=512)
    q: int = Field(default=2, ge=1, le=4)
    max_bucket: int = Field(default=300, ge=2)
    bucket_window: int = Field(default=10, ge=1)
    sn_window: int = Field(default=4, ge=2, le=100)
    sn_passes: tuple[tuple[str, ...], ...] = (
        ("dob", "soundex_last"),
        ("dob", "soundex_first"),
        ("soundex_last", "year", "soundex_first"),
    )
    seed: int = 7

    @model_validator(mode="after")
    def _bands_divide_permutations(self) -> BlockingConfig:
        if self.num_perm % self.bands:
            raise ValueError("num_perm must be a multiple of bands")
        return self


@dataclass(frozen=True, slots=True)
class BlockingResult:
    left: Int64Array
    right: Int64Array
    lsh_pairs: int
    sn_pairs: int
    seconds: dict[str, float] = field(default_factory=dict)

    @property
    def n_pairs(self) -> int:
        return int(self.left.size)


def dedupe_pairs(left: Int64Array, right: Int64Array, n: int) -> tuple[Int64Array, Int64Array]:
    """Order each pair as ``(min, max)``, drop self-pairs and duplicates; sorted by key."""
    lo = np.minimum(left, right).astype(np.int64)
    hi = np.maximum(left, right).astype(np.int64)
    keep = lo != hi
    keys = np.unique(lo[keep] * np.int64(n) + hi[keep])
    return keys // n, keys % n


def _run_pairs(
    order: Int64Array, run_lengths: Int64Array, window: int | None
) -> tuple[Int64Array, Int64Array]:
    """Pairs within consecutive runs of ``order``: each position with its next partners."""
    if order.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    run_starts = np.cumsum(run_lengths) - run_lengths
    run_of = np.repeat(np.arange(run_lengths.size), run_lengths)
    positions = np.arange(order.size, dtype=np.int64)
    after = run_starts[run_of] + run_lengths[run_of] - positions - 1
    if window is not None:
        after = np.minimum(after, window)
    total = int(after.sum())
    if total == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    left_pos = np.repeat(positions, after)
    first_out = np.cumsum(after) - after
    offsets = np.arange(total, dtype=np.int64) - np.repeat(first_out, after) + 1
    return order[left_pos], order[left_pos + offsets]


def bucket_pairs(
    keys: npt.NDArray[np.integer],
    valid: npt.NDArray[np.bool_],
    *,
    max_bucket: int,
    window: int,
    secondary: Int64Array | None = None,
) -> tuple[Int64Array, Int64Array]:
    """All pairs within equal-key buckets; buckets above ``max_bucket`` use a sorted window."""
    rows = np.flatnonzero(valid).astype(np.int64)
    if rows.size < 2:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    tie = secondary[rows] if secondary is not None else rows
    order = rows[np.lexsort((rows, tie, keys[rows]))]
    sorted_keys = keys[order]
    boundaries = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    lengths = np.diff(np.concatenate((starts, [order.size])))
    small = (lengths >= 2) & (lengths <= max_bucket)
    large = lengths > max_bucket
    parts_left: list[Int64Array] = []
    parts_right: list[Int64Array] = []
    for mask, limit in ((small, None), (large, window)):
        if not mask.any():
            continue
        member = np.repeat(mask, lengths)
        left, right = _run_pairs(order[member], lengths[mask], limit)
        parts_left.append(left)
        parts_right.append(right)
    if not parts_left:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(parts_left), np.concatenate(parts_right)


def lsh_pairs(records: PreparedRecords, config: BlockingConfig) -> tuple[Int64Array, Int64Array]:
    """MinHash LSH over ``"first last"`` bigram sets; one signature per distinct name string."""
    names = np.asarray(records.full_names(), dtype=object)
    if names.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    distinct, inverse = np.unique(names, return_inverse=True)
    inverse = inverse.reshape(-1)
    indptr, indices, vocabulary = shingle_sets([str(s) for s in distinct.tolist()], config.q)
    signatures = MinHasher(config.num_perm, config.seed).signatures(indptr, indices, vocabulary)
    valid = (signatures[:, 0] != EMPTY)[inverse]
    dob = records.sort_key("dob")
    left_parts: list[Int64Array] = []
    right_parts: list[Int64Array] = []
    acc_left = np.zeros(0, dtype=np.int64)
    acc_right = np.zeros(0, dtype=np.int64)
    for band, distinct_keys in band_keys(signatures, config.bands):
        keys = distinct_keys[inverse]
        left, right = bucket_pairs(
            keys, valid, max_bucket=config.max_bucket, window=config.bucket_window, secondary=dob
        )
        left_parts.append(left)
        right_parts.append(right)
        if (band + 1) % 5 == 0:
            acc_left, acc_right = dedupe_pairs(
                np.concatenate([acc_left, *left_parts]),
                np.concatenate([acc_right, *right_parts]),
                records.n,
            )
            left_parts, right_parts = [], []
    if left_parts:
        acc_left, acc_right = dedupe_pairs(
            np.concatenate([acc_left, *left_parts]),
            np.concatenate([acc_right, *right_parts]),
            records.n,
        )
    return acc_left, acc_right


def sorted_neighbourhood_pairs(
    records: PreparedRecords, keys: tuple[str, ...], window: int
) -> tuple[Int64Array, Int64Array]:
    """Pairs within ``window - 1`` positions after sorting by ``keys`` (first key is primary)."""
    columns = [records.sort_key(name) for name in keys]
    rows = np.arange(records.n, dtype=np.int64)
    order = np.lexsort((rows, *reversed(columns))).astype(np.int64)
    left_parts = [order[:-k] for k in range(1, window) if k < order.size]
    right_parts = [order[k:] for k in range(1, window) if k < order.size]
    if not left_parts:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    return dedupe_pairs(np.concatenate(left_parts), np.concatenate(right_parts), records.n)


def run_blocking(records: PreparedRecords, config: BlockingConfig) -> BlockingResult:
    timings: dict[str, float] = {}
    start = time.perf_counter()
    lsh_left, lsh_right = lsh_pairs(records, config)
    timings["lsh"] = time.perf_counter() - start
    start = time.perf_counter()
    sn_left: list[Int64Array] = []
    sn_right: list[Int64Array] = []
    for keys in config.sn_passes:
        left, right = sorted_neighbourhood_pairs(records, keys, config.sn_window)
        sn_left.append(left)
        sn_right.append(right)
    sn_l, sn_r = dedupe_pairs(
        np.concatenate(sn_left) if sn_left else np.zeros(0, dtype=np.int64),
        np.concatenate(sn_right) if sn_right else np.zeros(0, dtype=np.int64),
        records.n,
    )
    timings["sorted_neighbourhood"] = time.perf_counter() - start
    left, right = dedupe_pairs(
        np.concatenate([lsh_left, sn_l]), np.concatenate([lsh_right, sn_r]), records.n
    )
    return BlockingResult(left, right, int(lsh_left.size), int(sn_l.size), timings)


def reduction_ratio(n_pairs: int, n_records: int) -> float:
    total = n_records * (n_records - 1) / 2
    return 1.0 - n_pairs / total if total > 0 else 0.0


def true_pair_count(entity_ids: Int64Array) -> int:
    _, sizes = np.unique(entity_ids, return_counts=True)
    return int(np.sum(sizes * (sizes - 1) // 2))


def pair_completeness(left: Int64Array, right: Int64Array, entity_ids: Int64Array) -> float:
    """Share of all true matching pairs that are among the candidates."""
    total = true_pair_count(entity_ids)
    if total == 0:
        return 1.0
    return float(np.count_nonzero(entity_ids[left] == entity_ids[right]) / total)
