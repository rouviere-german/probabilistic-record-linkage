"""Comparison vectors: discrete agreement levels per field and comparison-pattern encoding (M3).

Level ``0`` of every field means "at least one value missing"; it contributes no evidence (a
Bayes factor of 1). Higher levels are categorical agreement grades whose last level is exact
agreement. A comparison *pattern* is the mixed-radix number formed by the levels of all fields, so
EM can run over the counts of distinct patterns (a few thousand at most) instead of over millions
of pairs.

Plaintext levels (``PLAIN_SPACE``):

- first name: else, Jaro-Winkler >= 0.80, JW >= 0.92, nickname of the same canonical name, exact
- last name: else, JW >= 0.80, JW >= 0.92, exact
- date of birth: else, one of day/month/year differs, day and month swapped, exact
- sex: else, exact
- address: else, same city, same postcode, similar street (typo) or same street with a different
  or missing postcode, exact street and postcode

The address is one hierarchical comparison rather than three independent fields, because street,
postcode and city change together when a person moves; treating them as independent would count
one move three times (see docs/adr/0003-address-as-one-hierarchical-comparison.md).

Jaro-Winkler similarity (Winkler 1990) comes from rapidfuzz and is computed once per distinct
(value, value) pair. Term-frequency adjustment (Winkler 1989 frequency-based matching; the form
used by Splink) scales the exact-agreement Bayes factor of a value ``v`` by
``sum_w tf(w)^2 / tf(v)``: agreeing on a rare surname is stronger evidence than agreeing on a
common one, and the average adjustment over random agreeing pairs is neutral.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
from rapidfuzz import process
from rapidfuzz.distance import JaroWinkler

from probabilistic_record_linkage.linkage.normalize import CodedColumn, PreparedRecords

__all__ = [
    "PLAIN_SPACE",
    "Comparator",
    "ComparisonSpace",
    "FieldSpec",
    "PlainComparator",
    "exact_agreement_rate",
    "jaro_winkler_pairs",
    "random_pairs",
    "u_from_random_pairs",
]

Int8Array = npt.NDArray[np.int8]
Int64Array = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """A compared field. ``levels`` labels levels ``1..L``; level 0 is always "missing"."""

    name: str
    levels: tuple[str, ...]
    tf: bool = False

    @property
    def n_levels(self) -> int:
        return len(self.levels) + 1

    @property
    def exact_level(self) -> int:
        return len(self.levels)


@dataclass(frozen=True, slots=True)
class ComparisonSpace:
    fields: tuple[FieldSpec, ...]

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("a comparison space needs at least one field")
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            raise ValueError("field names must be unique")
        if any(len(f.levels) < 2 for f in self.fields):
            raise ValueError("every field needs at least a disagreement and an exact level")

    @property
    def radices(self) -> Int64Array:
        sizes = [f.n_levels for f in self.fields]
        radices = np.ones(len(sizes), dtype=np.int64)
        for i in range(len(sizes) - 2, -1, -1):
            radices[i] = radices[i + 1] * sizes[i + 1]
        return radices

    @property
    def n_patterns(self) -> int:
        return int(math.prod(f.n_levels for f in self.fields))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def encode(self, levels: Int8Array) -> Int64Array:
        if levels.ndim != 2 or levels.shape[1] != len(self.fields):
            raise ValueError("levels must have one column per field")
        encoded: Int64Array = levels.astype(np.int64) @ self.radices
        return encoded

    def decode(self, patterns: Int64Array) -> Int8Array:
        out = np.zeros((patterns.size, len(self.fields)), dtype=np.int8)
        rest = patterns.astype(np.int64)
        for i, radix in enumerate(self.radices.tolist()):
            out[:, i] = rest // radix
            rest = rest % radix
        return out

    def describe(self) -> list[dict[str, object]]:
        return [{"name": f.name, "levels": ["missing", *f.levels], "tf": f.tf} for f in self.fields]


PLAIN_SPACE = ComparisonSpace(
    (
        FieldSpec("first_name", ("else", "jw>=0.80", "jw>=0.92", "nickname", "exact"), tf=True),
        FieldSpec("last_name", ("else", "jw>=0.80", "jw>=0.92", "exact"), tf=True),
        FieldSpec("dob", ("else", "one component differs", "day/month swapped", "exact")),
        FieldSpec("sex", ("else", "exact")),
        FieldSpec("address", ("else", "same city", "same postcode", "street similar", "exact")),
    )
)


class Comparator(Protocol):
    """Turns record pairs into agreement levels for one comparison space."""

    @property
    def space(self) -> ComparisonSpace: ...

    @property
    def n_records(self) -> int: ...

    def levels(self, left: Int64Array, right: Int64Array) -> Int8Array: ...

    def tf_adjustment(
        self, left: Int64Array, right: Int64Array, levels: Int8Array
    ) -> FloatArray: ...

    def random_u(self, n_pairs: int, rng: np.random.Generator) -> tuple[FloatArray, ...]: ...


def random_pairs(
    n_records: int, n_pairs: int, rng: np.random.Generator
) -> tuple[Int64Array, Int64Array]:
    """Uniform pairs of distinct records."""
    if n_records < 2:
        raise ValueError("random pairs need at least two records")
    left = rng.integers(0, n_records, size=n_pairs, dtype=np.int64)
    right = (left + rng.integers(1, n_records, size=n_pairs, dtype=np.int64)) % n_records
    return left, right


def exact_agreement_rate(keys: npt.NDArray[np.integer], valid: npt.NDArray[np.bool_]) -> float:
    """P(two distinct random records share a key | both keys valid), computed exactly."""
    present = keys[valid]
    k = present.size
    if k < 2:
        return 0.0
    _, counts = np.unique(present, return_counts=True)
    counts = counts.astype(np.float64)
    return float(np.sum(counts * (counts - 1.0)) / (k * (k - 1.0)))


def u_from_random_pairs(
    space: ComparisonSpace, sampled_levels: Int8Array, exact_rates: dict[str, float]
) -> tuple[FloatArray, ...]:
    """``u`` per field from sampled random pairs, with exact-level rates replaced by exact values.

    Exact agreement of random pairs is rare (about 1e-6 for addresses), so a sample of a few
    hundred thousand pairs cannot estimate it; the exact rate is computed from value counts over
    all records and the remaining levels are rescaled to fill the rest of the probability mass.
    """
    result: list[FloatArray] = []
    for f, spec in enumerate(space.fields):
        counts = np.bincount(sampled_levels[:, f].astype(np.int64), minlength=spec.n_levels)
        probs = counts.astype(np.float64) + 0.5
        probs[0] = 0.0
        probs /= probs[1:].sum()
        rate = exact_rates.get(spec.name)
        if rate is not None:
            rate = float(np.clip(rate, 1e-12, 1.0 - 1e-6))
            others = probs[1 : spec.exact_level]
            probs[1 : spec.exact_level] = others / others.sum() * (1.0 - rate)
            probs[spec.exact_level] = rate
        result.append(probs)
    return tuple(result)


def jaro_winkler_pairs(
    values: tuple[str, ...], a: npt.NDArray[np.int32], b: npt.NDArray[np.int32], *, workers: int = 1
) -> npt.NDArray[np.float32]:
    """Jaro-Winkler similarity for code pairs (codes must be >= 0), once per distinct pair."""
    if a.size == 0:
        return np.zeros(0, dtype=np.float32)
    n_values = np.int64(len(values))
    lo = np.minimum(a, b).astype(np.int64)
    hi = np.maximum(a, b).astype(np.int64)
    keys, inverse = np.unique(lo * n_values + hi, return_inverse=True)
    left = [values[i] for i in (keys // n_values).tolist()]
    right = [values[i] for i in (keys % n_values).tolist()]
    scores = np.asarray(
        process.cpdist(
            left,
            right,
            scorer=JaroWinkler.normalized_similarity,
            workers=workers,
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    result: npt.NDArray[np.float32] = scores[inverse.reshape(-1)]
    return result


class PlainComparator:
    """Agreement levels on normalized plaintext fields (``PLAIN_SPACE``)."""

    def __init__(
        self,
        records: PreparedRecords,
        *,
        jw_low: float = 0.80,
        jw_high: float = 0.92,
        street_similarity: float = 0.88,
        tf_clip_bits: float = 10.0,
        workers: int = 2,
        chunk_pairs: int = 1_000_000,
    ) -> None:
        if not 0.0 < jw_low < jw_high <= 1.0:
            raise ValueError("need 0 < jw_low < jw_high <= 1")
        self.records = records
        self.jw_low = jw_low
        self.jw_high = jw_high
        self.street_similarity = street_similarity
        self.tf_clip = tf_clip_bits * math.log(2.0)
        self.workers = workers
        self.chunk_pairs = chunk_pairs
        self._tf: dict[str, tuple[FloatArray, float]] = {}
        for name, column in (("first_name", records.first_name), ("last_name", records.last_name)):
            counts = column.counts().astype(np.float64)
            total = counts.sum()
            freq = counts / total if total > 0 else counts
            self._tf[name] = (freq, float(np.sum(freq**2)))

    @property
    def space(self) -> ComparisonSpace:
        return PLAIN_SPACE

    @property
    def n_records(self) -> int:
        return self.records.n

    def levels(self, left: Int64Array, right: Int64Array) -> Int8Array:
        out = np.zeros((left.size, len(PLAIN_SPACE.fields)), dtype=np.int8)
        for start in range(0, left.size, self.chunk_pairs):
            stop = min(left.size, start + self.chunk_pairs)
            out[start:stop] = self._levels(left[start:stop], right[start:stop])
        return out

    def _name_levels(
        self, column: CodedColumn, left: Int64Array, right: Int64Array
    ) -> tuple[Int8Array, npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        a, b = column.codes[left], column.codes[right]
        missing = (a < 0) | (b < 0)
        exact = (a == b) & ~missing
        fuzzy = ~missing & ~exact
        scores = np.zeros(left.size, dtype=np.float32)
        scores[fuzzy] = jaro_winkler_pairs(column.values, a[fuzzy], b[fuzzy], workers=self.workers)
        level = np.ones(left.size, dtype=np.int8)
        level[fuzzy & (scores >= self.jw_low)] = 2
        level[fuzzy & (scores >= self.jw_high)] = 3
        return level, missing, exact

    def _levels(self, left: Int64Array, right: Int64Array) -> Int8Array:
        r = self.records
        out = np.zeros((left.size, 5), dtype=np.int8)

        first, missing, exact = self._name_levels(r.first_name, left, right)
        ca, cb = r.first_canonical[left], r.first_canonical[right]
        nickname = np.zeros(left.size, dtype=bool)
        for x in range(2):
            for y in range(2):
                nickname |= (ca[:, x] >= 0) & (ca[:, x] == cb[:, y])
        first[nickname & ~exact & ~missing] = 4
        first[exact] = 5
        first[missing] = 0
        out[:, 0] = first

        last, missing, exact = self._name_levels(r.last_name, left, right)
        last[exact] = 4
        last[missing] = 0
        out[:, 1] = last

        ya, yb = r.year[left], r.year[right]
        ma, mb = r.month[left], r.month[right]
        da, db = r.day[left], r.day[right]
        missing = (ya < 0) | (yb < 0)
        eq_y, eq_m, eq_d = ya == yb, ma == mb, da == db
        dob = np.ones(left.size, dtype=np.int8)
        dob[(eq_y.astype(np.int8) + eq_m + eq_d) == 2] = 2
        dob[eq_y & (ma == db) & (da == mb) & (ma != da)] = 3
        dob[eq_y & eq_m & eq_d] = 4
        dob[missing] = 0
        out[:, 2] = dob

        sa, sb = r.sex.codes[left], r.sex.codes[right]
        sex = np.where(sa == sb, 2, 1).astype(np.int8)
        sex[(sa < 0) | (sb < 0)] = 0
        out[:, 3] = sex

        st_a, st_b = r.street.codes[left], r.street.codes[right]
        pc_a, pc_b = r.postcode.codes[left], r.postcode.codes[right]
        ci_a, ci_b = r.city.codes[left], r.city.codes[right]
        present_a = (st_a >= 0) | (pc_a >= 0) | (ci_a >= 0)
        present_b = (st_b >= 0) | (pc_b >= 0) | (ci_b >= 0)
        same_street = (st_a == st_b) & (st_a >= 0)
        same_postcode = (pc_a == pc_b) & (pc_a >= 0)
        same_city = (ci_a == ci_b) & (ci_a >= 0)
        check = same_postcode & ~same_street & (st_a >= 0) & (st_b >= 0)
        similar = np.zeros(left.size, dtype=bool)
        similar[check] = (
            jaro_winkler_pairs(r.street.values, st_a[check], st_b[check], workers=self.workers)
            >= self.street_similarity
        )
        address = np.ones(left.size, dtype=np.int8)
        address[same_city] = 2
        address[same_postcode] = 3
        address[similar | (same_street & ~same_postcode)] = 4
        address[same_street & same_postcode] = 5
        address[~present_a | ~present_b] = 0
        out[:, 4] = address
        return out

    def random_u(self, n_pairs: int, rng: np.random.Generator) -> tuple[FloatArray, ...]:
        r = self.records
        left, right = random_pairs(r.n, n_pairs, rng)
        sampled = self.levels(left, right)
        street_postcode = (
            r.street.codes.astype(np.int64) * max(1, r.postcode.n_values) + r.postcode.codes
        )
        has_both = (r.street.codes >= 0) & (r.postcode.codes >= 0)
        has_any = (r.street.codes >= 0) | (r.postcode.codes >= 0) | (r.city.codes >= 0)
        share_both = float(has_both.mean()) if r.n else 0.0
        share_any = float(has_any.mean()) if r.n else 1.0
        address_rate = exact_agreement_rate(street_postcode, has_both) * (
            share_both**2 / max(share_any**2, 1e-12)
        )
        rates = {
            "first_name": exact_agreement_rate(r.first_name.codes, r.first_name.codes >= 0),
            "last_name": exact_agreement_rate(r.last_name.codes, r.last_name.codes >= 0),
            "dob": exact_agreement_rate(r.sort_key("dob"), r.year >= 0),
            "sex": exact_agreement_rate(r.sex.codes, r.sex.codes >= 0),
            "address": address_rate,
        }
        return u_from_random_pairs(PLAIN_SPACE, sampled, rates)

    def tf_adjustment(self, left: Int64Array, right: Int64Array, levels: Int8Array) -> FloatArray:
        """Natural-log term-frequency adjustment added to the log Bayes factor of each pair."""
        del right  # exact agreement means both sides carry the same value
        total = np.zeros(left.size, dtype=np.float64)
        for spec_index, spec in enumerate(PLAIN_SPACE.fields):
            if not spec.tf:
                continue
            column = getattr(self.records, spec.name)
            freq, sum_sq = self._tf[spec.name]
            rows = np.flatnonzero(levels[:, spec_index] == spec.exact_level)
            if rows.size == 0 or sum_sq <= 0:
                continue
            value_freq = freq[column.codes[left[rows]]]
            adjustment = np.log(sum_sq / np.maximum(value_freq, 1e-12))
            total[rows] += np.clip(adjustment, -self.tf_clip, self.tf_clip)
        return total
