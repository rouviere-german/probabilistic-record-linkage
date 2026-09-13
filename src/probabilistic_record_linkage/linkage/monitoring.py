"""Comparison-pattern drift monitoring (M5) and linkage metrics (M6).

The distribution of comparison patterns over candidate pairs summarizes data quality: more typos
move mass from "exact" to fuzzy levels, more missing values move it to level 0. A new batch is
compared with the reference distribution recorded at training time using the Population Stability
Index over pattern buckets, ``sum (a - e) ln(a / e)``, where patterns below ``min_share`` of the
reference are pooled into one bucket so the index is not dominated by sparse cells. Per-field
level PSIs are reported alongside to say which field moved. A PSI above 0.2 is the conventional
"significant shift" alarm (Siddiqi 2006).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from probabilistic_record_linkage.linkage.comparison import ComparisonSpace

__all__ = [
    "LinkageMetrics",
    "PatternDriftMonitor",
    "PatternDriftReport",
    "categorical_psi",
    "create_linkage_metrics",
]

Int64Array = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


class DriftStateError(ValueError):
    """A stored drift reference is malformed."""


def categorical_psi(expected: FloatArray, actual: FloatArray, *, eps: float = 1e-4) -> float:
    """PSI between two count (or share) vectors over the same categories."""
    e = np.asarray(expected, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    if e.shape != a.shape or e.ndim != 1 or e.size == 0:
        raise ValueError("expected and actual must be non-empty vectors of the same length")
    if e.sum() <= 0 or a.sum() <= 0:
        raise ValueError("both distributions need positive mass")
    e = np.clip(e / e.sum(), eps, None)
    a = np.clip(a / a.sum(), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


@dataclass(frozen=True, slots=True)
class PatternDriftReport:
    psi: float
    field_psi: dict[str, float]
    drifted: bool
    n_pairs: int

    def as_dict(self) -> dict[str, object]:
        return {
            "psi": round(self.psi, 6),
            "field_psi": {k: round(v, 6) for k, v in self.field_psi.items()},
            "drifted": self.drifted,
            "n_pairs": self.n_pairs,
        }


class PatternDriftMonitor:
    def __init__(
        self,
        space: ComparisonSpace,
        reference_patterns: Int64Array,
        *,
        min_share: float = 0.002,
        threshold: float = 0.2,
    ) -> None:
        if reference_patterns.size == 0:
            raise ValueError("reference patterns are empty")
        self.space = space
        self.threshold = threshold
        distinct, counts = np.unique(reference_patterns, return_counts=True)
        shares = counts / counts.sum()
        self.buckets = distinct[shares >= min_share]
        self.reference = self._bucket_counts(reference_patterns)
        levels = space.decode(reference_patterns)
        self.reference_levels = [
            np.bincount(levels[:, f], minlength=spec.n_levels).astype(np.float64)
            for f, spec in enumerate(space.fields)
        ]

    def _bucket_counts(self, patterns: Int64Array) -> FloatArray:
        position = np.searchsorted(self.buckets, patterns)
        position = np.minimum(position, max(self.buckets.size - 1, 0))
        known = self.buckets.size > 0
        hit = (self.buckets[position] == patterns) if known else np.zeros(patterns.size, dtype=bool)
        index = np.where(hit, position, self.buckets.size)
        return np.bincount(index, minlength=self.buckets.size + 1).astype(np.float64)

    def check(self, patterns: Int64Array) -> PatternDriftReport:
        if patterns.size == 0:
            raise ValueError("no patterns to check")
        psi = categorical_psi(self.reference, self._bucket_counts(patterns))
        levels = self.space.decode(patterns)
        field_psi = {
            spec.name: categorical_psi(
                self.reference_levels[f],
                np.bincount(levels[:, f], minlength=spec.n_levels).astype(np.float64),
            )
            for f, spec in enumerate(self.space.fields)
        }
        return PatternDriftReport(psi, field_psi, psi > self.threshold, int(patterns.size))

    def to_json(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "buckets": [int(b) for b in self.buckets],
            "reference": [float(x) for x in self.reference],
            "reference_levels": [[float(x) for x in row] for row in self.reference_levels],
        }

    @classmethod
    def from_json(cls, space: ComparisonSpace, data: Mapping[str, object]) -> PatternDriftMonitor:
        monitor = cls.__new__(cls)
        monitor.space = space
        threshold = data["threshold"]
        buckets = data["buckets"]
        reference = data["reference"]
        levels = data["reference_levels"]
        if not isinstance(threshold, int | float) or not isinstance(buckets, list):
            raise DriftStateError("invalid drift monitor state")
        if not isinstance(reference, list) or not isinstance(levels, list):
            raise DriftStateError("invalid drift monitor state")
        monitor.threshold = float(threshold)
        monitor.buckets = np.asarray(buckets, dtype=np.int64)
        monitor.reference = np.asarray(reference, dtype=np.float64)
        monitor.reference_levels = [np.asarray(row, dtype=np.float64) for row in levels]
        if monitor.reference.size != monitor.buckets.size + 1:
            raise ValueError("drift monitor reference does not match its buckets")
        return monitor


@dataclass(frozen=True, slots=True)
class LinkageMetrics:
    registry: CollectorRegistry
    blocking_pairs_total: Counter
    reduction_ratio: Gauge
    match_rate: Gauge
    review_queue_depth: Gauge
    max_cluster_size: Gauge
    comparison_pattern_psi: Gauge
    em_iterations: Gauge
    stage_duration_seconds: Histogram
    entity_lookups_total: Counter
    review_decisions_total: Counter


def create_linkage_metrics(registry: CollectorRegistry | None = None) -> LinkageMetrics:
    """Linkage metric families. Labels never carry record ids or personal data."""
    reg = registry if registry is not None else CollectorRegistry()
    return LinkageMetrics(
        registry=reg,
        blocking_pairs_total=Counter(
            "blocking_pairs_total", "Candidate pairs produced by blocking.", registry=reg
        ),
        reduction_ratio=Gauge(
            "reduction_ratio", "1 - candidate pairs / all record pairs (last run).", registry=reg
        ),
        match_rate=Gauge(
            "match_rate", "Share of candidate pairs in the match band (last run).", registry=reg
        ),
        review_queue_depth=Gauge(
            "review_queue_depth", "Open items in the review queue.", registry=reg
        ),
        max_cluster_size=Gauge(
            "max_cluster_size", "Largest resolved cluster (last run).", registry=reg
        ),
        comparison_pattern_psi=Gauge(
            "comparison_pattern_psi",
            "PSI of comparison patterns against the model's reference.",
            registry=reg,
        ),
        em_iterations=Gauge(
            "em_iterations", "EM iterations of the selected initialization.", registry=reg
        ),
        stage_duration_seconds=Histogram(
            "linkage_stage_duration_seconds",
            "Pipeline stage duration.",
            ["stage"],
            registry=reg,
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 15.0, 60.0, 300.0),
        ),
        entity_lookups_total=Counter(
            "entity_lookups_total", "Entity lookups by PII access level.", ["pii"], registry=reg
        ),
        review_decisions_total=Counter(
            "review_decisions_total", "Reviewer decisions by outcome.", ["decision"], registry=reg
        ),
    )
