"""Baselines the Fellegi-Sunter pipeline is compared against in ``bench/bench_linkage.py``.

- Exact normalized-key matching: records with identical normalized first name, last name and date
  of birth form a cluster; records missing any key field stay alone. This is the "just join on a
  key" approach the problem statement starts from.
- Supervised logistic regression on a small labeled sample (1% of candidate pairs by default),
  over one-hot agreement levels, fit by Newton-Raphson (IRLS) with an L2 penalty. Scoring uses a
  per-field coefficient lookup, so millions of pairs never materialize a dense feature matrix.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import expit

from probabilistic_record_linkage.linkage.comparison import ComparisonSpace
from probabilistic_record_linkage.linkage.normalize import PreparedRecords

__all__ = ["LogisticLinkage", "exact_key_labels"]

Int8Array = npt.NDArray[np.int8]
Int64Array = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


def exact_key_labels(records: PreparedRecords) -> Int64Array:
    """Cluster label per record: smallest row sharing (first, last, dob); singleton otherwise."""
    n = records.n
    labels = np.arange(n, dtype=np.int64)
    first = records.first_name.codes.astype(np.int64)
    last = records.last_name.codes.astype(np.int64)
    dob = records.sort_key("dob")
    complete = np.flatnonzero((first >= 0) & (last >= 0) & (dob >= 0))
    if complete.size == 0:
        return labels
    order = complete[np.lexsort((complete, dob[complete], last[complete], first[complete]))]
    changed = (
        (np.diff(first[order]) != 0) | (np.diff(last[order]) != 0) | (np.diff(dob[order]) != 0)
    )
    group_start = np.concatenate(([0], np.flatnonzero(changed) + 1))
    group_of = np.repeat(np.arange(group_start.size), np.diff(np.append(group_start, order.size)))
    first_row = order[group_start]
    labels[order] = first_row[group_of]
    return labels


class LogisticLinkage:
    """L2-regularized logistic regression over one-hot comparison levels."""

    def __init__(
        self,
        space: ComparisonSpace,
        *,
        l2: float = 1.0,
        max_iterations: int = 50,
        tolerance: float = 1e-8,
    ) -> None:
        if l2 < 0:
            raise ValueError("l2 must be >= 0")
        self.space = space
        self.l2 = l2
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.offsets = np.cumsum([0] + [spec.n_levels for spec in space.fields])
        self.weights: FloatArray | None = None
        self.iterations = 0

    @property
    def n_features(self) -> int:
        return int(self.offsets[-1]) + 1

    def features(self, levels: Int8Array) -> FloatArray:
        matrix = np.zeros((levels.shape[0], self.n_features), dtype=np.float64)
        rows = np.arange(levels.shape[0])
        for f in range(len(self.space.fields)):
            matrix[rows, self.offsets[f] + levels[:, f].astype(np.int64)] = 1.0
        matrix[:, -1] = 1.0
        return matrix

    def fit(self, levels: Int8Array, labels: npt.NDArray[np.bool_]) -> LogisticLinkage:
        if levels.shape[0] != labels.size or levels.shape[0] == 0:
            raise ValueError("need one label per training pair and at least one pair")
        x = self.features(levels)
        y = labels.astype(np.float64)
        penalty = np.full(self.n_features, self.l2)
        penalty[-1] = 0.0
        weights = np.zeros(self.n_features, dtype=np.float64)
        for iteration in range(1, self.max_iterations + 1):
            probability = expit(x @ weights)
            gradient = x.T @ (y - probability) - penalty * weights
            curvature = probability * (1.0 - probability)
            hessian = (
                (x * curvature[:, None]).T @ x + np.diag(penalty) + 1e-9 * np.eye(self.n_features)
            )
            step = np.linalg.solve(hessian, gradient)
            weights = weights + step
            self.iterations = iteration
            if float(np.max(np.abs(step))) < self.tolerance:
                break
        self.weights = weights
        return self

    def predict_proba(self, levels: Int8Array) -> FloatArray:
        if self.weights is None:
            raise RuntimeError("fit the model first")
        logit = np.full(levels.shape[0], self.weights[-1], dtype=np.float64)
        for f in range(len(self.space.fields)):
            logit += self.weights[self.offsets[f] + levels[:, f].astype(np.int64)]
        probability: FloatArray = expit(logit)
        return probability
