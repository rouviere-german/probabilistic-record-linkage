"""Ground-truth evaluation: pairwise and B-cubed scores, calibration, merge rates.

All scores work on flat label arrays (one cluster label and one true entity id per record), via a
sparse contingency table, so they cost O(n log n) regardless of cluster sizes.

- Pairwise precision/recall/F1 over record pairs placed in the same cluster (Christen 2012, ch. 7).
- B-cubed precision/recall per record, averaged (Bagga and Baldwin 1998).
- Expected calibration error with equal-width probability bins (Naeini et al. 2015).
- Calibration restricted to a probability band (:func:`band_calibration`). Over all candidate
  pairs the ECE is dominated by the overwhelming majority of pairs the model puts near zero, so
  it stays small even when the uncertain pairs are mis-calibrated. Restricting to the pairs the
  model is unsure about, and to the pairs it commits to as matches (where the gap between the
  expected and the observed false-discovery rate is what a user actually pays), shows the
  conditional-independence trade-off that the aggregate hides.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "BandCalibration",
    "PairScores",
    "band_calibration",
    "bcubed_scores",
    "contingency",
    "expected_calibration_error",
    "group_pairwise_scores",
    "max_entities_per_cluster",
    "merge_rate",
    "pairwise_scores",
]

Int64Array = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PairScores:
    precision: float
    recall: float
    f1: float
    true_positives: int
    predicted_pairs: int
    true_pairs: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "true_positives": self.true_positives,
            "predicted_pairs": self.predicted_pairs,
            "true_pairs": self.true_pairs,
        }


def _choose2(counts: Int64Array) -> int:
    return int(np.sum(counts * (counts - 1) // 2))


def contingency(
    predicted: Int64Array, truth: Int64Array
) -> tuple[Int64Array, Int64Array, Int64Array, Int64Array, Int64Array]:
    """``(cell_counts, cell_pred, cell_true, pred_sizes, true_sizes)`` of the non-empty cells."""
    if predicted.shape != truth.shape:
        raise ValueError("predicted and truth must have the same shape")
    _, pred_codes = np.unique(predicted, return_inverse=True)
    true_values, true_codes = np.unique(truth, return_inverse=True)
    keys = pred_codes.astype(np.int64) * np.int64(max(1, true_values.size)) + true_codes
    cells, cell_counts = np.unique(keys, return_counts=True)
    cell_pred = cells // max(1, true_values.size)
    cell_true = cells % max(1, true_values.size)
    pred_sizes = np.bincount(pred_codes).astype(np.int64)
    true_sizes = np.bincount(true_codes).astype(np.int64)
    return cell_counts.astype(np.int64), cell_pred, cell_true, pred_sizes, true_sizes


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def pairwise_scores(predicted: Int64Array, truth: Int64Array) -> PairScores:
    if predicted.size == 0:
        return PairScores(1.0, 1.0, 1.0, 0, 0, 0)
    cells, _, _, pred_sizes, true_sizes = contingency(predicted, truth)
    tp = _choose2(cells)
    predicted_pairs = _choose2(pred_sizes)
    true_pairs = _choose2(true_sizes)
    precision = tp / predicted_pairs if predicted_pairs else 1.0
    recall = tp / true_pairs if true_pairs else 1.0
    return PairScores(precision, recall, _f1(precision, recall), tp, predicted_pairs, true_pairs)


def bcubed_scores(predicted: Int64Array, truth: Int64Array) -> tuple[float, float, float]:
    if predicted.size == 0:
        return 1.0, 1.0, 1.0
    cells, cell_pred, cell_true, pred_sizes, true_sizes = contingency(predicted, truth)
    squared = cells.astype(np.float64) ** 2
    precision = float(np.sum(squared / pred_sizes[cell_pred]) / predicted.size)
    recall = float(np.sum(squared / true_sizes[cell_true]) / predicted.size)
    return precision, recall, _f1(precision, recall)


def expected_calibration_error(
    probabilities: FloatArray, labels: npt.NDArray[np.bool_], *, bins: int = 10
) -> float:
    if probabilities.shape != labels.shape:
        raise ValueError("probabilities and labels must have the same shape")
    if probabilities.size == 0:
        return 0.0
    if bins < 1:
        raise ValueError("bins must be >= 1")
    index = np.minimum((probabilities * bins).astype(np.int64), bins - 1)
    counts = np.bincount(index, minlength=bins).astype(np.float64)
    confidence = np.bincount(index, weights=probabilities, minlength=bins)
    accuracy = np.bincount(index, weights=labels.astype(np.float64), minlength=bins)
    present = counts > 0
    gap = np.abs(accuracy[present] - confidence[present])
    return float(np.sum(gap) / probabilities.size)


@dataclass(frozen=True, slots=True)
class BandCalibration:
    """Calibration of the pairs whose match probability falls in ``[low, high]``."""

    pairs: int
    share: float  # of all scored pairs
    ece: float
    expected_fdr: float  # mean(1 - p): what the model says the error rate in the band is
    observed_fdr: float  # 1 - mean(label): what it actually is

    @property
    def fdr_gap(self) -> float:
        return self.observed_fdr - self.expected_fdr

    def as_dict(self) -> dict[str, float | int]:
        return {
            "pairs": self.pairs,
            "share": round(self.share, 6),
            "ece": round(self.ece, 6),
            "expected_fdr": round(self.expected_fdr, 6),
            "observed_fdr": round(self.observed_fdr, 6),
            "fdr_gap": round(self.fdr_gap, 6),
        }


def band_calibration(
    probabilities: FloatArray,
    labels: npt.NDArray[np.bool_],
    *,
    low: float = 0.0,
    high: float = 1.0,
    bins: int = 10,
) -> BandCalibration:
    """ECE and expected-versus-observed false-discovery rate inside one probability band."""
    if probabilities.shape != labels.shape:
        raise ValueError("probabilities and labels must have the same shape")
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError("the band must satisfy 0 <= low <= high <= 1")
    total = probabilities.size
    selected = (probabilities >= low) & (probabilities <= high)
    inside = probabilities[selected]
    truth = labels[selected]
    if inside.size == 0:
        return BandCalibration(0, 0.0, 0.0, 0.0, 0.0)
    return BandCalibration(
        pairs=int(inside.size),
        share=inside.size / total,
        ece=expected_calibration_error(inside, truth, bins=bins),
        expected_fdr=float(np.mean(1.0 - inside)),
        observed_fdr=1.0 - float(np.mean(truth.astype(np.float64))),
    )


def merge_rate(
    predicted: Int64Array, truth: Int64Array, entity_pairs: Iterable[tuple[int, int]]
) -> tuple[float, int]:
    """Share of the given (entity, entity) pairs that share at least one cluster, and the count."""
    clusters_of: dict[int, set[int]] = {}
    for cluster, entity in zip(predicted.tolist(), truth.tolist(), strict=True):
        clusters_of.setdefault(int(entity), set()).add(int(cluster))
    total = 0
    merged = 0
    for a, b in entity_pairs:
        if a not in clusters_of or b not in clusters_of:
            continue
        total += 1
        if clusters_of[a] & clusters_of[b]:
            merged += 1
    return (merged / total if total else 0.0), total


def max_entities_per_cluster(
    predicted: Int64Array, truth: Int64Array, *, clusters: Sequence[int] | None = None
) -> int:
    """Largest number of distinct true entities in one cluster (optionally among ``clusters``)."""
    if predicted.size == 0:
        return 0
    cells, cell_pred, _, _, _ = contingency(predicted, truth)
    del cells
    pred_values = np.unique(predicted)
    per_cluster = np.bincount(cell_pred, minlength=pred_values.size)
    if clusters is not None:
        wanted = np.isin(pred_values, np.asarray(list(clusters), dtype=np.int64))
        per_cluster = per_cluster[wanted]
    return int(per_cluster.max()) if per_cluster.size else 0


def group_pairwise_scores(
    predicted: Int64Array, truth: Int64Array, groups: Sequence[str]
) -> dict[str, PairScores]:
    """Pairwise scores restricted to the records of each group."""
    names = np.asarray(groups)
    return {
        str(group): pairwise_scores(predicted[names == group], truth[names == group])
        for group in sorted(set(groups))
    }
