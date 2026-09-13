from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from probabilistic_record_linkage.linkage.evaluation import (
    bcubed_scores,
    expected_calibration_error,
    group_pairwise_scores,
    max_entities_per_cluster,
    merge_rate,
    pairwise_scores,
)

pytestmark = pytest.mark.unit


@given(st.lists(st.integers(0, 4), min_size=1, max_size=25), st.data())
def test_scores_match_brute_force_definitions(predicted: list[int], data: st.DataObject) -> None:
    truth = data.draw(st.lists(st.integers(0, 4), min_size=len(predicted), max_size=len(predicted)))
    pred = np.asarray(predicted, dtype=np.int64)
    true = np.asarray(truth, dtype=np.int64)
    n = len(predicted)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    tp = sum(pred[i] == pred[j] and true[i] == true[j] for i, j in pairs)
    predicted_pairs = sum(pred[i] == pred[j] for i, j in pairs)
    true_pairs = sum(true[i] == true[j] for i, j in pairs)
    scores = pairwise_scores(pred, true)
    assert (scores.true_positives, scores.predicted_pairs, scores.true_pairs) == (
        tp,
        predicted_pairs,
        true_pairs,
    )
    assert scores.precision == pytest.approx(tp / predicted_pairs if predicted_pairs else 1.0)
    assert scores.recall == pytest.approx(tp / true_pairs if true_pairs else 1.0)

    precision = recall = 0.0
    for i in range(n):
        cluster = {j for j in range(n) if pred[j] == pred[i]}
        entity = {j for j in range(n) if true[j] == true[i]}
        precision += len(cluster & entity) / len(cluster)
        recall += len(cluster & entity) / len(entity)
    b3_precision, b3_recall, _ = bcubed_scores(pred, true)
    assert b3_precision == pytest.approx(precision / n)
    assert b3_recall == pytest.approx(recall / n)


@given(
    st.lists(st.integers(0, 6), min_size=1, max_size=30),
    st.lists(st.integers(0, 6), min_size=30, max_size=30),
)
def test_scores_ignore_how_clusters_are_labelled(predicted: list[int], truth: list[int]) -> None:
    pred = np.asarray(predicted, dtype=np.int64)
    true = np.asarray(truth[: len(predicted)], dtype=np.int64)
    assert pairwise_scores(pred * 7 + 3, true) == pairwise_scores(pred, true)
    assert bcubed_scores(pred, true * 5 + 1) == pytest.approx(bcubed_scores(pred, true))


def test_perfect_and_empty_clusterings() -> None:
    labels = np.asarray([0, 0, 1, 2, 2, 2], dtype=np.int64)
    perfect = pairwise_scores(labels, labels)
    assert (perfect.precision, perfect.recall, perfect.f1) == (1.0, 1.0, 1.0)
    assert bcubed_scores(labels, labels) == (1.0, 1.0, 1.0)
    empty = np.zeros(0, dtype=np.int64)
    assert pairwise_scores(empty, empty).f1 == 1.0
    assert bcubed_scores(empty, empty) == (1.0, 1.0, 1.0)
    assert perfect.as_dict()["f1"] == 1.0
    with pytest.raises(ValueError, match="same shape"):
        pairwise_scores(labels, labels[:3])


def test_expected_calibration_error_known_values() -> None:
    probabilities = np.asarray([0.1, 0.1, 0.9, 0.9], dtype=np.float64)
    labels = np.asarray([False, False, True, True], dtype=np.bool_)
    assert expected_calibration_error(probabilities, labels) == pytest.approx(0.1)
    edges = np.asarray([0.0, 1.0], dtype=np.float64)
    assert expected_calibration_error(edges, np.asarray([False, True], dtype=np.bool_)) == 0.0
    assert expected_calibration_error(np.zeros(0), np.zeros(0, dtype=bool)) == 0.0
    with pytest.raises(ValueError, match="same shape"):
        expected_calibration_error(probabilities, labels[:2])
    with pytest.raises(ValueError, match="bins"):
        expected_calibration_error(probabilities, labels, bins=0)


def test_merge_rates_and_entities_per_cluster() -> None:
    predicted = np.asarray([0, 0, 1, 1], dtype=np.int64)
    truth = np.asarray([10, 11, 12, 12], dtype=np.int64)
    assert merge_rate(predicted, truth, [(10, 11), (10, 12), (99, 100)]) == (0.5, 2)
    assert merge_rate(predicted, truth, []) == (0.0, 0)
    assert max_entities_per_cluster(predicted, truth) == 2
    assert max_entities_per_cluster(predicted, truth, clusters=[1]) == 1
    assert max_entities_per_cluster(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)) == 0
    groups = group_pairwise_scores(predicted, truth, ["a", "a", "b", "b"])
    assert groups["a"].precision == 0.0
    assert groups["b"].f1 == 1.0
