from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest
from scipy.special import expit

from probabilistic_record_linkage.linkage.baselines import LogisticLinkage, exact_key_labels
from probabilistic_record_linkage.linkage.comparison import ComparisonSpace, FieldSpec
from probabilistic_record_linkage.linkage.generator import SOURCE_SCHEMA
from probabilistic_record_linkage.linkage.normalize import normalize_sources, prepare_records

pytestmark = pytest.mark.unit

SPACE = ComparisonSpace(
    (FieldSpec("a", ("else", "exact")), FieldSpec("b", ("else", "mid", "exact")))
)


def test_exact_key_matching_groups_identical_normalized_keys() -> None:
    base = {"first_name": "Anna", "last_name": "Brandor", "dob": date(1980, 1, 2)}
    rows = [
        {"record_id": "s-1", **base},
        {"record_id": "s-2", **base, "first_name": "ANNA"},
        {"record_id": "s-3", **base, "first_name": "Anne"},
        {"record_id": "s-4", **base, "dob": None},
        {"record_id": "s-5", **base},
    ]
    records = prepare_records(normalize_sources({"s": pl.DataFrame(rows, schema=SOURCE_SCHEMA)}))
    assert exact_key_labels(records).tolist() == [0, 0, 2, 3, 0]
    empty = prepare_records(
        normalize_sources(
            {"s": pl.DataFrame([{"record_id": "s-1"}, {"record_id": "s-2"}], schema=SOURCE_SCHEMA)}
        )
    )
    assert exact_key_labels(empty).tolist() == [0, 1]


def test_logistic_regression_learns_level_effects() -> None:
    rng = np.random.default_rng(0)
    n = 20_000
    levels = np.stack([rng.integers(0, 3, n), rng.integers(0, 4, n)], axis=1).astype(np.int8)
    true_logit = (
        -2.0 + 3.0 * (levels[:, 0] == 2) + 2.0 * (levels[:, 1] == 3) - 1.0 * (levels[:, 1] == 1)
    )
    labels = rng.random(n) < expit(true_logit)
    model = LogisticLinkage(SPACE, l2=0.01).fit(levels, labels)
    predicted = model.predict_proba(levels)
    assert np.abs(predicted - expit(true_logit)).mean() < 0.02
    np.testing.assert_allclose(model.features(levels[:3]).sum(axis=1), [3.0, 3.0, 3.0])
    assert model.iterations >= 1


def test_logistic_regression_validates_its_inputs() -> None:
    model = LogisticLinkage(SPACE)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_proba(np.zeros((1, 2), dtype=np.int8))
    with pytest.raises(ValueError, match="one label"):
        model.fit(np.zeros((2, 2), dtype=np.int8), np.zeros(3, dtype=bool))
    with pytest.raises(ValueError, match="l2"):
        LogisticLinkage(SPACE, l2=-1.0)
