from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from probabilistic_record_linkage.linkage.comparison import PLAIN_SPACE
from probabilistic_record_linkage.linkage.privacy import PRIVATE_SPACE
from probabilistic_record_linkage.linkage.thresholds import (
    DEFAULT_REVIEW_RULES,
    ReviewRule,
    ThresholdConfig,
    Thresholds,
    assign_bands,
    select_thresholds,
)

pytestmark = pytest.mark.unit


@given(
    st.lists(
        st.sampled_from([0.0, 0.1, 0.5, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]) | st.floats(0, 1),
        min_size=1,
        max_size=200,
    ),
    st.floats(0.001, 0.3),
    st.floats(0.001, 0.3),
    st.integers(0, 50),
)
def test_threshold_invariants(
    probabilities: list[float], fdr: float, missed: float, capacity: int
) -> None:
    p = np.asarray(probabilities)
    config = ThresholdConfig(
        fdr_target=fdr,
        missed_match_target=missed,
        review_capacity=capacity,
        min_match_probability=0.0,
    )
    thresholds = select_thresholds(p, config)
    assert 0.0 <= thresholds.review <= thresholds.match <= 1.0
    band = (p >= thresholds.review) & (p < thresholds.match)
    assert int(band.sum()) <= capacity
    assert thresholds.review_pairs == int(band.sum())
    in_match = p >= thresholds.match
    assert thresholds.match_pairs == int(in_match.sum())
    if in_match.any():
        assert float(np.mean(1.0 - p[in_match])) <= fdr + 1e-9


def test_ties_at_the_match_threshold_do_not_break_the_fdr_target() -> None:
    p = np.asarray([0.95, 0.85, 0.85])
    thresholds = select_thresholds(p, ThresholdConfig(fdr_target=0.1, min_match_probability=0.0))
    assert thresholds.match_pairs == 1
    assert thresholds.expected_fdr <= 0.1


def test_degenerate_probability_inputs() -> None:
    empty = select_thresholds(np.zeros(0), ThresholdConfig())
    assert (empty.match, empty.review, empty.match_pairs) == (1.0, 1.0, 0)
    for bad in ([float("nan")], [1.5], [-0.1]):
        with pytest.raises(ValueError, match="within"):
            select_thresholds(np.asarray(bad), ThresholdConfig())


def test_the_twin_rule_moves_twin_like_matches_to_review() -> None:
    levels = np.asarray(
        [
            [1, 4, 4, 2, 5],  # clearly different first names, same surname, date and address
            [5, 4, 4, 2, 5],  # the same first name: a duplicate
            [1, 4, 4, 2, 1],  # twins who moved apart: the address is not required
            [1, 4, 3, 2, 5],  # date of birth with a day/month swap
            [1, 3, 2, 2, 5],  # surname typo (JW >= 0.92) and a one-component date error
            [1, 2, 4, 2, 5],  # surname only loosely similar: not a twin signature
            [1, 4, 1, 2, 5],  # different date of birth
            [0, 4, 4, 2, 5],  # missing first name: no evidence either way
            [2, 4, 4, 2, 5],  # similar first names look like a typo, not like twins
        ],
        dtype=np.int8,
    )
    rule = DEFAULT_REVIEW_RULES[0]
    expected = [True, False, True, True, True, False, False, False, False]
    assert rule.mask(PLAIN_SPACE, levels).tolist() == expected
    thresholds = Thresholds(
        match=0.9,
        review=0.5,
        expected_fdr=0.0,
        expected_missed_share=0.0,
        match_pairs=0,
        review_pairs=0,
    )
    probabilities = np.asarray([0.99, 0.99, 0.6, 0.99, 0.99, 0.99, 0.99, 0.99, 0.1])
    bands, hits = assign_bands(probabilities, levels, PLAIN_SPACE, thresholds, DEFAULT_REVIEW_RULES)
    assert bands.tolist() == [1, 2, 1, 1, 1, 2, 2, 2, 0]
    assert hits == {"possible_twin": 3}  # only match-band pairs count as routed
    private = np.asarray([[1, 3, 2, 2, 5], [1, 2, 2, 2, 5], [1, 3, 1, 2, 5]], dtype=np.int8)
    assert rule.mask(PRIVATE_SPACE, private).tolist() == [True, False, True]


def test_rule_aliases_resolve_per_comparison_space() -> None:
    rule = ReviewRule(name="missing_dob", levels={"dob": ("@missing",), "first_name": ("@exact",)})
    assert rule.mask(PLAIN_SPACE, np.asarray([[5, 0, 0, 0, 0]], dtype=np.int8)).tolist() == [True]
    assert rule.mask(PRIVATE_SPACE, np.asarray([[4, 0, 0, 0, 0]], dtype=np.int8)).tolist() == [True]
    assert rule.mask(PRIVATE_SPACE, np.asarray([[5, 0, 0, 0, 0]], dtype=np.int8)).tolist() == [
        False
    ]
    top = ReviewRule(name="near", levels={"last_name": ("@top2",), "dob": ("@top9",)})
    last_levels = np.asarray([[0, 4, 1, 0, 0], [0, 3, 1, 0, 0], [0, 2, 4, 0, 0]], dtype=np.int8)
    assert top.mask(PLAIN_SPACE, last_levels).tolist() == [True, True, False]
    assert top.mask(PRIVATE_SPACE, last_levels).tolist() == [True, True, False]
    # @top<N> never reaches the missing level, however large N is.
    assert top.mask(PLAIN_SPACE, np.asarray([[0, 4, 0, 0, 0]], dtype=np.int8)).tolist() == [False]
    with pytest.raises(ValueError, match="unknown level"):
        ReviewRule(name="x", levels={"dob": ("@top0",)}).mask(PLAIN_SPACE, last_levels)
    with pytest.raises(ValueError, match="unknown level"):
        ReviewRule(name="x", levels={"dob": ("@topmost",)}).mask(PLAIN_SPACE, last_levels)


def test_invalid_rules_are_rejected() -> None:
    levels = np.zeros((1, 5), dtype=np.int8)
    with pytest.raises(ValueError, match="unknown field"):
        ReviewRule(name="x", levels={"shoe_size": ("@exact",)}).mask(PLAIN_SPACE, levels)
    with pytest.raises(ValueError, match="unknown level"):
        ReviewRule(name="x", levels={"dob": ("sideways",)}).mask(PLAIN_SPACE, levels)
    with pytest.raises(ValidationError):
        ReviewRule(name="Bad Name", levels={})
