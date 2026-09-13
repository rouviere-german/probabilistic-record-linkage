from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from rapidfuzz.distance import JaroWinkler

from probabilistic_record_linkage.linkage.comparison import (
    PLAIN_SPACE,
    ComparisonSpace,
    FieldSpec,
    PlainComparator,
    exact_agreement_rate,
    jaro_winkler_pairs,
    random_pairs,
    u_from_random_pairs,
)
from probabilistic_record_linkage.linkage.generator import SOURCE_SCHEMA
from probabilistic_record_linkage.linkage.nicknames import NicknameTable
from probabilistic_record_linkage.linkage.normalize import (
    PreparedRecords,
    normalize_sources,
    prepare_records,
)

pytestmark = pytest.mark.unit

BASE: dict[str, object] = {
    "record_id": "s-1",
    "first_name": "William",
    "last_name": "Brandor",
    "dob": date(1980, 3, 5),
    "sex": "M",
    "street": "12 Brandor St",
    "city": "Northfield",
    "postcode": "12345",
}


def _records(
    rows: list[dict[str, object]], nicknames: NicknameTable | None = None
) -> PreparedRecords:
    frame = pl.DataFrame(rows, schema=SOURCE_SCHEMA)
    return prepare_records(normalize_sources({"s": frame}), nicknames)


def test_plaintext_levels_for_hand_built_pairs() -> None:
    rows = [
        BASE,
        {**BASE, "record_id": "s-2"},
        {
            **BASE, "record_id": "s-3", "first_name": "Bill", "last_name": "Brandr",
            "dob": date(1980, 5, 3), "postcode": "12346",
        },
        {
            **BASE, "record_id": "s-4", "first_name": None, "dob": date(1980, 3, 6), "sex": None,
            "street": "99 Other Rd", "postcode": "54321",
        },
        {
            **BASE, "record_id": "s-5", "first_name": "Zed", "last_name": "Quux",
            "dob": date(1955, 11, 30), "sex": "F", "street": "1 Far Way", "city": "Elsewhere",
            "postcode": "99999",
        },
        {**BASE, "record_id": "s-6", "street": "12 Brandar St"},
    ]  # fmt: skip
    comparator = PlainComparator(_records(rows, NicknameTable({"william": ("bill",)})))
    left = np.zeros(5, dtype=np.int64)
    right = np.arange(1, 6, dtype=np.int64)
    levels = comparator.levels(left, right)
    # columns: first name, last name, date of birth, sex, address
    assert levels[0].tolist() == [5, 4, 4, 2, 5]
    assert levels[1].tolist() == [4, 3, 3, 2, 4]
    assert levels[2].tolist() == [0, 4, 2, 0, 2]
    assert levels[3].tolist() == [1, 1, 1, 1, 1]
    assert levels[4].tolist() == [5, 4, 4, 2, 4]
    assert comparator.space is PLAIN_SPACE
    assert comparator.n_records == 6


@given(st.data())
def test_pattern_encoding_round_trips(data: st.DataObject) -> None:
    row = st.tuples(*[st.integers(0, spec.n_levels - 1) for spec in PLAIN_SPACE.fields])
    rows = data.draw(st.lists(row, min_size=1, max_size=30))
    levels = np.asarray(rows, dtype=np.int8)
    patterns = PLAIN_SPACE.encode(levels)
    assert np.all((patterns >= 0) & (patterns < PLAIN_SPACE.n_patterns))
    np.testing.assert_array_equal(PLAIN_SPACE.decode(patterns), levels)
    assert len(set(patterns.tolist())) == len(set(rows))


def test_comparison_spaces_are_validated() -> None:
    with pytest.raises(ValueError, match="at least one field"):
        ComparisonSpace(())
    with pytest.raises(ValueError, match="unique"):
        ComparisonSpace((FieldSpec("a", ("else", "exact")), FieldSpec("a", ("else", "exact"))))
    with pytest.raises(ValueError, match="disagreement and an exact"):
        ComparisonSpace((FieldSpec("a", ("exact",)),))
    with pytest.raises(ValueError, match="one column per field"):
        PLAIN_SPACE.encode(np.zeros((2, 3), dtype=np.int8))
    levels = PLAIN_SPACE.describe()[0]["levels"]
    assert isinstance(levels, list)
    assert levels[0] == "missing"


@given(
    st.lists(
        st.text(alphabet="abcde", min_size=1, max_size=6), min_size=2, max_size=8, unique=True
    ),
    st.data(),
)
def test_jaro_winkler_pairs_match_rapidfuzz(values: list[str], data: st.DataObject) -> None:
    ordered = tuple(sorted(values))
    n = len(ordered)
    a = data.draw(st.lists(st.integers(0, n - 1), min_size=1, max_size=20))
    b = [data.draw(st.integers(0, n - 1)) for _ in a]
    got = jaro_winkler_pairs(ordered, np.asarray(a, dtype=np.int32), np.asarray(b, dtype=np.int32))
    expected = [
        JaroWinkler.normalized_similarity(ordered[i], ordered[j]) for i, j in zip(a, b, strict=True)
    ]
    np.testing.assert_allclose(got, expected, atol=1e-6)
    assert (
        jaro_winkler_pairs(ordered, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)).size
        == 0
    )


@given(st.lists(st.integers(-1, 4), max_size=25))
def test_exact_agreement_rate_matches_brute_force(keys: list[int]) -> None:
    present = [k for k in keys if k >= 0]
    pairs = [(i, j) for i in range(len(present)) for j in range(len(present)) if i != j]
    expected = sum(present[i] == present[j] for i, j in pairs) / len(pairs) if pairs else 0.0
    array = np.asarray(keys, dtype=np.int64)
    assert exact_agreement_rate(array, array >= 0) == pytest.approx(expected)


def test_random_pairs_never_pair_a_record_with_itself() -> None:
    left, right = random_pairs(5, 1_000, np.random.default_rng(0))
    assert np.all(left != right)
    assert set(np.concatenate([left, right]).tolist()) == set(range(5))
    with pytest.raises(ValueError, match="two records"):
        random_pairs(1, 3, np.random.default_rng(0))


def test_u_from_random_pairs_uses_exact_rates_and_sums_to_one() -> None:
    levels = np.asarray([[1, 1, 1, 1, 1], [2, 2, 2, 1, 3], [0, 1, 4, 2, 5]], dtype=np.int8)
    u = u_from_random_pairs(PLAIN_SPACE, levels, {"first_name": 0.01, "address": 1e-7})
    for probs, spec in zip(u, PLAIN_SPACE.fields, strict=True):
        assert probs[0] == 0.0
        assert probs[1:].sum() == pytest.approx(1.0)
        assert probs.shape == (spec.n_levels,)
    assert u[0][PLAIN_SPACE.fields[0].exact_level] == pytest.approx(0.01)
    assert u[4][PLAIN_SPACE.fields[4].exact_level] == pytest.approx(1e-7)


def test_term_frequency_adjustment_rewards_rare_values() -> None:
    surnames = ["Common"] * 6 + ["Rare"] * 2 + ["Solo", "Lone"]
    rows = [
        {**BASE, "record_id": f"s-{i}", "first_name": f"Name{chr(97 + i)}", "last_name": surname}
        for i, surname in enumerate(surnames)
    ]
    records = _records(rows)
    comparator = PlainComparator(records)
    left = np.asarray([6, 0], dtype=np.int64)
    right = np.asarray([7, 1], dtype=np.int64)
    levels = comparator.levels(left, right)
    adjustment = comparator.tf_adjustment(left, right, levels)
    sum_sq = 0.6**2 + 0.2**2 + 0.1**2 + 0.1**2
    assert adjustment[0] == pytest.approx(math.log(sum_sq / 0.2))
    assert adjustment[1] == pytest.approx(math.log(sum_sq / 0.6))
    clipped = PlainComparator(records, tf_clip_bits=0.1).tf_adjustment(left, right, levels)
    assert np.all(np.abs(clipped) <= 0.1 * math.log(2) + 1e-12)
    with pytest.raises(ValueError, match="jw_low"):
        PlainComparator(records, jw_low=0.95, jw_high=0.9)


def test_random_u_of_generated_records_is_a_distribution(small_dataset: object) -> None:
    from probabilistic_record_linkage.linkage.generator import SyntheticDataset

    assert isinstance(small_dataset, SyntheticDataset)
    records = prepare_records(
        normalize_sources(small_dataset.sources), small_dataset.world.nicknames
    )
    u = PlainComparator(records).random_u(20_000, np.random.default_rng(1))
    for probs in u:
        assert probs[1:].sum() == pytest.approx(1.0)
        assert np.all(probs[1:] > 0)
    exact_address = u[4][PLAIN_SPACE.fields[4].exact_level]
    assert 0 < exact_address < 1e-3
