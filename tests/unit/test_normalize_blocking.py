from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from probabilistic_record_linkage.linkage.blocking import (
    BlockingConfig,
    bucket_pairs,
    dedupe_pairs,
    pair_completeness,
    reduction_ratio,
    run_blocking,
    sorted_neighbourhood_pairs,
    true_pair_count,
)
from probabilistic_record_linkage.linkage.generator import (
    SOURCE_SCHEMA,
    SyntheticDataset,
    entity_labels,
)
from probabilistic_record_linkage.linkage.minhash import (
    EMPTY,
    MinHasher,
    band_keys,
    candidate_probability,
    jaccard,
    shingle_sets,
)
from probabilistic_record_linkage.linkage.nicknames import NicknameTable
from probabilistic_record_linkage.linkage.normalize import (
    CodedColumn,
    normalize_sources,
    prepare_records,
)
from probabilistic_record_linkage.linkage.phonetic import soundex

pytestmark = pytest.mark.unit


def _frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=SOURCE_SCHEMA)


def test_normalization_removes_formatting_differences() -> None:
    frame = _frame(
        [
            {
                "record_id": "a-1", "first_name": "  ANNA-Marie ", "last_name": "O'Neil",
                "dob": date(1980, 1, 2), "sex": " f ", "street": "12 Brandor Street",
                "city": "North  Field", "postcode": "1234",
            },
            {
                "record_id": "a-2", "first_name": "!!!", "last_name": None, "dob": None,
                "sex": "X", "street": "7 Main Avenue", "city": None, "postcode": "01234",
            },
        ]
    )  # fmt: skip
    out = normalize_sources({"a": frame})
    assert out["first_name"].to_list() == ["anna marie", None]
    assert out["last_name"].to_list() == ["oneil", None]
    assert out["street"].to_list() == ["12 brandor st", "7 main ave"]
    assert out["city"].to_list() == ["north field", None]
    assert out["postcode"].to_list() == [None, "01234"]
    assert out["sex"].to_list() == ["F", None]
    assert out["row"].to_list() == [0, 1]


def test_sources_are_concatenated_in_name_order() -> None:
    first = _frame([{"record_id": "b-1", "first_name": "x"}])
    second = _frame([{"record_id": "a-1", "first_name": "y"}])
    out = normalize_sources({"b": first, "a": second})
    assert out["source"].to_list() == ["a", "b"]
    with pytest.raises(ValueError, match="at least one source"):
        normalize_sources({})


@given(st.lists(st.one_of(st.none(), st.text(alphabet="abc", min_size=1, max_size=3)), max_size=40))
def test_coded_columns_round_trip(values: list[str | None]) -> None:
    column = CodedColumn.from_values(values)
    decoded = [column.values[code] if code >= 0 else None for code in column.codes.tolist()]
    assert decoded == values
    assert list(column.values) == sorted({v for v in values if v is not None})
    assert int(column.counts().sum()) == sum(v is not None for v in values)


def test_derived_columns_and_nickname_canonical_codes() -> None:
    column = CodedColumn.from_values(["robert", "rupert", None])
    derived = column.derived(soundex)
    assert derived.values == ("R163",)
    assert derived.codes.tolist() == [0, 0, -1]
    frame = _frame(
        [
            {"record_id": "a-1", "first_name": "William", "dob": date(1980, 1, 2)},
            {"record_id": "a-2", "first_name": "Bill"},
            {"record_id": "a-3", "first_name": "Willy"},
        ]
    )
    records = prepare_records(
        normalize_sources({"a": frame}), NicknameTable({"william": ("bill",)})
    )
    canonical = records.first_canonical
    assert canonical[0, 0] == canonical[1, 0] >= 0
    assert canonical[2].tolist() == [-1, -1]
    assert records.sort_key("dob").tolist() == [19800102, -1, -1]
    assert records.full_names() == ["william", "bill", "willy"]
    with pytest.raises(ValueError, match="unknown sort key"):
        records.sort_key("shoe_size")


@given(st.lists(st.text(alphabet="abcdef", max_size=8), min_size=1, max_size=6))
def test_minhash_signatures_depend_only_on_the_shingle_set(strings: list[str]) -> None:
    hasher = MinHasher(16, seed=3)
    signatures = hasher.signatures(*shingle_sets(strings, 2))
    reordered = [*reversed(strings), strings[0]]
    again = hasher.signatures(*shingle_sets(reordered, 2))
    np.testing.assert_array_equal(again[: len(strings)][::-1], signatures)
    for row, text in zip(signatures, strings, strict=True):
        assert np.all(row == EMPTY) if text == "" else not np.any(row == EMPTY)


def test_minhash_agreement_estimates_jaccard_similarity() -> None:
    rng = np.random.default_rng(0)
    vocabulary = [f"g{i}" for i in range(300)]
    table = MinHasher(1024, seed=5).permutation_table(vocabulary)
    for _ in range(10):
        a = set(rng.choice(300, size=60, replace=False).tolist())
        b = set(rng.choice(sorted(a), size=30, replace=False).tolist()) | set(
            rng.choice(300, size=30, replace=False).tolist()
        )
        sig_a = table[:, sorted(a)].min(axis=1)
        sig_b = table[:, sorted(b)].min(axis=1)
        exact = jaccard({str(x) for x in a}, {str(x) for x in b})
        assert float(np.mean(sig_a == sig_b)) == pytest.approx(exact, abs=0.08)


@given(st.floats(0.0, 1.0), st.floats(0.0, 1.0), st.integers(1, 30), st.integers(1, 8))
def test_lsh_candidate_probability_is_monotone(s1: float, s2: float, bands: int, rows: int) -> None:
    low, high = sorted((s1, s2))
    assert (
        candidate_probability(low, bands, rows) <= candidate_probability(high, bands, rows) + 1e-12
    )
    assert candidate_probability(0.0, bands, rows) == 0.0
    assert candidate_probability(1.0, bands, rows) == 1.0


def test_band_keys_validate_their_shape_and_are_deterministic() -> None:
    signatures = np.arange(24, dtype=np.uint32).reshape(2, 12)
    with pytest.raises(ValueError, match="multiple"):
        list(band_keys(signatures, 5))
    doubled = np.vstack([signatures[0], signatures[0]])
    for _, keys in band_keys(doubled, 4):
        assert keys[0] == keys[1]
    assert jaccard(set(), set()) == 1.0


@given(
    st.lists(st.integers(0, 5), max_size=40),
    st.integers(min_value=2, max_value=8),
    st.integers(min_value=1, max_value=4),
)
def test_bucket_pairs_match_a_brute_force_oracle(
    keys: list[int], max_bucket: int, window: int
) -> None:
    array = np.asarray(keys, dtype=np.int64)
    left, right = bucket_pairs(array, array != 0, max_bucket=max_bucket, window=window)
    got = {(min(a, b), max(a, b)) for a, b in zip(left.tolist(), right.tolist(), strict=True)}
    assert len(got) == left.size
    expected: set[tuple[int, int]] = set()
    for key in set(keys) - {0}:
        members = sorted(i for i, k in enumerate(keys) if k == key)
        for i, a in enumerate(members):
            for j in range(i + 1, len(members)):
                if len(members) <= max_bucket or j - i <= window:
                    expected.add((a, members[j]))
    assert got == expected


@given(st.lists(st.tuples(st.integers(0, 9), st.integers(0, 9)), max_size=30))
def test_dedupe_pairs_orders_and_removes_duplicates(pairs: list[tuple[int, int]]) -> None:
    left = np.asarray([p[0] for p in pairs], dtype=np.int64)
    right = np.asarray([p[1] for p in pairs], dtype=np.int64)
    out_left, out_right = dedupe_pairs(left, right, 10)
    result = list(zip(out_left.tolist(), out_right.tolist(), strict=True))
    assert result == sorted({(min(a, b), max(a, b)) for a, b in pairs if a != b})


def test_pair_completeness_and_reduction_ratio_formulas() -> None:
    labels = np.asarray([0, 0, 1, 1, 1, 2], dtype=np.int64)
    assert true_pair_count(labels) == 4
    left = np.asarray([0, 2], dtype=np.int64)
    right = np.asarray([1, 3], dtype=np.int64)
    assert pair_completeness(left, right, labels) == 0.5
    assert pair_completeness(left, right, np.arange(6, dtype=np.int64)) == 1.0
    assert reduction_ratio(10, 5) == 0.0
    assert reduction_ratio(0, 1) == 0.0
    with pytest.raises(ValidationError):
        BlockingConfig(num_perm=100, bands=30)


def test_blocking_on_generated_data_keeps_matches_and_drops_most_pairs(
    small_dataset: SyntheticDataset,
) -> None:
    normalized = normalize_sources(small_dataset.sources)
    records = prepare_records(normalized, small_dataset.world.nicknames)
    truth = entity_labels(small_dataset.truth, list(records.record_ids))
    result = run_blocking(records, BlockingConfig())
    assert result.n_pairs > 0
    assert np.all(result.left < result.right)
    assert pair_completeness(result.left, result.right, truth) >= 0.97
    assert reduction_ratio(result.n_pairs, records.n) >= 0.99
    assert set(result.seconds) == {"lsh", "sorted_neighbourhood"}
    sn_left, sn_right = sorted_neighbourhood_pairs(records, ("dob", "soundex_last"), 3)
    assert sn_left.size == sn_right.size <= records.n * 2
