from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from probabilistic_record_linkage.linkage.generator import SOURCE_SCHEMA, SyntheticDataset
from probabilistic_record_linkage.linkage.normalize import normalize_sources
from probabilistic_record_linkage.linkage.privacy import (
    PRIVATE_SPACE,
    BloomComparator,
    HardeningConfig,
    InsecureKeyFileError,
    PrivacyConfig,
    date_tokens,
    derive_key,
    dice_pairs,
    encode_records,
    load_key_file,
    name_salts,
    token_blocking,
)

pytestmark = pytest.mark.unit

KEY = b"k" * 32


def _row(
    index: int, first: str, *, dob: date = date(1980, 3, 5), last: str = "Brandor"
) -> dict[str, object]:
    return {
        "record_id": f"s-{index}",
        "first_name": first,
        "last_name": last,
        "dob": dob,
        "sex": "M",
        "street": "12 Brandor St",
        "city": "Northfield",
        "postcode": "12345",
    }


def _normalized(rows: list[dict[str, object]]) -> pl.DataFrame:
    return normalize_sources({"s": pl.DataFrame(rows, schema=SOURCE_SCHEMA)})


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    os.chmod(path, 0o600)
    return path


def test_key_files_are_validated(tmp_path: Path) -> None:
    with pytest.raises(InsecureKeyFileError, match="not found"):
        load_key_file(tmp_path / "missing.key")
    with pytest.raises(InsecureKeyFileError, match="between"):
        load_key_file(_write(tmp_path / "short.key", b"abc"))
    with pytest.raises(InsecureKeyFileError, match="regular file"):
        load_key_file(tmp_path)
    with pytest.raises(InsecureKeyFileError, match="too large"):
        load_key_file(_write(tmp_path / "big.key", b"x" * 9_000))
    assert load_key_file(_write(tmp_path / "good.key", b"a" * 64 + b"\n")) == b"a" * 64


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_key_files_readable_by_others_are_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "shared.key", b"a" * 64)
    os.chmod(path, 0o644)
    with pytest.raises(InsecureKeyFileError, match="group or others"):
        load_key_file(path)


def test_symlinked_key_files_are_refused(tmp_path: Path) -> None:
    target = _write(tmp_path / "real.key", b"a" * 64)
    link = tmp_path / "link.key"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("creating symlinks needs extra privileges on this platform")
    with pytest.raises(InsecureKeyFileError, match="regular file"):
        load_key_file(link)


def test_derived_keys_are_independent() -> None:
    assert derive_key(KEY, "a") != derive_key(KEY, "b")
    assert derive_key(KEY, "a") != derive_key(b"x" * 32, "a")
    assert len(derive_key(KEY, "a")) == 32


def test_encodings_are_deterministic_per_key_and_preserve_similarity() -> None:
    frame = _normalized(
        [_row(1, "William"), _row(2, "William"), _row(3, "Wiliam"), _row(4, "Zed", last="Quux")]
    )
    config = PrivacyConfig.plain()
    first = encode_records(frame, KEY, config)
    second = encode_records(frame, KEY, config)
    other_key = encode_records(frame, b"j" * 32, config)
    for name in first.fields:
        np.testing.assert_array_equal(first.fields[name], second.fields[name])
    assert not np.array_equal(first.fields["first_name"], other_key.fields["first_name"])
    left, right = np.zeros(3, dtype=np.int64), np.arange(1, 4, dtype=np.int64)
    dice = dice_pairs(first, "first_name", left, right)
    assert dice[0] == 1.0
    assert 0.55 <= dice[1] < 1.0
    assert dice[2] < dice[1]
    levels = BloomComparator(first).levels(left, right)
    assert levels[0].tolist() == [4, 4, 3, 2, 5]
    assert levels[1][0] in (2, 3)
    assert levels[2][:2].tolist() == [1, 1]


@settings(max_examples=25, deadline=None)
@given(st.lists(st.text(alphabet="abcdefghij", min_size=1, max_size=12), min_size=1, max_size=8))
def test_balanced_name_encodings_have_a_constant_hamming_weight(names: list[str]) -> None:
    frame = _normalized([_row(i, name) for i, name in enumerate(names)])
    encoded = encode_records(frame, KEY, PrivacyConfig())
    weights = encoded.popcounts["first_name"][encoded.present["first_name"]]
    assert np.all(weights == encoded.bits["first_name"] // 2)


def test_birth_year_salting_separates_equal_names_born_in_different_years() -> None:
    frame = _normalized(
        [_row(1, "William", dob=date(1980, 3, 5)), _row(2, "William", dob=date(1981, 3, 5))]
    )
    hardened = encode_records(frame, KEY, PrivacyConfig())
    plain = encode_records(frame, KEY, PrivacyConfig.plain())
    assert np.array_equal(plain.fields["first_name"][0], plain.fields["first_name"][1])
    assert not np.array_equal(hardened.fields["first_name"][0], hardened.fields["first_name"][1])
    assert PrivacyConfig().hardening.enabled
    assert not PrivacyConfig.plain().hardening.enabled
    assert PrivacyConfig().dice_cutpoints("dob") == (0.45, 0.45)


def test_name_salts_combine_the_configured_record_attributes() -> None:
    frame = _normalized(
        [_row(1, "William", dob=date(1980, 3, 5)), {**_row(2, "Anna"), "dob": None}]
    )
    assert name_salts(frame, HardeningConfig()) == ["1980", ""]
    every_part = HardeningConfig(name_salt_parts=("birth_year", "birth_month", "sex"))
    assert name_salts(frame, every_part) == ["1980|03|M", ""]
    assert name_salts(frame, HardeningConfig(salt_names_with_birth_year=False)) == ["", ""]


def test_date_tokens_keep_partial_similarity() -> None:
    tokens = date_tokens("19800305")
    assert tokens == {"y1980", "m03", "d05", "md0305"}
    assert len(tokens & date_tokens("19800503")) == 2
    assert not tokens & date_tokens("19550716")


def test_keyed_tokens_block_records_that_share_a_birth_date() -> None:
    frame = _normalized(
        [
            _row(1, "William"),
            _row(2, "Zed", last="Quux"),
            _row(3, "Anna", dob=date(1999, 1, 1), last="Other"),
        ]
    )
    encoded = encode_records(frame, KEY, PrivacyConfig())
    left, right = token_blocking(encoded)
    pairs = set(zip(left.tolist(), right.tolist(), strict=True))
    assert (0, 1) in pairs
    assert (0, 2) not in pairs
    assert (1, 2) not in pairs


def test_bloom_comparator_estimates_u_and_short_keys_are_refused(
    small_dataset: SyntheticDataset,
) -> None:
    frame = normalize_sources(small_dataset.sources)
    comparator = BloomComparator(encode_records(frame, KEY, PrivacyConfig()))
    u = comparator.random_u(5_000, np.random.default_rng(0))
    assert len(u) == len(PRIVATE_SPACE.fields)
    for probs in u:
        assert probs[1:].sum() == pytest.approx(1.0)
    zero = comparator.tf_adjustment(
        np.zeros(1, dtype=np.int64), np.ones(1, dtype=np.int64), np.zeros((1, 5), dtype=np.int8)
    )
    assert zero.tolist() == [0.0]
    assert comparator.n_records == small_dataset.n_records
    with pytest.raises(InsecureKeyFileError, match="at least"):
        encode_records(frame.head(2), b"short", PrivacyConfig())
