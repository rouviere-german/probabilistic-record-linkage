from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from rapidfuzz.distance import OSA

from probabilistic_record_linkage.linkage.corruption import (
    KEYBOARD_NEIGHBOURS,
    RATE_FIELDS,
    CorruptionConfig,
    date_typo,
    digit_typo,
    keyboard_typo,
    swap_day_month,
)

pytestmark = pytest.mark.unit

_words = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=14)
_seeds = st.integers(min_value=0, max_value=2**32 - 1)


@given(_words, _seeds, st.floats(min_value=0.0, max_value=1.0))
def test_keyboard_typo_always_changes_the_word_by_exactly_one_edit(
    word: str, seed: int, share: float
) -> None:
    typo = keyboard_typo(word, np.random.default_rng(seed), transposition_share=share)
    assert typo != word
    assert OSA.distance(word, typo) == 1


def test_keyboard_neighbours_are_adjacent_keys() -> None:
    assert set(KEYBOARD_NEIGHBOURS) == set("abcdefghijklmnopqrstuvwxyz")
    for key, near in KEYBOARD_NEIGHBOURS.items():
        assert key not in near
        assert len(near) >= 2
    assert set(KEYBOARD_NEIGHBOURS["s"]) == set("adewxz")


def test_words_without_letters_still_get_a_typo() -> None:
    assert keyboard_typo("123", np.random.default_rng(0)) == "123e"


@given(st.text(alphabet="0123456789", min_size=1, max_size=8), _seeds)
def test_digit_typo_changes_exactly_one_digit(code: str, seed: int) -> None:
    out = digit_typo(code, np.random.default_rng(seed))
    assert len(out) == len(code)
    assert sum(a != b for a, b in zip(code, out, strict=True)) == 1


def test_digit_typo_needs_a_digit() -> None:
    with pytest.raises(ValueError, match="no digits"):
        digit_typo("abc", np.random.default_rng(0))


@given(st.dates(min_value=date(1901, 1, 1), max_value=date(2090, 12, 31)), _seeds)
def test_date_typo_changes_exactly_one_component(value: date, seed: int) -> None:
    out = date_typo(value, np.random.default_rng(seed))
    assert out != value
    changed = (out.year != value.year, out.month != value.month, out.day != value.day)
    assert sum(changed) == 1


@given(st.dates())
def test_day_month_swap_is_an_involution_when_it_applies(value: date) -> None:
    swapped = swap_day_month(value)
    if value.day <= 12 and value.day != value.month:
        assert (swapped.month, swapped.day) == (value.day, value.month)
        assert swap_day_month(swapped) == value
    else:
        assert swapped == value


@given(
    share=st.floats(min_value=0.0, max_value=0.19),
    multiplier=st.floats(min_value=1.0, max_value=5.0),
)
def test_sloppiness_mixture_preserves_every_marginal_rate(share: float, multiplier: float) -> None:
    config = CorruptionConfig().correlated(share, multiplier)
    mixture = config.multipliers()
    assert sum(p for p, _ in mixture) == pytest.approx(1.0)
    assert sum(p * a for p, a in mixture) == pytest.approx(1.0)
    assert all(a >= 0 for _, a in mixture)


def test_invalid_corruption_configurations_are_rejected() -> None:
    with pytest.raises(ValidationError):
        CorruptionConfig(sloppy_share=0.5, sloppy_multiplier=3.0)
    with pytest.raises(ValidationError):
        CorruptionConfig(address_move=0.5, sloppy_share=0.1, sloppy_multiplier=4.0)
    with pytest.raises(ValidationError):
        CorruptionConfig(dob_swap=0.6, dob_typo=0.6)


def test_scaled_rates_stay_valid_and_change_the_hash() -> None:
    base = CorruptionConfig()
    doubled = base.scaled(2.0)
    assert doubled.first_name_typo == pytest.approx(2 * base.first_name_typo)
    assert base.sha256() != doubled.sha256()
    capped = base.scaled(100.0)
    assert all(getattr(capped, name) <= 0.45 for name in RATE_FIELDS)
    with pytest.raises(ValueError, match="factor"):
        base.scaled(-1.0)
