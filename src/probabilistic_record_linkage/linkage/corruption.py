"""Corruption model: keyboard typos, transpositions, nicknames, missing fields, date swaps, moves.

Every rate is a per-record, per-field probability. Operators are chosen so that the analytic
m-probabilities in :mod:`probabilistic_record_linkage.linkage.analytic` are exact: a typo always
changes the string (never a no-op transposition of equal letters), a date typo always yields a
different valid date, and the exclusive order of operations below is the order the analytic
formulas assume.

- first name: missing, else nickname (only names that have one), else typo, else clean
- last name: missing, else typo
- date of birth: missing, else day/month swap (a no-op when day > 12 or day == month), else a
  one-component typo, else clean
- sex: missing, else flipped
- address: the entity's alternative address with probability ``address_move``; then a street typo,
  a missing postcode or a postcode digit typo; the city may be missing

Correlated corruption: with ``sloppy_share = rho > 0`` a record is "sloppy" with probability rho
and all of its rates are multiplied by ``k = sloppy_multiplier``; the other records use
``(1 - rho k) / (1 - rho)`` so every marginal rate is unchanged. Only the dependence between fields
of one record changes, which is exactly the Fellegi-Sunter conditional-independence assumption
that the calibration benchmark stresses.
"""

from __future__ import annotations

from datetime import date

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from probabilistic_record_linkage.core.canonical import canonical_json, sha256_hex

__all__ = [
    "KEYBOARD_NEIGHBOURS",
    "RATE_FIELDS",
    "CorruptionConfig",
    "date_typo",
    "digit_typo",
    "keyboard_typo",
    "swap_day_month",
]

_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")


def _neighbours() -> dict[str, str]:
    table: dict[str, str] = {}
    for r, row in enumerate(_ROWS):
        for c, char in enumerate(row):
            near: list[str] = []
            for dr, dc in ((0, -1), (0, 1), (-1, 0), (-1, 1), (1, -1), (1, 0)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < len(_ROWS) and 0 <= cc < len(_ROWS[rr]):
                    near.append(_ROWS[rr][cc])
            table[char] = "".join(sorted(set(near) - {char}))
    return table


KEYBOARD_NEIGHBOURS: dict[str, str] = _neighbours()

RATE_FIELDS: tuple[str, ...] = (
    "first_name_missing",
    "first_name_nickname",
    "first_name_typo",
    "last_name_missing",
    "last_name_typo",
    "dob_missing",
    "dob_swap",
    "dob_typo",
    "sex_missing",
    "sex_flip",
    "address_move",
    "street_typo",
    "postcode_missing",
    "postcode_typo",
    "city_missing",
)


class CorruptionConfig(BaseModel):
    """Per-record corruption probabilities. Defaults are the "default corruption" of the README."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    first_name_missing: float = Field(default=0.02, ge=0.0, le=1.0)
    first_name_nickname: float = Field(default=0.08, ge=0.0, le=1.0)
    first_name_typo: float = Field(default=0.06, ge=0.0, le=1.0)
    last_name_missing: float = Field(default=0.01, ge=0.0, le=1.0)
    last_name_typo: float = Field(default=0.06, ge=0.0, le=1.0)
    dob_missing: float = Field(default=0.02, ge=0.0, le=1.0)
    dob_swap: float = Field(default=0.04, ge=0.0, le=1.0)
    dob_typo: float = Field(default=0.03, ge=0.0, le=1.0)
    sex_missing: float = Field(default=0.02, ge=0.0, le=1.0)
    sex_flip: float = Field(default=0.005, ge=0.0, le=1.0)
    address_move: float = Field(default=0.10, ge=0.0, le=1.0)
    street_typo: float = Field(default=0.04, ge=0.0, le=1.0)
    postcode_missing: float = Field(default=0.03, ge=0.0, le=1.0)
    postcode_typo: float = Field(default=0.02, ge=0.0, le=1.0)
    city_missing: float = Field(default=0.02, ge=0.0, le=1.0)
    transposition_share: float = Field(default=0.25, ge=0.0, le=1.0)
    sloppy_share: float = Field(default=0.0, ge=0.0, lt=1.0)
    sloppy_multiplier: float = Field(default=4.0, ge=1.0, le=50.0)

    @model_validator(mode="after")
    def _rates_stay_probabilities(self) -> CorruptionConfig:
        top = max(multiplier for _, multiplier in self.multipliers())
        if self.sloppy_share * self.sloppy_multiplier >= 1.0:
            raise ValueError("sloppy_share * sloppy_multiplier must be < 1")
        for name in RATE_FIELDS:
            if getattr(self, name) * top > 1.0:
                raise ValueError(f"{name} times the sloppy multiplier exceeds 1")
        if (self.dob_swap + self.dob_typo) * top > 1.0:
            raise ValueError("dob_swap + dob_typo times the sloppy multiplier exceeds 1")
        return self

    def multipliers(self) -> tuple[tuple[float, float], ...]:
        """``(probability, rate multiplier)`` mixture of record types; marginals are preserved."""
        rho, k = self.sloppy_share, self.sloppy_multiplier
        if rho == 0.0:
            return ((1.0, 1.0),)
        return ((rho, k), (1.0 - rho, (1.0 - rho * k) / (1.0 - rho)))

    def scaled(self, factor: float) -> CorruptionConfig:
        """All rates times ``factor``, capped so they stay valid (for the robustness sweep)."""
        if factor < 0:
            raise ValueError("factor must be >= 0")
        top = max(multiplier for _, multiplier in self.multipliers())
        cap = 0.45 / top
        values = {name: min(getattr(self, name) * factor, cap) for name in RATE_FIELDS}
        return self.model_copy(update=values)

    def correlated(self, share: float = 0.1, multiplier: float = 5.0) -> CorruptionConfig:
        """Same marginal rates, but errors concentrate in a ``share`` of sloppy records."""
        return CorruptionConfig.model_validate(
            {**self.model_dump(), "sloppy_share": share, "sloppy_multiplier": multiplier}
        )

    def sha256(self) -> str:
        return sha256_hex(canonical_json(self.model_dump(mode="json")))


def keyboard_typo(word: str, rng: np.random.Generator, *, transposition_share: float = 0.25) -> str:
    """Return ``word`` with exactly one edit: a transposition or a keyboard-adjacent key error.

    The result always differs from ``word`` and is at optimal-string-alignment distance 1.
    Letters outside ``a-z`` are never edited.
    """
    letters = list(word)
    editable = [i for i, ch in enumerate(letters) if ch in KEYBOARD_NEIGHBOURS]
    if not editable:
        return word + "e"
    if rng.random() < transposition_share:
        swaps = [
            i
            for i in range(len(letters) - 1)
            if letters[i] != letters[i + 1]
            and letters[i] in KEYBOARD_NEIGHBOURS
            and letters[i + 1] in KEYBOARD_NEIGHBOURS
        ]
        if swaps:
            i = swaps[int(rng.integers(len(swaps)))]
            letters[i], letters[i + 1] = letters[i + 1], letters[i]
            return "".join(letters)
    i = editable[int(rng.integers(len(editable)))]
    near = KEYBOARD_NEIGHBOURS[letters[i]]
    replacement = near[int(rng.integers(len(near)))]
    operation = rng.random()
    if operation < 0.6 or len(letters) <= 3:
        letters[i] = replacement
    elif operation < 0.8:
        letters.insert(i + 1, replacement)
    else:
        del letters[i]
    return "".join(letters)


def digit_typo(code: str, rng: np.random.Generator) -> str:
    """Replace one digit of ``code`` with a different digit."""
    digits = [i for i, ch in enumerate(code) if ch.isdigit()]
    if not digits:
        raise ValueError("code has no digits")
    i = digits[int(rng.integers(len(digits)))]
    new = (int(code[i]) + int(rng.integers(1, 10))) % 10
    return code[:i] + str(new) + code[i + 1 :]


def swap_day_month(value: date) -> date:
    """Day/month transposition; unchanged when the swap is impossible or a no-op."""
    if value.day <= 12 and value.day != value.month:
        return date(value.year, value.day, value.month)
    return value


def date_typo(value: date, rng: np.random.Generator, *, min_year: int = 1900) -> date:
    """A different valid date that differs from ``value`` in exactly one of day, month or year."""
    for _ in range(64):
        component = rng.random()
        year, month, day = value.year, value.month, value.day
        if component < 0.5:
            day = day + int(rng.choice([-3, -2, -1, 1, 2, 3]))
        elif component < 0.75:
            month = month + int(rng.choice([-1, 1]))
        else:
            year = year + int(rng.choice([-2, -1, 1, 2]))
        if year < min_year:
            continue
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if candidate != value:
            return candidate
    return date(value.year, value.month, value.day - 1 if value.day > 1 else value.day + 1)
