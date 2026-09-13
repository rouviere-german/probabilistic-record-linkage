"""Field normalization (Polars) and integer-coded columns for vectorized comparison.

Normalization removes formatting differences that are not errors: case, punctuation, whitespace
and long versus short street suffixes (``Street`` -> ``st``). Values that normalize to an empty
string become null. After normalization every text column is dictionary-encoded into ``int32``
codes (``-1`` for null) over its sorted distinct values, so exact agreement is an integer compare
and string similarity only needs to run once per distinct value pair.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from probabilistic_record_linkage.linkage.nicknames import NicknameTable
from probabilistic_record_linkage.linkage.phonetic import soundex
from probabilistic_record_linkage.linkage.vocab import STREET_SUFFIXES

__all__ = [
    "RECORD_COLUMNS",
    "CodedColumn",
    "PreparedRecords",
    "normalize_sources",
    "prepare_records",
]

RECORD_COLUMNS = (
    "record_id",
    "first_name",
    "last_name",
    "dob",
    "sex",
    "street",
    "city",
    "postcode",
)
Int32Array = npt.NDArray[np.int32]
Int64Array = npt.NDArray[np.int64]


def _text(column: str, keep: str) -> pl.Expr:
    expr = (
        pl.col(column)
        .str.to_lowercase()
        .str.replace_all(r"['`]", "")
        .str.replace_all(keep, " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )
    return pl.when(expr.str.len_chars() > 0).then(expr).otherwise(None)


def _street() -> pl.Expr:
    expr = _text("street", r"[^a-z0-9]+")
    for short, long in STREET_SUFFIXES:
        if short != long:
            expr = expr.str.replace(rf" {long}$", f" {short}")
    return expr


def normalize_sources(sources: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """Concatenate sources (sorted by name), normalize every field, add ``source`` and ``row``."""
    if not sources:
        raise ValueError("at least one source is required")
    frames = [
        sources[name].select(RECORD_COLUMNS).with_columns(pl.lit(name).alias("source"))
        for name in sorted(sources)
    ]
    raw = pl.concat(frames, how="vertical")
    postcode = pl.col("postcode").str.replace_all(r"[^0-9]", "")
    sex = pl.col("sex").str.strip_chars().str.to_uppercase()
    return raw.with_columns(
        _text("first_name", r"[^a-z]+").alias("first_name"),
        _text("last_name", r"[^a-z]+").alias("last_name"),
        _street().alias("street"),
        _text("city", r"[^a-z]+").alias("city"),
        pl.when(postcode.str.len_chars() == 5).then(postcode).otherwise(None).alias("postcode"),
        pl.when(sex.is_in(["F", "M"])).then(sex).otherwise(None).alias("sex"),
    ).with_row_index("row")


@dataclass(frozen=True, slots=True)
class CodedColumn:
    """Dictionary encoding of a nullable text column."""

    codes: Int32Array
    values: tuple[str, ...]

    @classmethod
    def from_series(cls, series: pl.Series) -> CodedColumn:
        distinct = series.drop_nulls().unique().sort()
        if distinct.len() == 0:
            return cls(np.full(series.len(), -1, dtype=np.int32), ())
        codes = (
            series.replace_strict(
                distinct,
                pl.Series(np.arange(distinct.len(), dtype=np.int32)),
                default=-1,
                return_dtype=pl.Int32,
            )
            .fill_null(-1)
            .to_numpy()
            .astype(np.int32, copy=False)
        )
        return cls(codes, tuple(distinct.to_list()))

    @classmethod
    def from_values(cls, values: list[str | None]) -> CodedColumn:
        return cls.from_series(pl.Series(values, dtype=pl.Utf8))

    @property
    def n_values(self) -> int:
        return len(self.values)

    def counts(self) -> Int64Array:
        present = self.codes[self.codes >= 0]
        counts: Int64Array = np.bincount(present, minlength=self.n_values).astype(np.int64)
        return counts

    def value(self, row: int) -> str | None:
        code = int(self.codes[row])
        return self.values[code] if code >= 0 else None

    def derived(self, function: object) -> CodedColumn:
        """Encode ``function(value)`` of every distinct value (empty results become null)."""
        if not callable(function):
            raise TypeError("function must be callable")
        mapped = [str(function(value)) for value in self.values]
        distinct = sorted({value for value in mapped if value})
        index = {value: i for i, value in enumerate(distinct)}
        per_value = np.asarray([index.get(value, -1) for value in mapped], dtype=np.int32)
        if per_value.size == 0:
            return CodedColumn(np.full(self.codes.size, -1, dtype=np.int32), ())
        codes = np.where(self.codes >= 0, per_value[np.maximum(self.codes, 0)], -1).astype(np.int32)
        return CodedColumn(codes, tuple(distinct))


@dataclass(frozen=True, slots=True)
class PreparedRecords:
    """Normalized, integer-coded records in pipeline row order."""

    record_ids: tuple[str, ...]
    source: CodedColumn
    first_name: CodedColumn
    last_name: CodedColumn
    street: CodedColumn
    city: CodedColumn
    postcode: CodedColumn
    sex: CodedColumn
    year: Int32Array
    month: Int32Array
    day: Int32Array
    first_canonical: Int32Array  # (n, 2) canonical-name codes, -1 when unknown
    soundex_first: CodedColumn
    soundex_last: CodedColumn

    @property
    def n(self) -> int:
        return len(self.record_ids)

    def full_names(self) -> list[str]:
        """``"first last"`` per record (missing parts omitted), the MinHash shingle source."""
        first = [
            self.first_name.values[c] if c >= 0 else "" for c in self.first_name.codes.tolist()
        ]
        last = [self.last_name.values[c] if c >= 0 else "" for c in self.last_name.codes.tolist()]
        return [f"{a} {b}".strip() for a, b in zip(first, last, strict=True)]

    def sort_key(self, name: str) -> Int64Array:
        """Integer sort key for sorted-neighbourhood blocking (``-1`` sorts nulls first)."""
        if name == "dob":
            present = self.year >= 0
            value = self.year.astype(np.int64) * 10_000 + self.month * 100 + self.day
            key: Int64Array = np.where(present, value, -1).astype(np.int64)
            return key
        columns: dict[str, CodedColumn] = {
            "soundex_first": self.soundex_first,
            "soundex_last": self.soundex_last,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "postcode": self.postcode,
            "street": self.street,
            "city": self.city,
        }
        if name == "year":
            return self.year.astype(np.int64)
        if name not in columns:
            raise ValueError(f"unknown sort key {name!r}")
        return columns[name].codes.astype(np.int64)


def prepare_records(
    normalized: pl.DataFrame, nicknames: NicknameTable | None = None
) -> PreparedRecords:
    """Encode a frame produced by :func:`normalize_sources`."""
    first = CodedColumn.from_series(normalized["first_name"])
    last = CodedColumn.from_series(normalized["last_name"])
    dob = normalized["dob"]
    year = dob.dt.year().fill_null(-1).to_numpy().astype(np.int32)
    month = dob.dt.month().fill_null(-1).to_numpy().astype(np.int32)
    day = dob.dt.day().fill_null(-1).to_numpy().astype(np.int32)

    canonical_pairs = np.full((first.n_values, 2), -1, dtype=np.int32)
    if nicknames is not None and first.n_values:
        per_value = [nicknames.canonical_names(value)[:2] for value in first.values]
        universe = sorted({name for names in per_value for name in names})
        index = {name: i for i, name in enumerate(universe)}
        for row, names in enumerate(per_value):
            for slot, name in enumerate(names):
                canonical_pairs[row, slot] = index[name]
    if first.n_values:
        first_canonical = np.where(
            (first.codes >= 0)[:, None], canonical_pairs[np.maximum(first.codes, 0)], -1
        ).astype(np.int32)
    else:
        first_canonical = np.full((normalized.height, 2), -1, dtype=np.int32)

    return PreparedRecords(
        record_ids=tuple(normalized["record_id"].to_list()),
        source=CodedColumn.from_series(normalized["source"]),
        first_name=first,
        last_name=last,
        street=CodedColumn.from_series(normalized["street"]),
        city=CodedColumn.from_series(normalized["city"]),
        postcode=CodedColumn.from_series(normalized["postcode"]),
        sex=CodedColumn.from_series(normalized["sex"]),
        year=year,
        month=month,
        day=day,
        first_canonical=first_canonical,
        soundex_first=first.derived(soundex),
        soundex_last=last.derived(soundex),
    )
