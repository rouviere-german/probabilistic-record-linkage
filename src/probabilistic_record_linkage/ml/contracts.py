"""pandera (Polars backend) contracts for source records, with PII-safe failure reporting.

The source contract is strict: a frame with any extra column (for example a leaked ``entity_id``
from the generator's ground truth) is rejected, so true identities cannot reach the linker by
accident. Failure summaries report column, check and count only. The failing values themselves are
never copied into the exception, because they are exactly the personal data being validated.
"""

from __future__ import annotations

from datetime import date

import pandera.polars as pa
import polars as pl
from pandera.errors import SchemaError, SchemaErrors

__all__ = [
    "RECORD_ID_PATTERN",
    "DataContractError",
    "SourceRecordSchema",
    "validate_source",
]

RECORD_ID_PATTERN = r"^[a-z][a-z0-9]{0,7}-[0-9]{1,12}$"
MAX_NAME_CHARS = 64
MAX_STREET_CHARS = 128


class DataContractError(ValueError):
    """A frame violated its contract. ``failures`` holds (column, check, count) summaries."""

    def __init__(self, failures: list[dict[str, str | int]]) -> None:
        total = sum(int(item["count"]) for item in failures)
        super().__init__(
            f"data contract violated: {total} failure(s) across {len(failures)} check(s)"
        )
        self.failures = failures


class SourceRecordSchema(pa.DataFrameModel):
    """One person record as delivered by a source system."""

    record_id: str = pa.Field(unique=True, str_matches=RECORD_ID_PATTERN)
    first_name: str = pa.Field(
        nullable=True, str_length={"min_value": 1, "max_value": MAX_NAME_CHARS}
    )
    last_name: str = pa.Field(
        nullable=True, str_length={"min_value": 1, "max_value": MAX_NAME_CHARS}
    )
    dob: pl.Date = pa.Field(nullable=True, ge=date(1900, 1, 1), le=date(2030, 12, 31))
    sex: str = pa.Field(nullable=True, isin=["F", "M"])
    street: str = pa.Field(
        nullable=True, str_length={"min_value": 1, "max_value": MAX_STREET_CHARS}
    )
    city: str = pa.Field(nullable=True, str_length={"min_value": 1, "max_value": MAX_NAME_CHARS})
    postcode: str = pa.Field(nullable=True, str_matches=r"^[0-9]{5}$")

    class Config:
        strict = True
        name = "source_record"


def validate_source(frame: pl.DataFrame, *, max_rows: int = 2_000_000) -> pl.DataFrame:
    """Validate lazily (all failures at once) and raise a summarized :class:`DataContractError`."""
    if frame.height > max_rows:
        raise DataContractError(
            [{"column": "*", "check": f"max_rows({max_rows})", "count": frame.height - max_rows}]
        )
    try:
        return SourceRecordSchema.validate(frame, lazy=True)
    except SchemaErrors as exc:
        cases = exc.failure_cases
        summary = (
            cases.select(
                pl.col("column").cast(pl.Utf8).fill_null("*"),
                pl.col("check").cast(pl.Utf8).fill_null("unknown"),
            )
            .group_by(["column", "check"])
            .len()
            .sort(["column", "check"])
        )
        failures: list[dict[str, str | int]] = [
            {"column": str(row[0]), "check": str(row[1]), "count": int(row[2])}
            for row in summary.iter_rows()
        ]
        raise DataContractError(failures) from None
    except SchemaError as exc:  # raised for frame-level problems even in lazy mode
        check = getattr(exc, "check", None)
        column = getattr(exc, "column_name", None)
        raise DataContractError(
            [{"column": str(column or "*"), "check": str(check or "schema"), "count": 1}]
        ) from None
