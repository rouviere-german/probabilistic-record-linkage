from __future__ import annotations

import io
import json
import time
from datetime import date
from typing import Any

import jwt
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from probabilistic_record_linkage.core.logging import configure_logging, get_logger
from probabilistic_record_linkage.core.redaction import REDACTED, is_sensitive_key, redact_text
from probabilistic_record_linkage.core.security import (
    SCOPES,
    ApiKeyVerifier,
    CompositeVerifier,
    JwtVerifier,
    hash_api_key,
    issue_token,
    require_scopes,
)
from probabilistic_record_linkage.linkage.generator import SOURCE_SCHEMA, SyntheticDataset
from probabilistic_record_linkage.ml.contracts import DataContractError, validate_source

pytestmark = pytest.mark.unit

SECRET = b"s" * 40
ISSUER = "issuer"
AUDIENCE = "audience"
NOW = int(time.time())


def _verifier() -> JwtVerifier:
    return JwtVerifier(SECRET, issuer=ISSUER, audience=AUDIENCE, max_ttl_seconds=3600)


def _claims(**overrides: Any) -> dict[str, Any]:
    claims: dict[str, Any] = {
        "sub": "alice",
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": NOW,
        "exp": NOW + 600,
        "scope": "entity:read",
    }
    claims.update(overrides)
    return {key: value for key, value in claims.items() if value is not None}


def _token(**overrides: Any) -> str:
    return issue_token(
        SECRET,
        subject="alice",
        scopes=["entity:read"],
        issuer=ISSUER,
        audience=AUDIENCE,
        **overrides,
    )


# --- scoped JWTs ------------------------------------------------------------------------------


def test_valid_tokens_carry_only_known_scopes() -> None:
    token = issue_token(
        SECRET,
        subject="alice",
        scopes=["entity:read", "pii:read"],
        issuer=ISSUER,
        audience=AUDIENCE,
    )
    principal = _verifier().verify(token)
    assert principal is not None
    assert principal.subject == "alice"
    assert principal.scopes == {"entity:read", "pii:read"}
    assert len(principal.key_id) == 12
    assert "alice" not in principal.key_id
    extra = _verifier().verify(
        jwt.encode(_claims(scope="entity:read admin root"), SECRET, algorithm="HS256")
    )
    assert extra is not None
    assert extra.scopes == {"entity:read"}
    no_scope = _verifier().verify(jwt.encode(_claims(scope=None), SECRET, algorithm="HS256"))
    assert no_scope is not None
    assert no_scope.scopes == frozenset()
    assert {"link:run", "entity:read", "review:write", "pii:read"} == SCOPES


@pytest.mark.parametrize(
    "overrides",
    [
        {"iat": NOW - 700, "exp": NOW - 100},
        {"aud": "other"},
        {"iss": "other"},
        {"sub": None},
        {"sub": ""},
        {"sub": "x" * 200},
        {"exp": None},
        {"iat": NOW, "exp": NOW + 7_200},
        {"tenant": "acme corp"},
        {"tenant": ""},
        {"tenant": "-acme"},
        {"tenant": "t" * 65},
        {"tenant": 7},
        {"tenant": ["acme"]},
    ],
    ids=[
        "expired",
        "audience",
        "issuer",
        "no-subject",
        "empty-subject",
        "long-subject",
        "no-exp",
        "long-lived",
        "tenant-with-space",
        "empty-tenant",
        "tenant-leading-dash",
        "long-tenant",
        "numeric-tenant",
        "list-tenant",
    ],
)
def test_invalid_claims_are_refused(overrides: dict[str, Any]) -> None:
    assert _verifier().verify(jwt.encode(_claims(**overrides), SECRET, algorithm="HS256")) is None


def test_the_tenant_claim_defines_the_owner_of_link_jobs() -> None:
    without = _verifier().verify(_token())
    assert without is not None
    assert without.tenant is None
    assert without.owner == f"principal:{without.key_id}"
    with_tenant = _verifier().verify(_token(tenant="acme.eu-1"))
    assert with_tenant is not None
    assert with_tenant.tenant == "acme.eu-1"
    assert with_tenant.owner == "tenant:acme.eu-1"
    other_subject = _verifier().verify(
        jwt.encode(_claims(sub="bob", tenant="acme.eu-1"), SECRET, algorithm="HS256")
    )
    assert other_subject is not None
    assert other_subject.owner == with_tenant.owner
    assert other_subject.key_id != with_tenant.key_id
    with pytest.raises(ValueError, match="tenant"):
        _token(tenant="not valid")


def test_tampered_foreign_unsigned_and_malformed_tokens_are_refused() -> None:
    header, payload, signature = _token().split(".")
    flipped = signature[:-2] + ("A" if signature[-2] != "A" else "B") + signature[-1]
    assert _verifier().verify(f"{header}.{payload}.{flipped}") is None
    foreign = issue_token(
        b"t" * 40, subject="alice", scopes=["entity:read"], issuer=ISSUER, audience=AUDIENCE
    )
    assert _verifier().verify(foreign) is None
    assert _verifier().verify(jwt.encode(_claims(), None, algorithm="none")) is None
    assert _verifier().verify(jwt.encode(_claims(), SECRET * 2, algorithm="HS512")) is None
    for bad in (None, "", "garbage", "x" * 5_000):
        assert _verifier().verify(bad) is None
    unconfigured = JwtVerifier(None, issuer=ISSUER, audience=AUDIENCE)
    assert not unconfigured.configured
    assert unconfigured.verify(_token()) is None


def test_unknown_scope_names_are_rejected_when_issuing_or_requiring() -> None:
    with pytest.raises(ValueError, match="unknown scopes"):
        issue_token(SECRET, subject="a", scopes=["admin"], issuer=ISSUER, audience=AUDIENCE)
    with pytest.raises(ValueError, match="unknown scopes"):
        require_scopes("admin")


def test_composite_verifier_tries_api_keys_then_tokens() -> None:
    api_key = "k" * 30
    composite = CompositeVerifier(ApiKeyVerifier([hash_api_key(api_key)]), _verifier())
    by_key = composite.verify(api_key)
    assert by_key is not None
    assert by_key.subject is None
    by_token = composite.verify(_token())
    assert by_token is not None
    assert by_token.subject == "alice"
    assert composite.verify("nope") is None


# --- PII redaction in logs ---------------------------------------------------------------------

RECORD = {
    "record_id": "s1-0000001",
    "first_name": "Brandor",
    "last_name": "Keldmar",
    "dob": "1984-07-19",
    "sex": "M",
    "street": "12 Vantolio Street",
    "city": "Northfield",
    "postcode": "40917",
}


@pytest.mark.usefixtures("restore_logging")
def test_structured_record_fields_are_masked_in_logs() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    log = get_logger("fixture")
    log.info("record_loaded", **RECORD)
    log.info("pair_reviewed", left_fields=RECORD, right_fields=dict(RECORD, first_name="Brandy"))
    first, second = (json.loads(line) for line in stream.getvalue().splitlines())
    for key in ("first_name", "last_name", "dob", "street", "city", "postcode"):
        assert first[key] == REDACTED
    assert first["record_id"] == "s1-0000001"
    assert second["left_fields"] == REDACTED
    assert second["right_fields"] == REDACTED
    output = stream.getvalue()
    for leaked in ("Brandor", "Keldmar", "Brandy", "1984-07-19", "Vantolio", "Northfield"):
        assert leaked not in output


@pytest.mark.usefixtures("restore_logging")
def test_dates_and_street_addresses_are_masked_in_free_text() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    get_logger().warning("lookup failed for 1984-07-19 or 19.07.1984 living at 12 Vantolio Street")
    output = stream.getvalue()
    for leaked in ("1984-07-19", "19.07.1984", "Vantolio"):
        assert leaked not in output
    record = json.loads(output)
    assert "[REDACTED:DATE]" in record["event"]
    assert "[REDACTED:ADDRESS]" in record["event"]
    assert "T" in record["timestamp"]


@pytest.mark.parametrize(
    ("key", "sensitive"),
    [
        ("first_name", True), ("lastName", True), ("left_first_name", True), ("dob", True),
        ("date_of_birth", True), ("street", True), ("postcode", True), ("city", True),
        ("fields", True), ("right_fields", True), ("service_name", False), ("model_name", False),
        ("review_capacity", False), ("record_id", False), ("match_threshold", False),
        ("source", False),
    ],
)  # fmt: skip
def test_record_field_keys_are_recognized(key: str, sensitive: bool) -> None:
    assert is_sensitive_key(key) is sensitive


@pytest.mark.parametrize(
    "text", ["at 2026-09-13T10:00:00Z", "version 1.2.3", "order 12345 shipped", "took 12 ms"]
)
def test_benign_text_survives(text: str) -> None:
    assert redact_text(text) == text


@given(st.text(alphabet="0123456789-./ abcdStreetRdAve", max_size=60))
def test_date_and_address_patterns_keep_redaction_idempotent(text: str) -> None:
    once = redact_text(text)
    assert redact_text(once) == once


# --- source data contract ----------------------------------------------------------------------

GOOD: dict[str, object] = {
    "record_id": "s1-0000001",
    "first_name": "Anna",
    "last_name": "Brandor",
    "dob": date(1980, 1, 2),
    "sex": "F",
    "street": "1 Main St",
    "city": "Northfield",
    "postcode": "12345",
}


def _failures(frame: pl.DataFrame, **kwargs: Any) -> DataContractError:
    with pytest.raises(DataContractError) as info:
        validate_source(frame, **kwargs)
    return info.value


def test_generated_sources_satisfy_the_contract(small_dataset: SyntheticDataset) -> None:
    for frame in small_dataset.sources.values():
        assert validate_source(frame).height == frame.height


def test_leaked_entity_ids_are_rejected(small_dataset: SyntheticDataset) -> None:
    frame = small_dataset.sources["s1"].with_columns(pl.lit(1).alias("entity_id"))
    assert _failures(frame).failures


def test_field_formats_lengths_and_uniqueness_are_enforced_without_echoing_values() -> None:
    bad = {"first_name": "x" * 65, "postcode": "12a45", "sex": "X", "dob": date(1850, 1, 1)}
    rows = [GOOD, {**GOOD, "record_id": "s1-0000002", **bad}, {**GOOD, "record_id": "S1_bad"}, GOOD]
    error = _failures(pl.DataFrame(rows, schema=SOURCE_SCHEMA))
    columns = {item["column"] for item in error.failures}
    assert {"first_name", "postcode", "sex", "dob", "record_id"} <= columns
    for leaked in ("x" * 65, "12a45", "1850"):
        assert leaked not in str(error)
        assert leaked not in repr(error.failures)


def test_wrong_types_missing_columns_and_row_limits_are_rejected() -> None:
    frame = pl.DataFrame([GOOD], schema=SOURCE_SCHEMA)
    assert _failures(frame.with_columns(pl.col("dob").cast(pl.Utf8))).failures
    assert _failures(frame.drop("city")).failures
    two = pl.DataFrame([GOOD, {**GOOD, "record_id": "s1-0000002"}], schema=SOURCE_SCHEMA)
    assert _failures(two, max_rows=1).failures == [
        {"column": "*", "check": "max_rows(1)", "count": 1}
    ]
