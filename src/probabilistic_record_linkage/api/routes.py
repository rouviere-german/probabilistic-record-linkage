"""Linkage routes. Every route requires a JWT with the named scope.

| Route | Scope |
|---|---|
| ``POST /v1/link-jobs`` | ``link:run`` |
| ``GET /v1/link-jobs/{job_id}`` | ``link:run`` |
| ``GET /v1/link-jobs/{job_id}/entities/{record_id}`` | ``entity:read`` (+ ``pii:read``) |
| ``GET /v1/link-jobs/{job_id}/review-items`` | ``review:write`` (+ ``pii:read``) |

| ``POST /v1/link-jobs/{job_id}/review-items/{item_id}/decision`` | ``review:write`` |

With ``pii:read`` the responses include the record fields; without it, only ids and probabilities.
Every ``{job_id}`` route is scoped to the job's owner (the token's ``tenant`` claim, or its
subject when there is none): other callers get the same 404 as for an unknown id, so a leaked
job id reveals nothing.

Uploads are limited three ways: the request body size (middleware), the row count (a static
schema maximum plus ``APP_UPLOAD_MAX_ROWS``) and per-field lengths and formats (the pydantic
model here, then the pandera source contract). Contract violations are reported as counts, never
with the offending values.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal

import polars as pl
from fastapi import APIRouter, Depends, Query
from fastapi import Path as PathParam
from pydantic import BaseModel, ConfigDict, Field

from probabilistic_record_linkage.core.config import Settings
from probabilistic_record_linkage.core.errors import AppError
from probabilistic_record_linkage.core.security import Principal, require_scopes
from probabilistic_record_linkage.linkage.generator import SOURCE_SCHEMA
from probabilistic_record_linkage.linkage.service import (
    JobNotFoundError,
    LinkageService,
    TooManyJobsError,
)
from probabilistic_record_linkage.linkage.store import ReviewConflictError
from probabilistic_record_linkage.ml.contracts import (
    MAX_NAME_CHARS,
    MAX_STREET_CHARS,
    RECORD_ID_PATTERN,
    DataContractError,
)

__all__ = ["HARD_MAX_ROWS", "DecisionRequest", "LinkJobRequest", "UploadRecord", "build_router"]

HARD_MAX_ROWS = 100_000
JOB_ID = r"^[0-9a-f]{32}$"
ITEM_ID = r"^[0-9a-f]{24}$"
SOURCE = r"^[a-z][a-z0-9]{0,7}$"


class UploadRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(pattern=SOURCE)
    record_id: str = Field(max_length=21, pattern=RECORD_ID_PATTERN)
    first_name: str | None = Field(default=None, min_length=1, max_length=MAX_NAME_CHARS)
    last_name: str | None = Field(default=None, min_length=1, max_length=MAX_NAME_CHARS)
    dob: date | None = None
    sex: Literal["F", "M"] | None = None
    street: str | None = Field(default=None, min_length=1, max_length=MAX_STREET_CHARS)
    city: str | None = Field(default=None, min_length=1, max_length=MAX_NAME_CHARS)
    postcode: str | None = Field(default=None, pattern=r"^[0-9]{5}$")


class LinkJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[UploadRecord] = Field(min_length=2, max_length=HARD_MAX_ROWS)
    use_registered_model: bool = False


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["match", "non_match"]


JobId = Annotated[str, PathParam(pattern=JOB_ID)]
RecordId = Annotated[str, PathParam(max_length=21, pattern=RECORD_ID_PATTERN)]
ItemId = Annotated[str, PathParam(pattern=ITEM_ID)]
# Module level on purpose: with postponed annotations FastAPI resolves these names through the
# module globals, so aliases defined inside build_router would silently become query parameters.
LinkRun = Annotated[Principal, Depends(require_scopes("link:run"))]
EntityRead = Annotated[Principal, Depends(require_scopes("entity:read"))]
ReviewWrite = Annotated[Principal, Depends(require_scopes("review:write"))]


def build_router(service: LinkageService, settings: Settings) -> APIRouter:
    router = APIRouter(prefix="/v1")

    @router.post("/link-jobs", status_code=202)
    def create_link_job(body: LinkJobRequest, principal: LinkRun) -> dict[str, str]:
        if len(body.records) > settings.upload_max_rows:
            raise AppError(413, "too_many_rows", "The upload exceeds the row limit.")
        if len({record.record_id for record in body.records}) != len(body.records):
            raise AppError(422, "duplicate_record_ids", "Record ids must be unique.")
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in body.records:
            grouped.setdefault(record.source, []).append(record.model_dump(exclude={"source"}))
        sources = {name: pl.DataFrame(rows, schema=SOURCE_SCHEMA) for name, rows in grouped.items()}
        try:
            job_id = service.submit(
                principal.key_id,
                sources,
                owner=principal.owner,
                use_registered_model=body.use_registered_model,
            )
        except TooManyJobsError:
            raise AppError(
                429,
                "too_many_jobs",
                "A job concurrency limit was reached.",
                headers={"Retry-After": "5"},
            ) from None
        except DataContractError as exc:
            raise AppError(
                422, "data_contract_violation", f"{len(exc.failures)} contract check(s) failed."
            ) from None
        return {"job_id": job_id, "status": "running"}

    @router.get("/link-jobs/{job_id}")
    def get_link_job(job_id: JobId, principal: LinkRun) -> dict[str, Any]:
        try:
            return service.job(job_id, owner=principal.owner)
        except JobNotFoundError:
            raise AppError(404, "not_found", "Unknown job.") from None

    @router.get("/link-jobs/{job_id}/entities/{record_id}")
    def get_entity(job_id: JobId, record_id: RecordId, principal: EntityRead) -> dict[str, Any]:
        try:
            return service.entity(
                job_id,
                record_id,
                include_pii="pii:read" in principal.scopes,
                owner=principal.owner,
            )
        except JobNotFoundError:
            raise AppError(404, "not_found", "Unknown job or record.") from None

    @router.get("/link-jobs/{job_id}/review-items")
    def list_review_items(
        job_id: JobId,
        principal: ReviewWrite,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> dict[str, Any]:
        try:
            items = service.review_queue(
                job_id,
                include_pii="pii:read" in principal.scopes,
                limit=limit,
                owner=principal.owner,
            )
        except JobNotFoundError:
            raise AppError(404, "not_found", "Unknown job.") from None
        return {"job_id": job_id, "items": items}

    @router.post("/link-jobs/{job_id}/review-items/{item_id}/decision")
    def decide_review_item(
        job_id: JobId, item_id: ItemId, body: DecisionRequest, principal: ReviewWrite
    ) -> dict[str, Any]:
        try:
            return service.decide(
                job_id,
                item_id,
                decision=body.decision,
                reviewer=principal.subject or principal.key_id,
                owner=principal.owner,
            )
        except ReviewConflictError:
            raise AppError(
                409, "review_conflict", "The review item is not open for this job."
            ) from None
        except JobNotFoundError:
            raise AppError(404, "not_found", "Unknown job.") from None

    return router
