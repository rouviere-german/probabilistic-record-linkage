"""Model artifacts: m/u tables, thresholds and configuration as JSON in the registry.

A registered version holds one file, ``model.json`` (canonical JSON, never pickle), plus a run
manifest with everything needed to reproduce it: the generator seed and corruption-config hash,
content hashes of the input sources, the pipeline-config hash, every EM initialization of every
training session, the ``uv.lock`` hash and the deterministic run hash. The registry verifies the
SHA-256 of the file and of the version record on every load, so a tampered artifact is refused.
Term-frequency tables are deliberately *not* stored: they are counts of names, and scoring
recomputes them from the batch being linked.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from probabilistic_record_linkage.core.canonical import canonical_json
from probabilistic_record_linkage.linkage.comparison import PLAIN_SPACE, ComparisonSpace
from probabilistic_record_linkage.linkage.fellegi_sunter import FSParams
from probabilistic_record_linkage.linkage.monitoring import PatternDriftMonitor
from probabilistic_record_linkage.linkage.pipeline import FittedModel, LinkageRun, PipelineConfig
from probabilistic_record_linkage.linkage.privacy import PRIVATE_SPACE, PrivacyConfig
from probabilistic_record_linkage.linkage.thresholds import Thresholds
from probabilistic_record_linkage.ml.manifest import create_manifest, sha256_file
from probabilistic_record_linkage.ml.registry import ModelRegistry, ModelVersion

__all__ = [
    "LINKAGE_PACKAGES",
    "LinkageModel",
    "ModelFormatError",
    "load_model",
    "model_from_run",
    "register_run",
]

LINKAGE_PACKAGES = ("duckdb", "numpy", "pandera", "polars", "pydantic", "rapidfuzz", "scipy")
MODEL_FILE = "model.json"


class ModelFormatError(ValueError):
    """The artifact does not describe a comparison space this code can score."""


class LinkageModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    mode: Literal["plaintext", "privacy"]
    fields: list[dict[str, Any]]
    params: dict[str, Any]
    em_strategy: str
    em_sessions: list[dict[str, Any]]
    thresholds: Thresholds
    pipeline: PipelineConfig
    privacy: PrivacyConfig | None = None
    drift_reference: dict[str, Any]

    def space(self) -> ComparisonSpace:
        space = PLAIN_SPACE if self.mode == "plaintext" else PRIVATE_SPACE
        if space.describe() != self.fields:
            raise ModelFormatError("the artifact's comparison levels do not match this code")
        return space

    def fitted(self) -> FittedModel:
        space = self.space()
        return FittedModel(
            params=FSParams.from_json(self.params, space),
            strategy=self.em_strategy,
            sessions=tuple(self.em_sessions),
        )

    def drift_monitor(self) -> PatternDriftMonitor:
        return PatternDriftMonitor.from_json(self.space(), self.drift_reference)

    def to_bytes(self) -> bytes:
        return (canonical_json(self.model_dump(mode="json")) + "\n").encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> LinkageModel:
        return cls.model_validate_json(data)


def model_from_run(run: LinkageRun, privacy: PrivacyConfig | None = None) -> LinkageModel:
    if (run.mode == "privacy") != (privacy is not None):
        raise ModelFormatError("privacy configuration is required exactly for privacy-mode runs")
    return LinkageModel(
        mode="plaintext" if run.mode == "plaintext" else "privacy",
        fields=run.comparator.space.describe(),
        params=run.model.params.to_json(),
        em_strategy=run.model.strategy,
        em_sessions=[dict(session) for session in run.model.sessions],
        thresholds=run.thresholds,
        pipeline=run.config,
        privacy=privacy,
        drift_reference=run.drift.to_json(),
    )


def register_run(
    registry: ModelRegistry,
    run: LinkageRun,
    *,
    name: str = "person-linkage",
    generator_manifest: Mapping[str, JsonValue] | None = None,
    lockfile: Path | None = None,
    datasets: Sequence[Path] = (),
    privacy: PrivacyConfig | None = None,
    clock: Callable[[], datetime] | None = None,
) -> ModelVersion:
    """Register ``run``'s model with a reproducibility manifest."""
    model = model_from_run(run, privacy)
    summary = run.summary()
    metrics = {
        key: float(value)
        for key, value in summary.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
    params: dict[str, JsonValue] = {
        "run_hash": run.run_hash(),
        "pipeline_sha256": run.config.sha256(),
        "input_sha256": dict(sorted(run.input_sha256.items())),
        "generator": dict(generator_manifest or {}),
        "lockfile_sha256": sha256_file(lockfile)
        if lockfile is not None and lockfile.is_file()
        else None,
        "em_initializations": [dict(session) for session in run.model.sessions],
        "em_strategy": run.model.strategy,
    }
    kwargs: dict[str, Any] = {"clock": clock} if clock is not None else {}
    manifest = create_manifest(
        seed=run.config.seed,
        datasets=list(datasets),
        params=params,
        metrics=metrics,
        packages=LINKAGE_PACKAGES,
        run_id=run.run_hash()[:32],
        **kwargs,
    )
    return registry.register(name, {MODEL_FILE: model.to_bytes()}, manifest, metrics=metrics)


def load_model(
    registry: ModelRegistry, name: str, ref: str = "production"
) -> tuple[LinkageModel, ModelVersion]:
    """Load and verify a registered model (blob and record hashes are checked by the registry)."""
    version = registry.get(name, ref)
    files = registry.load_files(version)
    if MODEL_FILE not in files:
        raise ModelFormatError("the version has no model.json")
    return LinkageModel.from_bytes(files[MODEL_FILE]), version
