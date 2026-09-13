"""End-to-end linkage: validate, normalize, block, compare, fit, score, threshold, cluster (M6).

Every stage runs inside an OpenTelemetry span ``linkage.<stage>`` whose attributes are counts and
never personal data, and its duration is recorded in ``LinkageRun.timings`` and, when a metrics
object is supplied, in ``linkage_stage_duration_seconds``.

Model fitting (see docs/adr/0002-u-from-random-pairs-and-training-blocks.md):

- ``u`` comes from random record pairs, with exact-agreement levels computed exactly from value
  counts;
- ``m`` is fitted by EM over comparison-pattern counts on *training blocks* (pairs sharing a date
  of birth, and pairs sharing a first and last name), with the block's own key fields masked and
  ``u`` held fixed. Inside such a block, non-matches are random with respect to the other fields,
  which keeps the conditional-independence assumption close to true. Estimates from the blocks
  are averaged, weighted by expected matches;
- the prior ``lambda`` of a random record pair comes from a one-parameter EM over all pairs
  (:func:`fellegi_sunter.estimate_prior`).

``EMConfig(strategy="candidates")`` instead fits ``m``, ``u`` and ``lambda`` by EM on the candidate
pairs themselves. It is kept for comparison: blocking makes non-matching candidates agree on the
blocked fields far more often than random pairs, and the benchmark shows the resulting bias.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from opentelemetry import trace
from opentelemetry.trace import Span, Tracer
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from probabilistic_record_linkage.core.canonical import canonical_json, sha256_hex
from probabilistic_record_linkage.linkage.blocking import (
    BlockingConfig,
    BlockingResult,
    bucket_pairs,
    reduction_ratio,
    run_blocking,
)
from probabilistic_record_linkage.linkage.clustering import (
    ClusterConfig,
    Resolution,
    resolve_clusters,
)
from probabilistic_record_linkage.linkage.comparison import Comparator, PlainComparator
from probabilistic_record_linkage.linkage.fellegi_sunter import (
    FSParams,
    combine_session_m,
    default_inits,
    estimate_prior,
    fit_em_multistart,
    log_bayes_factor_table,
    mask_fields,
    match_probability,
    pattern_table,
)
from probabilistic_record_linkage.linkage.monitoring import LinkageMetrics, PatternDriftMonitor
from probabilistic_record_linkage.linkage.nicknames import NicknameTable
from probabilistic_record_linkage.linkage.normalize import (
    PreparedRecords,
    normalize_sources,
    prepare_records,
)
from probabilistic_record_linkage.linkage.privacy import (
    BloomComparator,
    EncodedRecords,
    token_blocking,
)
from probabilistic_record_linkage.linkage.thresholds import (
    BAND_MATCH,
    BAND_REVIEW,
    ThresholdConfig,
    Thresholds,
    assign_bands,
    select_thresholds,
)
from probabilistic_record_linkage.ml.contracts import validate_source

__all__ = [
    "EMConfig",
    "FittedModel",
    "LinkageRun",
    "PipelineConfig",
    "ReviewCandidate",
    "ScoredPairs",
    "TrainingBlock",
    "encoded_training_blocks",
    "frame_sha256",
    "link_encoded",
    "link_sources",
    "plaintext_training_blocks",
    "score_pairs",
    "train_model",
]

Int8Array = npt.NDArray[np.int8]
Int64Array = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


class EMConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: Literal["training_blocks", "candidates"] = "training_blocks"
    max_iterations: int = Field(default=500, ge=1, le=10_000)
    tolerance: float = Field(default=1e-7, gt=0.0)
    random_pairs: int = Field(default=300_000, ge=1_000)
    lambdas: tuple[float, ...] = (0.02, 0.2)
    exact_masses: tuple[float, ...] = (0.9, 0.6)
    training_max_bucket: int = Field(default=50, ge=2)
    training_window: int = Field(default=5, ge=1)


class PipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = 7
    blocking: BlockingConfig = Field(default_factory=BlockingConfig)
    em: EMConfig = Field(default_factory=EMConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    clustering: ClusterConfig = Field(default_factory=ClusterConfig)

    def sha256(self) -> str:
        return sha256_hex(canonical_json(self.model_dump(mode="json")))


@dataclass(frozen=True, slots=True)
class TrainingBlock:
    name: str
    left: Int64Array
    right: Int64Array
    masked: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FittedModel:
    params: FSParams
    strategy: str
    sessions: tuple[dict[str, JsonValue], ...]
    prior_iterations: int = 0

    @property
    def em_iterations(self) -> int:
        total = 0
        for session in self.sessions:
            value = session.get("iterations", 0)
            total += int(value) if isinstance(value, int) else 0
        return total


@dataclass(frozen=True, slots=True)
class ScoredPairs:
    left: Int64Array
    right: Int64Array
    levels: Int8Array
    patterns: Int64Array
    log_bayes_factor: FloatArray
    probability: FloatArray


@dataclass(frozen=True, slots=True)
class ReviewCandidate:
    left: int
    right: int
    probability: float
    reason: str


def plaintext_training_blocks(records: PreparedRecords, config: EMConfig) -> list[TrainingBlock]:
    dob = records.sort_key("dob")
    dob_left, dob_right = bucket_pairs(
        dob, records.year >= 0, max_bucket=config.training_max_bucket, window=config.training_window
    )
    names = (
        records.first_name.codes.astype(np.int64) * max(1, records.last_name.n_values)
        + records.last_name.codes
    )
    valid = (records.first_name.codes >= 0) & (records.last_name.codes >= 0)
    name_left, name_right = bucket_pairs(
        names, valid, max_bucket=config.training_max_bucket, window=config.training_window
    )
    return [
        TrainingBlock("same_dob", dob_left, dob_right, ("dob",)),
        TrainingBlock("same_name", name_left, name_right, ("first_name", "last_name")),
    ]


def encoded_training_blocks(encoded: EncodedRecords, config: EMConfig) -> list[TrainingBlock]:
    blocks: list[TrainingBlock] = []
    for name, token, masked in (
        ("same_dob_token", "dob", ("dob",)),
        ("same_name_token", "names", ("first_name", "last_name")),
    ):
        keys = encoded.tokens[token]
        left, right = bucket_pairs(
            keys, keys >= 0, max_bucket=config.training_max_bucket, window=config.training_window
        )
        blocks.append(TrainingBlock(name, left, right, masked))
    return blocks


def train_model(
    comparator: Comparator,
    *,
    candidate_left: Int64Array,
    candidate_right: Int64Array,
    candidate_levels: Int8Array,
    blocks: Sequence[TrainingBlock],
    config: EMConfig,
    rng: np.random.Generator,
) -> FittedModel:
    space = comparator.space
    u_random = comparator.random_u(config.random_pairs, rng)
    inits = default_inits(space, u_random, lambdas=config.lambdas, exact_masses=config.exact_masses)
    if config.strategy == "candidates":
        levels, counts = pattern_table(space.encode(candidate_levels), space)
        best, runs = fit_em_multistart(
            space,
            levels,
            counts,
            inits,
            max_iterations=config.max_iterations,
            tolerance=config.tolerance,
        )
        session: dict[str, JsonValue] = {
            "name": "candidates",
            "masked": [],
            "pairs": int(counts.sum()),
            "selected": best.label,
            "iterations": best.iterations,
            "runs": [run.summary() for run in runs],  # type: ignore[misc]
        }
        return FittedModel(best.params, "candidates", (session,))

    fitted: list[tuple[object, int, Sequence[str]]] = []
    sessions: list[dict[str, JsonValue]] = []
    for block in blocks:
        if block.left.size == 0:
            continue
        levels = mask_fields(space, comparator.levels(block.left, block.right), block.masked)
        distinct, counts = pattern_table(space.encode(levels), space)
        best, runs = fit_em_multistart(
            space,
            distinct,
            counts,
            inits,
            max_iterations=config.max_iterations,
            tolerance=config.tolerance,
            fix_u=True,
        )
        fitted.append((best, int(block.left.size), block.masked))
        sessions.append(
            {
                "name": block.name,
                "masked": list(block.masked),
                "pairs": int(block.left.size),
                "selected": best.label,
                "iterations": best.iterations,
                "session_lambda": round(best.params.lam, 8),
                "runs": [run.summary() for run in runs],  # type: ignore[misc]
            }
        )
    try:
        m = combine_session_m(space, fitted)  # type: ignore[arg-type]
    except ValueError:
        # Too few records for a training block (small uploads): fall back to EM on candidates.
        if candidate_levels.shape[0] == 0:
            raise
        return train_model(
            comparator,
            candidate_left=candidate_left,
            candidate_right=candidate_right,
            candidate_levels=candidate_levels,
            blocks=(),
            config=config.model_copy(update={"strategy": "candidates"}),
            rng=rng,
        )
    provisional = FSParams(0.5, m, u_random)
    table = log_bayes_factor_table(provisional, space)
    log_bf = table[space.encode(candidate_levels)] + comparator.tf_adjustment(
        candidate_left, candidate_right, candidate_levels
    )
    n = comparator.n_records
    lam, iterations = estimate_prior(log_bf, n * (n - 1) / 2.0)
    return FittedModel(FSParams(lam, m, u_random), "training_blocks", tuple(sessions), iterations)


def score_pairs(
    comparator: Comparator,
    left: Int64Array,
    right: Int64Array,
    params: FSParams,
    levels: Int8Array | None = None,
) -> ScoredPairs:
    space = comparator.space
    lv = comparator.levels(left, right) if levels is None else levels
    patterns = space.encode(lv)
    table = log_bayes_factor_table(params, space)
    log_bf = table[patterns] + comparator.tf_adjustment(left, right, lv)
    return ScoredPairs(left, right, lv, patterns, log_bf, match_probability(params, log_bf))


def frame_sha256(frame: pl.DataFrame) -> str:
    """Content hash of a frame (CSV serialization, so it is independent of Parquet metadata)."""
    return sha256_hex(frame.write_csv())


@dataclass(slots=True)
class LinkageRun:
    mode: str
    config: PipelineConfig
    record_ids: tuple[str, ...]
    record_sources: tuple[str, ...]
    comparator: Comparator
    blocking: BlockingResult
    scored: ScoredPairs
    model: FittedModel
    thresholds: Thresholds
    drift: PatternDriftMonitor
    input_sha256: dict[str, str]
    timings: dict[str, float]
    resolution: Resolution
    band: npt.NDArray[np.int8]
    rule_hits: dict[str, int]
    must_link: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    cannot_link: frozenset[tuple[int, int]] = field(default_factory=frozenset)

    @property
    def n_records(self) -> int:
        return len(self.record_ids)

    def row_of(self, record_id: str) -> int:
        try:
            return self.record_ids.index(record_id)
        except ValueError:
            raise KeyError("unknown record id") from None

    def oracle(self, left: Int64Array, right: Int64Array) -> FloatArray:
        """Match probability of arbitrary record pairs under this run's model."""
        return score_pairs(self.comparator, left, right, self.model.params).probability

    def bands(self) -> npt.NDArray[np.int8]:
        """0 non-match, 1 review (probability band or review rule), 2 match."""
        return self.band

    def drift_patterns(self, involving: str | None = None) -> Int64Array:
        """Comparison patterns of likely matches (review band and above).

        Corruption changes how matching records agree; non-matching candidate pairs, which are
        the vast majority, look the same under any corruption rate and would dilute the index.
        ``involving`` restricts the pairs to those with at least one record from that source.
        """
        keep = self.band >= BAND_REVIEW
        if involving is not None:
            origin = np.asarray(self.record_sources)
            keep &= (origin[self.scored.left] == involving) | (
                origin[self.scored.right] == involving
            )
        patterns: Int64Array = self.scored.patterns[keep]
        return patterns

    def resolve(
        self,
        must_link: Iterable[tuple[int, int]] = (),
        cannot_link: Iterable[tuple[int, int]] = (),
    ) -> Resolution:
        must = frozenset((min(a, b), max(a, b)) for a, b in must_link)
        cannot = frozenset((min(a, b), max(a, b)) for a, b in cannot_link)
        self.resolution = resolve_clusters(
            self.n_records,
            self.scored.left,
            self.scored.right,
            self.scored.probability,
            match_threshold=self.thresholds.match,
            config=self.config.clustering,
            must_link=must,
            cannot_link=cannot,
            conflict_oracle=self.oracle,
            edge_mask=self.band == BAND_MATCH,
        )
        self.must_link, self.cannot_link = must, cannot
        return self.resolution

    def review_candidates(self) -> list[ReviewCandidate]:
        """Review-band pairs, pairs cut by the guard, and one pair per pending cluster."""
        decided = self.must_link | self.cannot_link
        bands = self.bands()
        items: dict[tuple[int, int], ReviewCandidate] = {}
        review_rows = np.flatnonzero(bands == BAND_REVIEW)
        rule_reason = np.full(review_rows.size, "review_band", dtype=object)
        for rule in reversed(self.config.thresholds.review_rules):
            hit = rule.mask(self.comparator.space, self.scored.levels[review_rows])
            hit &= self.scored.probability[review_rows] >= self.thresholds.match
            rule_reason[hit] = f"rule_{rule.name}"
        for i, reason in zip(review_rows.tolist(), rule_reason.tolist(), strict=True):
            key = (int(self.scored.left[i]), int(self.scored.right[i]))
            if key not in decided:
                items[key] = ReviewCandidate(*key, float(self.scored.probability[i]), str(reason))
        for edge in self.resolution.removed_edges:
            key = (edge.left, edge.right)
            if key not in decided:
                items[key] = ReviewCandidate(
                    edge.left, edge.right, edge.probability, f"guard_{edge.reason}"
                )
        labels = self.resolution.labels
        match = bands == BAND_MATCH
        for cluster, reasons in sorted(self.resolution.reasons.items()):
            inside = (
                match
                & (labels[self.scored.left] == cluster)
                & (labels[self.scored.right] == cluster)
            )
            rows = np.flatnonzero(inside)
            if rows.size == 0:
                continue
            weakest = rows[int(np.argmin(self.scored.probability[rows]))]
            key = (int(self.scored.left[weakest]), int(self.scored.right[weakest]))
            if key not in decided:
                items[key] = ReviewCandidate(
                    *key, float(self.scored.probability[weakest]), "pending_" + "+".join(reasons)
                )
        return sorted(
            items.values(), key=lambda item: (item.reason, -item.probability, item.left, item.right)
        )

    def summary(self) -> dict[str, JsonValue]:
        bands = self.bands()
        labels = self.resolution.labels
        return {
            "mode": self.mode,
            "records": self.n_records,
            "candidate_pairs": self.blocking.n_pairs,
            "lsh_pairs": self.blocking.lsh_pairs,
            "sorted_neighbourhood_pairs": self.blocking.sn_pairs,
            "reduction_ratio": reduction_ratio(self.blocking.n_pairs, self.n_records),
            "em_strategy": self.model.strategy,
            "em_iterations": self.model.em_iterations,
            "prior_lambda": self.model.params.lam,
            "match_threshold": self.thresholds.match,
            "review_threshold": self.thresholds.review,
            "match_pairs": int(np.count_nonzero(bands == BAND_MATCH)),
            "review_pairs": int(np.count_nonzero(bands == BAND_REVIEW)),
            "review_rule_hits": dict(self.rule_hits),
            "clusters": int(np.unique(labels).size) if labels.size else 0,
            "max_cluster_size": self.resolution.max_cluster_size,
            "pending_records": int(self.resolution.pending.sum()),
            "guard_removed_edges": len(self.resolution.removed_edges),
            "timings_seconds": {k: round(v, 4) for k, v in self.timings.items()},
        }

    def run_hash(self) -> str:
        """Deterministic digest of inputs, configuration, fitted parameters and the clustering."""
        params = self.model.params.to_json()
        rounded = {
            "lambda": round(float(params["lambda"]), 10),  # type: ignore[arg-type]
            "m": [[round(x, 10) for x in row] for row in params["m"]],  # type: ignore[attr-defined]
            "u": [[round(x, 10) for x in row] for row in params["u"]],  # type: ignore[attr-defined]
        }
        payload = {
            "mode": self.mode,
            "config": self.config.sha256(),
            "inputs": dict(sorted(self.input_sha256.items())),
            "params": rounded,
            "thresholds": [round(self.thresholds.match, 10), round(self.thresholds.review, 10)],
            "pairs": sha256_hex(self.scored.left.tobytes() + self.scored.right.tobytes()),
            "labels": sha256_hex(self.resolution.labels.astype(np.int64).tobytes()),
            "pending": sha256_hex(self.resolution.pending.astype(np.uint8).tobytes()),
        }
        return sha256_hex(canonical_json(payload))


@contextmanager
def _stage(
    tracer: Tracer, name: str, timings: dict[str, float], metrics: LinkageMetrics | None
) -> Iterator[Span]:
    start = time.perf_counter()
    with tracer.start_as_current_span(f"linkage.{name}") as span:
        yield span
    elapsed = time.perf_counter() - start
    timings[name] = elapsed
    if metrics is not None:
        metrics.stage_duration_seconds.labels(name).observe(elapsed)


def _drift_reference(
    comparator: Comparator, scored: ScoredPairs, band: npt.NDArray[np.int8]
) -> PatternDriftMonitor:
    likely = scored.patterns[band >= BAND_REVIEW]
    if likely.size == 0:
        likely = scored.patterns if scored.patterns.size else np.zeros(1, dtype=np.int64)
    return PatternDriftMonitor(comparator.space, likely)


def _rows(record_ids: Sequence[str], pairs: Iterable[tuple[str, str]]) -> list[tuple[int, int]]:
    index = {record_id: row for row, record_id in enumerate(record_ids)}
    rows: list[tuple[int, int]] = []
    for a, b in pairs:
        if a not in index or b not in index:
            raise KeyError("constraint refers to an unknown record id")
        rows.append((index[a], index[b]))
    return rows


def _finish(
    *,
    mode: str,
    config: PipelineConfig,
    record_ids: tuple[str, ...],
    record_sources: tuple[str, ...],
    comparator: Comparator,
    blocking: BlockingResult,
    blocks: Sequence[TrainingBlock] | None,
    input_sha256: dict[str, str],
    timings: dict[str, float],
    tracer: Tracer,
    metrics: LinkageMetrics | None,
    fitted: FittedModel | None,
    thresholds: Thresholds | None,
    must_link: Iterable[tuple[str, str]],
    cannot_link: Iterable[tuple[str, str]],
) -> LinkageRun:
    rng = np.random.default_rng([config.seed, 11])
    with _stage(tracer, "compare", timings, metrics) as span:
        levels = comparator.levels(blocking.left, blocking.right)
        span.set_attribute("linkage.pairs", int(levels.shape[0]))
    with _stage(tracer, "fit_em", timings, metrics) as span:
        model = fitted or train_model(
            comparator,
            candidate_left=blocking.left,
            candidate_right=blocking.right,
            candidate_levels=levels,
            blocks=blocks or (),
            config=config.em,
            rng=rng,
        )
        span.set_attribute("linkage.em_iterations", model.em_iterations)
        span.set_attribute("linkage.em_strategy", model.strategy)
    with _stage(tracer, "score", timings, metrics):
        scored = score_pairs(comparator, blocking.left, blocking.right, model.params, levels)
    with _stage(tracer, "threshold", timings, metrics) as span:
        selected = thresholds or select_thresholds(scored.probability, config.thresholds)
        span.set_attribute("linkage.match_threshold", selected.match)
        span.set_attribute("linkage.review_threshold", selected.review)
        band, rule_hits = assign_bands(
            scored.probability,
            scored.levels,
            comparator.space,
            selected,
            config.thresholds.review_rules,
        )
    n = len(record_ids)
    placeholder = Resolution(np.arange(n, dtype=np.int64), np.zeros(n, dtype=bool))
    run = LinkageRun(
        mode=mode,
        config=config,
        record_ids=record_ids,
        record_sources=record_sources,
        comparator=comparator,
        blocking=blocking,
        scored=scored,
        model=model,
        thresholds=selected,
        drift=_drift_reference(comparator, scored, band),
        input_sha256=input_sha256,
        timings=timings,
        resolution=placeholder,
        band=band,
        rule_hits=rule_hits,
    )
    with _stage(tracer, "cluster", timings, metrics) as span:
        resolution = run.resolve(_rows(record_ids, must_link), _rows(record_ids, cannot_link))
        span.set_attribute("linkage.clusters", resolution.n_clusters)
        span.set_attribute("linkage.pending_records", int(resolution.pending.sum()))
    if metrics is not None:
        summary = run.summary()
        metrics.blocking_pairs_total.inc(blocking.n_pairs)
        metrics.reduction_ratio.set(float(summary["reduction_ratio"]))  # type: ignore[arg-type]
        metrics.match_rate.set(
            float(summary["match_pairs"]) / blocking.n_pairs if blocking.n_pairs else 0.0  # type: ignore[arg-type]
        )
        metrics.max_cluster_size.set(resolution.max_cluster_size)
        metrics.em_iterations.set(model.em_iterations)
    return run


def link_sources(
    sources: Mapping[str, pl.DataFrame],
    *,
    nicknames: NicknameTable | None = None,
    config: PipelineConfig | None = None,
    tracer: Tracer | None = None,
    metrics: LinkageMetrics | None = None,
    fitted: FittedModel | None = None,
    thresholds: Thresholds | None = None,
    must_link: Iterable[tuple[str, str]] = (),
    cannot_link: Iterable[tuple[str, str]] = (),
    validate: bool = True,
    max_rows: int = 2_000_000,
) -> LinkageRun:
    """Link plaintext sources. ``fitted``/``thresholds`` score with an existing model instead."""
    cfg = config or PipelineConfig()
    active_tracer = tracer or trace.get_tracer("probabilistic_record_linkage.pipeline")
    timings: dict[str, float] = {}
    with _stage(active_tracer, "validate", timings, metrics) as span:
        if validate:
            for name in sorted(sources):
                validate_source(sources[name], max_rows=max_rows)
        hashes = {name: frame_sha256(sources[name]) for name in sorted(sources)}
        span.set_attribute("linkage.sources", len(sources))
    with _stage(active_tracer, "normalize", timings, metrics) as span:
        normalized = normalize_sources(sources)
        records = prepare_records(normalized, nicknames)
        span.set_attribute("linkage.records", records.n)
    if records.n < 2:
        raise ValueError("linkage needs at least two records")
    if len(set(records.record_ids)) != records.n:
        raise ValueError("record ids must be unique across sources")
    comparator = PlainComparator(records)
    with _stage(active_tracer, "block", timings, metrics) as span:
        blocking = run_blocking(records, cfg.blocking)
        span.set_attribute("linkage.pairs", blocking.n_pairs)
    blocks = None if fitted is not None else plaintext_training_blocks(records, cfg.em)
    return _finish(
        mode="plaintext",
        config=cfg,
        record_ids=records.record_ids,
        record_sources=tuple(normalized["source"].to_list()),
        comparator=comparator,
        blocking=blocking,
        blocks=blocks,
        input_sha256=hashes,
        timings=timings,
        tracer=active_tracer,
        metrics=metrics,
        fitted=fitted,
        thresholds=thresholds,
        must_link=must_link,
        cannot_link=cannot_link,
    )


def link_encoded(
    encoded: EncodedRecords,
    record_sources: Sequence[str],
    *,
    config: PipelineConfig | None = None,
    tracer: Tracer | None = None,
    metrics: LinkageMetrics | None = None,
    fitted: FittedModel | None = None,
    thresholds: Thresholds | None = None,
) -> LinkageRun:
    """Link Bloom-filter encodings (privacy mode). The linkage unit never sees plaintext."""
    cfg = config or PipelineConfig()
    if len(record_sources) != encoded.n:
        raise ValueError("record_sources must have one entry per encoded record")
    if encoded.n < 2:
        raise ValueError("linkage needs at least two records")
    active_tracer = tracer or trace.get_tracer("probabilistic_record_linkage.pipeline")
    timings: dict[str, float] = {}
    comparator = BloomComparator(encoded)
    with _stage(active_tracer, "block", timings, metrics) as span:
        left, right = token_blocking(encoded)
        blocking = BlockingResult(left, right, 0, int(left.size))
        span.set_attribute("linkage.pairs", blocking.n_pairs)
    digest = sha256_hex(b"".join(encoded.fields[name].tobytes() for name in sorted(encoded.fields)))
    return _finish(
        mode="privacy",
        config=cfg,
        record_ids=encoded.record_ids,
        record_sources=tuple(record_sources),
        comparator=comparator,
        blocking=blocking,
        blocks=None if fitted is not None else encoded_training_blocks(encoded, cfg.em),
        input_sha256={"encodings": digest},
        timings=timings,
        tracer=active_tracer,
        metrics=metrics,
        fitted=fitted,
        thresholds=thresholds,
        must_link=(),
        cannot_link=(),
    )
