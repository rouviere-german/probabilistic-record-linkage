from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import generate_latest

from probabilistic_record_linkage.core.tracing import build_tracer_provider
from probabilistic_record_linkage.linkage.analytic import analytic_exact_agreement
from probabilistic_record_linkage.linkage.blocking import pair_completeness
from probabilistic_record_linkage.linkage.clustering import ClusterConfig
from probabilistic_record_linkage.linkage.demo import entity_pairs
from probabilistic_record_linkage.linkage.evaluation import (
    max_entities_per_cluster,
    merge_rate,
    pairwise_scores,
)
from probabilistic_record_linkage.linkage.generator import (
    GeneratorConfig,
    SyntheticDataset,
    entity_labels,
    generate,
    household_chains,
    make_bridge_records,
)
from probabilistic_record_linkage.linkage.model import (
    LinkageModel,
    ModelFormatError,
    load_model,
    model_from_run,
    register_run,
)
from probabilistic_record_linkage.linkage.monitoring import create_linkage_metrics
from probabilistic_record_linkage.linkage.normalize import normalize_sources
from probabilistic_record_linkage.linkage.pipeline import (
    EMConfig,
    LinkageRun,
    PipelineConfig,
    link_encoded,
    link_sources,
)
from probabilistic_record_linkage.linkage.privacy import PrivacyConfig, encode_records
from probabilistic_record_linkage.linkage.thresholds import BAND_MATCH, band_of
from probabilistic_record_linkage.linkage.vocab import World
from probabilistic_record_linkage.ml.registry import ModelRegistry, RegistryIntegrityError

pytestmark = pytest.mark.integration

STAGES = ("validate", "normalize", "block", "compare", "fit_em", "score", "threshold", "cluster")


@pytest.fixture(scope="module")
def linked(small_dataset: SyntheticDataset) -> LinkageRun:
    return link_sources(small_dataset.sources, nicknames=small_dataset.world.nicknames)


def test_linkage_matches_the_ground_truth(
    small_dataset: SyntheticDataset, linked: LinkageRun
) -> None:
    truth = entity_labels(small_dataset.truth, list(linked.record_ids))
    assert pairwise_scores(linked.resolution.labels, truth).f1 >= 0.93
    assert pair_completeness(linked.blocking.left, linked.blocking.right, truth) >= 0.97
    analytic = analytic_exact_agreement(small_dataset)
    for f, spec in enumerate(linked.comparator.space.fields):
        fitted = float(linked.model.params.m[f][spec.exact_level])
        assert abs(fitted - analytic[spec.name]) <= 0.05, spec.name
    summary = linked.summary()
    assert summary["records"] == small_dataset.n_records
    assert set(summary["timings_seconds"]) == set(STAGES)  # type: ignore[arg-type]
    prefixes = {candidate.reason.split("_")[0] for candidate in linked.review_candidates()}
    assert prefixes <= {"review", "rule", "guard", "pending"}


def test_run_hash_is_deterministic_and_sensitive_to_inputs_and_config(
    small_dataset: SyntheticDataset, linked: LinkageRun
) -> None:
    nicknames = small_dataset.world.nicknames
    assert link_sources(small_dataset.sources, nicknames=nicknames).run_hash() == linked.run_hash()
    other_seed = link_sources(
        small_dataset.sources, nicknames=nicknames, config=PipelineConfig(seed=8)
    )
    assert other_seed.run_hash() != linked.run_hash()
    fewer = dict(small_dataset.sources)
    fewer["s1"] = fewer["s1"].head(fewer["s1"].height - 1)
    assert link_sources(fewer, nicknames=nicknames).run_hash() != linked.run_hash()


def test_every_stage_emits_a_span_without_personal_data(small_dataset: SyntheticDataset) -> None:
    exporter = InMemorySpanExporter()
    provider = build_tracer_provider("test", extra_exporter=exporter)
    link_sources(
        small_dataset.sources,
        nicknames=small_dataset.world.nicknames,
        tracer=provider.get_tracer("t"),
    )
    spans = exporter.get_finished_spans()
    assert {span.name for span in spans} == {f"linkage.{stage}" for stage in STAGES}
    names = set(small_dataset.sources["s1"]["first_name"].drop_nulls().head(50).to_list())
    for span in spans:
        for value in (span.attributes or {}).values():
            assert str(value) not in names


def test_linkage_metrics_are_published_without_record_ids(small_dataset: SyntheticDataset) -> None:
    metrics = create_linkage_metrics()
    link_sources(small_dataset.sources, nicknames=small_dataset.world.nicknames, metrics=metrics)
    payload = generate_latest(metrics.registry).decode()
    assert 'linkage_stage_duration_seconds_count{stage="cluster"} 1.0' in payload
    for name in (
        "blocking_pairs_total",
        "reduction_ratio",
        "match_rate",
        "max_cluster_size",
        "em_iterations",
    ):
        assert name in payload
    for record_id in small_dataset.sources["s1"]["record_id"].head(20).to_list():
        assert record_id not in payload


def test_reviewer_constraints_change_the_resolution(linked: LinkageRun) -> None:
    before = linked.resolution.labels.copy()
    rows = np.flatnonzero(linked.bands() == BAND_MATCH)
    a, b = int(linked.scored.left[rows[0]]), int(linked.scored.right[rows[0]])
    assert before[a] == before[b]
    separated = linked.resolve(cannot_link=[(a, b)])
    assert separated.labels[a] != separated.labels[b] or separated.pending[a]
    far = linked.n_records - 1
    joined = linked.resolve(must_link=[(0, far)])
    assert joined.labels[0] == joined.labels[far]
    linked.resolve()
    np.testing.assert_array_equal(linked.resolution.labels, before)


def test_scoring_with_an_existing_model_reproduces_the_clusters(
    small_dataset: SyntheticDataset, linked: LinkageRun
) -> None:
    rescored = link_sources(
        small_dataset.sources,
        nicknames=small_dataset.world.nicknames,
        fitted=linked.model,
        thresholds=linked.thresholds,
    )
    assert rescored.model is linked.model
    np.testing.assert_array_equal(rescored.resolution.labels, linked.resolution.labels)


def test_the_candidate_em_strategy_is_selectable(small_dataset: SyntheticDataset) -> None:
    config = PipelineConfig(em=EMConfig(strategy="candidates"))
    run = link_sources(
        small_dataset.sources, nicknames=small_dataset.world.nicknames, config=config
    )
    assert run.model.strategy == "candidates"
    assert run.model.sessions[0]["name"] == "candidates"


def test_link_inputs_are_validated(small_dataset: SyntheticDataset) -> None:
    source = small_dataset.sources["s1"]
    with pytest.raises(ValueError, match="unique"):
        link_sources({"a": source, "b": source})
    with pytest.raises(ValueError, match="at least two"):
        link_sources({"s1": source.head(1)})


def test_twin_pairs_are_routed_to_review_and_rarely_merged(world: World) -> None:
    dataset = generate(GeneratorConfig(seed=world.seed, n_records=20_000), world)
    run = link_sources(dataset.sources, nicknames=world.nicknames)
    truth = entity_labels(dataset.truth, list(run.record_ids))
    twins, _ = entity_pairs(dataset)
    assert len(twins) >= 40
    with_rules = merge_rate(run.resolution.labels, truth, twins)[0]
    f1_with_rules = pairwise_scores(run.resolution.labels, truth).f1
    run.band = band_of(run.scored.probability, run.thresholds)  # probability bands only
    without_rules = merge_rate(run.resolve().labels, truth, twins)[0]
    assert run.rule_hits["possible_twin"] > 0
    assert without_rules >= 0.5  # the model alone merges most twins
    assert with_rules <= 0.25
    assert with_rules <= without_rules / 3
    assert f1_with_rules >= 0.98


def test_the_guard_limits_adversarial_bridge_chains_to_two_entities(world: World) -> None:
    dataset = generate(GeneratorConfig(seed=world.seed, n_records=12_000), world)
    base = link_sources(dataset.sources, nicknames=world.nicknames)
    truth = entity_labels(dataset.truth, list(base.record_ids))
    clusters_of: dict[int, set[int]] = {}
    for label, entity in zip(base.resolution.labels.tolist(), truth.tolist(), strict=True):
        clusters_of.setdefault(entity, set()).add(label)
    chains = []
    for chain in household_chains(dataset, min_members=3, min_records=2):
        clusters = [clusters_of.get(entity, set()) for entity in chain]
        if all(len(c) == 1 for c in clusters) and len(set().union(*clusters)) == len(chain):
            chains.append(chain)
    assert chains
    frame, bridge_truth = make_bridge_records(dataset, chains[:15])
    all_truth = pl.concat([dataset.truth, bridge_truth])
    worst_committed: dict[bool, int] = {}
    for guard in (True, False):
        run = link_sources(
            {**dataset.sources, "adv": frame},
            nicknames=world.nicknames,
            config=PipelineConfig(clustering=ClusterConfig(guard=guard)),
        )
        labels = entity_labels(all_truth, list(run.record_ids))
        origin = np.asarray(run.record_sources)
        real = origin != "adv"
        touched = np.unique(run.resolution.labels[~real]).tolist()
        # Clusters the guard could not split safely are held for review, never auto-committed.
        committed = [c for c in touched if run.resolution.status_of(int(c)) == "committed"]
        worst_committed[guard] = max_entities_per_cluster(
            run.resolution.labels[real], labels[real], clusters=committed
        )
    assert worst_committed[True] <= 2
    assert worst_committed[False] > 2


def test_privacy_mode_links_encodings(small_dataset: SyntheticDataset) -> None:
    normalized = normalize_sources(small_dataset.sources)
    encoded = encode_records(normalized, b"p" * 32, PrivacyConfig())
    sources = normalized["source"].to_list()
    run = link_encoded(encoded, sources)
    truth = entity_labels(small_dataset.truth, list(run.record_ids))
    assert run.mode == "privacy"
    assert pairwise_scores(run.resolution.labels, truth).f1 >= 0.85
    assert set(run.input_sha256) == {"encodings"}
    rescored = link_encoded(encoded, sources, fitted=run.model, thresholds=run.thresholds)
    np.testing.assert_array_equal(rescored.resolution.labels, run.resolution.labels)
    with pytest.raises(ValueError, match="one entry per"):
        link_encoded(encoded, sources[:1])


def test_models_round_trip_through_the_registry_with_a_manifest(
    tmp_path: Path, linked: LinkageRun, repo_root: Path
) -> None:
    registry = ModelRegistry(tmp_path)
    version = register_run(
        registry,
        linked,
        generator_manifest={"generator_sha256": "g" * 64},
        lockfile=repo_root / "uv.lock",
        clock=lambda: datetime(2026, 9, 13, tzinfo=UTC),
    )
    params = version.manifest.params
    for key in (
        "run_hash",
        "pipeline_sha256",
        "input_sha256",
        "generator",
        "lockfile_sha256",
        "em_initializations",
    ):
        assert key in params
    assert params["run_hash"] == linked.run_hash()
    assert isinstance(params["lockfile_sha256"], str)
    registry.set_alias("person-linkage", "production", version.digest)
    model, loaded = load_model(registry, "person-linkage")
    assert loaded.digest == version.digest
    fitted = model.fitted()
    assert fitted.params.lam == linked.model.params.lam
    np.testing.assert_allclose(fitted.params.m[0], linked.model.params.m[0])
    assert model.thresholds == linked.thresholds
    assert model.drift_monitor().check(linked.drift_patterns()).psi == pytest.approx(0.0, abs=1e-9)
    blob = version.files["model.json"]
    (tmp_path / "blobs" / "sha256" / blob[:2] / blob).write_bytes(b"{}")
    with pytest.raises(RegistryIntegrityError):
        load_model(registry, "person-linkage")


def test_model_artifacts_refuse_mismatched_spaces(linked: LinkageRun) -> None:
    model = model_from_run(linked)
    assert LinkageModel.from_bytes(model.to_bytes()) == model
    tampered = LinkageModel.model_validate({**model.model_dump(), "fields": model.fields[:-1]})
    with pytest.raises(ModelFormatError):
        tampered.space()
    with pytest.raises(ModelFormatError):
        model_from_run(linked, PrivacyConfig())
