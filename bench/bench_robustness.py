"""Robustness, calibration, over-merge guard, drift and privacy-mode benchmark.

Usage:
  uv run python bench/bench_robustness.py --check
  uv run python bench/bench_robustness.py --records 20000 --privacy-records 50000

Sections:

- corruption sweep: pairwise F1 with every corruption rate scaled by 0.5, 1, 1.5 and 2;
- calibration: ECE of match probabilities with independent corruption versus correlated
  corruption (same marginal rates, errors concentrated in 10% "sloppy" records), over all
  candidate pairs and restricted to the uncertain band (0.01 <= p <= 0.99) and the match band,
  where the expected and the observed false-discovery rate are compared;
- over-merge guard: adversarial bridge chains between household members; the largest number of
  true identities in any cluster that contains a bridge record, with and without the guard, plus
  the guard's F1 cost on clean data;
- drift: comparison-pattern PSI of a new source with the same corruption versus 3x corruption;
- privacy mode: F1 of plaintext, plain Bloom-filter and hardened Bloom-filter linkage on the same
  records, and the frequency-alignment attack on the top 100 first-name encodings.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from probabilistic_record_linkage.linkage.attack import frequency_alignment_attack
from probabilistic_record_linkage.linkage.clustering import ClusterConfig
from probabilistic_record_linkage.linkage.corruption import CorruptionConfig
from probabilistic_record_linkage.linkage.evaluation import (
    band_calibration,
    expected_calibration_error,
    max_entities_per_cluster,
    pairwise_scores,
)
from probabilistic_record_linkage.linkage.generator import (
    GeneratorConfig,
    emit_source,
    entity_labels,
    generate,
    household_chains,
    make_bridge_records,
)
from probabilistic_record_linkage.linkage.normalize import normalize_sources
from probabilistic_record_linkage.linkage.pipeline import PipelineConfig, link_encoded, link_sources
from probabilistic_record_linkage.linkage.privacy import (
    HardeningConfig,
    PrivacyConfig,
    encode_records,
)
from probabilistic_record_linkage.linkage.vocab import build_world

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results"
THRESHOLDS = ROOT / "bench" / "robustness_thresholds.json"
KEY = b"benchmark-only-key-" + b"0" * 13


def _r(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def corruption_sweep(records: int, seed: int) -> list[dict[str, float]]:
    world = build_world(seed)
    rows = []
    for factor in (0.5, 1.0, 1.5, 2.0):
        config = GeneratorConfig(
            seed=seed, n_records=records, corruption=CorruptionConfig().scaled(factor)
        )
        dataset = generate(config, world)
        run = link_sources(dataset.sources, nicknames=world.nicknames)
        scores = pairwise_scores(
            run.resolution.labels, entity_labels(dataset.truth, list(run.record_ids))
        )
        rows.append(
            {
                "factor": factor,
                "precision": _r(scores.precision),
                "recall": _r(scores.recall),
                "f1": _r(scores.f1),
            }
        )
    return rows


def calibration(records: int, seed: int) -> dict[str, Any]:
    """ECE over all candidate pairs, and the two bands where calibration actually costs something.

    Over all candidate pairs the ECE is dominated by the clear non-matches, so it barely moves
    when conditional independence breaks. The ``uncertain`` band (0.01 <= p <= 0.99) and the
    ``match`` band (p >= the selected match threshold, where the gap between the expected and the
    observed false-discovery rate is what a user pays) are reported next to it.
    """
    world = build_world(seed)
    result: dict[str, Any] = {}
    for label, corruption in (
        ("independent", CorruptionConfig()),
        ("correlated", CorruptionConfig().correlated(0.1, 5.0)),
    ):
        dataset = generate(
            GeneratorConfig(seed=seed, n_records=records, corruption=corruption), world
        )
        run = link_sources(dataset.sources, nicknames=world.nicknames)
        truth = entity_labels(dataset.truth, list(run.record_ids))
        is_match = truth[run.blocking.left] == truth[run.blocking.right]
        probability = run.scored.probability
        result[f"{label}_ece"] = _r(expected_calibration_error(probability, is_match), 5)
        result[f"{label}_f1"] = _r(pairwise_scores(run.resolution.labels, truth).f1)
        result[f"{label}_uncertain"] = band_calibration(
            probability, is_match, low=0.01, high=0.99
        ).as_dict()
        result[f"{label}_match_band"] = band_calibration(
            probability, is_match, low=run.thresholds.match
        ).as_dict()
    return result


def over_merge_guard(records: int, seed: int) -> dict[str, float]:
    world = build_world(seed)
    dataset = generate(GeneratorConfig(seed=seed, n_records=records), world)
    base = link_sources(dataset.sources, nicknames=world.nicknames)
    truth = entity_labels(dataset.truth, list(base.record_ids))
    unguarded = link_sources(
        dataset.sources,
        nicknames=world.nicknames,
        config=PipelineConfig(clustering=ClusterConfig(guard=False)),
    )
    clusters_of: dict[int, set[int]] = {}
    for label, entity in zip(base.resolution.labels.tolist(), truth.tolist(), strict=True):
        clusters_of.setdefault(entity, set()).add(label)
    chains = [
        chain
        for chain in household_chains(dataset, min_members=3, min_records=2)
        if all(len(clusters_of.get(e, set())) == 1 for e in chain)
        and len(set().union(*(clusters_of[e] for e in chain))) == len(chain)
    ][:40]
    frame, bridge_truth = make_bridge_records(dataset, chains)
    all_truth = pl.concat([dataset.truth, bridge_truth])
    worst: dict[bool, int] = {}
    worst_real: dict[bool, int] = {}
    worst_committed: dict[bool, int] = {}
    pending_touched: dict[bool, int] = {}
    for guard in (True, False):
        run = link_sources(
            {**dataset.sources, "adv": frame},
            nicknames=world.nicknames,
            config=PipelineConfig(clustering=ClusterConfig(guard=guard)),
        )
        labels = entity_labels(all_truth, list(run.record_ids))
        origin = np.asarray(run.record_sources)
        real = origin != "adv"
        clusters = run.resolution.labels
        touched = np.unique(clusters[~real]).tolist()
        committed = [c for c in touched if run.resolution.status_of(int(c)) == "committed"]
        worst[guard] = max_entities_per_cluster(clusters, labels, clusters=touched)
        worst_real[guard] = max_entities_per_cluster(clusters[real], labels[real], clusters=touched)
        worst_committed[guard] = max_entities_per_cluster(
            clusters[real], labels[real], clusters=committed
        )
        pending_touched[guard] = len(touched) - len(committed)
    return {
        "chains": len(chains),
        "bridge_records": frame.height,
        "max_real_entities_committed_with_guard": worst_committed[True],
        "max_real_entities_committed_without_guard": worst_committed[False],
        "max_real_entities_including_pending_with_guard": worst_real[True],
        "touched_clusters_pending_review_with_guard": pending_touched[True],
        "max_identities_with_guard": worst[True],
        "max_identities_without_guard": worst[False],
        "clean_f1_with_guard": _r(pairwise_scores(base.resolution.labels, truth).f1),
        "clean_f1_without_guard": _r(pairwise_scores(unguarded.resolution.labels, truth).f1),
    }


def drift(records: int, seed: int) -> dict[str, Any]:
    world = build_world(seed)
    dataset = generate(GeneratorConfig(seed=seed, n_records=records), world)
    base = link_sources(dataset.sources, nicknames=world.nicknames)
    rng = np.random.default_rng([seed, 5])
    result: dict[str, Any] = {}
    for label, corruption in (
        ("same", dataset.config.corruption),
        ("shifted_3x", dataset.config.corruption.scaled(3.0)),
    ):
        entities = rng.choice(len(dataset.population), size=records // 5, replace=False).astype(
            np.int64
        )
        frame, _ = emit_source(
            dataset.world, dataset.population, entities, "new", corruption=corruption, rng=rng
        )
        run = link_sources(
            {**dataset.sources, "new": frame},
            nicknames=world.nicknames,
            config=base.config,
            fitted=base.model,
            thresholds=base.thresholds,
        )
        report = base.drift.check(run.drift_patterns(involving="new"))
        result[f"{label}_psi"] = _r(report.psi)
        result[f"{label}_drifted"] = report.drifted
    return result


def _salt_frequencies(config: PrivacyConfig, years: range) -> dict[str, float]:
    """The attacker's public distribution of salt values for this configuration."""
    values = {
        "birth_year": [str(year) for year in years],
        "birth_month": [f"{month:02d}" for month in range(1, 13)],
        "sex": ["F", "M"],
    }
    combos = list(itertools.product(*(values[part] for part in config.hardening.name_salt_parts)))
    return {"|".join(combo): 1.0 / len(combos) for combo in combos}


def privacy(records: int, seed: int) -> dict[str, Any]:
    world = build_world(seed)
    dataset = generate(GeneratorConfig(seed=seed, n_records=records), world)
    normalized = normalize_sources(dataset.sources)
    truth = entity_labels(dataset.truth, normalized["record_id"].to_list())
    plaintext = link_sources(dataset.sources, nicknames=world.nicknames)
    plaintext_f1 = pairwise_scores(
        plaintext.resolution.labels, entity_labels(dataset.truth, list(plaintext.record_ids))
    ).f1
    public = world.first_name_frequencies()
    years = range(dataset.config.birth_years[0], dataset.config.birth_years[1] + 1)
    result: dict[str, Any] = {"records": dataset.n_records, "plaintext_f1": _r(plaintext_f1)}
    year_month = PrivacyConfig(
        hardening=HardeningConfig(name_salt_parts=("birth_year", "birth_month"))
    )
    for label, config in (
        ("plain_bloom", PrivacyConfig.plain()),
        ("hardened", PrivacyConfig()),
        ("hardened_year_month", year_month),
    ):
        salts = _salt_frequencies(config, years)
        start = time.perf_counter()
        encoded = encode_records(normalized, KEY, config)
        encode_seconds = time.perf_counter() - start
        run = link_encoded(encoded, normalized["source"].to_list())
        f1 = pairwise_scores(run.resolution.labels, truth).f1
        attacks = frequency_alignment_attack(
            encoded.fields["first_name"],
            encoded.present["first_name"],
            normalized["first_name"].to_list(),
            public,
            bits=encoded.bits["first_name"],
            hashes=config.name_hashes,
            q=config.q,
            top_k=100,
            salt_frequencies=salts,
        )
        result[label] = {
            "f1": _r(f1),
            "f1_drop_pp": _r(100 * (plaintext_f1 - f1), 2),
            "encode_seconds": _r(encode_seconds, 2),
            "attack_top100": {attack.strategy: _r(attack.rate) for attack in attacks},
            "worst_attack_rate": _r(max(attack.rate for attack in attacks)),
        }
    return result


def _lookup(results: dict[str, Any], dotted: str) -> float:
    value: Any = results
    for part in dotted.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return float(value)


def check(results: dict[str, Any]) -> list[str]:
    thresholds: dict[str, dict[str, float]] = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    problems = []
    for dotted, limit in thresholds.get("min", {}).items():
        if _lookup(results, dotted) < limit:
            problems.append(f"{dotted} = {_lookup(results, dotted)} is below {limit}")
    for dotted, limit in thresholds.get("max", {}).items():
        if _lookup(results, dotted) > limit:
            problems.append(f"{dotted} = {_lookup(results, dotted)} exceeds {limit}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="robustness, guard, drift and privacy benchmark")
    parser.add_argument("--records", type=int, default=20_000)
    parser.add_argument("--privacy-records", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument(
        "--check", action="store_true", help="enforce bench/robustness_thresholds.json"
    )
    args = parser.parse_args(argv)
    start = time.perf_counter()
    results: dict[str, Any] = {
        "machine": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
        },
        "records": args.records,
        "corruption_sweep": corruption_sweep(args.records, args.seed),
        "calibration": calibration(args.records, args.seed),
        "over_merge_guard": over_merge_guard(args.records, args.seed),
        "drift": drift(args.records, args.seed),
        "privacy": privacy(args.privacy_records, args.seed),
    }
    results["seconds"] = _r(time.perf_counter() - start, 1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "robustness.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "machine"}, indent=2))
    if args.check:
        problems = check(results)
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all robustness metrics within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
