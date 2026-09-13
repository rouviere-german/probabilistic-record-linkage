"""End-to-end linkage quality and throughput benchmark, with regression thresholds.

Usage:
  uv run python bench/bench_linkage.py --records 20000 --check    # CI job e2e-20k
  uv run python bench/bench_linkage.py --records 200000           # README results (a few minutes)

Everything is measured against the generator's ground truth:

- blocking pair completeness and reduction ratio;
- pairwise precision/recall/F1 and B-cubed F1 of the guarded clustering;
- baselines: exact normalized-key matching and logistic regression trained on 1% labeled pairs;
- EM recovery: |fitted m(exact) - analytic m(exact)| per field, for the default training-block
  estimator and for EM on the candidate pairs;
- calibration (ECE of match probabilities over candidate pairs);
- twin and household false-merge rates and per-name-origin-group pairwise scores;
- throughput of ``link_sources`` (records per second, all stages).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from probabilistic_record_linkage.linkage.analytic import analytic_exact_agreement
from probabilistic_record_linkage.linkage.baselines import LogisticLinkage, exact_key_labels
from probabilistic_record_linkage.linkage.blocking import pair_completeness, reduction_ratio
from probabilistic_record_linkage.linkage.clustering import connected_labels
from probabilistic_record_linkage.linkage.demo import entity_pairs
from probabilistic_record_linkage.linkage.evaluation import (
    bcubed_scores,
    expected_calibration_error,
    group_pairwise_scores,
    merge_rate,
    pairwise_scores,
)
from probabilistic_record_linkage.linkage.generator import GeneratorConfig, entity_labels, generate
from probabilistic_record_linkage.linkage.normalize import normalize_sources, prepare_records
from probabilistic_record_linkage.linkage.pipeline import EMConfig, link_sources, train_model

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results"
THRESHOLDS = ROOT / "bench" / "linkage_thresholds.json"


def machine() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or "unknown",
    }


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def measure(records: int, seed: int) -> dict[str, Any]:
    start = time.perf_counter()
    dataset = generate(GeneratorConfig(seed=seed, n_records=records))
    generation_seconds = time.perf_counter() - start

    start = time.perf_counter()
    run = link_sources(dataset.sources, nicknames=dataset.world.nicknames)
    link_seconds = time.perf_counter() - start

    truth = entity_labels(dataset.truth, list(run.record_ids))
    blocking = run.blocking
    scores = pairwise_scores(run.resolution.labels, truth)
    _, _, b3_f1 = bcubed_scores(run.resolution.labels, truth)
    is_match = truth[blocking.left] == truth[blocking.right]
    space = run.comparator.space

    analytic = analytic_exact_agreement(dataset)
    per_field = {
        spec.name: _round(abs(float(run.model.params.m[f][spec.exact_level]) - analytic[spec.name]))
        for f, spec in enumerate(space.fields)
    }
    candidates_model = train_model(
        run.comparator,
        candidate_left=blocking.left,
        candidate_right=blocking.right,
        candidate_levels=run.scored.levels,
        blocks=(),
        config=EMConfig(strategy="candidates"),
        rng=np.random.default_rng(1),
    )
    per_field_candidates = {
        spec.name: _round(
            abs(float(candidates_model.params.m[f][spec.exact_level]) - analytic[spec.name])
        )
        for f, spec in enumerate(space.fields)
    }

    records_prepared = prepare_records(normalize_sources(dataset.sources), dataset.world.nicknames)
    exact = pairwise_scores(exact_key_labels(records_prepared), truth)
    sample = np.random.default_rng(seed).random(blocking.n_pairs) < 0.01
    logistic = LogisticLinkage(space).fit(run.scored.levels[sample], is_match[sample])
    keep = logistic.predict_proba(run.scored.levels) >= 0.5
    logistic_scores = pairwise_scores(
        connected_labels(run.n_records, blocking.left[keep], blocking.right[keep]), truth
    )

    twins, households = entity_pairs(dataset)
    group_of = dict(
        zip(dataset.truth["record_id"].to_list(), dataset.truth["group"].to_list(), strict=True)
    )
    groups = group_pairwise_scores(
        run.resolution.labels, truth, [group_of[record_id] for record_id in run.record_ids]
    )
    summary = run.summary()
    return {
        "records": dataset.n_records,
        "entities": len(dataset.population),
        "seed": seed,
        "generation_seconds": _round(generation_seconds, 2),
        "link_seconds": _round(link_seconds, 2),
        "records_per_second": round(dataset.n_records / link_seconds),
        "stage_seconds": {k: _round(v, 2) for k, v in run.timings.items()},
        "blocking": {
            "candidate_pairs": blocking.n_pairs,
            "pair_completeness": _round(pair_completeness(blocking.left, blocking.right, truth)),
            "reduction_ratio": round(reduction_ratio(blocking.n_pairs, run.n_records), 6),
        },
        "linkage": {
            "precision": _round(scores.precision),
            "recall": _round(scores.recall),
            "f1": _round(scores.f1),
            "bcubed_f1": _round(b3_f1),
            "ece": _round(expected_calibration_error(run.scored.probability, is_match), 5),
        },
        "baselines": {
            "exact_key_precision": _round(exact.precision),
            "exact_key_recall": _round(exact.recall),
            "exact_key_f1": _round(exact.f1),
            "logistic_1pct_f1": _round(logistic_scores.f1),
            "logistic_training_pairs": int(sample.sum()),
        },
        "em_recovery": {
            "training_blocks_max_abs_error": max(per_field.values()),
            "candidates_max_abs_error": max(per_field_candidates.values()),
            "training_blocks": per_field,
            "candidates": per_field_candidates,
        },
        "hard_negatives": {
            "twin_merge_rate": _round(merge_rate(run.resolution.labels, truth, twins)[0]),
            "household_merge_rate": _round(merge_rate(run.resolution.labels, truth, households)[0]),
        },
        "groups": {
            name: {"precision": _round(s.precision), "recall": _round(s.recall), "f1": _round(s.f1)}
            for name, s in groups.items()
        },
        "review": {
            "review_pairs": summary["review_pairs"],
            "rule_hits": summary["review_rule_hits"],
            "pending_records": summary["pending_records"],
            "guard_removed_edges": summary["guard_removed_edges"],
        },
    }


def _lookup(results: dict[str, Any], dotted: str) -> float:
    value: Any = results
    for part in dotted.split("."):
        value = value[part]
    return float(value)


def check(results: dict[str, Any]) -> list[str]:
    thresholds: dict[str, dict[str, dict[str, float]]] = json.loads(
        THRESHOLDS.read_text(encoding="utf-8")
    )
    scale = thresholds.get(str(results["requested_records"]))
    if scale is None:
        return [f"no thresholds for {results['requested_records']} records"]
    problems = []
    for dotted, limit in scale.get("min", {}).items():
        if _lookup(results, dotted) < limit:
            problems.append(f"{dotted} = {_lookup(results, dotted)} is below {limit}")
    for dotted, limit in scale.get("max", {}).items():
        if _lookup(results, dotted) > limit:
            problems.append(f"{dotted} = {_lookup(results, dotted)} exceeds {limit}")
    return problems


def print_tables(results: dict[str, Any]) -> None:
    b, lk, base, em = (
        results["blocking"],
        results["linkage"],
        results["baselines"],
        results["em_recovery"],
    )
    print(f"records={results['records']} entities={results['entities']} seed={results['seed']}")
    print("| metric | value |")
    print("|---|---:|")
    rows = [
        ("candidate pairs", b["candidate_pairs"]),
        ("pair completeness", b["pair_completeness"]),
        ("reduction ratio", b["reduction_ratio"]),
        ("pairwise precision", lk["precision"]),
        ("pairwise recall", lk["recall"]),
        ("pairwise F1", lk["f1"]),
        ("B-cubed F1", lk["bcubed_f1"]),
        ("ECE (candidate pairs)", lk["ece"]),
        ("exact-key baseline F1", base["exact_key_f1"]),
        ("logistic regression (1% labels) F1", base["logistic_1pct_f1"]),
        ("EM max abs m error (training blocks)", em["training_blocks_max_abs_error"]),
        ("EM max abs m error (EM on candidates)", em["candidates_max_abs_error"]),
        ("twin false-merge rate", results["hard_negatives"]["twin_merge_rate"]),
        ("household false-merge rate", results["hard_negatives"]["household_merge_rate"]),
        ("link seconds (all stages)", results["link_seconds"]),
        ("records per second", results["records_per_second"]),
    ]
    for name, value in rows:
        print(f"| {name} | {value} |")
    print("| group | precision | recall | F1 |")
    print("|---|---:|---:|---:|")
    for name, scores in results["groups"].items():
        print(f"| {name} | {scores['precision']} | {scores['recall']} | {scores['f1']} |")
    print("stage seconds:", json.dumps(results["stage_seconds"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="linkage quality and throughput benchmark")
    parser.add_argument("--records", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument(
        "--check", action="store_true", help="enforce bench/linkage_thresholds.json"
    )
    args = parser.parse_args(argv)
    results = {
        "machine": machine(),
        "requested_records": args.records,
        **measure(args.records, args.seed),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"linkage-{args.records}.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print_tables(results)
    if args.check:
        problems = check(results)
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all linkage metrics within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
