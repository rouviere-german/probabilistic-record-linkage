"""Micro-benchmarks for hot paths, with CI regression thresholds.

Usage:
  uv run python bench/bench_baseline.py          # run, write bench/results/latest.json
  uv run python bench/bench_baseline.py --check   # also exit 1 if bench/thresholds.json is exceeded

Thresholds are deliberately loose (several times the numbers measured on a laptop) because shared
CI runners are noisy; they catch order-of-magnitude regressions, not small drifts. The quality
benchmarks live in ``bench/bench_linkage.py`` and ``bench/bench_robustness.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from probabilistic_record_linkage.core.ratelimit import RateLimiter
from probabilistic_record_linkage.core.redaction import redact_text
from probabilistic_record_linkage.linkage.comparison import PLAIN_SPACE, jaro_winkler_pairs
from probabilistic_record_linkage.linkage.fellegi_sunter import (
    default_inits,
    fit_em_multistart,
    pattern_table,
    u_from_levels,
)
from probabilistic_record_linkage.linkage.minhash import MinHasher, shingle_sets
from probabilistic_record_linkage.ml.drift import DriftMonitor

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / "latest.json"
THRESHOLDS = ROOT / "bench" / "thresholds.json"


def _timed(function: Callable[[], object], runs: int, warmup: int = 3) -> dict[str, float]:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        function()
        samples.append(time.perf_counter_ns() - start)
    micros = np.asarray(samples, dtype=np.float64) / 1_000.0
    return {
        "runs": runs,
        "p50_us": round(float(np.percentile(micros, 50)), 1),
        "p95_us": round(float(np.percentile(micros, 95)), 1),
    }


def bench_redaction() -> dict[str, float]:
    line = (
        "user jane.doe@example.com born 1984-07-19 at 12 Vantolio Street paid with "
        "4111 1111 1111 1111; " + "lorem ipsum dolor sit amet " * 35
    )
    return _timed(lambda: redact_text(line), runs=2_000)


def bench_rate_limiter() -> dict[str, float]:
    limiter = RateLimiter(rate=1e9, burst=1_000_000_000)
    return _timed(lambda: limiter.check("client"), runs=20_000)


def bench_drift() -> dict[str, float]:
    rng = np.random.default_rng(0)
    monitor = DriftMonitor({"x": rng.normal(size=10_000)})
    current = {"x": rng.normal(loc=0.1, size=10_000)}
    return _timed(lambda: monitor.check(current), runs=30, warmup=2)


def bench_em_fit() -> dict[str, float]:
    rng = np.random.default_rng(1)
    columns = [rng.integers(0, spec.n_levels, size=500_000) for spec in PLAIN_SPACE.fields]
    levels = np.stack(columns, axis=1).astype(np.int8)
    distinct, counts = pattern_table(PLAIN_SPACE.encode(levels), PLAIN_SPACE)
    inits = default_inits(PLAIN_SPACE, u_from_levels(PLAIN_SPACE, levels))
    return _timed(
        lambda: fit_em_multistart(PLAIN_SPACE, distinct, counts, inits, max_iterations=50),
        runs=10,
        warmup=1,
    )


def bench_minhash() -> dict[str, float]:
    rng = np.random.default_rng(2)
    letters = np.asarray(list("abcdefghijklmnopqrstuvwxyz"))
    names = ["".join(rng.choice(letters, size=int(rng.integers(8, 16)))) for _ in range(10_000)]
    indptr, indices, vocabulary = shingle_sets(names, 2)
    hasher = MinHasher(120, seed=7)
    return _timed(lambda: hasher.signatures(indptr, indices, vocabulary), runs=10, warmup=1)


def bench_jaro_winkler() -> dict[str, float]:
    rng = np.random.default_rng(3)
    letters = np.asarray(list("abcdefghijklmnopqrstuvwxyz"))
    values = tuple(sorted({"".join(rng.choice(letters, size=7)) for _ in range(5_000)}))
    a = rng.integers(0, len(values), size=100_000).astype(np.int32)
    b = rng.integers(0, len(values), size=100_000).astype(np.int32)
    return _timed(lambda: jaro_winkler_pairs(values, a, b, workers=2), runs=10, warmup=1)


def machine() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or "unknown",
    }


def run() -> dict[str, dict[str, float]]:
    return {
        "redact_text_1kb": bench_redaction(),
        "rate_limiter_check": bench_rate_limiter(),
        "drift_check_10k_vs_10k": bench_drift(),
        "em_fit_500k_pairs_4_inits": bench_em_fit(),
        "minhash_signatures_10k_names": bench_minhash(),
        "jaro_winkler_100k_pairs": bench_jaro_winkler(),
    }


def check_thresholds(results: dict[str, dict[str, float]]) -> list[str]:
    thresholds: dict[str, dict[str, float]] = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    problems = []
    for name, limits in thresholds.items():
        if name not in results:
            problems.append(f"{name}: benchmark missing")
            continue
        for metric, limit in limits.items():
            value = results[name][metric]
            if value > limit:
                problems.append(f"{name}.{metric} = {value} exceeds threshold {limit}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="micro-benchmarks")
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    args = parser.parse_args(argv)

    results = run()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps({"machine": machine(), "results": results}, indent=2) + "\n", encoding="utf-8"
    )
    print("| benchmark | runs | p50 (us) | p95 (us) |")
    print("|---|---:|---:|---:|")
    for name, stats in results.items():
        print(f"| {name} | {stats['runs']:.0f} | {stats['p50_us']} | {stats['p95_us']} |")
    if args.check:
        problems = check_thresholds(results)
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all benchmarks within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
