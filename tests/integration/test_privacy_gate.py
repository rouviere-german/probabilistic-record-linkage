"""The attacks on name encodings gate the privacy-mode hardening defaults, at benchmark scale.

The attack rates grow with the number of records (more encodings per (name, birth year) pair
give the salt-aware alignment more signal, and a denser similarity graph helps graph matching)
and vary with the synthetic world, so the gate runs on the same 50,000 records, seed and key as
``bench/bench_robustness.py`` rather than on a smaller sample where the defaults look better.
"""

from __future__ import annotations

import itertools

import pytest

from probabilistic_record_linkage.linkage.attack import all_attacks
from probabilistic_record_linkage.linkage.generator import GeneratorConfig, generate
from probabilistic_record_linkage.linkage.normalize import normalize_sources
from probabilistic_record_linkage.linkage.privacy import (
    HardeningConfig,
    PrivacyConfig,
    encode_records,
)
from probabilistic_record_linkage.linkage.vocab import build_world

pytestmark = pytest.mark.integration

RECORDS = 50_000
SEED = 20260913  # the benchmark seed: the world and the records are the same as in the README
KEY = b"benchmark-only-key-" + b"0" * 13
GOAL = 0.05  # this project's own goal for the default: at most 5 of the top 100 names recovered
REGRESSION_LIMIT = 0.15  # CI guard at the measured level (bench/robustness_thresholds.json)
CONFIGS: dict[str, PrivacyConfig] = {
    "plain": PrivacyConfig.plain(),
    "hardened": PrivacyConfig(),
    "hardened_year_month": PrivacyConfig(
        hardening=HardeningConfig(name_salt_parts=("birth_year", "birth_month"))
    ),
}


def _salt_frequencies(config: PrivacyConfig, years: range) -> dict[str, float]:
    values = {
        "birth_year": [str(year) for year in years],
        "birth_month": [f"{month:02d}" for month in range(1, 13)],
        "sex": ["F", "M"],
    }
    combos = list(itertools.product(*(values[part] for part in config.hardening.name_salt_parts)))
    return {"|".join(combo): 1.0 / len(combos) for combo in combos}


@pytest.fixture(scope="module")
def attack_rates() -> dict[str, dict[str, float]]:
    world = build_world(SEED)
    dataset = generate(GeneratorConfig(seed=SEED, n_records=RECORDS), world)
    normalized = normalize_sources(dataset.sources)
    public = world.first_name_frequencies()
    years = range(dataset.config.birth_years[0], dataset.config.birth_years[1] + 1)
    rates: dict[str, dict[str, float]] = {}
    for label, config in CONFIGS.items():
        encoded = encode_records(normalized, KEY, config)
        reports = all_attacks(
            encoded.fields["first_name"],
            encoded.present["first_name"],
            normalized["first_name"].to_list(),
            public,
            bits=encoded.bits["first_name"],
            hashes=config.name_hashes,
            q=config.q,
            top_k=100,
            salt_frequencies=_salt_frequencies(config, years),
        )
        rates[label] = {report.strategy: report.rate for report in reports}
    return rates


def test_the_defaults_are_hardened() -> None:
    defaults = PrivacyConfig().hardening
    assert defaults.salt_names_with_birth_year
    assert defaults.name_salt_parts == ("birth_year",)
    assert defaults.xor_fold
    assert defaults.balance


def test_graph_matching_breaks_unhardened_encodings(
    attack_rates: dict[str, dict[str, float]],
) -> None:
    """The attacks keep their power: unhardened keyed Bloom filters are re-identifiable."""
    plain = attack_rates["plain"]
    assert plain["graph_matching"] >= 0.5
    assert max(plain.values()) >= 0.5


def test_default_hardening_stays_within_the_regression_limit(
    attack_rates: dict[str, dict[str, float]],
) -> None:
    """Graph matching finds no name structure; the salt-aware alignment is the worst case."""
    hardened = attack_rates["hardened"]
    assert max(hardened.values()) <= REGRESSION_LIMIT
    assert hardened["graph_matching"] <= REGRESSION_LIMIT
    assert max(hardened.values()) < attack_rates["plain"]["graph_matching"] / 4


@pytest.mark.xfail(
    strict=True,
    reason="Known: on the benchmark records the birth-year salt (default) lets the salt-aware"
    " alignment recover 10 of the top 100 names; the goal of at most 5 is only met by the"
    " birth-year plus birth-month salt, which costs 4.52 F1 points instead of 2.59 (README).",
)
def test_default_hardening_meets_the_five_percent_goal(
    attack_rates: dict[str, dict[str, float]],
) -> None:
    assert max(attack_rates["hardened"].values()) <= GOAL


def test_the_year_month_salt_meets_the_five_percent_goal(
    attack_rates: dict[str, dict[str, float]],
) -> None:
    assert max(attack_rates["hardened_year_month"].values()) <= GOAL
