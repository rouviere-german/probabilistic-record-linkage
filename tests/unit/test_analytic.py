"""The analytic m-probabilities must agree with the records the generator actually emits."""

from __future__ import annotations

import numpy as np
import pytest

from probabilistic_record_linkage.linkage.analytic import analytic_exact_agreement
from probabilistic_record_linkage.linkage.blocking import bucket_pairs
from probabilistic_record_linkage.linkage.comparison import PLAIN_SPACE, PlainComparator
from probabilistic_record_linkage.linkage.corruption import CorruptionConfig
from probabilistic_record_linkage.linkage.generator import (
    GeneratorConfig,
    entity_labels,
    generate,
)
from probabilistic_record_linkage.linkage.normalize import normalize_sources, prepare_records
from probabilistic_record_linkage.linkage.vocab import World

pytestmark = pytest.mark.unit


def _empirical_exact_rates(
    config: GeneratorConfig, world: World
) -> tuple[dict[str, float], dict[str, float]]:
    dataset = generate(config, world)
    normalized = normalize_sources(dataset.sources)
    records = prepare_records(normalized, dataset.world.nicknames)
    truth = entity_labels(dataset.truth, list(records.record_ids))
    left, right = bucket_pairs(truth, np.ones(truth.size, dtype=bool), max_bucket=10**9, window=1)
    levels = PlainComparator(records).levels(left, right)
    empirical = {}
    for f, spec in enumerate(PLAIN_SPACE.fields):
        present = levels[:, f] > 0
        empirical[spec.name] = float(np.mean(levels[present, f] == spec.exact_level))
    return empirical, analytic_exact_agreement(dataset)


@pytest.mark.parametrize(
    "corruption",
    [CorruptionConfig(), CorruptionConfig().correlated(0.1, 5.0)],
    ids=["independent", "correlated"],
)
def test_analytic_exact_agreement_matches_simulation(
    corruption: CorruptionConfig, world: World
) -> None:
    config = GeneratorConfig(seed=world.seed, n_records=20_000, corruption=corruption)
    empirical, analytic = _empirical_exact_rates(config, world)
    for name, expected in analytic.items():
        assert empirical[name] == pytest.approx(expected, abs=0.015), name
