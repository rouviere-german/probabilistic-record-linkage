"""Analytic m-probabilities of the "exact agreement" level, derived from the corruption model.

For two records of the same entity, corrupted independently, the probability that a field agrees
exactly (given both values are present) follows from the per-record state probabilities. With a
sloppiness mixture ``{(pi_a, a)}`` the state probabilities of one record are averaged over ``a``
*before* squaring, because the two records draw their multipliers independently:

- last name: ``q_clean = E_a[(1 - a pm)(1 - a pt)]`` and ``m = q_clean^2 / q_present^2``
- first name with ``l`` nicknames: ``q_clean = E_a[(1 - a pm)(1 - a pn)(1 - a pt)]``,
  ``q_nick = E_a[(1 - a pm) a pn] / l`` and ``m = (q_clean^2 + l q_nick^2) / q_present^2``
  (two records that both use the same nickname agree exactly)
- date of birth: a swap is a no-op unless ``day <= 12 and day != month``
- sex: ``m = (q_same^2 + q_flip^2) / q_present^2``
- address (street and postcode): the entity shows its primary or its alternative address

Entity-level terms (nickname count, swappable date) are averaged with weight ``k (k - 1) / 2``,
the number of within-entity record pairs. These values describe *all* matching pairs, while EM is
fitted on blocks of likely matches by default (``EMConfig(strategy="training_blocks")``) or on the
candidate pairs, both of which miss a few heavily corrupted matches and so bias the fitted values
slightly upward; ``bench/bench_linkage.py`` reports the gap for both estimators (target within
0.03), next to the error of the fitted ``m`` and ``u`` over every level against the empirical
distributions of the labelled candidate pairs.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from probabilistic_record_linkage.linkage.generator import SyntheticDataset

__all__ = ["analytic_exact_agreement"]


def analytic_exact_agreement(dataset: SyntheticDataset) -> dict[str, float]:
    """Expected P(exact agreement | match, both present) per compared field."""
    c = dataset.config.corruption
    mixture = c.multipliers()

    def mix(function: Callable[[float], float]) -> float:
        return sum(weight * function(a) for weight, a in mixture)

    truth = dataset.truth.filter(pl.col("kind") != "bridge")
    sizes = truth.group_by("entity_id").len()
    entities = sizes["entity_id"].to_numpy()
    k = sizes["len"].to_numpy().astype(np.float64)
    weights = k * (k - 1.0) / 2.0
    if weights.sum() <= 0:
        raise ValueError("the dataset has no within-entity record pairs")
    weights = weights / weights.sum()
    pop, world = dataset.population, dataset.world

    nick_counts = np.asarray(
        [len(world.nicknames.nicknames_of(pop.first_name(world, int(e)))) for e in entities],
        dtype=np.float64,
    )
    present = mix(lambda a: 1.0 - a * c.first_name_missing)
    clean_plain = mix(lambda a: (1.0 - a * c.first_name_missing) * (1.0 - a * c.first_name_typo))
    clean_nick = mix(
        lambda a: (
            (1.0 - a * c.first_name_missing)
            * (1.0 - a * c.first_name_nickname)
            * (1.0 - a * c.first_name_typo)
        )
    )
    nick_total = mix(lambda a: (1.0 - a * c.first_name_missing) * a * c.first_name_nickname)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_entity_first = np.where(
            nick_counts > 0,
            (clean_nick**2 + nick_total**2 / np.maximum(nick_counts, 1.0)) / present**2,
            clean_plain**2 / present**2,
        )
    first = float(np.sum(weights * per_entity_first))

    last = (
        mix(lambda a: (1.0 - a * c.last_name_missing) * (1.0 - a * c.last_name_typo)) ** 2
        / mix(lambda a: 1.0 - a * c.last_name_missing) ** 2
    )

    day = pop.day[entities]
    month = pop.month[entities]
    swappable = ((day <= 12) & (day != month)).astype(np.float64)
    dob_present = mix(lambda a: 1.0 - a * c.dob_missing)
    unchanged_if_swappable = mix(
        lambda a: (1.0 - a * c.dob_missing) * (1.0 - a * c.dob_swap - a * c.dob_typo)
    )
    unchanged_if_fixed = mix(lambda a: (1.0 - a * c.dob_missing) * (1.0 - a * c.dob_typo))
    swapped = mix(lambda a: (1.0 - a * c.dob_missing) * a * c.dob_swap)
    per_entity_dob = np.where(
        swappable > 0,
        (unchanged_if_swappable**2 + swapped**2) / dob_present**2,
        unchanged_if_fixed**2 / dob_present**2,
    )
    dob = float(np.sum(weights * per_entity_dob))

    sex_present = mix(lambda a: 1.0 - a * c.sex_missing)
    sex_same = mix(lambda a: (1.0 - a * c.sex_missing) * (1.0 - a * c.sex_flip))
    sex_flip = mix(lambda a: (1.0 - a * c.sex_missing) * a * c.sex_flip)
    sex = (sex_same**2 + sex_flip**2) / sex_present**2

    def clean_address(a: float) -> float:
        return (
            (1.0 - a * c.street_typo) * (1.0 - a * c.postcode_missing) * (1.0 - a * c.postcode_typo)
        )

    primary = mix(lambda a: (1.0 - a * c.address_move) * clean_address(a))
    alternative = mix(lambda a: a * c.address_move * clean_address(a))
    address = primary**2 + alternative**2

    return {
        "first_name": first,
        "last_name": float(last),
        "dob": dob,
        "sex": float(sex),
        "address": float(address),
    }
