from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import expit

from probabilistic_record_linkage.linkage.comparison import ComparisonSpace, FieldSpec
from probabilistic_record_linkage.linkage.fellegi_sunter import (
    EMRun,
    FSParams,
    ParameterFormatError,
    combine_session_m,
    default_inits,
    estimate_prior,
    field_weight_table,
    fit_em,
    fit_em_multistart,
    log_bayes_factor_table,
    mask_fields,
    match_probability,
    pattern_table,
    u_from_levels,
)

pytestmark = pytest.mark.unit

SPACE = ComparisonSpace(
    (
        FieldSpec("a", ("else", "exact")),
        FieldSpec("b", ("else", "mid", "exact")),
        FieldSpec("c", ("else", "exact")),
    )
)
TRUE = FSParams(
    0.3,
    (np.array([0, 0.1, 0.9]), np.array([0, 0.05, 0.15, 0.8]), np.array([0, 0.2, 0.8])),
    (np.array([0, 0.9, 0.1]), np.array([0, 0.7, 0.2, 0.1]), np.array([0, 0.6, 0.4])),
)


def _simulate(
    params: FSParams, n: int, rng: np.random.Generator, missing: float = 0.05
) -> npt.NDArray[np.int8]:
    match = rng.random(n) < params.lam
    levels = np.zeros((n, len(SPACE.fields)), dtype=np.int8)
    for f, spec in enumerate(SPACE.fields):
        choices = np.arange(1, spec.n_levels)
        from_m = rng.choice(choices, size=n, p=params.m[f][1:])
        from_u = rng.choice(choices, size=n, p=params.u[f][1:])
        column = np.where(match, from_m, from_u)
        column[rng.random(n) < missing] = 0
        levels[:, f] = column
    return levels


def _random_params(rng: np.random.Generator, lam: float) -> FSParams:
    def draw(size: int) -> npt.NDArray[np.float64]:
        return np.concatenate([[0.0], rng.dirichlet(np.ones(size - 1))])

    return FSParams(
        lam,
        tuple(draw(spec.n_levels) for spec in SPACE.fields),
        tuple(draw(spec.n_levels) for spec in SPACE.fields),
    )


@settings(max_examples=30)
@given(seed=st.integers(0, 10_000), lam=st.floats(0.01, 0.99))
def test_em_log_likelihood_never_decreases(seed: int, lam: float) -> None:
    rng = np.random.default_rng(seed)
    levels = _simulate(_random_params(rng, 0.4), 400, rng)
    distinct, counts = pattern_table(SPACE.encode(levels), SPACE)
    run = fit_em(
        SPACE, distinct, counts, _random_params(rng, lam), max_iterations=60, tolerance=0.0
    )
    trace = np.asarray(run.trace)
    assert np.all(np.diff(trace) >= -1e-7 * np.abs(trace[1:]))
    assert run.iterations == 60


def test_em_recovers_the_generating_parameters() -> None:
    rng = np.random.default_rng(42)
    levels = _simulate(TRUE, 200_000, rng)
    distinct, counts = pattern_table(SPACE.encode(levels), SPACE)
    best, runs = fit_em_multistart(
        SPACE, distinct, counts, default_inits(SPACE, u_from_levels(SPACE, levels))
    )
    assert len(runs) == 4
    assert best.log_likelihood == max(run.log_likelihood for run in runs)
    assert best.params.lam == pytest.approx(TRUE.lam, abs=0.02)
    for f in range(len(SPACE.fields)):
        np.testing.assert_allclose(best.params.m[f][1:], TRUE.m[f][1:], atol=0.03)
        np.testing.assert_allclose(best.params.u[f][1:], TRUE.u[f][1:], atol=0.03)


def test_fixed_u_training_on_a_masked_block_recovers_the_other_fields() -> None:
    rng = np.random.default_rng(7)
    levels = mask_fields(SPACE, _simulate(TRUE, 100_000, rng), ["a"])
    assert np.all(levels[:, 0] == 0)
    distinct, counts = pattern_table(SPACE.encode(levels), SPACE)
    init = default_inits(SPACE, TRUE.u, lambdas=(0.1,), exact_masses=(0.6,))[0][1]
    run = fit_em(SPACE, distinct, counts, init, fix_u=True)
    for f in (1, 2):
        np.testing.assert_allclose(run.params.m[f][1:], TRUE.m[f][1:], atol=0.03)
        np.testing.assert_array_equal(run.params.u[f], TRUE.u[f])


def test_label_switching_is_undone_for_free_u() -> None:
    rng = np.random.default_rng(3)
    levels = _simulate(TRUE, 50_000, rng)
    distinct, counts = pattern_table(SPACE.encode(levels), SPACE)
    swapped = FSParams(0.7, TRUE.u, TRUE.m)
    run = fit_em(SPACE, distinct, counts, swapped)
    evidence = sum(
        run.params.m[f][spec.exact_level] - run.params.u[f][spec.exact_level]
        for f, spec in enumerate(SPACE.fields)
    )
    assert evidence > 0
    assert run.params.lam == pytest.approx(0.3, abs=0.03)


def test_session_estimates_are_weighted_by_expected_matches() -> None:
    m1 = (np.array([0, 0.5, 0.5]), np.array([0, 0.2, 0.3, 0.5]), np.array([0, 0.4, 0.6]))
    m2 = (np.array([0, 0.1, 0.9]), np.array([0, 0.1, 0.1, 0.8]), np.array([0, 0.3, 0.7]))
    run1 = EMRun("one", FSParams(0.5, m1, TRUE.u), 1, True, 0.0, (0.0,))
    run2 = EMRun("two", FSParams(0.25, m2, TRUE.u), 1, True, 0.0, (0.0,))
    m = combine_session_m(SPACE, [(run1, 100, ("a",)), (run2, 400, ("c",))])
    np.testing.assert_allclose(m[0], m2[0], atol=1e-6)
    np.testing.assert_allclose(m[2], m1[2], atol=1e-6)
    np.testing.assert_allclose(m[1], (50 * m1[1] + 100 * m2[1]) / 150, atol=1e-6)
    with pytest.raises(ValueError, match="no training session"):
        combine_session_m(SPACE, [(run1, 100, ("a",))])


def test_prior_estimate_is_a_self_consistent_fixed_point() -> None:
    rng = np.random.default_rng(0)
    total = 1e7
    log_bf = np.concatenate([rng.normal(20.0, 2.0, 1_000), rng.normal(-5.0, 2.0, 9_000)])
    lam, iterations = estimate_prior(log_bf, total)
    posterior = expit(np.log(lam) - np.log1p(-lam) + log_bf)
    assert lam == pytest.approx(posterior.sum() / total, rel=1e-6)
    assert lam == pytest.approx(1_000 / total, rel=0.05)
    assert iterations >= 1
    assert estimate_prior(np.zeros(0), 10.0) == (1.0 / 11.0, 0)
    with pytest.raises(ValueError, match="total_pairs"):
        estimate_prior(log_bf, 0.5)


def test_parameters_are_validated_and_round_trip_through_json() -> None:
    restored = FSParams.from_json(TRUE.to_json(), SPACE)
    assert restored.lam == TRUE.lam
    for x, y in zip((*restored.m, *restored.u), (*TRUE.m, *TRUE.u), strict=True):
        np.testing.assert_allclose(x, y)
    with pytest.raises(ValueError, match="lambda"):
        FSParams(0.0, TRUE.m, TRUE.u).check(SPACE)
    with pytest.raises(ValueError, match="entries"):
        FSParams(0.5, (np.array([0, 1.0]), *TRUE.m[1:]), TRUE.u).check(SPACE)
    with pytest.raises(ValueError, match="sum to 1"):
        FSParams(0.5, (np.array([0, 0.5, 0.2]), *TRUE.m[1:]), TRUE.u).check(SPACE)
    with pytest.raises(ValueError, match="one array per field"):
        FSParams(0.5, TRUE.m[:2], TRUE.u).check(SPACE)
    with pytest.raises(ParameterFormatError):
        FSParams.from_json({"lambda": "x", "m": [], "u": []}, SPACE)
    with pytest.raises(ValueError, match="same patterns"):
        fit_em(SPACE, np.zeros((2, 3), dtype=np.int8), np.ones(3), TRUE)
    with pytest.raises(ValueError, match="at least one initialization"):
        fit_em_multistart(SPACE, np.zeros((1, 3), dtype=np.int8), np.ones(1), [])


@given(st.lists(st.floats(-30, 30), min_size=2, max_size=20))
def test_match_probability_is_monotone_in_the_bayes_factor(values: list[float]) -> None:
    ordered = np.sort(np.asarray(values))
    probabilities = match_probability(TRUE, ordered)
    assert np.all(np.diff(probabilities) >= 0)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_bayes_factor_table_and_weight_report_agree() -> None:
    table = log_bayes_factor_table(TRUE, SPACE)
    assert table.size == SPACE.n_patterns
    all_exact = SPACE.encode(np.asarray([[2, 3, 2]], dtype=np.int8))[0]
    expected = sum(np.log(TRUE.m[f][lvl] / TRUE.u[f][lvl]) for f, lvl in enumerate((2, 3, 2)))
    assert table[all_exact] == pytest.approx(expected)
    assert table[0] == 0.0
    rows = field_weight_table(TRUE, SPACE)
    assert len(rows) == sum(spec.n_levels - 1 for spec in SPACE.fields)
    assert rows[1]["log2_bayes_factor"] == pytest.approx(np.log2(0.9 / 0.1), abs=1e-3)
