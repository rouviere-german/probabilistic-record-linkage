"""Unsupervised Fellegi-Sunter model fit by EM over comparison-pattern counts (M3).

Model (Fellegi and Sunter 1969): a candidate pair is a match with prior probability ``lambda``;
given its class, field agreement levels are independent with probabilities ``m_f(l)`` for
matches and ``u_f(l)`` for non-matches. A missing field (level 0) is ignored in both classes.

EM (Dempster, Laird and Rubin 1977; Winkler 1988 and Jaro 1989 for record linkage) runs on the
table of distinct patterns ``gamma`` with counts ``n_gamma``:

- E step: ``g(gamma) = lambda P(gamma|M) / (lambda P(gamma|M) + (1 - lambda) P(gamma|U))``
- M step: ``lambda = sum n g / sum n``; ``m_f(l) = sum_{gamma_f = l} n g / sum_{gamma_f > 0} n g``
  and the same for ``u`` with ``1 - g``.

Each iteration costs O(patterns x fields), independent of the number of pairs, so a fit over
millions of candidate pairs takes milliseconds. The observed-data log-likelihood never decreases
(``tests/unit/test_fellegi_sunter.py::test_em_log_likelihood_never_decreases``). ``u`` is
initialized from random record pairs, which are almost all non-matches; several initializations
of ``m`` and ``lambda`` are run and the one with the highest log-likelihood is kept, with every
start recorded in the manifest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.special import expit

from probabilistic_record_linkage.linkage.comparison import ComparisonSpace

__all__ = [
    "EMRun",
    "FSParams",
    "combine_session_m",
    "default_inits",
    "e_step",
    "estimate_prior",
    "field_weight_table",
    "fit_em",
    "fit_em_multistart",
    "log_bayes_factor_table",
    "log_likelihood",
    "m_step",
    "mask_fields",
    "match_probability",
    "pattern_table",
    "u_from_levels",
]

FloatArray = npt.NDArray[np.float64]
Int8Array = npt.NDArray[np.int8]
Int64Array = npt.NDArray[np.int64]

_FLOOR = 1e-9


class ParameterFormatError(ValueError):
    """Serialized model parameters are malformed."""


@dataclass(frozen=True, slots=True)
class FSParams:
    """``lam`` plus per-field ``m`` and ``u`` arrays of length ``n_levels`` (index 0 unused)."""

    lam: float
    m: tuple[FloatArray, ...]
    u: tuple[FloatArray, ...]

    def check(self, space: ComparisonSpace) -> FSParams:
        if not 0.0 < self.lam < 1.0:
            raise ValueError("lambda must be in (0, 1)")
        if len(self.m) != len(space.fields) or len(self.u) != len(space.fields):
            raise ValueError("m and u need one array per field")
        for spec, m, u in zip(space.fields, self.m, self.u, strict=True):
            for label, probs in (("m", m), ("u", u)):
                if probs.shape != (spec.n_levels,):
                    raise ValueError(f"{label} for {spec.name} must have {spec.n_levels} entries")
                if np.any(probs[1:] <= 0) or not np.isclose(probs[1:].sum(), 1.0, atol=1e-6):
                    raise ValueError(f"{label} for {spec.name} must be positive and sum to 1")
        return self

    def to_json(self) -> dict[str, object]:
        return {
            "lambda": float(self.lam),
            "m": [[float(x) for x in probs] for probs in self.m],
            "u": [[float(x) for x in probs] for probs in self.u],
        }

    @classmethod
    def from_json(cls, data: dict[str, object], space: ComparisonSpace) -> FSParams:
        lam = data["lambda"]
        m, u = data["m"], data["u"]
        if not isinstance(lam, int | float) or not isinstance(m, list) or not isinstance(u, list):
            raise ParameterFormatError("invalid Fellegi-Sunter parameters")
        return cls(
            float(lam),
            tuple(np.asarray(row, dtype=np.float64) for row in m),
            tuple(np.asarray(row, dtype=np.float64) for row in u),
        ).check(space)


@dataclass(frozen=True, slots=True)
class EMRun:
    label: str
    params: FSParams
    iterations: int
    converged: bool
    log_likelihood: float
    trace: tuple[float, ...]

    def summary(self) -> dict[str, object]:
        return {
            "label": self.label,
            "iterations": self.iterations,
            "converged": self.converged,
            "log_likelihood": round(self.log_likelihood, 6),
            "lambda": round(self.params.lam, 8),
        }


def pattern_table(patterns: Int64Array, space: ComparisonSpace) -> tuple[Int8Array, FloatArray]:
    """Distinct patterns (decoded to levels) and their counts."""
    distinct, counts = np.unique(patterns, return_counts=True)
    return space.decode(distinct), counts.astype(np.float64)


def _normalize(probs: FloatArray) -> FloatArray:
    out = np.maximum(probs, _FLOOR)
    out[0] = 0.0
    total = out[1:].sum()
    result: FloatArray = out / total
    return result


def _log_class_likelihoods(params: FSParams, levels: Int8Array) -> tuple[FloatArray, FloatArray]:
    log_m = np.zeros(levels.shape[0], dtype=np.float64)
    log_u = np.zeros(levels.shape[0], dtype=np.float64)
    for f, (m, u) in enumerate(zip(params.m, params.u, strict=True)):
        level = levels[:, f].astype(np.int64)
        present = level > 0
        with np.errstate(divide="ignore"):
            log_m[present] += np.log(m[level[present]])
            log_u[present] += np.log(u[level[present]])
    return log_m, log_u


def log_likelihood(params: FSParams, levels: Int8Array, counts: FloatArray) -> float:
    log_m, log_u = _log_class_likelihoods(params, levels)
    mixed = np.logaddexp(np.log(params.lam) + log_m, np.log1p(-params.lam) + log_u)
    return float(np.sum(counts * mixed))


def e_step(params: FSParams, levels: Int8Array) -> FloatArray:
    """Posterior match probability of each pattern."""
    log_m, log_u = _log_class_likelihoods(params, levels)
    posterior: FloatArray = expit((np.log(params.lam) + log_m) - (np.log1p(-params.lam) + log_u))
    return posterior


def m_step(
    space: ComparisonSpace,
    levels: Int8Array,
    counts: FloatArray,
    posterior: FloatArray,
    *,
    fixed_u: tuple[FloatArray, ...] | None = None,
) -> FSParams:
    match_weight = counts * posterior
    nonmatch_weight = counts * (1.0 - posterior)
    total = counts.sum()
    lam = float(np.clip(match_weight.sum() / total, 1e-9, 1.0 - 1e-9))
    m_list: list[FloatArray] = []
    u_list: list[FloatArray] = []
    for f, spec in enumerate(space.fields):
        level = levels[:, f].astype(np.int64)
        m_counts = np.bincount(level, weights=match_weight, minlength=spec.n_levels)
        m_list.append(_normalize(m_counts.astype(np.float64)))
        if fixed_u is None:
            u_counts = np.bincount(level, weights=nonmatch_weight, minlength=spec.n_levels)
            u_list.append(_normalize(u_counts.astype(np.float64)))
        else:
            u_list.append(fixed_u[f])
    return FSParams(lam, tuple(m_list), tuple(u_list))


def _max_change(a: FSParams, b: FSParams) -> float:
    change = abs(a.lam - b.lam)
    for x, y in zip((*a.m, *a.u), (*b.m, *b.u), strict=True):
        change = max(change, float(np.max(np.abs(x - y))))
    return change


def _orient(space: ComparisonSpace, params: FSParams) -> FSParams:
    """Label switching guard: the match class must agree exactly more often than non-matches."""
    evidence = sum(
        float(m[spec.exact_level] - u[spec.exact_level])
        for spec, m, u in zip(space.fields, params.m, params.u, strict=True)
    )
    if evidence >= 0:
        return params
    return FSParams(1.0 - params.lam, params.u, params.m)


def fit_em(
    space: ComparisonSpace,
    levels: Int8Array,
    counts: FloatArray,
    init: FSParams,
    *,
    max_iterations: int = 500,
    tolerance: float = 1e-7,
    fix_u: bool = False,
    label: str = "init",
) -> EMRun:
    if levels.shape[0] != counts.size:
        raise ValueError("levels and counts must describe the same patterns")
    if counts.sum() <= 0:
        raise ValueError("EM needs at least one pair")
    params = init.check(space)
    trace = [log_likelihood(params, levels, counts)]
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):  # noqa: B007 - the final value is reported
        posterior = e_step(params, levels)
        updated = m_step(space, levels, counts, posterior, fixed_u=params.u if fix_u else None)
        trace.append(log_likelihood(updated, levels, counts))
        change = _max_change(params, updated)
        params = updated
        if change < tolerance:
            converged = True
            break
    if not fix_u:  # with u held fixed the classes cannot switch labels
        params = _orient(space, params)
    return EMRun(label, params, iterations, converged, trace[-1], tuple(trace))


def fit_em_multistart(
    space: ComparisonSpace,
    levels: Int8Array,
    counts: FloatArray,
    inits: Sequence[tuple[str, FSParams]],
    **kwargs: object,
) -> tuple[EMRun, list[EMRun]]:
    """Fit from every initialization; return the highest-likelihood run and all runs."""
    if not inits:
        raise ValueError("at least one initialization is required")
    runs = [
        fit_em(space, levels, counts, params, label=label, **kwargs)  # type: ignore[arg-type]
        for label, params in inits
    ]
    best = max(runs, key=lambda run: (run.log_likelihood, -run.iterations))
    return best, runs


def u_from_levels(space: ComparisonSpace, levels: Int8Array) -> tuple[FloatArray, ...]:
    """Level frequencies (missing excluded) of a sample of pairs, e.g. random record pairs."""
    result: list[FloatArray] = []
    for f, spec in enumerate(space.fields):
        counts = np.bincount(levels[:, f].astype(np.int64), minlength=spec.n_levels)
        result.append(_normalize(counts.astype(np.float64) + 0.5))
    return tuple(result)


def default_inits(
    space: ComparisonSpace,
    u: tuple[FloatArray, ...],
    *,
    lambdas: Sequence[float] = (0.02, 0.2),
    exact_masses: Sequence[float] = (0.9, 0.6),
) -> list[tuple[str, FSParams]]:
    """Initializations: ``u`` from random pairs, ``m`` concentrated on exact agreement."""
    inits: list[tuple[str, FSParams]] = []
    for lam in lambdas:
        for mass in exact_masses:
            m_list: list[FloatArray] = []
            for spec in space.fields:
                probs = np.zeros(spec.n_levels, dtype=np.float64)
                probs[spec.exact_level] = mass
                rest = spec.n_levels - 2
                if rest > 0:
                    geometric = 0.5 ** np.arange(rest, 0, -1, dtype=np.float64)
                    probs[1 : spec.exact_level] = (1.0 - mass) * geometric / geometric.sum()
                else:
                    probs[1] = 1.0 - mass
                m_list.append(_normalize(probs))
            inits.append((f"lambda={lam},exact_m={mass}", FSParams(lam, tuple(m_list), u)))
    return inits


def mask_fields(space: ComparisonSpace, levels: Int8Array, fields: Sequence[str]) -> Int8Array:
    """Copy of ``levels`` with ``fields`` set to level 0 (ignored), e.g. a training block's key."""
    masked = levels.copy()
    for name in fields:
        masked[:, space.index(name)] = 0
    return masked


def combine_session_m(
    space: ComparisonSpace, sessions: Sequence[tuple[EMRun, int, Sequence[str]]]
) -> tuple[FloatArray, ...]:
    """Average ``m`` over training sessions, weighted by each session's expected matches.

    ``sessions`` holds ``(run, number of pairs, masked fields)``. A field masked in every session
    has no estimate and raises ``ValueError``.
    """
    totals = [np.zeros(spec.n_levels, dtype=np.float64) for spec in space.fields]
    weights = np.zeros(len(space.fields), dtype=np.float64)
    for run, n_pairs, masked in sessions:
        expected = run.params.lam * n_pairs
        for f, spec in enumerate(space.fields):
            if spec.name in masked:
                continue
            totals[f] += expected * run.params.m[f]
            weights[f] += expected
    result: list[FloatArray] = []
    for f, spec in enumerate(space.fields):
        if weights[f] <= 0:
            raise ValueError(f"no training session estimates m for {spec.name}")
        result.append(_normalize(totals[f] / weights[f]))
    return tuple(result)


def estimate_prior(
    log_bayes_factor: FloatArray,
    total_pairs: float,
    *,
    max_iterations: int = 500,
    tolerance: float = 1e-10,
) -> tuple[float, int]:
    """Prior match probability of a random record pair: EM for ``lambda`` alone over all pairs.

    Pairs outside the candidate set are treated as having negligible posterior (the blocking
    completeness assumption), so the update is ``lambda <- sum_candidates posterior / total_pairs``.
    Returns ``(lambda, iterations)``.
    """
    if total_pairs < 1:
        raise ValueError("total_pairs must be >= 1")
    if log_bayes_factor.size == 0:
        return 1.0 / (total_pairs + 1.0), 0
    lam = float(np.clip(np.count_nonzero(log_bayes_factor > 0) / total_pairs, 1e-15, 0.5))
    iterations = 0
    for iterations in range(1, max_iterations + 1):  # noqa: B007 - reported
        posterior = expit(np.log(lam) - np.log1p(-lam) + log_bayes_factor)
        updated = float(np.clip(posterior.sum() / total_pairs, 1e-15, 0.5))
        if abs(updated - lam) <= tolerance * lam:
            lam = updated
            break
        lam = updated
    return lam, iterations


def log_bayes_factor_table(params: FSParams, space: ComparisonSpace) -> FloatArray:
    """Natural-log Bayes factor of every possible pattern (indexable by pattern id)."""
    levels = space.decode(np.arange(space.n_patterns, dtype=np.int64))
    log_m, log_u = _log_class_likelihoods(params, levels)
    table: FloatArray = log_m - log_u
    return table


def match_probability(params: FSParams, log_bayes_factor: FloatArray) -> FloatArray:
    prior = np.log(params.lam) - np.log1p(-params.lam)
    probability: FloatArray = expit(prior + log_bayes_factor)
    return probability


def field_weight_table(params: FSParams, space: ComparisonSpace) -> list[dict[str, object]]:
    """Human-readable m, u and log2 Bayes factor per field level (for reports and the API)."""
    rows: list[dict[str, object]] = []
    for spec, m, u in zip(space.fields, params.m, params.u, strict=True):
        for level in range(1, spec.n_levels):
            rows.append(
                {
                    "field": spec.name,
                    "level": spec.levels[level - 1],
                    "m": round(float(m[level]), 6),
                    "u": round(float(u[level]), 8),
                    "log2_bayes_factor": round(float(np.log2(m[level] / u[level])), 4),
                }
            )
    return rows
