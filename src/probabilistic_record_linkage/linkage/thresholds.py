"""Label-free match / review / non-match thresholds from the model's own probabilities (M4).

With calibrated match probabilities ``p``, the expected number of false matches among the pairs
above a threshold is ``sum(1 - p)`` over those pairs. Sorting pairs by ``p`` descending makes the
expected false discovery rate of the top ``k`` pairs non-decreasing in ``k``, so the match
threshold is the lowest probability that keeps the expected FDR at or below ``fdr_target``.

The review threshold keeps the expected number of matches left below it (``sum p`` of those
pairs) under ``missed_match_target`` of all expected matches, and is then raised if needed so that
at most ``review_capacity`` pairs fall into the review band. Whether the model's probabilities are
calibrated enough for these estimates is measured, not assumed: see the ECE rows in the README.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from probabilistic_record_linkage.linkage.comparison import ComparisonSpace

__all__ = [
    "BAND_MATCH",
    "BAND_NON_MATCH",
    "BAND_REVIEW",
    "DEFAULT_REVIEW_RULES",
    "ReviewRule",
    "ThresholdConfig",
    "Thresholds",
    "assign_bands",
    "band_of",
    "select_thresholds",
]

FloatArray = npt.NDArray[np.float64]
Int8Array = npt.NDArray[np.int8]

BAND_NON_MATCH = 0
BAND_REVIEW = 1
BAND_MATCH = 2


class ReviewRule(BaseModel):
    """Route match-band pairs with a known hard-negative signature to human review.

    ``levels`` maps a field to the level labels that trigger the rule; ``@exact``, ``@else`` and
    ``@missing`` stand for the exact, lowest-agreement and missing levels of whichever comparison
    space is in use, and ``@top<N>`` for the exact level and the ``N - 1`` agreement levels below
    it (never the missing level), so one rule works for plaintext and privacy mode. A pair
    triggers the rule only when every listed field is at one of its listed levels.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,40}$")
    levels: dict[str, tuple[str, ...]]

    def mask(self, space: ComparisonSpace, levels: Int8Array) -> npt.NDArray[np.bool_]:
        hit = np.ones(levels.shape[0], dtype=bool)
        for field_name, labels in self.levels.items():
            if field_name not in space.names:
                raise ValueError(f"review rule {self.name} names unknown field {field_name}")
            spec = space.fields[space.index(field_name)]
            allowed: set[int] = set()
            for label in labels:
                if label == "@exact":
                    allowed.add(spec.exact_level)
                elif label == "@else":
                    allowed.add(1)
                elif label == "@missing":
                    allowed.add(0)
                elif label.startswith("@top") and label[4:].isdigit() and int(label[4:]) >= 1:
                    allowed.update(
                        range(max(1, spec.exact_level - int(label[4:]) + 1), spec.exact_level + 1)
                    )
                elif label in spec.levels:
                    allowed.add(spec.levels.index(label) + 1)
                else:
                    raise ValueError(f"review rule {self.name}: unknown level {label!r}")
            hit &= np.isin(levels[:, space.index(field_name)], sorted(allowed))
        return hit


DEFAULT_REVIEW_RULES: tuple[ReviewRule, ...] = (
    # Twins: clearly different first names with the same, or nearly the same, surname and date
    # of birth. The surname may carry a typo (one level below exact) and the date a swap or a
    # one-component error (two levels below), and the address is not required: an earlier form of
    # the rule that needed all three exact let twin pairs through whenever one field of one record
    # was corrupted (twin false-merge rate 0.38 at 20,000 records against 0.12 with this rule, at
    # the same pairwise F1; see bench/bench_linkage.py). Pairs with a missing first name are not
    # routed: they are far more often duplicates than twins.
    ReviewRule(
        name="possible_twin",
        levels={"first_name": ("@else",), "last_name": ("@top2",), "dob": ("@top3",)},
    ),
)


class ThresholdConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fdr_target: float = Field(default=0.01, gt=0.0, lt=1.0)
    missed_match_target: float = Field(default=0.002, gt=0.0, lt=1.0)
    review_capacity: int = Field(default=5_000, ge=0)
    min_match_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    review_rules: tuple[ReviewRule, ...] = DEFAULT_REVIEW_RULES


class Thresholds(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    match: float = Field(ge=0.0, le=1.0)
    review: float = Field(ge=0.0, le=1.0)
    expected_fdr: float
    expected_missed_share: float
    match_pairs: int
    review_pairs: int


def select_thresholds(probabilities: FloatArray, config: ThresholdConfig) -> Thresholds:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.size == 0:
        return Thresholds(
            match=1.0, review=1.0, expected_fdr=0.0, expected_missed_share=0.0, match_pairs=0,
            review_pairs=0,
        )  # fmt: skip
    if np.any((p < 0) | (p > 1)) or not np.all(np.isfinite(p)):
        raise ValueError("probabilities must be finite and within [0, 1]")
    descending = np.sort(p)[::-1]
    fdr = np.cumsum(1.0 - descending) / np.arange(1, p.size + 1)
    k = int(np.searchsorted(fdr, config.fdr_target, side="right"))
    match = float(descending[k - 1]) if k > 0 else 1.0
    if 0 < k < p.size and descending[k] == match:
        # The tie group at the cut would enter the band as a whole and could push the FDR over
        # the target; keep only the strictly more probable pairs.
        match = float(np.nextafter(match, 2.0))
    match = min(max(match, config.min_match_probability), 1.0)
    in_match = p >= match
    expected_fdr = float(np.mean(1.0 - p[in_match])) if in_match.any() else 0.0

    total_expected = float(p.sum())
    ascending = np.sort(p)
    cumulative = np.cumsum(ascending)
    allowed = config.missed_match_target * total_expected
    j = int(np.searchsorted(cumulative, allowed, side="right"))
    review = float(ascending[j]) if j < p.size else match
    review = min(review, match)
    band = (p >= review) & (p < match)
    if int(band.sum()) > config.review_capacity:
        below = np.sort(p[p < match])[::-1]
        review = float(below[config.review_capacity - 1]) if config.review_capacity > 0 else match
        review = (
            float(np.nextafter(review, 1.0))
            if np.sum(below >= review) > config.review_capacity
            else review
        )
        band = (p >= review) & (p < match)
    missed = float(p[p < review].sum() / total_expected) if total_expected > 0 else 0.0
    return Thresholds(
        match=match,
        review=review,
        expected_fdr=expected_fdr,
        expected_missed_share=missed,
        match_pairs=int(in_match.sum()),
        review_pairs=int(band.sum()),
    )


def band_of(probabilities: FloatArray, thresholds: Thresholds) -> npt.NDArray[np.int8]:
    """0 non-match, 1 review, 2 match."""
    bands = np.zeros(probabilities.size, dtype=np.int8)
    bands[probabilities >= thresholds.review] = BAND_REVIEW
    bands[probabilities >= thresholds.match] = BAND_MATCH
    return bands


def assign_bands(
    probabilities: FloatArray,
    levels: Int8Array,
    space: ComparisonSpace,
    thresholds: Thresholds,
    rules: tuple[ReviewRule, ...],
) -> tuple[Int8Array, dict[str, int]]:
    """Probability bands, with match-band pairs that trigger a review rule moved to review."""
    bands = band_of(probabilities, thresholds)
    hits: dict[str, int] = {}
    for rule in rules:
        hit = (bands == BAND_MATCH) & rule.mask(space, levels)
        hits[rule.name] = int(hit.sum())
        bands[hit] = BAND_REVIEW
    return bands, hits
