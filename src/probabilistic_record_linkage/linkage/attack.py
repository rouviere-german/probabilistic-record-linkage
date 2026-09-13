"""Frequency-alignment attack on Bloom-filter encodings of names (M5).

Threat: the linkage unit (or anyone holding the encodings) knows the public name-frequency table
of the population and the encoding parameters, but not the key. Identical names produce identical
encodings, so it can count how often each distinct encoding occurs, sort, and align that ranking
with the public ranking of names (Kuzu et al. 2011; Christen et al. 2018 survey the stronger
pattern-mining variants, which are not implemented here).

Strategies, all of which are run; the report keeps each and the worst case for the defender:

- ``global``: align the encoding-frequency ranking with the public name ranking;
- ``length``: first group encodings by the q-gram count implied by their Hamming weight
  (``w = L (1 - (1 - 1/L)^{k n})`` inverted for ``n``), then align within each group. Balancing
  makes every weight equal and collapses this to ``global``;
- ``salt_aware``: an attacker who knows names are salted with the birth year aligns with the public
  ranking of (name, birth year) combinations, ``p(name) p(year)``, and guesses the name part.

The metric is the share of the ``top_k`` most frequent encodings whose guessed plaintext is right.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "AttackReport",
    "frequency_alignment_attack",
    "qgram_count",
]

UInt64Array = npt.NDArray[np.uint64]


@dataclass(frozen=True, slots=True)
class AttackReport:
    strategy: str
    top_k: int
    attacked: int
    correct: int

    @property
    def rate(self) -> float:
        return self.correct / self.attacked if self.attacked else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "top_k": self.top_k,
            "attacked": self.attacked,
            "correct": self.correct,
            "rate": round(self.rate, 4),
        }


def qgram_count(value: str, q: int = 2) -> int:
    padded = f"#{value}#" if q > 1 else value
    return len({padded[i : i + q] for i in range(max(1, len(padded) - q + 1))})


def _estimated_qgrams(weight: int, bits: int, hashes: int) -> int:
    if weight <= 0:
        return 0
    if weight >= bits:
        return 10**6
    return round(math.log(1.0 - weight / bits) / (hashes * math.log(1.0 - 1.0 / bits)))


def _align(ranked_groups: Sequence[int], ranked_names: Sequence[str]) -> dict[int, str]:
    return dict(zip(ranked_groups, ranked_names, strict=False))


def frequency_alignment_attack(
    packed: UInt64Array,
    present: npt.NDArray[np.bool_],
    plaintext: Sequence[str | None],
    public_frequencies: Mapping[str, float],
    *,
    bits: int,
    hashes: int,
    q: int = 2,
    top_k: int = 100,
    salt_frequencies: Mapping[str, float] | None = None,
) -> list[AttackReport]:
    """Run every strategy against one encoded name field; ``plaintext`` is used only for scoring."""
    rows = np.flatnonzero(present)
    if rows.size == 0:
        return [AttackReport("global", top_k, 0, 0)]
    distinct, first_row, inverse, counts = np.unique(
        packed[rows], axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    del inverse
    order = np.lexsort((first_row, -counts))
    targets = order[:top_k]
    truth = {int(g): plaintext[int(rows[first_row[g]])] for g in targets}
    weights = np.bitwise_count(distinct).sum(axis=1)
    names_ranked = sorted(public_frequencies, key=lambda name: (-public_frequencies[name], name))

    def score(label: str, guesses: Mapping[int, str]) -> AttackReport:
        correct = sum(1 for g in targets if guesses.get(int(g)) == truth[int(g)])
        return AttackReport(label, top_k, int(targets.size), correct)

    reports = [score("global", _align([int(g) for g in order], names_ranked))]

    by_length: dict[int, list[str]] = {}
    for name in names_ranked:
        by_length.setdefault(qgram_count(name, q), []).append(name)
    groups_by_length: dict[int, list[int]] = {}
    for g in order.tolist():
        estimate = _estimated_qgrams(int(weights[g]), bits, hashes)
        groups_by_length.setdefault(estimate, []).append(int(g))
    length_guesses: dict[int, str] = {}
    for estimate, groups in groups_by_length.items():
        length_guesses.update(_align(groups, by_length.get(estimate, [])))
    reports.append(score("length", length_guesses))

    if salt_frequencies:
        combos = sorted(
            (
                (name, salt)
                for name in names_ranked[: max(top_k * 4, 400)]
                for salt in salt_frequencies
            ),
            key=lambda item: (
                -public_frequencies[item[0]] * salt_frequencies[item[1]],
                item[0],
                item[1],
            ),
        )
        reports.append(score("salt_aware", _align([int(g) for g in order], [c[0] for c in combos])))
    return reports
