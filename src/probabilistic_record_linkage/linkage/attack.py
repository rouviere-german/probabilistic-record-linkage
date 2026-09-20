"""Attacks on Bloom-filter encodings of names (M5).

Threat: the linkage unit (or anyone holding the encodings) knows the public name-frequency table
of the population and the encoding parameters, but not the key. Two families of attack are
implemented; the benchmark and the privacy gate run all of them and report the worst case for
the defender.

Frequency alignment (:func:`frequency_alignment_attack`; Kuzu et al. 2011): identical names
produce identical encodings, so the attacker counts how often each distinct encoding occurs,
sorts, and aligns that ranking with the public ranking of names.

- ``global``: align the encoding-frequency ranking with the public name ranking;
- ``length``: first group encodings by the q-gram count implied by their Hamming weight
  (``w = L (1 - (1 - 1/L)^{k n})`` inverted for ``n``), then align within each group. Balancing
  makes every weight equal and collapses this to ``global``;
- ``salt_aware``: an attacker who knows names are salted with the birth year aligns with the public
  ranking of (name, birth year) combinations, ``p(name) p(year)``, and guesses the name part.

Graph matching (:func:`graph_matching_attack`; Vidanage, Christen, Ranbaduge and Schnell 2020, "A
graph matching attack on privacy-preserving record linkage"): a much stronger attacker does not
only use how often an encoding occurs but how *similar* encodings are to each other. Without
hardening the Dice similarity of two filters is a monotone, noisy function of the Dice similarity
of the two names' q-gram sets, which the attacker can compute exactly from the public name list.
Re-identification then becomes the problem of matching the weighted similarity graph of the most
frequent encodings to the weighted similarity graph of the most frequent names. It is solved here
with graduated assignment (Gold and Rangarajan 1996): a doubly stochastic soft assignment is
refined by ``P <- W * (S P T)`` and Sinkhorn normalization, then rounded with the Hungarian
algorithm. ``W`` is a prior from the two signals a per-node attacker already has, the estimated
q-gram count and the frequency.

Each hardening measure attacks a premise of this: salting makes two records with the same name
encode differently, so encoding frequency and encoding similarity stop mirroring the name's;
XOR-folding and balancing add similarity noise that is independent of the names. The measured
contrast (see ``bench/bench_robustness.py`` and ``tests/integration/test_privacy_gate.py``) is
what the hardening defaults are justified by.

The pattern-mining family (Christen et al. 2017) is deliberately *not* implemented: it needs to
isolate the ``hashes`` bit positions of one q-gram from co-occurrence counts, and at this
project's parameters (512 bits, 12 hashes, about 400 distinct q-grams, so roughly nine q-grams
per bit position) the positions of different q-grams co-occur as often as those of the same one.
Graph matching does not depend on that separation.

The metric of every attack is the share of the ``top_k`` most frequent encodings whose guessed
plaintext is right.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

__all__ = [
    "AttackReport",
    "all_attacks",
    "frequency_alignment_attack",
    "graph_matching_attack",
    "qgram_count",
    "qgrams",
    "similarity_correlation",
]

UInt64Array = npt.NDArray[np.uint64]
BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]
Int64Array = npt.NDArray[np.int64]

_TINY = 1e-300


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


def qgrams(value: str, q: int = 2) -> frozenset[str]:
    """Padded q-grams of ``value`` (the same tokenization the encoder uses)."""
    padded = f"#{value}#" if q > 1 else value
    return frozenset(padded[i : i + q] for i in range(max(1, len(padded) - q + 1)))


def qgram_count(value: str, q: int = 2) -> int:
    return len(qgrams(value, q))


def _estimated_qgrams(weight: int, bits: int, hashes: int) -> int:
    if weight <= 0:
        return 0
    if weight >= bits:
        return 10**6
    return round(math.log(1.0 - weight / bits) / (hashes * math.log(1.0 - 1.0 / bits)))


def _align(ranked_groups: Sequence[int], ranked_names: Sequence[str]) -> dict[int, str]:
    return dict(zip(ranked_groups, ranked_names, strict=False))


@dataclass(frozen=True, slots=True)
class _Targets:
    """Distinct encodings, their record counts and the ``top_k`` most frequent ones."""

    distinct: UInt64Array
    counts: Int64Array
    order: Int64Array  # distinct encodings by descending count (first row breaks ties)
    targets: Int64Array
    truth: dict[int, str | None]


def _targets(
    packed: UInt64Array, present: BoolArray, plaintext: Sequence[str | None], top_k: int
) -> _Targets | None:
    rows = np.flatnonzero(present)
    if rows.size == 0:
        return None
    distinct, first_row, counts = np.unique(
        packed[rows], axis=0, return_index=True, return_counts=True
    )
    order = np.lexsort((first_row, -counts))
    targets = order[:top_k]
    truth = {int(g): plaintext[int(rows[first_row[g]])] for g in targets}
    return _Targets(distinct, counts.astype(np.int64), order, targets, truth)


def _score(label: str, top_k: int, targets: _Targets, guesses: Mapping[int, str]) -> AttackReport:
    correct = sum(1 for g in targets.targets if guesses.get(int(g)) == targets.truth[int(g)])
    return AttackReport(label, top_k, int(targets.targets.size), correct)


def _ranked_names(public_frequencies: Mapping[str, float]) -> list[str]:
    return sorted(public_frequencies, key=lambda name: (-public_frequencies[name], name))


def frequency_alignment_attack(
    packed: UInt64Array,
    present: BoolArray,
    plaintext: Sequence[str | None],
    public_frequencies: Mapping[str, float],
    *,
    bits: int,
    hashes: int,
    q: int = 2,
    top_k: int = 100,
    salt_frequencies: Mapping[str, float] | None = None,
) -> list[AttackReport]:
    """Run every alignment strategy against one encoded name field.

    ``plaintext`` is used only for scoring.
    """
    found = _targets(packed, present, plaintext, top_k)
    if found is None:
        return [AttackReport("global", top_k, 0, 0)]
    order = found.order
    weights = np.bitwise_count(found.distinct).sum(axis=1)
    names_ranked = _ranked_names(public_frequencies)

    reports = [_score("global", top_k, found, _align([int(g) for g in order], names_ranked))]

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
    reports.append(_score("length", top_k, found, length_guesses))

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
        guesses = _align([int(g) for g in order], [c[0] for c in combos])
        reports.append(_score("salt_aware", top_k, found, guesses))
    return reports


# --- graph matching -------------------------------------------------------------------------


def _unpack(packed: UInt64Array, bits: int) -> BoolArray:
    """Bit matrix of packed encodings (big-endian 64-bit words, most significant bit first)."""
    words = np.ascontiguousarray(packed.astype(">u8"))
    unpacked = np.unpackbits(words.view(np.uint8).reshape(packed.shape[0], -1), axis=1)
    return unpacked[:, :bits].astype(bool)


def _standardized(similarity: FloatArray) -> FloatArray:
    """Similarity matrix centred and scaled by its off-diagonal moments, diagonal zeroed."""
    size = similarity.shape[0]
    off = ~np.eye(size, dtype=bool)
    values = similarity[off]
    if values.size == 0:
        return np.zeros_like(similarity)
    spread = float(values.std())
    if spread <= 0.0:
        return np.zeros_like(similarity)
    standardized: FloatArray = (similarity - float(values.mean())) / spread
    np.fill_diagonal(standardized, 0.0)
    return standardized


def _bit_dice(matrix: BoolArray) -> FloatArray:
    """Pairwise Dice similarity of bit vectors (diagonal zeroed)."""
    dense: Float32Array = matrix.astype(np.float32)
    weights = dense.sum(axis=1).astype(np.float64)
    overlap = (dense @ dense.T).astype(np.float64)
    denominator = weights[:, None] + weights[None, :]
    similarity: FloatArray = np.divide(
        2.0 * overlap, denominator, out=np.zeros_like(overlap), where=denominator > 0
    )
    np.fill_diagonal(similarity, 0.0)
    return similarity


def _qgram_dice(names: Sequence[str], q: int) -> FloatArray:
    """Pairwise Dice similarity of the names' q-gram sets (diagonal zeroed)."""
    gram_sets = [qgrams(name, q) for name in names]
    grams = sorted(set().union(*gram_sets)) if gram_sets else []
    index = {gram: i for i, gram in enumerate(grams)}
    incidence = np.zeros((len(names), len(grams)), dtype=np.float32)
    for row, gram_set in enumerate(gram_sets):
        for gram in gram_set:
            incidence[row, index[gram]] = 1.0
    sizes = incidence.sum(axis=1).astype(np.float64)
    overlap = (incidence @ incidence.T).astype(np.float64)
    denominator = sizes[:, None] + sizes[None, :]
    similarity: FloatArray = np.divide(
        2.0 * overlap, denominator, out=np.zeros_like(overlap), where=denominator > 0
    )
    np.fill_diagonal(similarity, 0.0)
    return similarity


def _prior(
    *,
    hamming: FloatArray,
    counts: Int64Array,
    names: Sequence[str],
    public_frequencies: Mapping[str, float],
    bits: int,
    hashes: int,
    q: int,
    length_weight: float,
    frequency_weight: float,
) -> FloatArray:
    """Per-node compatibility from the two signals a frequency attacker already has.

    Balanced encodings all have the same Hamming weight, which flattens the length term; salting
    flattens the frequency term, because an encoding then counts one (name, salt) pair.
    """
    estimated = np.asarray(
        [float(_estimated_qgrams(int(weight), bits, hashes)) for weight in hamming]
    )
    sizes = np.asarray([float(qgram_count(name, q)) for name in names])
    observed = counts.astype(np.float64)
    observed /= max(float(observed.sum()), _TINY)
    public = np.asarray([public_frequencies[name] for name in names], dtype=np.float64)
    public /= max(float(public.sum()), _TINY)
    length_cost = np.abs(estimated[:, None] - sizes[None, :])
    frequency_cost = np.abs(np.log(observed[:, None] + _TINY) - np.log(public[None, :] + _TINY))
    prior: FloatArray = np.exp(-length_weight * length_cost - frequency_weight * frequency_cost)
    total = float(prior.sum())
    return prior / total if total > 0 else np.full_like(prior, 1.0 / prior.size)


def _sinkhorn(matrix: FloatArray, rounds: int = 8) -> FloatArray:
    """Approximate the nearest doubly stochastic matrix by alternating normalization."""
    scaled = np.maximum(matrix, 0.0)
    total = float(scaled.sum())
    if total <= 0.0:
        return np.full_like(matrix, 1.0 / matrix.size)
    scaled = scaled / total
    for _ in range(rounds):
        scaled = scaled / (scaled.sum(axis=1, keepdims=True) + _TINY)
        scaled = scaled / (scaled.sum(axis=0, keepdims=True) + _TINY)
    return scaled


def _graduated_assignment(
    encoding_similarity: FloatArray,
    name_similarity: FloatArray,
    prior: FloatArray,
    iterations: int,
) -> Int64Array:
    """Soft-assign, refine with the two similarity graphs, then round with the Hungarian rule."""
    soft = _sinkhorn(prior)
    for _ in range(iterations):
        agreement = encoding_similarity @ soft @ name_similarity
        spread = float(agreement.max()) - float(agreement.min())
        if spread <= 0.0:  # no usable structure: keep the prior
            break
        candidate = prior * (agreement - float(agreement.min()))
        if float(candidate.sum()) <= 0.0:
            break
        soft = _sinkhorn(candidate)
    rows, columns = linear_sum_assignment(-np.log(soft + _TINY))
    assignment = np.zeros(soft.shape[0], dtype=np.int64)
    assignment[rows] = columns
    return assignment


def graph_matching_attack(
    packed: UInt64Array,
    present: BoolArray,
    plaintext: Sequence[str | None],
    public_frequencies: Mapping[str, float],
    *,
    bits: int,
    hashes: int,
    q: int = 2,
    top_k: int = 100,
    nodes: int = 400,
    candidates: int = 800,
    iterations: int = 6,
    length_weight: float = 3.0,
    frequency_weight: float = 1.0,
) -> AttackReport:
    """Match the encoding-similarity graph to the name-similarity graph and read off the names.

    ``nodes`` most frequent distinct encodings are matched against the ``candidates`` most
    frequent public names; only the ``top_k`` most frequent encodings are scored, so the extra
    nodes and candidates are context that makes the matching identifiable.
    """
    found = _targets(packed, present, plaintext, top_k)
    names = _ranked_names(public_frequencies)[: max(candidates, top_k)]
    if found is None or len(names) < 2 or found.order.size < 2:
        attacked = 0 if found is None else int(found.targets.size)
        return AttackReport("graph_matching", top_k, attacked, 0)
    chosen = found.order[: max(min(nodes, found.order.size), top_k)]
    matrix = _unpack(found.distinct, bits)[chosen]
    encoding_similarity = _standardized(_bit_dice(matrix))
    name_similarity = _standardized(_qgram_dice(names, q))
    prior = _prior(
        hamming=matrix.sum(axis=1).astype(np.float64),
        counts=found.counts[chosen],
        names=names,
        public_frequencies=public_frequencies,
        bits=bits,
        hashes=hashes,
        q=q,
        length_weight=length_weight,
        frequency_weight=frequency_weight,
    )
    assignment = _graduated_assignment(
        encoding_similarity, name_similarity, prior, max(iterations, 1)
    )
    guesses = {
        int(encoding): names[int(name)]
        for encoding, name in zip(chosen.tolist(), assignment.tolist(), strict=True)
    }
    return _score("graph_matching", top_k, found, guesses)


def similarity_correlation(
    packed: UInt64Array,
    present: BoolArray,
    plaintext: Sequence[str | None],
    *,
    bits: int,
    q: int = 2,
    nodes: int = 300,
) -> float:
    """Pearson correlation of filter Dice with q-gram Dice over the most frequent encodings.

    This is the premise :func:`graph_matching_attack` runs on, measured directly. It uses the
    plaintext, so it is a diagnostic for the defender rather than something an attacker computes:
    a correlation near zero means the similarity graph carries no name structure to match.
    """
    found = _targets(packed, present, plaintext, nodes)
    if found is None or found.order.size < 3:
        return 0.0
    chosen = found.order[: min(nodes, found.order.size)]
    names = [found.truth.get(int(g)) or "" for g in chosen.tolist()]
    filters = _bit_dice(_unpack(found.distinct, bits)[chosen])
    grams = _qgram_dice(names, q)
    off = ~np.eye(chosen.size, dtype=bool)
    left, right = filters[off], grams[off]
    if left.std() <= 0.0 or right.std() <= 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def all_attacks(
    packed: UInt64Array,
    present: BoolArray,
    plaintext: Sequence[str | None],
    public_frequencies: Mapping[str, float],
    *,
    bits: int,
    hashes: int,
    q: int = 2,
    top_k: int = 100,
    salt_frequencies: Mapping[str, float] | None = None,
) -> list[AttackReport]:
    """Every implemented attack on one encoded name field (frequency alignment, graph matching)."""
    reports = frequency_alignment_attack(
        packed,
        present,
        plaintext,
        public_frequencies,
        bits=bits,
        hashes=hashes,
        q=q,
        top_k=top_k,
        salt_frequencies=salt_frequencies,
    )
    reports.append(
        graph_matching_attack(
            packed,
            present,
            plaintext,
            public_frequencies,
            bits=bits,
            hashes=hashes,
            q=q,
            top_k=top_k,
        )
    )
    return reports
