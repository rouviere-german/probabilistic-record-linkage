from __future__ import annotations

import json

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from prometheus_client import generate_latest

from probabilistic_record_linkage.linkage import attack
from probabilistic_record_linkage.linkage.attack import (
    all_attacks,
    frequency_alignment_attack,
    graph_matching_attack,
    qgram_count,
    qgrams,
)
from probabilistic_record_linkage.linkage.comparison import PLAIN_SPACE
from probabilistic_record_linkage.linkage.monitoring import (
    DriftStateError,
    PatternDriftMonitor,
    categorical_psi,
    create_linkage_metrics,
)
from probabilistic_record_linkage.linkage.vocab import World

pytestmark = pytest.mark.unit


def _encodings(labels: list[int], weights: dict[int, int] | None = None) -> npt.NDArray[np.uint64]:
    rows = np.zeros((len(labels), 1), dtype=np.uint64)
    for i, label in enumerate(labels):
        rows[i, 0] = np.uint64((1 << weights[label]) - 1) if weights else np.uint64(label + 1)
    return rows


def _population(names: list[str], counts: list[int]) -> tuple[list[str], list[int]]:
    plaintext = [name for name, count in zip(names, counts, strict=True) for _ in range(count)]
    labels = [i for i, count in enumerate(counts) for _ in range(count)]
    return plaintext, labels


def test_frequency_alignment_recovers_deterministic_encodings() -> None:
    names = ["anna", "bert", "carl", "dora", "emil"]
    counts = [50, 40, 30, 20, 10]
    plaintext, labels = _population(names, counts)
    public = {name: count / 150 for name, count in zip(names, counts, strict=True)}
    present = np.ones(len(labels), dtype=bool)
    reports = frequency_alignment_attack(
        _encodings(labels), present, plaintext, public, bits=64, hashes=1, top_k=5
    )
    assert reports[0].strategy == "global"
    assert (reports[0].attacked, reports[0].correct, reports[0].rate) == (5, 5, 1.0)
    assert reports[0].as_dict()["rate"] == 1.0
    salted = frequency_alignment_attack(
        _encodings(labels),
        present,
        plaintext,
        public,
        bits=64,
        hashes=1,
        top_k=5,
        salt_frequencies={"1980": 0.5, "1981": 0.5},
    )
    assert [report.strategy for report in salted] == ["global", "length", "salt_aware"]


def test_unique_encodings_defeat_frequency_alignment() -> None:
    names = ["anna", "bert", "carl", "dora", "emil"]
    plaintext, _ = _population(names, [50, 40, 30, 20, 10])
    order = np.random.default_rng(0).permutation(len(plaintext))
    shuffled = [plaintext[i] for i in order]
    public = dict.fromkeys(names, 0.2)
    unique = _encodings(list(range(len(shuffled))))
    reports = frequency_alignment_attack(
        unique, np.ones(len(shuffled), dtype=bool), shuffled, public, bits=64, hashes=1, top_k=5
    )
    assert reports[0].rate <= 0.6


def test_hamming_weight_strata_reveal_name_length() -> None:
    names = ["abc", "abcd", "abcde"]
    assert [qgram_count(name) for name in names] == [4, 5, 6]
    plaintext, _ = _population(["abcd", "abc", "abcde"], [50, 40, 30])
    labels = [{"abc": 0, "abcd": 1, "abcde": 2}[name] for name in plaintext]
    weights = {
        0: 8,
        1: 9,
        2: 11,
    }  # expected set bits for 4, 5 and 6 bigrams with 64 bits and 2 hashes
    public = {"abc": 0.4, "abcd": 0.35, "abcde": 0.25}
    reports = frequency_alignment_attack(
        _encodings(labels, weights),
        np.ones(len(labels), dtype=bool),
        plaintext,
        public,
        bits=64,
        hashes=2,
        top_k=3,
    )
    by_strategy = {report.strategy: report.correct for report in reports}
    assert by_strategy == {"global": 1, "length": 3}


def test_attack_edge_cases() -> None:
    empty = frequency_alignment_attack(
        np.zeros((2, 1), dtype=np.uint64),
        np.zeros(2, dtype=bool),
        [None, None],
        {"a": 1.0},
        bits=64,
        hashes=1,
    )
    assert [(r.strategy, r.attacked, r.rate) for r in empty] == [("global", 0, 0.0)]
    assert qgram_count("ann", q=1) == 2
    nothing = graph_matching_attack(
        np.zeros((2, 1), dtype=np.uint64),
        np.zeros(2, dtype=bool),
        [None, None],
        {"a": 1.0},
        bits=64,
        hashes=1,
    )
    assert (nothing.strategy, nothing.attacked, nothing.rate) == ("graph_matching", 0, 0.0)
    # One candidate name and one distinct encoding leave the graph with no structure at all.
    single = graph_matching_attack(
        np.ones((2, 1), dtype=np.uint64),
        np.ones(2, dtype=bool),
        ["a", "a"],
        {"a": 1.0},
        bits=64,
        hashes=1,
    )
    assert (single.strategy, single.attacked, single.correct) == ("graph_matching", 1, 0)
    assert attack._standardized(np.ones((3, 3))).tolist() == [[0.0] * 3] * 3
    assert attack._standardized(np.zeros((1, 1))).tolist() == [[0.0]]
    assert attack._sinkhorn(np.zeros((2, 2))).tolist() == [[0.25, 0.25], [0.25, 0.25]]
    assert attack._estimated_qgrams(0, 64, 2) == 0
    assert attack._estimated_qgrams(64, 64, 2) == 10**6
    # Graduated assignment falls back to the prior when the graphs carry nothing: either the
    # agreement is flat, or the prior rules out every pair the agreement prefers.
    swap = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    flat = attack._graduated_assignment(swap, np.zeros((2, 2)), np.full((2, 2), 0.25), 3)
    assert flat.tolist() in ([0, 1], [1, 0])
    blocked = attack._graduated_assignment(swap, swap, np.asarray([[1.0, 0.0], [0.0, 0.0]]), 3)
    assert blocked.tolist() == [0, 1]


class _SimulatedBloom:
    """A keyless stand-in for the encoder: every (q-gram, salt) gets ``hashes`` random positions."""

    def __init__(self, bits: int, hashes: int, seed: int) -> None:
        self.bits, self.hashes = bits, hashes
        self.rng = np.random.default_rng(seed)
        self.positions: dict[tuple[str, str], npt.NDArray[np.int64]] = {}

    def positions_of(self, gram: str, salt: str = "") -> npt.NDArray[np.int64]:
        key = (gram, salt)
        if key not in self.positions:
            self.positions[key] = self.rng.integers(0, self.bits, size=self.hashes)
        return self.positions[key]

    def encode(self, names: list[str], salts: list[str]) -> npt.NDArray[np.uint64]:
        rows = np.zeros((len(names), self.bits // 64), dtype=np.uint64)
        for row, (name, salt) in enumerate(zip(names, salts, strict=True)):
            vector = np.zeros(self.bits, dtype=bool)
            for gram in qgrams(name, 2):
                vector[self.positions_of(gram, salt)] = True
            rows[row] = np.frombuffer(np.packbits(vector).tobytes(), dtype=">u8").astype(np.uint64)
        return rows


def test_unpacking_follows_the_encoder_bit_order() -> None:
    vector = np.zeros(128, dtype=bool)
    vector[[0, 7, 8, 63, 64, 127]] = True
    packed = np.frombuffer(np.packbits(vector).tobytes(), dtype=">u8").astype(np.uint64)
    unpacked = attack._unpack(packed.reshape(1, -1), 128)
    assert np.array_equal(unpacked[0], vector)
    assert attack._unpack(packed.reshape(1, -1), 70).shape == (1, 70)


def test_graph_matching_beats_frequency_alignment_on_unsalted_encodings(world: World) -> None:
    public = world.first_name_frequencies()
    names = sorted(public)
    probabilities = np.asarray([public[name] for name in names])
    rng = np.random.default_rng(3)
    plaintext = [
        names[i] for i in rng.choice(len(names), size=8_000, p=probabilities / probabilities.sum())
    ]
    bloom = _SimulatedBloom(bits=512, hashes=12, seed=5)
    present = np.ones(len(plaintext), dtype=bool)

    unsalted = bloom.encode(plaintext, [""] * len(plaintext))
    reports = all_attacks(unsalted, present, plaintext, public, bits=512, hashes=12)
    assert [report.strategy for report in reports] == ["global", "length", "graph_matching"]
    by_strategy = {report.strategy: report.rate for report in reports}
    assert by_strategy["graph_matching"] >= 0.3
    # The similarity graph carries far more than the frequency ranking and the length strata.
    assert by_strategy["graph_matching"] >= 2 * max(by_strategy["global"], by_strategy["length"])
    report = graph_matching_attack(unsalted, present, plaintext, public, bits=512, hashes=12)
    assert (report.strategy, report.attacked) == ("graph_matching", 100)
    assert report.correct == reports[-1].correct

    # A per-record salt makes every encoding unique and its similarities meaningless: the graph
    # carries no name structure and the attacker is left with the prior.
    salted = bloom.encode(plaintext, [str(i) for i in range(len(plaintext))])
    assert graph_matching_attack(salted, present, plaintext, public, bits=512, hashes=12).rate <= (
        0.05
    )


@given(st.lists(st.integers(1, 50), min_size=2, max_size=10))
def test_psi_is_zero_for_identical_distributions_and_never_negative(counts: list[int]) -> None:
    expected = np.asarray(counts, dtype=np.float64)
    assert categorical_psi(expected, expected) == pytest.approx(0.0, abs=1e-12)
    assert categorical_psi(expected, np.roll(expected, 1)) >= 0.0


def test_psi_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="same length"):
        categorical_psi(np.ones(3), np.ones(2))
    with pytest.raises(ValueError, match="positive mass"):
        categorical_psi(np.zeros(3), np.ones(3))


def _levels(rng: np.random.Generator, n: int, exact_share: float) -> npt.NDArray[np.int8]:
    columns = []
    for spec in PLAIN_SPACE.fields:
        exact = rng.random(n) < exact_share
        other = rng.integers(1, spec.exact_level, size=n)
        columns.append(np.where(exact, spec.exact_level, other))
    return np.stack(columns, axis=1).astype(np.int8)


def test_pattern_drift_monitor_detects_shifts_and_round_trips() -> None:
    rng = np.random.default_rng(0)
    reference = PLAIN_SPACE.encode(_levels(rng, 20_000, 0.9))
    monitor = PatternDriftMonitor(PLAIN_SPACE, reference)
    same = monitor.check(PLAIN_SPACE.encode(_levels(rng, 20_000, 0.9)))
    shifted = monitor.check(PLAIN_SPACE.encode(_levels(rng, 20_000, 0.6)))
    assert not same.drifted
    assert same.psi < 0.1
    assert shifted.drifted
    assert set(shifted.field_psi) == set(PLAIN_SPACE.names)
    assert shifted.as_dict()["n_pairs"] == 20_000
    restored = PatternDriftMonitor.from_json(PLAIN_SPACE, json.loads(json.dumps(monitor.to_json())))
    assert restored.check(PLAIN_SPACE.encode(_levels(rng, 5_000, 0.6))).drifted
    with pytest.raises(DriftStateError):
        PatternDriftMonitor.from_json(
            PLAIN_SPACE, {"threshold": "x", "buckets": [], "reference": [], "reference_levels": []}
        )
    with pytest.raises(ValueError, match="does not match"):
        PatternDriftMonitor.from_json(
            PLAIN_SPACE,
            {"threshold": 0.2, "buckets": [1, 2], "reference": [1.0], "reference_levels": []},
        )
    with pytest.raises(ValueError, match="empty"):
        PatternDriftMonitor(PLAIN_SPACE, np.zeros(0, dtype=np.int64))
    with pytest.raises(ValueError, match="no patterns"):
        monitor.check(np.zeros(0, dtype=np.int64))


def test_linkage_metrics_use_the_documented_names() -> None:
    metrics = create_linkage_metrics()
    metrics.blocking_pairs_total.inc(10)
    metrics.reduction_ratio.set(0.999)
    metrics.match_rate.set(0.1)
    metrics.review_queue_depth.set(3)
    metrics.max_cluster_size.set(5)
    metrics.comparison_pattern_psi.set(0.01)
    metrics.em_iterations.set(12)
    metrics.stage_duration_seconds.labels("block").observe(0.5)
    payload = generate_latest(metrics.registry).decode()
    for expected in (
        "blocking_pairs_total 10.0",
        "reduction_ratio 0.999",
        "match_rate 0.1",
        "review_queue_depth 3.0",
        "max_cluster_size 5.0",
        "comparison_pattern_psi 0.01",
        "em_iterations 12.0",
        'linkage_stage_duration_seconds_count{stage="block"} 1.0',
    ):
        assert expected in payload
