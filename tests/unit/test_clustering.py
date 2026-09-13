from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st

from probabilistic_record_linkage.linkage.clustering import (
    STATUS_COMMITTED,
    STATUS_PENDING,
    ClusterConfig,
    Resolution,
    connected_labels,
    find_articulation_points,
    find_bridges,
    resolve_clusters,
)

pytestmark = pytest.mark.unit

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _reach(n: int, edges: list[tuple[int, int]], start: int, removed: int = -1) -> set[int]:
    adjacency: dict[int, list[int]] = {i: [] for i in range(n)}
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    seen, stack = {start}, [start]
    while stack:
        for neighbour in adjacency[stack.pop()]:
            if neighbour != removed and neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return seen


_graphs = st.integers(1, 9).flatmap(
    lambda n: st.tuples(
        st.just(n), st.lists(st.tuples(st.integers(0, n - 1), st.integers(0, n - 1)), max_size=16)
    )
)


@given(_graphs)
def test_bridges_match_a_brute_force_oracle(graph: tuple[int, list[tuple[int, int]]]) -> None:
    n, edges = graph
    bridges, tin, tout = find_bridges(n, edges)
    expected = set()
    for index, (a, b) in enumerate(edges):
        others = [e for i, e in enumerate(edges) if i != index]
        if a != b and b not in _reach(n, others, a):
            expected.add(index)
    assert {index for index, _ in bridges} == expected
    for index, child in bridges:
        others = [e for i, e in enumerate(edges) if i != index]
        subtree = {x for x in range(n) if tin[child] <= tin[x] <= tout[child]}
        assert _reach(n, others, child) == subtree


@given(_graphs)
def test_articulation_points_match_a_brute_force_oracle(
    graph: tuple[int, list[tuple[int, int]]],
) -> None:
    n, edges = graph
    expected = []
    for vertex in range(n):
        neighbours = sorted(
            {b for a, b in edges if a == vertex and b != vertex}
            | {a for a, b in edges if b == vertex and a != vertex}
        )
        if len(neighbours) < 2:
            continue
        reached = _reach(n, edges, neighbours[0], removed=vertex)
        if any(other not in reached for other in neighbours[1:]):
            expected.append(vertex)
    assert find_articulation_points(n, edges) == expected


def _pairs(n: int) -> st.SearchStrategy[list[tuple[int, int]]]:
    pair = st.tuples(st.integers(0, n - 1), st.integers(0, n - 1)).filter(lambda t: t[0] != t[1])
    return st.lists(pair, max_size=30, unique_by=lambda t: (min(t), max(t)))


@given(st.integers(2, 12), st.data())
def test_resolution_is_a_partition_that_respects_constraints(n: int, data: st.DataObject) -> None:
    pairs = data.draw(_pairs(n))
    probabilities = np.asarray(
        data.draw(st.lists(st.floats(0, 1), min_size=len(pairs), max_size=len(pairs))),
        dtype=np.float64,
    )
    left = np.asarray([min(p) for p in pairs], dtype=np.int64)
    right = np.asarray([max(p) for p in pairs], dtype=np.int64)
    cannot = {(min(p), max(p)) for p in data.draw(_pairs(n))[:3]}
    must = [p for p in data.draw(_pairs(n))[:3] if (min(p), max(p)) not in cannot]
    config = ClusterConfig(size_cap=4)

    def oracle(a: IntArray, b: IntArray) -> FloatArray:
        return np.full(a.size, 0.5)

    result = resolve_clusters(
        n, left, right, probabilities, match_threshold=0.5, config=config,
        must_link=must, cannot_link=cannot, conflict_oracle=oracle,
    )  # fmt: skip
    unguarded = resolve_clusters(
        n, left, right, probabilities, match_threshold=0.5, config=ClusterConfig(guard=False),
        must_link=must, cannot_link=cannot,
    ).labels  # fmt: skip
    labels = result.labels
    for cluster in np.unique(labels):
        members = np.flatnonzero(labels == cluster)
        assert cluster == members.min()
        assert np.unique(unguarded[members]).size == 1  # the guard only refines components
        if members.size > config.size_cap:
            assert result.pending[members].all()
    for a, b in must:
        assert labels[a] == labels[b]
    for a, b in cannot:
        if labels[a] == labels[b]:
            assert result.pending[a]
            assert result.pending[b]


def _run(
    n: int, edges: list[tuple[int, int, float]], **kwargs: Any
) -> tuple[list[set[int]], Resolution]:
    left = np.asarray([e[0] for e in edges], dtype=np.int64)
    right = np.asarray([e[1] for e in edges], dtype=np.int64)
    probability = np.asarray([e[2] for e in edges], dtype=np.float64)
    result = resolve_clusters(n, left, right, probability, match_threshold=0.5, **kwargs)
    groups: dict[int, set[int]] = {}
    for row, label in enumerate(result.labels.tolist()):
        groups.setdefault(label, set()).add(row)
    return sorted(groups.values(), key=min), result


def test_a_bridge_record_between_two_groups_is_detached() -> None:
    edges = [
        (0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0),  # group A
        (4, 5, 1.0), (4, 6, 1.0), (5, 6, 1.0),  # group B
        (3, 0, 0.99), (3, 1, 0.99), (3, 2, 0.99),  # bridge record 3 -> A
        (3, 4, 0.995), (3, 5, 0.995),  # bridge record 3 -> B (stronger)
    ]  # fmt: skip
    groups, result = _run(7, edges)
    assert groups == [{0, 1, 2}, {3, 4, 5, 6}]
    assert [edge.reason for edge in result.removed_edges] == ["bridge_record"] * 3
    unguarded, _ = _run(7, edges, config=ClusterConfig(guard=False))
    assert unguarded == [set(range(7))]


def test_a_chain_of_bridge_records_is_split_into_pairs_of_identities() -> None:
    def clique(nodes: list[int]) -> list[tuple[int, int, float]]:
        return [(a, b, 1.0) for i, a in enumerate(nodes) for b in nodes[i + 1 :]]

    edges = [*clique([0, 1]), *clique([3, 4]), *clique([6, 7])]
    edges += [
        (2, 0, 0.99),
        (2, 1, 0.99),
        (2, 3, 0.999),
        (2, 4, 0.999),
    ]  # record 2 bridges 0-1 and 3-4
    edges += [
        (5, 3, 0.99),
        (5, 4, 0.99),
        (5, 6, 0.999),
        (5, 7, 0.999),
    ]  # record 5 bridges 3-4 and 6-7
    groups, _ = _run(8, edges)
    assert groups == [{0, 1}, {2, 3, 4}, {5, 6, 7}]


def test_dense_clusters_joined_by_weak_edges_stay_together() -> None:
    edges = [(a, b, 0.99) for a in range(4) for b in range(a + 1, 4)]
    groups, result = _run(4, edges)
    assert groups == [{0, 1, 2, 3}]
    assert not result.pending.any()


def test_weak_single_record_links_are_cut() -> None:
    edges = [(0, 1, 1.0), (1, 2, 0.9), (2, 3, 1.0)]
    groups, result = _run(4, edges)
    assert groups == [{0, 1}, {2, 3}]
    assert [edge.reason for edge in result.removed_edges] == ["weak_bridge"]


def test_groups_joined_only_by_weak_links_are_split() -> None:
    edges = [
        (0, 1, 1.0), (2, 3, 1.0), (0, 2, 0.6), (1, 3, 0.6),  # two households joined weakly
        (4, 0, 0.7),  # a lone record with only a weak link joins its strongest neighbour
    ]  # fmt: skip
    groups, result = _run(5, edges)
    assert groups == [{0, 1, 4}, {2, 3}]
    assert [edge.reason for edge in result.removed_edges] == ["weak_link"] * 2


def test_cluster_probabilities_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="weak_probability"):
        ClusterConfig(weak_probability=0.99999, strong_probability=0.9999)


def test_conflicting_records_are_separated_at_the_weakest_bridge() -> None:
    def oracle(a: IntArray, b: IntArray) -> FloatArray:
        return np.where((a == 0) & (b == 2), 1e-6, 0.5)

    groups, result = _run(3, [(0, 1, 0.9999), (1, 2, 0.9)], conflict_oracle=oracle)
    assert groups == [{0, 1}, {2}]
    assert [edge.reason for edge in result.removed_edges] == ["conflict"]


def test_cannot_links_inside_a_strong_core_are_sent_to_review() -> None:
    edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
    groups, result = _run(3, edges, cannot_link=[(2, 0)])
    assert groups == [{0, 1, 2}]
    assert result.pending.all()
    assert result.reasons == {0: ("unresolved_conflict",)}
    assert result.status_of(0) == STATUS_PENDING


def test_oversized_clusters_are_cut_or_marked_pending() -> None:
    clique = [(a, b, 1.0) for a in range(5) for b in range(a + 1, 5)]
    groups, result = _run(5, clique, config=ClusterConfig(size_cap=3))
    assert groups == [set(range(5))]
    assert result.reasons == {0: ("size_cap",)}
    weak_path = [(0, 1, 0.9), (1, 2, 0.6), (2, 3, 0.9), (3, 4, 0.9)]
    groups, result = _run(5, weak_path, config=ClusterConfig(size_cap=3))
    assert groups == [{0, 1}, {2, 3, 4}]
    assert not result.pending.any()
    assert result.max_cluster_size == 3
    solid_path = [(i, i + 1, 0.95) for i in range(5)]
    groups, result = _run(6, solid_path, config=ClusterConfig(size_cap=3))
    assert max(len(group) for group in groups) <= 3
    assert not result.pending.any()


def test_must_links_join_records_and_contradictions_are_rejected() -> None:
    groups, result = _run(3, [], must_link=[(0, 2)])
    assert groups == [{0, 2}, {1}]
    assert result.status_of(0) == STATUS_COMMITTED
    with pytest.raises(ValueError, match="both must-link and cannot-link"):
        _run(3, [], must_link=[(0, 1)], cannot_link=[(1, 0)])


def test_edge_masks_replace_the_probability_threshold() -> None:
    left = np.asarray([0, 1], dtype=np.int64)
    right = np.asarray([1, 2], dtype=np.int64)
    probability = np.asarray([0.99, 0.99], dtype=np.float64)
    mask = np.asarray([True, False], dtype=np.bool_)
    result = resolve_clusters(3, left, right, probability, match_threshold=0.5, edge_mask=mask)
    assert result.labels.tolist() == [0, 0, 2]
    assert result.n_clusters == 2
    with pytest.raises(ValueError, match="edge_mask"):
        resolve_clusters(3, left, right, probability, match_threshold=0.5, edge_mask=mask[:1])


def test_connected_labels_use_the_smallest_row() -> None:
    labels = connected_labels(
        4, np.asarray([3, 1], dtype=np.int64), np.asarray([1, 0], dtype=np.int64)
    )
    assert labels.tolist() == [0, 0, 2, 0]
    assert connected_labels(0, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)).size == 0
