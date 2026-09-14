"""Constraint-aware cluster resolution with an over-merge guard (M4).

1. Match edges plus reviewer must-links form a graph; reviewer cannot-link pairs are removed from
   it. Connected components (scipy) are the initial clusters.
2. Every component with three or more records (or containing a cannot-link) is checked, and the
   first applicable rule splits it; the parts are checked again. *Solid* edges are must-links and
   edges with probability >= ``weak_probability``; the others are weak links.

   - *weak cohesion*: if the solid edges alone leave two or more groups of at least
     ``min_side_records`` records, the component was held together only by weak links (typically
     household members who share a surname and an address). It is split into those groups; groups
     smaller than ``min_side_records`` join the group they are most strongly linked to.
   - *bridge record* (articulation point of the solid graph, Tarjan 1972): if removing one record
     splits the rest into at least two groups of ``min_side_records`` or more records, only that
     record holds them together, which is the signature of a record that copies fields from two
     people (``generator.make_bridge_records``). The record stays with the group it is most
     strongly linked to and its links to the other groups are cut.
   - *bridge edge* of the graph condensed by strong edges (probability >= ``strong_probability``
     or a must-link): an edge whose removal separates a *conflicting* pair (a reviewer cannot-link,
     or two records whose own match probability is at most ``conflict_probability``), an edge in a
     component above ``size_cap``, or a link through a single record between two groups of at
     least ``min_side_records`` records. The lowest-probability such edge is cut.

3. A cluster that still exceeds ``size_cap`` or still contains a conflict is never auto-committed:
   its status is ``pending_review``. Must-links are never cut. Every removed edge is reported so it
   can be queued for review.

The guard's work is bounded so that adversarial inputs (long chains, stars, ladders of matching
records) cannot stall a job: a component with more than ``max_check_size`` records is held for
review as a whole (reason ``too_large_to_check``) instead of being split, and each split pass is
linear in the component's size (one depth-first search gives every articulation point together
with the sizes of the parts it separates). Cannot-link pairs are respected even with the guard
switched off: a component that contains one is ``pending_review``, never auto-committed.

Cluster ids are the smallest row index in the cluster, so a resolution is deterministic.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

__all__ = [
    "STATUS_COMMITTED",
    "STATUS_PENDING",
    "ClusterConfig",
    "RemovedEdge",
    "Resolution",
    "connected_labels",
    "find_articulation_points",
    "find_bridges",
    "resolve_clusters",
]

Int64Array = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
ConflictOracle = Callable[[Int64Array, Int64Array], FloatArray]
LocalEdge = tuple[int, int, float, bool]

STATUS_COMMITTED = "committed"
STATUS_PENDING = "pending_review"


class ClusterConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    size_cap: int = Field(default=8, ge=2)
    strong_probability: float = Field(default=0.9999, gt=0.0, le=1.0)
    weak_probability: float = Field(default=0.9, gt=0.0, lt=1.0)
    conflict_probability: float = Field(default=1e-3, ge=0.0, lt=1.0)
    min_side_records: int = Field(default=2, ge=1)
    max_check_size: int = Field(default=200, ge=3)
    guard: bool = True

    @model_validator(mode="after")
    def _ordered_probabilities(self) -> ClusterConfig:
        if not self.conflict_probability < self.weak_probability <= self.strong_probability:
            raise ValueError("need conflict_probability < weak_probability <= strong_probability")
        return self


@dataclass(frozen=True, slots=True)
class RemovedEdge:
    left: int
    right: int
    probability: float
    reason: str


@dataclass(frozen=True, slots=True)
class Resolution:
    labels: Int64Array
    pending: BoolArray  # per record: its cluster awaits human review
    reasons: dict[int, tuple[str, ...]] = field(default_factory=dict)  # cluster id -> reasons
    removed_edges: tuple[RemovedEdge, ...] = ()
    components_checked: int = 0

    @property
    def n_clusters(self) -> int:
        return int(np.unique(self.labels).size)

    @property
    def max_cluster_size(self) -> int:
        return int(np.bincount(self.labels).max()) if self.labels.size else 0

    def status_of(self, cluster: int) -> str:
        return STATUS_PENDING if cluster in self.reasons else STATUS_COMMITTED


def connected_labels(n: int, left: Int64Array, right: Int64Array) -> Int64Array:
    """Component label per node (smallest node index in the component)."""
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    graph = coo_matrix((np.ones(left.size, dtype=np.int8), (left, right)), shape=(n, n)).tocsr()
    _, labels = connected_components(graph, directed=False)
    labels = labels.astype(np.int64)
    first = np.full(int(labels.max()) + 1, n, dtype=np.int64)
    np.minimum.at(first, labels, np.arange(n, dtype=np.int64))
    result: Int64Array = first[labels]
    return result


def _adjacency(n: int, edges: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for index, (a, b) in enumerate(edges):
        if a != b:
            adjacency[a].append((b, index))
            adjacency[b].append((a, index))
    return adjacency


@dataclass(slots=True)
class _Dfs:
    tin: list[int]
    low: list[int]
    tout: list[int]
    parent: list[int]  # DFS tree parent, -1 for roots
    root: list[int]  # DFS root of the node's component
    bridges: list[tuple[int, int]]  # (edge index, child)
    articulation: set[int]

    def subtree_size(self, node: int) -> int:
        return self.tout[node] - self.tin[node] + 1

    def in_subtree(self, node: int, ancestor: int) -> bool:
        return self.tin[ancestor] <= self.tin[node] <= self.tout[ancestor]


def _tarjan(n: int, edges: list[tuple[int, int]]) -> _Dfs:
    """One iterative DFS computing bridges and articulation points of a multigraph."""
    adjacency = _adjacency(n, edges)
    state = _Dfs([-1] * n, [0] * n, [0] * n, [-1] * n, [-1] * n, [], set())
    tin, low = state.tin, state.low
    timer = 0
    for root in range(n):
        if tin[root] != -1:
            continue
        tin[root] = low[root] = timer
        state.root[root] = root
        timer += 1
        root_children = 0
        stack: list[tuple[int, int, int]] = [(root, -1, 0)]
        while stack:
            node, parent_edge, cursor = stack[-1]
            neighbours = adjacency[node]
            if cursor < len(neighbours):
                stack[-1] = (node, parent_edge, cursor + 1)
                neighbour, edge_index = neighbours[cursor]
                if edge_index == parent_edge:
                    continue
                if tin[neighbour] == -1:
                    tin[neighbour] = low[neighbour] = timer
                    state.parent[neighbour] = node
                    state.root[neighbour] = root
                    timer += 1
                    root_children += node == root
                    stack.append((neighbour, edge_index, 0))
                else:
                    low[node] = min(low[node], tin[neighbour])
                continue
            stack.pop()
            state.tout[node] = timer - 1
            if stack:
                parent = stack[-1][0]
                low[parent] = min(low[parent], low[node])
                if low[node] > tin[parent]:
                    state.bridges.append((parent_edge, node))
                if parent != root and low[node] >= tin[parent]:
                    state.articulation.add(parent)
        if root_children > 1:
            state.articulation.add(root)
    return state


def find_bridges(
    n: int, edges: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Bridges of an undirected multigraph (iterative Tarjan).

    Returns ``(bridges, tin, tout)`` where each bridge is ``(edge index, child node)``: removing
    the edge disconnects the DFS subtree of ``child``, i.e. the nodes ``x`` with
    ``tin[child] <= tin[x] <= tout[child]``. Parallel edges are never bridges.
    """
    state = _tarjan(n, edges)
    return state.bridges, state.tin, state.tout


def find_articulation_points(n: int, edges: list[tuple[int, int]]) -> list[int]:
    """Vertices whose removal increases the number of connected components."""
    return sorted(_tarjan(n, edges).articulation)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


@dataclass(slots=True)
class _Component:
    nodes: list[int]  # global row ids, sorted
    edges: list[LocalEdge]  # local a, local b, probability, must-link
    conflicts: set[tuple[int, int]]  # local pairs


@dataclass(frozen=True, slots=True)
class _Condensed:
    core_of: dict[int, int]  # local node -> core index
    core_size: list[int]
    keys: list[tuple[int, int]]  # condensed edges, (core a, core b) with a < b
    cut_edges: dict[tuple[int, int], list[tuple[int, int, float]]]


@dataclass(frozen=True, slots=True)
class _Cut:
    weight: float
    key: tuple[int, int]
    reason: str


@dataclass(frozen=True, slots=True)
class _Split:
    parts: list[list[int]]
    removed: list[tuple[int, int, float, str]]


def _components(nodes: list[int], edges: list[tuple[int, int]]) -> list[frozenset[int]]:
    adjacency: dict[int, list[int]] = {node: [] for node in nodes}
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    seen: set[int] = set()
    parts: list[frozenset[int]] = []
    for start in nodes:
        if start in seen:
            continue
        group = {start}
        stack = [start]
        while stack:
            for neighbour in adjacency[stack.pop()]:
                if neighbour not in group:
                    group.add(neighbour)
                    stack.append(neighbour)
        seen |= group
        parts.append(frozenset(group))
    return parts


def _is_solid(edge: LocalEdge, config: ClusterConfig) -> bool:
    return edge[3] or edge[2] >= config.weak_probability


def _finish_split(
    subset: list[int],
    edges: list[LocalEdge],
    anchors: list[frozenset[int]],
    reason: str,
) -> _Split:
    """Assign every node to an anchor group (small leftovers join their strongest neighbour)."""
    assignment: dict[int, int] = {node: i for i, group in enumerate(anchors) for node in group}
    unassigned = [node for node in subset if node not in assignment]
    while unassigned:
        best: tuple[float, int, int] | None = None  # (probability, -anchor, node)
        for a, b, probability, _ in edges:
            for node, other in ((a, b), (b, a)):
                if node in assignment or other not in assignment:
                    continue
                candidate = (probability, -assignment[other], node)
                if best is None or candidate > best:
                    best = candidate
        if best is None:  # not reachable through edges; keep with the first anchor
            assignment[unassigned[0]] = 0
        else:
            assignment[best[2]] = -best[1]
        unassigned = [node for node in subset if node not in assignment]
    parts = [sorted(node for node in subset if assignment[node] == i) for i in range(len(anchors))]
    removed = [
        (a, b, probability, reason)
        for a, b, probability, _ in edges
        if assignment[a] != assignment[b]
    ]
    return _Split([part for part in parts if part], removed)


def _weak_cohesion_split(
    subset: list[int], edges: list[LocalEdge], config: ClusterConfig
) -> _Split | None:
    if len(subset) < 2 * config.min_side_records:
        return None
    solid = [(a, b) for a, b, p, must in edges if _is_solid((a, b, p, must), config)]
    groups = _components(subset, solid)
    anchors = [group for group in groups if len(group) >= config.min_side_records]
    if len(anchors) < 2:
        return None
    return _finish_split(subset, edges, sorted(anchors, key=min), "weak_link")


_REST = -1  # the part of a component that stays connected to the DFS root after a cut


def _separated_parts(dfs: _Dfs, children: list[list[int]], point: int) -> list[int]:
    """Parts left by removing ``point``: DFS children whose subtree it separates, plus ``_REST``.

    In a DFS tree of an undirected graph every non-tree edge joins a node to an ancestor, so
    removing ``point`` disconnects exactly the subtrees of its children ``c`` with
    ``low[c] >= tin[point]`` (all of them for a root); everything else stays with the root.
    """
    parts = [child for child in children[point] if dfs.low[child] >= dfs.tin[point]]
    if dfs.parent[point] != -1:
        parts.append(_REST)
    return parts


def _part_of(dfs: _Dfs, point: int, separated: list[int], node: int) -> int:
    """Which part of ``_separated_parts`` a neighbour of ``point`` falls into."""
    for child in separated:
        if child != _REST and dfs.in_subtree(node, child):
            return child
    return _REST


def _bridge_record_split(
    subset: list[int], edges: list[LocalEdge], config: ClusterConfig
) -> _Split | None:
    """Detach the articulation record (of the solid graph) whose cut links are weakest.

    One depth-first search (:func:`_tarjan`) gives every articulation point together with the
    parts its removal leaves, sized from DFS intervals, so the whole pass is linear in the size
    of the component instead of recomputing components once per articulation point.
    """
    if len(subset) < 2 * config.min_side_records + 1:
        return None
    solid = [edge for edge in edges if _is_solid(edge, config)]
    position = {node: i for i, node in enumerate(subset)}
    simple = [(position[a], position[b]) for a, b, _, _ in solid]
    n_local = len(subset)
    dfs = _tarjan(n_local, simple)
    children: list[list[int]] = [[] for _ in range(n_local)]
    for node in range(n_local):
        if dfs.parent[node] != -1:
            children[dfs.parent[node]].append(node)
    incident: list[list[tuple[int, float, bool]]] = [[] for _ in range(n_local)]
    for a, b, probability, must in solid:
        incident[position[a]].append((position[b], probability, must))
        incident[position[b]].append((position[a], probability, must))
    best: tuple[float, int, list[int], int, dict[int, tuple[float, bool]]] | None = None
    for point in sorted(dfs.articulation):
        separated = _separated_parts(dfs, children, point)
        component_size = dfs.subtree_size(dfs.root[point])
        sizes = {
            part: dfs.subtree_size(part) if part != _REST else 0 for part in separated
        }
        if _REST in sizes:
            sizes[_REST] = component_size - 1 - sum(
                size for part, size in sizes.items() if part != _REST
            )
        big = [part for part in separated if sizes[part] >= config.min_side_records]
        if len(big) < 2:
            continue
        attachment: dict[int, tuple[float, bool]] = {}
        for other, probability, must in incident[point]:
            part = _part_of(dfs, point, separated, other)
            if part not in big:
                continue
            strongest, has_must = attachment.get(part, (0.0, False))
            attachment[part] = (max(strongest, probability), has_must or must)
        must_parts = [part for part, (_, must) in attachment.items() if must]
        if len(must_parts) > 1 or len(attachment) < 2:
            continue
        strengths = sorted((strength for strength, _ in attachment.values()), reverse=True)
        if must_parts:
            weight = max(
                strength for part, (strength, _) in attachment.items() if part != must_parts[0]
            )
        else:
            weight = strengths[1] if len(attachment) > 1 else strengths[0]
        vertex = subset[point]
        if best is None or (weight, vertex) < best[:2]:
            best = (weight, vertex, big, point, attachment)
    if best is None:
        return None
    _, _, big, point, attachment = best
    order = sorted(range(n_local), key=lambda node: dfs.tin[node])
    members: dict[int, set[int]] = {}
    root = dfs.root[point]
    component = set(order[dfs.tin[root] : dfs.tout[root] + 1])
    component.discard(point)
    for part in big:
        if part != _REST:
            members[part] = set(order[dfs.tin[part] : dfs.tout[part] + 1])
            component -= members[part]
    for part in _separated_parts(dfs, children, point):
        if part != _REST and part not in members:
            component -= set(order[dfs.tin[part] : dfs.tout[part] + 1])
    if _REST in big:
        members[_REST] = component
    groups = {part: frozenset(subset[node] for node in nodes) for part, nodes in members.items()}
    must_parts = [part for part, (_, must) in attachment.items() if must]
    keep = (
        must_parts[0]
        if must_parts
        else max(attachment, key=lambda part: (attachment[part][0], -min(groups[part])))
    )
    anchors = [groups[keep] | {subset[point]}, *(group for part, group in groups.items() if part != keep)]
    return _finish_split(subset, edges, sorted(anchors, key=min), "bridge_record")


def _condense(
    subset: list[int], edges: list[LocalEdge], n_local: int, strong_probability: float
) -> _Condensed:
    union_find = _UnionFind(n_local)
    for a, b, probability, must in edges:
        if must or probability >= strong_probability:
            union_find.union(a, b)
    roots = sorted({union_find.find(x) for x in subset})
    index = {root: i for i, root in enumerate(roots)}
    core_of = {x: index[union_find.find(x)] for x in subset}
    core_size = [0] * len(roots)
    for x in subset:
        core_size[core_of[x]] += 1
    cut_edges: dict[tuple[int, int], list[tuple[int, int, float]]] = {}
    for a, b, probability, _must in edges:
        ca, cb = core_of[a], core_of[b]
        if ca != cb:
            cut_edges.setdefault((min(ca, cb), max(ca, cb)), []).append((a, b, probability))
    return _Condensed(core_of, core_size, sorted(cut_edges), cut_edges)


def _choose_cut(
    subset_size: int,
    condensed: _Condensed,
    conflicts: list[tuple[int, int]],
    *,
    oversize: bool,
    config: ClusterConfig,
) -> _Cut | None:
    """The lowest-probability removable bridge of the condensed graph, if any.

    Linear in the condensed graph: the records on the far side of a bridge are a DFS subtree,
    so their count is a prefix-sum difference, and whether a bridge separates a conflicting pair
    is checked on the pair's DFS numbers. Cuts forced by the size cap never need the conflict
    check except to name the reason of the chosen cut.
    """
    n_cores = len(condensed.core_size)
    bridges, tin, tout = find_bridges(n_cores, condensed.keys)
    size_by_tin = np.zeros(n_cores + 1, dtype=np.int64)
    for core, size in enumerate(condensed.core_size):
        size_by_tin[tin[core] + 1] = size
    prefix = np.cumsum(size_by_tin)
    conflict_tin = np.asarray(
        [(tin[condensed.core_of[x]], tin[condensed.core_of[y]]) for x, y in conflicts],
        dtype=np.int64,
    ).reshape(-1, 2)

    def separates(child: int) -> bool:
        if conflict_tin.size == 0:
            return False
        lo, hi = tin[child], tout[child]
        inside = (conflict_tin >= lo) & (conflict_tin <= hi)
        return bool(np.any(inside[:, 0] != inside[:, 1]))

    best: _Cut | None = None
    for edge_index, child in bridges:
        key = condensed.keys[edge_index]
        members = condensed.cut_edges[key]
        weight = max(probability for _, _, probability in members)
        side = int(prefix[tout[child] + 1] - prefix[tin[child]])
        first_side = {a if condensed.core_of[a] == key[0] else b for a, b, _ in members}
        second_side = {b if condensed.core_of[a] == key[0] else a for a, b, _ in members}
        weak_link = (
            weight < config.strong_probability
            and min(len(first_side), len(second_side)) == 1
            and min(side, subset_size - side) >= config.min_side_records
        )
        if not (oversize or weak_link or separates(child)):
            continue
        if best is None or (weight, key) < (best.weight, best.key):
            reason = (
                "conflict" if separates(child) else ("size_cap" if oversize else "weak_bridge")
            )
            best = _Cut(weight, key, reason)
    return best


def _split_component(
    comp: _Component, config: ClusterConfig
) -> tuple[list[tuple[list[int], tuple[str, ...]]], list[tuple[int, int, float, str]]]:
    """Return ``(parts, removed)``: parts as (local node ids, pending reasons)."""
    results: list[tuple[list[int], tuple[str, ...]]] = []
    removed: list[tuple[int, int, float, str]] = []
    if len(comp.nodes) > config.max_check_size:
        # Held for review as a whole: splitting work is bounded by max_check_size per component.
        reasons = ["unresolved_conflict"] if comp.conflicts else []
        if len(comp.nodes) > config.size_cap:
            reasons.append("size_cap")
        reasons.append("too_large_to_check")
        return [(list(range(len(comp.nodes))), tuple(reasons))], removed
    work: list[list[int]] = [list(range(len(comp.nodes)))]
    while work:
        subset = work.pop()
        members = set(subset)
        edges = [e for e in comp.edges if e[0] in members and e[1] in members]
        conflicts = [c for c in comp.conflicts if c[0] in members and c[1] in members]
        oversize = len(subset) > config.size_cap
        if len(subset) < 3 and not conflicts:
            results.append((subset, ()))
            continue
        split = _weak_cohesion_split(subset, edges, config) or _bridge_record_split(
            subset, edges, config
        )
        if split is not None:
            removed.extend(split.removed)
            work.extend(split.parts)
            continue
        condensed = _condense(subset, edges, len(comp.nodes), config.strong_probability)
        inside_core = any(condensed.core_of[x] == condensed.core_of[y] for x, y in conflicts)
        cut = (
            None
            if inside_core
            else _choose_cut(len(subset), condensed, conflicts, oversize=oversize, config=config)
        )
        if cut is not None:
            cut_members = condensed.cut_edges[cut.key]
            removed.extend((a, b, probability, cut.reason) for a, b, probability in cut_members)
            cut_pairs = {(a, b) for a, b, _ in cut_members}
            remaining = [(a, b) for a, b, _, _ in edges if (a, b) not in cut_pairs]
            work.extend(sorted(part) for part in _components(subset, remaining))
            continue
        reasons: list[str] = []
        if conflicts:
            reasons.append("unresolved_conflict")
        if oversize:
            reasons.append("size_cap")
        results.append((subset, tuple(reasons)))
    return results, removed


def _pair_set(pairs: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(min(a, b), max(a, b)) for a, b in pairs if a != b}


@dataclass(frozen=True, slots=True)
class _Edges:
    left: Int64Array
    right: Int64Array
    probability: FloatArray
    must: BoolArray


def _edge_list(
    n: int,
    left: Int64Array,
    right: Int64Array,
    probability: FloatArray,
    *,
    must: set[tuple[int, int]],
    cannot: set[tuple[int, int]],
) -> _Edges:
    edge_left = left.astype(np.int64)
    edge_right = right.astype(np.int64)
    edge_prob = probability.astype(np.float64)
    if cannot:
        lo, hi = np.minimum(edge_left, edge_right), np.maximum(edge_left, edge_right)
        banned = np.asarray([a * n + b for a, b in cannot], dtype=np.int64)
        allowed = ~np.isin(lo * np.int64(n) + hi, banned)
        edge_left, edge_right, edge_prob = (
            edge_left[allowed],
            edge_right[allowed],
            edge_prob[allowed],
        )
    must_flag = np.zeros(edge_left.size, dtype=bool)
    if must:
        pairs = np.asarray(sorted(must), dtype=np.int64)
        edge_left = np.concatenate([edge_left, pairs[:, 0]])
        edge_right = np.concatenate([edge_right, pairs[:, 1]])
        edge_prob = np.concatenate([edge_prob, np.ones(len(must))])
        must_flag = np.concatenate([must_flag, np.ones(len(must), dtype=bool)])
    return _Edges(edge_left, edge_right, edge_prob, must_flag)


def _components_to_check(
    labels: Int64Array, edges: _Edges, cannot: set[tuple[int, int]]
) -> dict[int, _Component]:
    sizes = np.bincount(labels, minlength=labels.size)
    cannot_by_label: dict[int, list[tuple[int, int]]] = {}
    for a, b in sorted(cannot):
        if labels[a] == labels[b]:
            cannot_by_label.setdefault(int(labels[a]), []).append((a, b))
    check = sorted({int(c) for c in np.flatnonzero(sizes >= 3)} | set(cannot_by_label))
    if not check:
        return {}
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    comps: dict[int, _Component] = {}
    local_index: dict[int, int] = {}
    for label in check:
        start = int(np.searchsorted(sorted_labels, label, side="left"))
        stop = int(np.searchsorted(sorted_labels, label, side="right"))
        nodes = sorted(order[start:stop].tolist())
        comps[label] = _Component(nodes, [], set())
        for position, node in enumerate(nodes):
            local_index[node] = position
    in_check = np.isin(labels, np.asarray(check, dtype=np.int64))
    for a, b, probability, must_edge in zip(
        edges.left.tolist(),
        edges.right.tolist(),
        edges.probability.tolist(),
        edges.must.tolist(),
        strict=True,
    ):
        if in_check[a]:
            comps[int(labels[a])].edges.append(
                (local_index[a], local_index[b], probability, must_edge)
            )
    for label, pairs in cannot_by_label.items():
        for a, b in pairs:
            comps[label].conflicts.add((local_index[a], local_index[b]))
    return comps


def _add_implicit_conflicts(
    comps: dict[int, _Component], oracle: ConflictOracle, config: ClusterConfig
) -> None:
    """Score every non-edge pair inside checked components; very unlikely pairs conflict."""
    query_left: list[int] = []
    query_right: list[int] = []
    owners: list[tuple[int, tuple[int, int]]] = []
    for label, comp in comps.items():
        size = len(comp.nodes)
        if size > config.max_check_size:
            continue
        linked = {(min(a, b), max(a, b)) for a, b, _, _ in comp.edges}
        for i in range(size):
            for j in range(i + 1, size):
                if (i, j) not in linked:
                    query_left.append(comp.nodes[i])
                    query_right.append(comp.nodes[j])
                    owners.append((label, (i, j)))
    if not query_left:
        return
    scores = oracle(np.asarray(query_left, dtype=np.int64), np.asarray(query_right, dtype=np.int64))
    for (label, pair), score in zip(owners, scores.tolist(), strict=True):
        if score <= config.conflict_probability:
            comps[label].conflicts.add(pair)


def resolve_clusters(
    n: int,
    left: Int64Array,
    right: Int64Array,
    probability: FloatArray,
    *,
    match_threshold: float,
    config: ClusterConfig | None = None,
    must_link: Iterable[tuple[int, int]] = (),
    cannot_link: Iterable[tuple[int, int]] = (),
    conflict_oracle: ConflictOracle | None = None,
    edge_mask: BoolArray | None = None,
) -> Resolution:
    """Resolve records ``0..n-1`` into clusters. See the module docstring for the rules.

    ``edge_mask`` selects the match edges explicitly (for example after review rules moved some
    pairs to review); by default every pair with ``probability >= match_threshold`` is an edge.
    """
    cfg = config or ClusterConfig()
    must = _pair_set(must_link)
    cannot = _pair_set(cannot_link)
    if must & cannot:
        raise ValueError("a pair cannot be both must-link and cannot-link")
    if edge_mask is not None and edge_mask.shape != probability.shape:
        raise ValueError("edge_mask must have one entry per pair")
    keep = edge_mask if edge_mask is not None else probability >= match_threshold
    edges = _edge_list(n, left[keep], right[keep], probability[keep], must=must, cannot=cannot)
    labels = connected_labels(n, edges.left, edges.right)
    pending = np.zeros(n, dtype=bool)
    if n == 0:
        return Resolution(labels, pending)
    if not cfg.guard:
        # No splitting, but a reviewer's cannot-link is still never auto-committed.
        conflicted: dict[int, tuple[str, ...]] = {}
        for a, b in sorted(cannot):
            if labels[a] == labels[b]:
                conflicted[int(labels[a])] = ("unresolved_conflict",)
        for cluster in conflicted:
            pending[labels == cluster] = True
        return Resolution(labels, pending, conflicted)
    comps = _components_to_check(labels, edges, cannot)
    if not comps:
        return Resolution(labels, pending)
    if conflict_oracle is not None:
        _add_implicit_conflicts(comps, conflict_oracle, cfg)

    new_labels = labels.copy()
    reasons: dict[int, tuple[str, ...]] = {}
    removed_edges: list[RemovedEdge] = []
    for comp in comps.values():
        parts, removed = _split_component(comp, cfg)
        for local_nodes, part_reasons in parts:
            rows = [comp.nodes[i] for i in local_nodes]
            cluster = min(rows)
            new_labels[rows] = cluster
            if part_reasons:
                reasons[cluster] = part_reasons
                pending[rows] = True
        for a, b, edge_probability, reason in removed:
            ga, gb = comp.nodes[a], comp.nodes[b]
            removed_edges.append(RemovedEdge(min(ga, gb), max(ga, gb), edge_probability, reason))
    removed_edges.sort(key=lambda edge: (edge.left, edge.right))
    return Resolution(new_labels, pending, reasons, tuple(removed_edges), len(comps))
