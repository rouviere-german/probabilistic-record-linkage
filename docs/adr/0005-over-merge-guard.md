# 5. Over-merge guard: weak cohesion, bridge records and bridge edges

- Status: accepted
- Date: 2026-09-13

## Context

Connected components over match edges are transitive: one wrong edge merges two people, and a
record that looks like two people at once (for example first name and sex of one household
member, date of birth and address of another) merges both. The specification asks for splitting by
cluster size and by weak bridge edges, and for an adversarial test in which no cluster merges more
than two true entities.

The first guard condensed records joined by strong edges (probability >= 0.9999) into cores and
cut the weakest *bridge edge* of the condensed graph. On 15 adversarial chains injected into a
12,000-record dataset it failed: a cluster still contained 6 true identities. Development
diagnostics (the same data, `ClusterConfig` variants) showed why:

- bridge-record edges scored logit 0.2 to 12.5 and genuine duplicate edges mostly 10 to 22, so
  the strong threshold either welded the bridge record into both people's cores or, when raised,
  fragmented genuine duplicates into cliques;
- a bridge record is a *cut vertex*, not a cut edge: removing any one of its edges leaves the
  others, so no bridge edge exists;
- household members were already joined by direct edges just above the match threshold
  (probability about 0.6), giving a 2-edge-connected graph with no articulation point at all.

Adding an articulation-point rule reduced the worst case to 4 identities; adding the weak-cohesion
rule reduced it to 3, with at most 2 real (non-injected) people per cluster.

## Decision

Apply three rules in order to every multi-record component, recursively, with "solid" edges being
must-links and edges with probability >= `weak_probability` (0.9):

1. **Weak cohesion**: if the solid edges alone leave two or more groups of at least
   `min_side_records` records, split into those groups.
2. **Bridge record**: if one record is an articulation point of the solid graph and its removal
   leaves two or more such groups, keep it with the group it is most strongly linked to and cut
   its links to the others.
3. **Bridge edge** of the strong-core condensation, for conflicts (reviewer cannot-links or pairs
   with probability <= 0.001), oversized clusters, or single-record weak links.

Clusters that still exceed the size cap or still contain a conflict are `pending_review` and never
auto-committed. Must-links are never cut, and every cut edge goes to the review queue.

## Consequences

- The adversarial benchmark measures the maximum number of real people in any cluster touched by
  an injected record (target <= 2) and, separately, the maximum number of identities including the
  injected records. The second can be 3 when one person is bridged from both sides: a record that
  matches a single person strongly is indistinguishable from a genuine duplicate, so the guard
  keeps it.
- The guard is not a proof against adversaries who inject several bridge records per pair of
  people (no single articulation point, no single weak group boundary). This residual risk is in
  the threat model.
- On the clean 12,000-record development dataset the guard raised pairwise F1 from 0.972 to 0.990
  (mainly by separating household members), so it does not cost recall there; the benchmarks
  report the with/without numbers on every run.
- The rules add Python work per component with three or more records; clustering stays about one
  second for 200,000 records.

## Amendment (2026-09-14): bounded work and cannot-links without the guard

An adversarial review showed that the bridge-record rule recomputed connected components once
per articulation point and that `_split_component` looped without a work bound: a chain of
matching records cost about 8x more time per doubling (0.09 s, 0.68 s and 5.23 s for 100, 200
and 400 records), so a few thousand chained records in one upload would have tied up a job worker
for hours. It also showed that `ClusterConfig(guard=False)` auto-committed clusters containing a
reviewer's cannot-link.

Changes:

- one depth-first search per pass now yields every articulation point together with the sizes
  of the parts it separates (DFS intervals), and bridge sides are prefix sums over DFS order, so
  each split pass is linear in the component;
- a component with more than `max_check_size` records (200) is held for review as a whole with
  reason `too_large_to_check` instead of being split, which bounds the guard's work per
  component; `tests/unit/test_clustering.py::test_guard_work_stays_bounded_on_large_adversarial_components`
  checks chains, stars and ladders of 800 records;
- with the guard off, a component containing a cannot-link pair is `pending_review`
  (`tests/unit/test_clustering.py::test_cannot_links_are_never_auto_committed_even_without_the_guard`).

The clean-data results of the benchmarks did not change: no component in the 20,000- or
200,000-record runs exceeds 200 records.
