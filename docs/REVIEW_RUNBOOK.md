# Review runbook

Operational guide for the people who work the clerical-review queue. The queue exists because the
model is deliberately not allowed to decide some pairs on its own.

## What lands in the queue

| Reason | Meaning | What to look at first |
|---|---|---|
| `review_band` | Match probability between the review and match thresholds | All fields; the model is genuinely unsure |
| `rule_possible_twin` | High probability, but first names clearly differ while surname, date of birth and address agree | First names and sex: twins look exactly like this |
| `guard_weak_link` | The over-merge guard cut a weak link between two otherwise separate groups (often household members) | Dates of birth and first names of the two groups |
| `guard_bridge_record` | One record linked two groups that do not match each other; its links to the weaker group were cut | Whether the record copies fields from two different people |
| `guard_weak_bridge`, `guard_conflict`, `guard_size_cap` | Other guard cuts: a single weak connecting edge, a conflicting pair, or an oversized cluster | The records on both sides of the cut |
| `pending_unresolved_conflict`, `pending_size_cap` | A cluster that is not auto-committed at all | The whole cluster (entity lookup with `pii:read`) |
| `pending_too_large_to_check` (usually with `size_cap`) | A component of more than 200 records: the guard held it for review as a whole instead of splitting it, which bounds the work an adversarial upload can cause | Whether the source contains chained or copied records (possible injection) |

Items are listed per job with `GET /v1/link-jobs/{job_id}/review-items` (scope `review:write`;
add `pii:read` to see the record fields) or `probabilistic-record-linkage review list`.

## Deciding

- `match` becomes a must-link: the two records always end up in the same cluster.
- `non_match` becomes a cannot-link: the records are never auto-committed together. If other
  evidence still connects them the cluster is marked `pending_review` instead of being merged.
- The latest decision per pair wins. Decisions are final for the item (a second decision on the
  same item is refused with 409); to correct a mistake, record the opposite decision on a new item
  for the same pair or escalate.

Every decision is appended to the hash-chained audit log *before* clusters are recomputed. The
response contains the new `audit_head`.

## Checks after a review session

1. Verify the audit chain and compare the head with the last `audit_head` you received:
   `probabilistic-record-linkage audit verify var/audit/review-decisions.jsonl --expect-head <head>`.
   Store the head outside the service (ticket, chat message); only an anchored head reveals a
   truncated log.
2. Watch `review_queue_depth` and `review_decisions_total{decision=...}` in `/metrics`. A queue
   that grows run after run means thresholds or rules need attention, not more reviewers.
3. If `comparison_pattern_psi` is above 0.2 for a batch, the data changed (more typos, more
   missing fields). Do not register a model fitted on that batch as `production` before looking
   at the per-field PSI in the job summary.

## Escalation

- Many `guard_bridge_record` items touching the same people: possible deliberate record
  injection. Stop auto-commits for the affected source and raise a security incident.
- A reviewer disagrees with most `rule_possible_twin` items: the twin rule may be too broad for
  this population; record the observation for a threshold/rule change rather than bulk-deciding.
