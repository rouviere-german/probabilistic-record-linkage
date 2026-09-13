# 3. Compare street, postcode and city as one hierarchical address field

- Status: accepted
- Date: 2026-09-13

## Context

Street, postcode and city change together: when a person moves, all three usually differ, and
when two records share a street they almost always share the postcode and city. Fellegi-Sunter
multiplies per-field Bayes factors under conditional independence, so three separate address
fields would count one piece of evidence (the same or a different address) up to three times.
That inflates match weights for household members (same surname and address, different person)
and deflates them for the same person after a move, and it distorts the probabilities the
label-free threshold selection relies on.

## Decision

The address is one comparison with ordered, mutually exclusive levels:

`missing` (all three parts absent) < `else` < `same city` < `same postcode` < `street similar`
(Jaro-Winkler >= 0.88 within the same postcode, or the same street with a different or missing
postcode) < `exact` (same street and postcode).

The same hierarchy is used in privacy mode, where "same" means identical keyed encodings and
"similar" means Dice similarity above the calibrated cutpoint.

## Consequences

- One `m`/`u` pair per address level; the analytic `m(exact)` of the address follows directly from
  the move, street-typo and postcode corruption rates and is checked by
  `tests/unit/test_analytic.py`.
- The remaining dependence between fields (surname and address within households, surname and
  birth date for twins) is not modelled. It is what the review rule for twins and the over-merge
  guard exist for, and its effect on calibration is measured with the correlated-corruption
  scenario in `bench/bench_robustness.py`.
- House-number differences on the same street fall into `street similar`; neighbours are a small
  source of false evidence that EM absorbs into `u` for that level.
