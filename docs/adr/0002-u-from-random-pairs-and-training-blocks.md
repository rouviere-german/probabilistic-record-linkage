# 2. Estimate u from random pairs and m on training blocks, not EM on candidate pairs

- Status: accepted
- Date: 2026-09-13

## Context

The specification asks for a Fellegi-Sunter model fit by EM over comparison-pattern counts, with
`u` initialized from random record pairs. The first implementation ran EM with `m`, `u` and
`lambda` all free on the *candidate* pairs produced by blocking. At 20,000 records it recovered
the analytic exact-agreement `m` of every field within 0.018. At 200,000 records it did not:

- the fitted address `m(exact)` was 0.758 against an analytic 0.683 (off by 0.075, target 0.03);
- `u(first name exact)` among candidates was 0.28 and `u(date of birth exact)` 0.06, because LSH
  proposes pairs that share a name and sorted neighbourhood proposes pairs that share a birth date;
- pairs agreeing on first name, surname and birth date but with a moved address were scored as
  non-matches, and pairwise recall fell to 0.91 (F1 0.944, target 0.95).

The cause is that blocking makes non-matching candidates agree on the blocked fields far more
often than random pairs, and in a correlated way (name agreement and birth-date agreement come
from different blocking passes). EM under conditional independence cannot represent that.

## Decision

Follow the estimation strategy popularized by Splink (Linacre et al., UK Ministry of Justice):

1. `u` is estimated from random record pairs and then held fixed. Exact-agreement levels, which
   random pairs almost never hit (about 1e-6 for addresses), are computed exactly from value
   counts over all records; the other levels come from a sample of 300,000 random pairs.
2. `m` is estimated by EM over pattern counts on *training blocks*: all pairs sharing a date of
   birth (with the date field masked) and all pairs sharing first and last name (with the names
   masked). Inside such a block the non-matches are random with respect to the unmasked fields.
   Estimates are averaged, weighted by each block's expected number of matches, and every
   initialization of every block is recorded in the model manifest.
3. The prior `lambda` of a random record pair comes from a one-parameter EM over all pairs, with
   pairs outside the candidate set treated as having negligible posterior.
4. Scoring uses these parameters for every candidate pair; blocking only decides which pairs are
   scored. Term-frequency adjustment uses the form that is exact when `u(exact)` is the random-pair
   rate.

EM on candidates remains available as `EMConfig(strategy="candidates")`, and the linkage benchmark
reports the recovery error of both estimators on the same run.

## Consequences

- At 200,000 records the training-block estimator recovered every field's `m(exact)` within
  0.018 and pairwise F1 rose to 0.979 (see the README results for the current numbers).
- This deviates from the literal specification ("u is initialized from random pairs" and refined
  by EM); `u` is not refined. The deviation is recorded in the README non-goals.
- Very small batches can lack training blocks; the pipeline then falls back to EM on candidates,
  or scores with a registered model (`use_registered_model`).
- Probabilities are calibrated to random record pairs, not to the candidate set; the threshold
  selection and ECE reported in the benchmark are computed on candidate pairs regardless.
