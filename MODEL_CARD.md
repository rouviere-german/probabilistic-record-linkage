# Model card: person-linkage (Fellegi-Sunter)

Structure follows Mitchell et al., "Model Cards for Model Reporting" (FAT* 2019). Every number was
copied from benchmark output together with the command that produced it (see the README Results).

## Model details

- **Name:** `person-linkage`, registered by `probabilistic-record-linkage link --register` or the
  demo. The registry digest (`model_version`) is printed by those commands; it covers the manifest
  timestamp, so it differs between runs while the deterministic `run_hash` does not.
- **Type:** unsupervised Fellegi-Sunter record-linkage model over five comparison fields (first name,
  last name, date of birth, sex, hierarchical address), with term-frequency adjustment for names,
  label-free thresholds, a twin review rule and an over-merge-guarded clustering step.
- **Estimation:** `u` from random record pairs, with exact levels computed exactly from value counts.
  `m` comes from EM over comparison-pattern counts on training blocks (same date of birth, same
  name), with 4 initializations per block recorded in the manifest. The prior is a one-parameter EM
  over all record pairs ([ADR 0002](docs/adr/0002-u-from-random-pairs-and-training-blocks.md)).
- **Owner:** Rouviere German. **Date:** 2026-09-13. **License:** MIT (code). All data is synthetic.
- **Artifact format:** canonical JSON (`model.json`) in a content-addressed registry with SHA-256
  verification; never pickle.
- **Run manifest:** the generator config and corruption-config hashes, source file hashes,
  pipeline-config hash, every EM initialization, the `uv.lock` hash and the run hash.
  Term-frequency tables are not stored; they are recomputed from each batch.

## Intended use

- **Primary use:** grouping person records from several internal systems into entities, with
  uncertain pairs, twin-like pairs and unsafe merges routed to human review.
- **Intended users:** data-quality and master-data teams who review the queue
  ([docs/REVIEW_RUNBOOK.md](docs/REVIEW_RUNBOOK.md)).
- **Out-of-scope uses:** decisions about individuals without human review of uncertain cases;
  linking across organizations in plaintext when privacy mode is required; any population whose
  names, dates or addresses follow patterns the model was not evaluated on (see caveats); stable
  identifiers across batches (incremental resolution is a non-goal).

## Training data

- **Source:** `linkage/generator.py`, seed 20260913, default `CorruptionConfig`: 5 sources, twins
  (1% of people), households (5% of people with 1-3 extra members) and within-source duplicates
  (3%).
- **Names:** three synthetic name-origin groups (`north` 60%, `south` 25%, `east` 15%) built from
  Zipfian syllable models. They differ in name length and surname concentration: `east` has short
  names and only 600 surnames.
- **Contract:** `SourceRecordSchema` (pandera, Polars backend, strict). Violations are rejected
  with column/check/count summaries.
- **Known gaps:** no transliteration, no multi-part or hyphenated surnames, no marriage name
  changes, no phone or e-mail fields, a fixed set of error types, and synthetic geography.

## Evaluation

Command: `uv run python bench/bench_linkage.py --records 200000 --check` (199,684 records, 79,674
people). Metrics are computed against generator ground truth.

| Metric | Value | Why |
|---|---:|---|
| Blocking pair completeness / reduction ratio | 0.9918 / 0.999735 | Ceiling on recall / cost of comparison |
| Pairwise precision / recall / F1 | 0.9898 / 0.9912 / 0.9905 | Standard linkage quality |
| B-cubed F1 | 0.9950 | Per-record cluster quality |
| ECE over candidate pairs | 0.00125 | Probability calibration (weak probe, see caveats) |
| Max abs m(exact) error vs analytic | 0.0178 | Whether EM recovers the true agreement rates |
| Twin / household false-merge rate | 0.2802 / 0.0181 | Hard negatives |
| Exact-key baseline F1 / logistic regression (1% labels) F1 | 0.7458 / 0.9870 | Context |

### Per-group error rates (199,684 records)

| Name-origin group | Pairwise precision | Pairwise recall | F1 |
|---|---:|---:|---:|
| north | 0.9909 | 0.9914 | 0.9912 |
| south | 0.9879 | 0.9911 | 0.9895 |
| east | 0.9890 | 0.9906 | 0.9898 |

The spread between groups is about 0.2 F1 points on this data. The `south` group has the lowest
precision; `east`, with its concentrated surnames, has the lowest recall. Real populations can differ
far more; re-measure per group before deployment.

## Assumptions and calibration

- **Conditional independence.** Fields are assumed independent given match status. Address parts
  are compared as one hierarchical field to avoid counting a move three times
  ([ADR 0003](docs/adr/0003-address-as-one-hierarchical-comparison.md)). Remaining dependence
  (surname and address within households, surname and date of birth for twins) is handled by the
  review rule and the guard, not by the model.
- **Correlated errors.** With errors concentrated in 10% of records at 5x the rates (same marginal
  rates; `bench/bench_robustness.py`, 20,000 records), F1 fell from 0.9927 to 0.9894. ECE over
  candidate pairs went from 0.00349 to 0.00290, so no degradation was visible there. Most candidate
  pairs are clear non-matches, which dominate that ECE.
- **Blocking completeness.** The prior assumes pairs outside the candidate set are non-matches;
  at 0.9918 pair completeness, about 0.8% of true pairs can never be found.

## Robustness, drift and monitoring

- **Corruption sweep** (`bench/bench_robustness.py`, 20,000 records): F1 0.9973 / 0.9927 / 0.9880 /
  0.9825 at 0.5x / 1x / 1.5x / 2x corruption rates.
- **Adversarial bridge records** (105 records in 40 chains): with the guard, at most 2 real people
  in an auto-committed cluster touched by a bridge record (5 without); 3 touched clusters were held
  for review, one of them with 3 real people.
- **Drift monitor:** PSI of comparison patterns of likely matches against the reference stored in
  the model, alarm above 0.2. Measured PSI is 0.016 for a new source with the same corruption and
  0.554 for 3x corruption. It is exported as `comparison_pattern_psi`, and per-field PSI is in the
  job summary.
- **Operational metrics:** `blocking_pairs_total`, `reduction_ratio`, `match_rate`,
  `review_queue_depth`, `max_cluster_size`, `em_iterations`, `linkage_stage_duration_seconds`.
- **Retraining trigger:** PSI above 0.2 on consecutive batches, a growing review queue, or reviewer
  decisions contradicting the model on more than a small share of review-band items. Refit, compare
  the benchmark metrics, then move the `production` alias.

## Privacy mode

Keyed Bloom-filter encodings, 50,206 records (`bench/bench_robustness.py`):

| Encoding | F1 | Drop vs plaintext (0.9916) | Frequency-attack recovery (top 100) |
|---|---:|---:|---:|
| Unhardened | 0.9817 | 0.99 pp | 0.20 |
| Hardened, birth-year salt (default) | 0.9638 | 2.79 pp | 0.10 |
| Hardened, birth-year + month salt | 0.9446 | 4.70 pp | 0.01 |

No setting meets both targets of at most 3 points F1 loss and at most 5% recovery at this scale.
Stronger attacks (pattern mining, graph matching) were not evaluated.

## Ethical considerations

Linking records concentrates information about people: an entity lookup reveals which systems know
a person even without field values. A false merge can attach one person's history to another; a
false split can hide a duplicate from someone who is entitled to see it. Twins and household
members carry most of the false-merge risk, and short names from concentrated surname pools behave
differently from long names. Mitigations here are human review of uncertain and unsafe merges,
scoped access to personal fields, redacted telemetry, audited decisions and a privacy mode whose
leakage is measured rather than assumed.

## Caveats and recommendations

- All numbers are on synthetic data with a known, fixed corruption model; they are not estimates
  of accuracy on any real dataset.
- Refit on each new population and re-run the benchmarks' per-group and hard-negative measurements
  on labelled samples before trusting auto-commits.
- Keep the review queue staffed; the twin rule and the guard exist because the model alone merges
  twins.
- Anchor the audit head outside the host and encrypt the DuckDB volume.
