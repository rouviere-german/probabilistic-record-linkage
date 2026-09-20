# 4. Privacy mode: harden name encodings with birth-year salts, block on keyed tokens

- Status: accepted
- Date: 2026-09-13

## Context

In privacy mode each party encodes fields as HMAC-keyed Bloom filters and the linkage unit
compares bit vectors. The specification requires a frequency-attack test that gates the hardening
defaults, and an F1 drop against plaintext linkage of at most 3 percentage points.

Measured at 20,000 records during development:

- Unhardened encodings of first names are deterministic: identical names give identical filters,
  so encoding frequencies mirror name frequencies, and filter Hamming weights reveal name length.
  A frequency-alignment attack stratified by Hamming weight recovered 15-17 of the 100 most
  frequent first-name encodings. F1 dropped 0.77 points against plaintext.
- XOR-folding and balancing alone do not help against that attack: they change bits, not the fact
  that equal names encode equally.
- Balancing makes every filter half full, which destroys the Jaccard signal MinHash LSH needs.
- Applying balancing to dates pushed one-component date errors below every Dice cutpoint, and the
  first hardened configuration lost 3.06 F1 points.

## Decision

1. Name fields are salted with the record's birth year inside the HMAC, so equal names of people
   born in different years encode differently and encoding frequencies flatten.
2. XOR-folding and balancing are applied to the variable-length, frequency-skewed string fields
   (first name, last name, street, city). Fixed-format fields (date, sex, postcode) keep plain keyed
   encodings; dates are encoded as component tokens (year, month, day, unordered month/day) so
   swaps and one-component errors stay partially similar.
3. Blocking uses keyed tokens computed by each party: HMAC of the birth date, and of Soundex codes
   combined with each other or with the birth year. MinHash LSH is not used on encodings.
4. The key is only read from a mounted secret file with owner-only permissions; there is no
   environment-variable key path. Hardening is on by default, and
   `tests/integration/test_privacy_gate.py` fails if the default configuration lets the attack
   recover more than 5 of the top 100 names (or if the unhardened baseline stops being attackable,
   which would mean the test lost its power). *Superseded by the 2026-09-20 amendment below: the
   gate moved to the benchmark's scale, where the default recovers 10 and the 5% goal is a strict
   xfail.*

## Measured trade-off at scale

At 20,000 records the default hardening held the attack to 2 of the top 100 names. Repeating the
measurement on 50,206 records (seed 20260913, `scratchpad` experiment later folded into
`bench/bench_robustness.py`) showed that a salt-aware attacker, who ranks (name, birth year)
combinations by expected frequency, gets better as the sample grows:

| Name salt | F1 drop vs plaintext | Worst attack rate (top 100) |
|---|---:|---:|
| birth year (default) | 2.80 pp | 0.10 |
| birth year + sex | 3.49 pp | 0.08 |
| birth year + birth month | 4.68 pp | 0.00 |

No setting meets both targets of the specification (F1 drop at most 3 pp *and* attack recovery at
most 5%) at 50,000 records. The default keeps the birth-year salt, which meets the F1 target and
cuts the unhardened attack rate (0.20 at the same scale) in half; birth year + birth month is
available as `HardeningConfig(name_salt_parts=("birth_year", "birth_month"))` for deployments that
value re-identification resistance over recall. Both are measured on every robustness run. The CI
threshold for the default's attack rate (0.15) is a regression guard at the measured level, not a
claim that the 5% target is met.

## Consequences

- The specification's 5% re-identification target is not met by the default configuration at
  50,000 records; this is recorded as a deviation and in the README.
- Records whose birth year is missing or mistyped lose name similarity against their duplicates;
  this is the main source of the hardened F1 drop, and the benchmark reports it on every run.
  Every extra salt part (month, sex) multiplies that loss.
- Keyed blocking tokens are themselves deterministic. Their frequencies reveal how birth dates and
  Soundex codes are distributed; the threat model lists this as residual risk, together with the
  stronger pattern-mining and graph-matching attacks that are not implemented here.
- No term-frequency adjustment in privacy mode: computing frequencies of encodings is exactly the
  information the attack uses, so the linkage unit does not rely on it.

## Amendment (2026-09-20): gate at benchmark scale, and a stronger attack

An adversarial review made two points about the gate described above. First, it ran on 20,000
records and asserted 5%, while the benchmark measured 10% for the same default on 50,206 records:
the gate passed only at a scale where the attack is weaker. Second, frequency alignment is a weak
adversary, so "hardening defeats the attack" said little about hardening.

Changes:

- `tests/integration/test_privacy_gate.py` now runs on the same 50,000 records, seed and key as
  `bench/bench_robustness.py`. The 5% goal for the default is a `strict=True` xfail with the
  measured value in its reason, so the suite fails if the default ever meets it and the xfail is
  left behind; the birth-year + birth-month salt is asserted to meet it. The 0.15 assertion is
  named `REGRESSION_LIMIT` in the test, matching `bench/robustness_thresholds.json`.
- A graph-matching attack (Vidanage, Christen, Ranbaduge and Schnell 2020, solved with the
  graduated assignment of Gold and Rangarajan 1996) was added in `linkage/attack.py` and runs in
  every benchmark, demo and gate. It exploits the fact that without hardening the Dice similarity
  of two filters tracks the Dice similarity of the two names' q-gram sets: Pearson 0.87 between
  the two matrices on the benchmark records (0.868, reported as `similarity_correlation` by
  `bench/bench_robustness.py`), against 0.006 after hardening.
- Measured on the 50,206 benchmark records: it names 77 of the 100 most frequent unhardened
  encodings (the best frequency alignment gets 20) and 0 of the hardened ones. The gate's floor for
  the unhardened case is 0.5 and `bench/robustness_thresholds.json` holds a matching minimum, so
  the contrast cannot silently lose its power.

Pattern-mining cryptanalysis (Christen et al. 2017) was tried and dropped: it has to isolate the
12 bit positions of one q-gram from co-occurrence counts, and with 512 bits and about 400 distinct
q-grams in the name vocabulary roughly nine q-grams share every position, so positions of
different q-grams co-occur as often as positions of the same one. This is recorded in the README
Non-goals rather than claimed as a defence.
