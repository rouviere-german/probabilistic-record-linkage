"""Probabilistic record linkage: synthetic data, blocking, Fellegi-Sunter EM, clustering, privacy.

Modules, in pipeline order:

- ``vocab``, ``nicknames``, ``corruption``, ``generator``: the seeded synthetic world and its
  ground truth (M1).
- ``normalize``, ``phonetic``, ``minhash``, ``blocking``: field normalization and candidate
  generation with MinHash LSH plus sorted neighbourhood (M2).
- ``comparison``, ``fellegi_sunter``, ``analytic``: agreement levels, EM over comparison-pattern
  counts, term-frequency adjustment and the analytic m-probabilities used to check EM (M3).
- ``thresholds``, ``clustering``, ``review``, ``audit``: label-free thresholds, constraint-aware
  clustering with the over-merge guard, and the audited review queue (M4).
- ``privacy``, ``attack``, ``monitoring``: keyed Bloom-filter linkage, the frequency-alignment
  attack and comparison-pattern drift (M5).
- ``model``, ``pipeline``, ``store``, ``evaluation``, ``baselines``: artifacts, orchestration,
  DuckDB persistence and ground-truth evaluation (M6).
"""
