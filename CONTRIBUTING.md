# Contributing

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv can install it).

```bash
uv sync --frozen
uv run python tasks.py check      # lint, typecheck, test, audit, baseline
```

If the repository sits on a small or slow drive, keep the virtual environment elsewhere:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "C:/Users/<you>/.venvs/probabilistic-record-linkage"
uv sync --frozen
```

## Tasks

`tasks.py` is the cross-platform task runner; the Makefile only delegates to it.

| Target | What it runs |
|---|---|
| `lint` | `ruff check` and `ruff format --check` |
| `format` | `ruff format` and `ruff check --fix` |
| `typecheck` | `mypy --strict` over src, tests, scripts, bench and tasks.py |
| `test` | pytest with branch coverage; fails below 85% |
| `bench` | micro-benchmarks, the 20,000-record linkage benchmark and the robustness benchmark, each against its thresholds file |
| `demo` | the offline end-to-end demo |
| `audit` | bandit and pip-audit |
| `baseline` | `scripts/check_baseline.py` |

## Rules for changes

- **Tests first for behavior.** Mark tests `unit` or `integration`. Use hypothesis for invariants
  (EM likelihood monotonicity, clustering constraints, brute-force oracles for bridges, scores and
  blocking) rather than hand-picked examples alone.
- **Offline by default.** Tests, demo and bench never touch the network; `tests/conftest.py` blocks
  non-loopback sockets. All data comes from the seeded generator.
- **Ground truth stays out of the linker.** Only `generator.py`, `analytic.py`, the demo, tests and
  benchmarks may read `truth/`. The source contract rejects any extra column.
- **Treat synthetic records as personal data.** Never log field values; log counts and
  pseudonymized ids. New record-level fields need a redaction rule and a fixture test.
- **Evidence for claims.** A new README claim needs a row in `CLAIMS.md` pointing to a test, bench
  script or CI job. Numbers must be copied from real `bench/` output together with the command.
- **Threat model.** New entry points or artifacts need a STRIDE row in `docs/THREAT_MODEL.md` with
  a control and a test.
- **Decisions.** Record significant trade-offs as ADRs in `docs/adr/`.
- **No pickle.** Artifacts are JSON (models, manifests) or Parquet (datasets).
- **No secrets in git.** Configuration goes in `.env` (ignored); document new settings in
  `.env.example` (a test enforces that it lists every setting). Key files belong in `secrets/`
  (ignored).
- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `ci:`).
