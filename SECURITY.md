# Security policy

## Supported versions

Only the latest commit on `main` is supported. This is a portfolio project, not a product with
release branches.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private vulnerability
reporting ("Report a vulnerability" under the repository's Security tab). Include:

- the affected commit and component;
- steps to reproduce, ideally as a failing test;
- the impact you expect.

You can expect an acknowledgement within 7 days and a fix or a written assessment within 30 days.
Credit is given in the fix commit unless you ask otherwise.

## Scope

In scope: the code in this repository, its container image and its CI configuration, including
the privacy-mode encodings and the frequency-attack test that gates their defaults.
Out of scope: denial of service by traffic volume, findings that need a compromised host, attacks
that require the shared Bloom-filter key, and issues in third-party dependencies that are already
publicly tracked (please report those upstream).

## How this repository is hardened

- Threats, controls and the tests that prove them: [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).
- Every security claim is linked to evidence in [CLAIMS.md](CLAIMS.md).
- All data in this repository is synthetic, but it is handled as personal data: record fields are
  masked in logs, record ids and principals are pseudonymized in traces and stored reviewer
  columns, and fields are returned only to tokens with the `pii:read` scope.
- CI runs ruff, mypy --strict, pytest with a coverage floor, bandit, semgrep (registry packs plus
  `.semgrep/rules.yml`), pip-audit, gitleaks, hadolint, a trivy image scan and a syft CycloneDX
  SBOM, plus the linkage and robustness benchmarks against their thresholds.
- Secrets come only from the environment or mounted files; only `.env.example` is committed.
- Artifacts are pickle-free (JSON models and manifests with SHA-256 verification, Parquet data).
