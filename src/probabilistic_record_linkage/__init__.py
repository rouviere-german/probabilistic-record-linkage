"""Probabilistic record linkage for messy person records.

Sub-packages:
- ``core``: settings, logging with PII redaction, tracing, metrics, authentication and the
  hardened FastAPI app factory (shared portfolio baseline).
- ``ml``: data contracts, run manifests, content-addressed registry, drift monitors and
  pickle-free serialization (shared portfolio baseline).
- ``linkage``: synthetic data, blocking, the Fellegi-Sunter model, clustering, review and privacy
  mode (this project's pattern).
- ``api``: the HTTP routes of the linkage service.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
