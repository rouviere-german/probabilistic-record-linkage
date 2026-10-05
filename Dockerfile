# syntax=docker/dockerfile:1
#
# Multi-stage, non-root image. Base images are pinned by digest (resolved 2026-09-13); Dependabot's
# docker ecosystem proposes digest bumps. python:3.12-slim is multi-arch, so the index digest works
# on amd64 and arm64.
#
# Runtime layout: the root filesystem is meant to run read-only. /data (DuckDB store, audit log,
# model registry) must be a volume or tmpfs owned by UID 10001. The Bloom-filter key for privacy
# mode is mounted as a secret file and referenced by APP_PPRL_KEY_FILE; it is never baked in.

ARG PYTHON_IMAGE=python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

FROM ghcr.io/astral-sh/uv:0.12.23@sha256:61d393e44e249f2e4b526b6c7ddcecce245946826e608e11c93ad4f5bba55b21 AS uv

FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build
# Dependencies first, so source edits do not invalidate the dependency layer.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="probabilistic-record-linkage" \
      org.opencontainers.image.licenses="MIT"
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --home-dir /nonexistent \
       --shell /usr/sbin/nologin app \
    && mkdir -p /data \
    && chown 10001:10001 /data
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    APP_DATA_DIR=/data
WORKDIR /app
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-m", "probabilistic_record_linkage.core.healthcheck"]
ENTRYPOINT ["python", "-m", "probabilistic_record_linkage"]
CMD ["serve"]
