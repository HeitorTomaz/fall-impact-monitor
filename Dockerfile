# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS base
COPY --from=ghcr.io/astral-sh/uv:0.9.30 /uv /uvx /bin/
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ORT_DISABLE_TELEMETRY=1 \
    PATH=/opt/venv/bin:$PATH
WORKDIR /app

# Development image: dependencies only; the source tree is bind-mounted at /app.
FROM base AS dev
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

# Runtime image: no dev dependencies, non-root user.
FROM base AS runtime
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY models.toml ./
# Model weights are downloaded from their original source and verified by SHA-256.
ARG FIM_MODELS_FETCH=yolox_m_humanart,rtmpose_s,yolox_s
RUN uv sync --frozen --no-dev \
    && fim models fetch --only "${FIM_MODELS_FETCH}" --models-dir /app/models \
    && useradd --create-home --uid 10001 fim \
    && mkdir -p /app/artifacts \
    && chown fim:fim /app/artifacts
USER fim
CMD ["fim", "--help"]
