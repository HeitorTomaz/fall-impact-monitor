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

# Model weights, downloaded from their original source and verified by SHA-256. This stage
# copies only the manifest and the fetch code (standard library only; the list is checked
# by tests/perception/test_models.py), so a source change does not download them again.
FROM base AS models
COPY models.toml ./
COPY src/fall_impact_monitor/__init__.py src/fall_impact_monitor/
COPY src/fall_impact_monitor/cli/__init__.py src/fall_impact_monitor/cli/models.py src/fall_impact_monitor/cli/
COPY src/fall_impact_monitor/perception/__init__.py src/fall_impact_monitor/perception/models.py src/fall_impact_monitor/perception/
ARG FIM_MODELS_FETCH=yolox_m_humanart,rtmpose_s,yolox_s
RUN PYTHONPATH=/app/src python -m fall_impact_monitor.cli.models \
    models fetch --only "${FIM_MODELS_FETCH}" --models-dir /app/models

# Runtime image: no dev dependencies, non-root user.
FROM base AS runtime
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY --from=models /app/models ./models
COPY models.toml ./
COPY src ./src
RUN uv sync --frozen --no-dev \
    && useradd --create-home --uid 10001 fim \
    && mkdir -p /app/artifacts \
    && chown fim:fim /app/artifacts
USER fim
CMD ["fim", "--help"]
