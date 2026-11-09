#!/bin/sh
# Full local gate: tests, lint, format, types.
set -e
uv run --frozen pytest -q "$@"
uv run --frozen ruff check
uv run --frozen ruff format --check
uv run --frozen mypy
