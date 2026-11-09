#!/bin/bash
# Host-side gate: auto-fix lint/format, then tests + lint + format + types in Docker.
export MSYS_NO_PATHCONV=1
cd "D:/Projetos/fall-impact-monitor" || exit 2
docker compose --profile dev run --rm dev sh -c "uv run --frozen ruff check --fix -q; uv run --frozen ruff format -q; sh .superpowers/check.sh $*" > .superpowers/last.log 2>&1
rc=$?
grep -E "passed|failed|[Ee]rror|Success:|All checks|^E  |-->|^FAILED" .superpowers/last.log | tail -25
echo "rc=$rc"
exit $rc
