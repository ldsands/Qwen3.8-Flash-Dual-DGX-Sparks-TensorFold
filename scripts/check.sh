#!/usr/bin/env bash
# Offline recipe tests only; never calls start, prepare, Docker or SSH against real services.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
for script in start.sh stop.sh scripts/*.sh; do bash -n "$script"; done
uv run python -m unittest discover -s tests -v
uv run python tools/test_bench_stats.py
