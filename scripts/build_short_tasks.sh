#!/usr/bin/env bash
SOURCE="data/generated/spatial-pilot-20261026"
OUTPUT="data/generated/short-tasks-$(date -u +%Y%m%dT%H%M%SZ)"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

set -euo pipefail
cd "$REPO_ROOT"
python -m src.dataset.short_tasks --source "$SOURCE" --output "$OUTPUT" --gpu 0
