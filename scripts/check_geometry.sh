#!/usr/bin/env bash
# Paths/configs below; simple options are written directly in the command.
TASK_CONFIG="config/tasks/pilot.yaml"
ENV_CONFIG="config/environments/habitat.yaml"
RUN_ID="$(date -u +%Y%m%dT%H%M%S%NZ)"
OUTPUT="cache/validation/${RUN_ID}-pilot-geometry"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

set -euo pipefail
cd "$REPO_ROOT"

python -m src.bench.checks.pilot_geometry --config "$ENV_CONFIG" --config "$TASK_CONFIG" \
  --gpu 1 --output "$OUTPUT"
