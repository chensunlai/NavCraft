#!/usr/bin/env bash
# Paths/configs below; simple options are written directly in the command.
PROVIDER_CONFIG="config/providers/221.yaml"
MODEL_CONFIG="config/models/sol.yaml"
TASK_CONFIG="config/tasks/scene_maps.yaml"
RUNTIME_CONFIG="config/build_maps/runtime.yaml"
STAGE_CONFIG="config/build_maps/walls.yaml"
RUN_ID="$(date -u +%Y%m%dT%H%M%S%NZ)"
OUTPUT="cache/build-maps/${RUN_ID}-sol-walls"
BUILD_CODEX_HOME="data/scene_maps/codex/${RUN_ID}-sol-walls"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

set -euo pipefail
cd "$REPO_ROOT"

# Change run to plan/probe as needed. Optional: --scene ID --limit N.
# Resume: reuse the exact OUTPUT/BUILD_CODEX_HOME and add --resume.
python -m src.dataset.build_maps run \
  --config "$MODEL_CONFIG" --config "$PROVIDER_CONFIG" --config "$TASK_CONFIG" \
  --config "$RUNTIME_CONFIG" --config "$STAGE_CONFIG" \
  --codex-home "$BUILD_CODEX_HOME" --output "$OUTPUT" \
  --concurrency 16 --gpu 0 --start 1
