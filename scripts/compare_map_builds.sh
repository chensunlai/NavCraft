#!/usr/bin/env bash
# Paths/configs below; simple options are written directly in the command.
PROVIDER_CONFIG="config/providers/221.yaml"
MODEL_CONFIGS=("config/models/sol.yaml" "config/models/astra.yaml")
TASK_CONFIG="config/tasks/scene_maps.yaml"
RUNTIME_CONFIG="config/build_maps/runtime.yaml"
STAGE_CONFIG="config/build_maps/full.yaml"
SCENE="LT9Jq6dN3Ea"
RUN_ID="$(date -u +%Y%m%dT%H%M%S%NZ)"
OUTPUT="cache/build-maps/${RUN_ID}-comparison"
BUILD_CODEX_HOME_ROOT="data/scene_maps/codex/${RUN_ID}-comparison"
ARCHIVE_ROOT="data/scene_maps/comparisons/${RUN_ID}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

set -euo pipefail
cd "$REPO_ROOT"

# Each model gets its own archive, raw workspace, Codex home and logs.
PIDS=()
for INDEX in "${!MODEL_CONFIGS[@]}"; do
  MODEL_CONFIG="${MODEL_CONFIGS[$INDEX]}"
  NAME="$(basename "$MODEL_CONFIG" .yaml)"
  python -m src.dataset.build_maps run \
    --config "$MODEL_CONFIG" --config "$PROVIDER_CONFIG" --config "$TASK_CONFIG" \
    --config "$RUNTIME_CONFIG" --config "$STAGE_CONFIG" --scene "$SCENE" \
    --gpu "$((INDEX % 2))" --concurrency 1 \
    --codex-home "$BUILD_CODEX_HOME_ROOT/$NAME" --output "$OUTPUT/$NAME" \
    --archive-root "$ARCHIVE_ROOT/$NAME" &
  PIDS+=("$!")
done
STATUS=0
for PID in "${PIDS[@]}"; do wait "$PID" || STATUS=1; done
exit "$STATUS"
