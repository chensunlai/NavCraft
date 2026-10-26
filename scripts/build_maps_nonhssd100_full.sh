#!/usr/bin/env bash
# Continue the existing 100 scene threads from walls to complete maps.
PROVIDER_CONFIG="config/providers/221.yaml"
MODEL_CONFIG="config/models/sol.yaml"
TASK_CONFIG="config/tasks/scene_maps_nonhssd100.yaml"
RUNTIME_CONFIG="config/build_maps/runtime.yaml"
STAGE_CONFIG="config/build_maps/full.yaml"
OUTPUT="cache/build-maps/20261026T052701Z-sol-nonhssd100-full"
BUILD_CODEX_HOME="cache/build-maps/codex/20261026-sol-nonhssd100-walls"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

set -euo pipefail
cd "$REPO_ROOT"

python -m src.dataset.build_maps run \
  --config "$MODEL_CONFIG" --config "$PROVIDER_CONFIG" --config "$TASK_CONFIG" \
  --config "$RUNTIME_CONFIG" --config "$STAGE_CONFIG" \
  --codex-home "$BUILD_CODEX_HOME" --output "$OUTPUT" \
  --concurrency 16 --gpus 0 1 --gpu 0 --max-turns 3 --turn-timeout 14400 --resume
