#!/usr/bin/env bash
COLLECTION="data/generated/short-tasks-20261027T115918Z"
QWEN_CONFIG="config/models/qwen27.yaml"
SOL_CONFIG="config/models/sol.yaml"
HARNESS_CONFIG="config/harness/entity_geometry.yaml"
OUTPUT="runs/$(date -u +%Y%m%dT%H%M%SZ)-short-tasks-qwen-sol"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

set -euo pipefail
cd "$REPO_ROOT"
python -m src.bench.short_batch --collection "$COLLECTION" --output "$OUTPUT" \
  --model-config "$QWEN_CONFIG" --model-config "$SOL_CONFIG" --repeats 4 2 \
  --harness-config "$HARNESS_CONFIG" --concurrency 8
