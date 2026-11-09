#!/usr/bin/env bash
# Paths and model settings are composed from the existing configs.
MODELS_CONFIG="config/models/pilot.yaml"
TASK_CONFIG="config/tasks/goat_single_floor.yaml"
ENV_CONFIG="config/environments/habitat.yaml"
BUDGET_CONFIG="config/budgets/navigation.yaml"
PROVIDER_CONFIG="config/providers/221.yaml"
HARNESS_CONFIG="config/harness/node_geometry.yaml"
RUNTIME_CONFIG="config/runtime/navigation.yaml"
EPISODE="goat:val_unseen:5cdEh9F2hJL:3"
RUN_ID="$(date -u +%Y%m%dT%H%M%S%NZ)"
OUTPUT="runs/${RUN_ID}-sol-node-goat"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

set -euo pipefail
cd "$REPO_ROOT"

python -m src.bench run --config "$MODELS_CONFIG" --config "$ENV_CONFIG" --config "$TASK_CONFIG" \
  --config "$BUDGET_CONFIG" \
  --model-config "$PROVIDER_CONFIG" --model-config "$HARNESS_CONFIG" \
  --model-config "$RUNTIME_CONFIG" \
  --model sol --episode "$EPISODE" --output "$OUTPUT"
