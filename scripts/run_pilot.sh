#!/usr/bin/env bash
# Paths/configs below; simple options are written directly in the command.
MODELS_CONFIG="config/models/pilot.yaml"
TASK_CONFIG="config/tasks/pilot.yaml"
ENV_CONFIG="config/environments/habitat.yaml"
BUDGET_CONFIG="config/budgets/navigation.yaml"
PROVIDER_CONFIG="config/providers/221.yaml"
TOOLS_CONFIG="config/tools/basic.yaml"
RUNTIME_CONFIG="config/runtime/navigation.yaml"
RUN_ID="$(date -u +%Y%m%dT%H%M%S%NZ)"
OUTPUT="runs/${RUN_ID}-pilot"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

set -euo pipefail
cd "$REPO_ROOT"

python -m src.bench pilot --config "$MODELS_CONFIG" --config "$ENV_CONFIG" \
  --config "$BUDGET_CONFIG" --config "$TASK_CONFIG" \
  --model-config "$PROVIDER_CONFIG" --model-config "$TOOLS_CONFIG" --model-config "$RUNTIME_CONFIG" \
  --model all --output "$OUTPUT"
