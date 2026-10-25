#!/usr/bin/env bash
# Paths/configs below; simple options are written directly in the command.
TASK_CONFIG="config/tasks/qa_sequence.yaml"
MODEL_CONFIG="config/models/astra.yaml"
PROVIDER_CONFIG="config/providers/221.yaml"
TOOLS_CONFIG="config/tools/super.yaml"
RUNTIME_CONFIG="config/runtime/navigation.yaml"
RUN_ID="$(date -u +%Y%m%dT%H%M%S%NZ)"
OUTPUT="runs/${RUN_ID}-sequence"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

set -euo pipefail
cd "$REPO_ROOT"

python -m src.bench.sequence --config "$TASK_CONFIG" \
  --model-config "$MODEL_CONFIG" --model-config "$PROVIDER_CONFIG" \
  --model-config "$TOOLS_CONFIG" --model-config "$RUNTIME_CONFIG" \
  --gpu 0 --output "$OUTPUT"
