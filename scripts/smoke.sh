#!/usr/bin/env bash
set -euo pipefail

TASK_CONFIGS=(config/task/objectnav.yaml config/task/r2r.yaml config/task/rxr.yaml config/task/goat.yaml)
ENV_CONFIGS=(config/environment/objectnav.yaml config/environment/r2r.yaml config/environment/rxr.yaml config/environment/goat.yaml)
MODEL_CONFIG=config/model/deepseek.yaml
HARNESS_CONFIG=config/harness/minimal.yaml
RUN_CONFIG=config/run/smoke.yaml

cd "$(dirname "${BASH_SOURCE[0]}")/.."
FAILED=0
for INDEX in "${!TASK_CONFIGS[@]}"; do
    TASK_CONFIG="${TASK_CONFIGS[$INDEX]}"
    ENV_CONFIG="${ENV_CONFIGS[$INDEX]}"
    if ! python -m src.bench.run \
        --task-config "$TASK_CONFIG" \
        --env-config "$ENV_CONFIG" \
        --model-config "$MODEL_CONFIG" \
        --harness-config "$HARNESS_CONFIG" \
        --run-config "$RUN_CONFIG"; then
        FAILED=1
    fi
done
exit "$FAILED"
