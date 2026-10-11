#!/usr/bin/env bash
set -euo pipefail

TASK_CONFIG=config/task/objectnav.yaml
ENV_CONFIG=config/environment/fake.yaml
MODEL_CONFIG=config/model/scripted.yaml
HARNESS_CONFIG=config/harness/minimal.yaml
RUN_CONFIG=config/run/fake.yaml

cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m src.bench.run \
    --task-config "$TASK_CONFIG" \
    --env-config "$ENV_CONFIG" \
    --model-config "$MODEL_CONFIG" \
    --harness-config "$HARNESS_CONFIG" \
    --run-config "$RUN_CONFIG"
