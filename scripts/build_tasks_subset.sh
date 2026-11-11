#!/usr/bin/env bash
set -euo pipefail

CONFIG="config/tasks/tasks_subset.yaml"
OUTPUT="cache/tasks_subset/subset"

conda run --no-capture-output -n vln python -m src.dataset.tasks_subset --config "$CONFIG" --output "$OUTPUT" --gpu 0
