#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG=config/generation/pilot.yaml

python -m src.dataset.generation prepare --config "$CONFIG"
python -m src.dataset.generation collect --config "$CONFIG"
python -m src.dataset.generation derive --config "$CONFIG"
python -m src.dataset.generation check --config "$CONFIG"
python -m src.dataset.generation solve --config "$CONFIG"
python -m src.dataset.generation export --config "$CONFIG"
