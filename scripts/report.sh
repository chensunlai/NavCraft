#!/usr/bin/env bash
# Paths/configs below; simple options are written directly in the command.
RUN="runs/20261024T012029Z-lite-pilot"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

set -euo pipefail
cd "$REPO_ROOT"

python -m src.bench report --output "$RUN"
