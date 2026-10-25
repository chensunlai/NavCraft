#!/usr/bin/env bash
# Paths/configs below; simple options are written directly in the command.
RUN="runs/20261024T165128Z-astra-point-memory-super/astra"
OUTPUT="outputs/sequence-viewer"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

set -euo pipefail
cd "$REPO_ROOT"

python -m src.viewer.export "$RUN" --output "$OUTPUT"
