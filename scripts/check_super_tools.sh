#!/usr/bin/env bash
# Paths/configs below; simple options are written directly in the command.
EPISODE="objnav:val:LT9Jq6dN3Ea:bed:14"
RUN_ID="$(date -u +%Y%m%dT%H%M%S%NZ)"
OUTPUT="cache/validation/${RUN_ID}-super-tools"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

set -euo pipefail
cd "$REPO_ROOT"

python -m src.bench.checks.super_tools --episode "$EPISODE" --gpu 1 --output "$OUTPUT"
