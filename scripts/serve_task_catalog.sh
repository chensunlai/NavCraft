#!/usr/bin/env bash
# Run inside the existing project Python environment.
COLLECTION="data/generated/spatial-pilot-20261026"
OUTPUT="outputs/task-catalog"
PUBLIC_HOST="10.82.1.221"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

set -euo pipefail
cd "$REPO_ROOT"

python -m src.viewer.tasks --collection "$COLLECTION" --output "$OUTPUT"
echo "Task catalog: http://${PUBLIC_HOST}:8771"
python -m src.viewer.server --directory "$OUTPUT" --host 0.0.0.0 --port 8771
