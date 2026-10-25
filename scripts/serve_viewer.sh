#!/usr/bin/env bash
# Paths/configs below; simple options are written directly in the command.
DIRECTORY="outputs/sequence-viewer"
PUBLIC_HOST="10.82.1.221"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

set -euo pipefail
cd "$REPO_ROOT"

echo "Viewer: http://${PUBLIC_HOST}:8770"
python -m src.viewer.server --directory "$DIRECTORY" --host 0.0.0.0 --port 8770
