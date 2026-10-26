#!/usr/bin/env bash
# Manual publication of finished scenes; this does not resume the build.
BUILD_RUN="cache/build-maps/20261026T052701Z-sol-nonhssd100-full"
SOURCE_ROOT="data/scene_maps"
OUTPUT_ROOT="data/scene_map_datasets"
QUEUE="data/samples/scene_build_queue.json"
EXTRA_QUEUE="data/samples/scene_build_queue_nonhssd_100.json"
REPORT="cache/map-publish/$(date -u +%Y%m%dT%H%M%S%NZ)/report.json"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

set -euo pipefail
cd "$REPO_ROOT"

python -m src.dataset.maps publish \
  --source-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT" \
  --build-run "$BUILD_RUN" --queue "$QUEUE" --queue "$EXTRA_QUEUE" \
  --report "$REPORT"
