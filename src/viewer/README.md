# Offline sequence viewer

Read saved continuous-task results without starting Bench, Habitat or model requests.

```bash
python -m src.viewer.export runs/<comparison-directory> --output outputs/sequence-viewer
python -m src.viewer.server --directory outputs/sequence-viewer --host 0.0.0.0 --port 8770
```

The comparison directory contains one model folder per run, with `result.json`, observation files and the saved trace. Export produces an independent static site with compressed, deduplicated JPEG assets. The server exposes only that export directory.

The site includes a model comparison, task matrix, failure filter, first-person replay with speed control, wall/trajectory view, visible assistant text, tool results, and before/after memory with evidence images. Use arrow keys to step through frames and Space to play/pause. Click a matrix cell to open that model and round; changing the model preserves the selected round.

Exploration is excluded from navigation SR. Arrival, explicit submission and required route events appear separately. The displayed full-round path is private offline verification; entity records are model claims, including potentially incorrect positions and connections. No hidden reasoning or HTTP request payloads are exported.

Single-goal benchmark episodes can be opened directly as well:

```bash
python -m src.viewer.export runs/<batch>/astra/000-objnav --output outputs/astra-map-bench
python -m src.viewer.server --directory outputs/astra-map-bench --host 0.0.0.0 --port 8770
```

This reads the actual episode, result and trace into one displayed task; it does not create a synthetic scenario or modify the original run. Multi-goal benchmark episode export is not supported yet.
