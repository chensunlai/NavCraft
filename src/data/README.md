# Wall-map exploration

This builder reads raw HM3D/MP3D meshes and optional semantic priors. It does not
import Harness or modify scene assets. Geometry, render evidence, rejected
proposals and model-call audits remain separate.

Use the `vln` Python environment and a fresh worker on physical GPU 0 or 1:

```bash
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  /home/chen/.local/anaconda3/envs/vln/bin/python -m src.data.map_exploration development \
  --manifest runs/map-exploration-20261024-geometric/final/development_manifest.json \
  --output runs/map-development-reproduction --gpu 1
python -m src.data.map_exploration audit --output runs/map-development-reproduction
python -m src.data.map_report --run runs/map-exploration-20261024-geometric
```

The frozen development manifest contains six HM3D training scenes and four
MP3D R2R training scenes, exact episode starts, source hashes and asset paths.
The CLI starts one native renderer per scene and releases it before the next.
`audit` and `map_report` use saved evidence without model or GPU requests.

- `map_geometry.py`: raw GLB node transforms, optional semantic priors, slices.
- `map_scene.py`: rebuilt robot NavMesh, real RGB/depth, calibration and probes.
- `map_exploration.py`: four geometric baselines, output files, comparisons.
- `map_openings.py`: geometric passage checks and candidate region partitions.
- `map_quality.py`, `map_leaks.py`: independent checks and detour diagnostics.
- `map_model_review.py`, `map_experiments.py`: six model experiment forms and
  geometry-gated proposals. `--modes active` limits a targeted recheck;
  `--score-existing` evaluates stored responses without additional inference.
- `map_report.py`: deduplicated request costs and outcome summaries.

Coordinates are Habitat X/Z, image right +X/down +Z, origin at the outer
upper-left pixel corner. Resolution is exactly 0.05m/pixel. No artificial
perimeter wall is added. Region fragments below 0.1m² are discarded, and
remaining regions are candidates, not room ground truth.

Read `docs/MapReport.md` before bulk generation. The old preliminary
`data/maps/map-exploration-v1` directory is not validated and is not used here.
The configured service returned `deepseek-v4.1-flash-1017` for requests named
`deepseek-flash`; those attempts are disclosed and excluded from an approved
Flash comparison. Do not continue that alias until its backend is corrected.
