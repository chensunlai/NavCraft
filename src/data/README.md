# Architectural wall-map reconstruction

This builder reads raw HM3D/MP3D meshes and optional semantic priors. It does not
import Harness or modify scene assets. Geometry, render evidence, rejected
proposals and model-call audits remain separate.

The current target is a connected architectural floor plan, not a raw wall
projection. Start with one saved development scene; final bulk generation is
outside the current scope:

```bash
python -m src.data.map_floorplan \
  --source runs/map-exploration-20261024-geometric/final/00-hm3d-DNWbUAJYsPy \
  --output runs/my-wall-fit/DNWbUAJYsPy
python -m src.data.map_structure_audit \
  --fitted runs/my-wall-fit/DNWbUAJYsPy --output runs/my-wall-fit/audit.json
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python -m src.data.map_gap_audit \
  --source runs/map-exploration-20261024-geometric/final/00-hm3d-DNWbUAJYsPy \
  --fitted runs/my-wall-fit/DNWbUAJYsPy --output runs/my-wall-fit/opening-audit --gpu 0
```

Use the `vln` Python for these commands. Fitting and structural auditing are
CPU-only; gap auditing creates one real Habitat instance on the selected GPU.
Each new output directory must be empty. Opening auditing updates that fitted
candidate's opening/region records and saves the previous records alongside it.

- `map_instances.py`: instance-level architectural semantic priors and sources.
- `map_wall_fit.py`: supported lines/arcs without forcing axis-aligned walls.
- `map_wall_structure.py`: evidence-gated line/arc joins, junction graph and
  PNG/SVG rendering; a straight wall can meet a curve without flattening it.
- `map_wall_recovery.py`: strict, aligned raw support for missing straight spans.
- `map_window_boundaries.py`: reconstruct raised-window wall continuity from
  a complete raw frame before testing its contained wall gaps. Relative support
  is required on all four edges, including short windows. Structural inference is recorded
  separately from measured wall surfaces; observed passage or walkable-space
  conflicts reject the proposed boundary.
- `map_floorplan.py`: one-scene fitting, independent raw checks and real-RGB overlay.
- `map_wall_cad.py`: canonical line/circle/arc coverage and overlap diagnostics.
  A continuous wall has one drawing entity, including internal T-junction nodes.
  Near repeated coverage is a hard pending issue, even when exact overlap is zero.
  Parallel fits only deduplicate after an independent same-surface check;
  distinct measured planes remain separate.
- `map_wall_axes.py`: one axis for two independently confirmed physical wall
  faces. Raw outward-facing surfaces, vertical extent, area, door and NavMesh
  checks prevent merging corridor sides. Finite confirmed slabs can absorb
  contained scan fragments without extending the wall or crossing a door.
- `map_wall_refit.py`: jointly fit original cuts after independent same-surface
  confirmation; bound source residual and old-line displacement, preserve the
  finite coverage union, then rebuild junctions. A close offset alone is not
  permission to refit two distinct measured surface sheets.
- `map_cad_review.py`: small, audited jamb-point and wall-pair tasks. Saved wall
  reviews bind to the exact source geometry and verified provider model; a
  visual decision cannot override missing geometry or a passage conflict.
- `map_gap_audit.py`: contiguous raw clearance, actual robot-width crossing,
  paired views and compressed depth evidence; unresolved gaps are retained.
- `map_structure_audit.py`: unresolved endpoints/gaps, without repairing the map.

All outputs remain development candidates until structural and visual review.
Door lintels alone do not establish a wall at body height. Window or semantic
labels alone do not justify filling scan holes. A supported rectangular window
surround can justify continuity through scan damage without claiming that every
point on the inferred boundary was directly measured. Missing wall thickness is
stored as null and is not invented by widening the rendered line.

PNG and SVG always use a single stroke. Measured physical thickness is metadata,
not two parallel outlines. Full circular walls use an explicit circle; partial
arcs remain partial. Removing an apparent spur requires evidence that it belongs
to a confirmed wall slab or a nonarchitectural surface, not just a length cutoff.

For the saved small wall-pair experiment, add
`--pair-reviews runs/map-cad-20261024/pair-review/results.json` to the fitting
command. It is an optional offline evidence file, not a live model loop. Changed
geometry invalidates old decisions. Point proposals are separately reviewed and
never write directly to maps. The CAD checks only establish coverage consistency;
room structure, semantic correctness and real doorways still require inspection.

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
