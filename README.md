# NavCraft Lite

Python navigation tools, spatial memory, Habitat execution and private benchmark scoring, driven by a local Pi Agent Core. A model receives RGB, continuous map pose, architectural walls, and its own initially empty entity memory.

## Setup

Use the existing Habitat Python environment and build the local Coding Agent SDK:

```bash
git submodule update --init src/agent-core/pi
npm ci --ignore-scripts
npm run build
conda activate vln
bash scripts/doctor.sh
```

`data/` and `model/` point to local assets. `.env.harness-test` supplies `NAVCRAFT_TEST_BASE_URL` and `NAVCRAFT_TEST_API_KEY`; neither credentials nor assets enter Git. Model and reasoning settings live in `config/models/`.

## Run

Reusable Bash entrypoints are listed in [scripts/README.md](scripts/README.md). They accept no arguments: edit paths/configs at the top and simple options in the command, then run `bash scripts/<name>.sh`. They use the active `python` environment, never a fixed interpreter path. Provider, model, task, environment, budget and tool configs compose in order; later values override earlier ones. The module commands below remain available for programmatic use.

```bash
# One frozen sample.
python -m src.bench run --model sol \
  --episode objnav:val:6s7QHgap2fW:chair:1

# Five ObjectNav and five R2R samples per model; dispatch alternates families.
python -m src.bench pilot --model all

# Refresh a report from saved files, including while a run is in progress.
python -m src.bench report --output runs/<run-name>
```

Only physical GPUs 0 and 1 are permitted for simulation. The fixed pilot uses DeepSeek xhigh / Qwen27 medium / Sol high, with per-model concurrency 1 / 4 / 2. GOAT uses the same persistent session across subgoals, but is not part of this pilot.

Programmatic callers use `src.harness.open_session(...)` and successive `session.prompt(...)` calls. Tool definitions and execution live in Python; the TS Core is task independent. `submit` ends one request and never returns an evaluation outcome. Native Coding tools can read scene, observations and memory; only scratch is writable. Spatial entities change through `memory_edit`; the model maintains notes and scripts freely under `scratch/`. The [Harness guide](src/harness/README.md) documents the seven tools and their shared references.

For a continuous development check, load a frozen local task chain:

```bash
python -m src.bench.sequence \
  --scenario data/datasets/navcraft/qa_campaign_v2/scenario.json \
  --map cache/wall_maps/QaLdnwvtxbs-L1/map.json \
  --model-config <model-config.json> --output runs/<new-run> --gpu 1
```

This preserves one environment, Pi session and memory across all requests. Explicit relocations preserve history and do not count as travel. Each round saves navigation scores and memory changes separately; exploration is excluded from navigation success. Local task chains are development checks, separate from the frozen public benchmark subsets.

## Boundaries and records

- `src/dataset`: official episode readers; no map construction during evaluation.
- `src/environment`: fresh observations, panorama, adaptive atomic motion, private pose and microstep path.
- `src/harness`: seven tools, entity memory and free workspace notes, map rendering, public state and session lifecycle.
- `src/agent-core`: Pi session, model transport, Coding sandbox, compaction and request ledger.
- `src/bench`: natural requests, scheduling, limits, submission and private SR/SPL/NE.

Each episode records `result.json`, public files under `workspace/`, and private prompts, scoring, actual model/tool events and content-addressed blobs under `private/`. Pi's native session is under `private/pi/sessions/`. Batch `summary.json` reports completion coverage as well as success; infrastructure errors remain explicit. Service outages retry unchanged model decisions without charging normal budgets or outage waiting time. Motion is never automatically replayed.

Scene data has one home: `data/scene_maps/<dataset>/<split?>/<scene>/`. Original `scene.json`, `walls.json`, `wall_real.json`, rooms, objects and evidence live directly in that directory; there is no `raw/` wrapper or published copy. Bench is configured through `config/maps/scene_dataset.yaml` and reads these original files through the shared dataset loader.

The loader selects one floor and wall layer, converts the declared archive `(X,-Z)` metres into the Harness `(X,Z)` geometry, and freezes only the public geometry in the current run. These are read-time coordinate calculations; they do not rewrite the source dataset. World/UV calculations share `src/common/coordinates.py`, and source hashes are stored in the run's private provenance. See [the data guide](src/dataset/README.md).

Missing maps, ambiguous floors and absent layers produce explicit errors, without substitutes or automatic repair. HSSD assets match a unique scene directory; no official HSSD task adapter is introduced. Original scene annotations remain outside the model workspace. Each new run reads the current source files; existing runs retain their own input snapshots.

## Checks

```bash
npm run check
npm run test:pi
python -m pytest -q
python -m src.bench.checks.harness local --gpu 0
python -m src.bench.checks.harness probe --model sol --gpu 0
python -m src.bench.checks.pilot_geometry --help
```

Integration probes and program-assisted physical references are recorded separately from navigation scores. Context is appended unchanged between native compactions. At a compaction boundary, retain a short summary and two complete recent rounds, then append a public checkpoint containing the current request, pose and current/target images. There is no rolling image deletion or per-turn replacement of the prefix. Actual server-side cache hits still depend on the provider.

## Basic and super tools

Both tiers expose `observe`, `move`, `map_view`, `view_images`, `memory_read`, `memory_edit`, and `submit`, plus native Coding tools. Super adds image-depth projection, RGB map overlays, nearby-memory queries, and `move` modes `face` / `approach`; it does not add another set of tools. Basic permits atomic movement and map references but cannot project RGB pixels through depth.

All point-consuming operations share entity, displayed-image-pixel or global-UV references. `move(after="panorama")` returns the panorama directly; map views attach at most 12 related record summaries, and detailed memory reads at most two evidence images. Further detail remains available on request. Formal memory still validates structure and references, not semantic truth.

```bash
# Real Habitat tool replay; no model requests or benchmark score.
python -m src.bench.checks.super_tools --gpu 0 \
  --output cache/validation/super-tools-check
```

Only an observed point with preserved provenance can use the bounded local NavMesh planner. An arbitrary map point uses straight motion. Geometric arrival refers to the model-selected point, never task success. Depth stays private; tool projections do not certify object identity, glass or mirrors.
