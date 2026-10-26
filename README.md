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

Programmatic callers use `src.harness.open_session(...)` and successive `session.prompt(...)` calls. Tool definitions and execution live in Python; the TS Core is task independent. `submit` ends one request and never returns an evaluation outcome. Native Coding tools can read scene, observations and memory; only scratch is writable. Spatial entities change through `memory_edit`; task notes change through `task_memory`.

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
- `src/harness`: eight tools, entity and task memory, map rendering, public state and session lifecycle.
- `src/agent-core`: Pi session, model transport, Coding sandbox, compaction and request ledger.
- `src/bench`: natural requests, scheduling, limits, submission and private SR/SPL/NE.

Each episode records `result.json`, public files under `workspace/`, and private prompts, scoring, actual model/tool events and content-addressed blobs under `private/`. Pi's native session is under `private/pi/sessions/`. Batch `summary.json` reports completion coverage as well as success; infrastructure errors remain explicit. Service outages retry unchanged model decisions without charging normal budgets or outage waiting time. Motion is never automatically replayed.

Bench reads published maps from `data/scene_map_datasets/`, configured in `config/maps/scene_dataset.yaml`. HM3D and MP3D scene-relative directories match the original assets under `data/nas/scene_datasets/`. The dataset adapter selects the start floor, converts `(X,-Z)` walls to the Harness `(X,Z)` map, and saves a geometry-only `private/map.json` plus `private/map_source.json` with source hashes. The source map files remain unchanged. `doctor` reports missing/invalid maps; evaluation refuses these jobs rather than falling back to an old cache.

The map layer is `walls` by default; set `maps.layer: wall_real` to use observed geometry. Floor matching currently requires exactly one floor within 0.5 m of the episode start height; existing single-floor task checks remain in force. This does not add multi-floor navigation. No descriptions, object/door IDs, evidence labels or goal locations enter the public map.

Sequence runs accept either a published scene directory or an explicitly supplied normalized `map.json` through `--map`. The old Qa development sequence still uses its explicit cached map until that scene is published in the new dataset. HSSD maps can be loaded through an explicit directory; no official HSSD task adapter is added here. Blank map space is not a traversability guarantee.

The former semantic wall-slice builder and its tests are archived in `cache/archive/` (local files only). Wall construction is no longer a Bench command or runtime dependency.

## Checks

```bash
npm run check
npm run test:pi
python -m pytest -q
python -m src.bench.checks.harness local --gpu 0
python -m src.bench.checks.harness probe --model sol --gpu 0
python -m src.bench.checks.pilot_geometry --help
```

Integration probes and program-assisted physical references are recorded separately from model navigation scores. The Core keeps three recent image rounds and uses native compaction with a short visible summary plus the public factual head and two complete interaction rounds.

## Optional geometric tools

Set `tool_tier: super` in a model YAML to add `locate`, `nav_face`, and `nav_goto`, plus ego maps, RGB-D overlays and pixel-based memory edits. Door locations use one point, like objects; room anchors remain distinct. Room and door markers and labels are emphasized, and map labels avoid overlap with leader lines. Observation images have no pixel-grid overlay. Default `basic` keeps the eight basic tools and RGB sensor condition. Bench passes the same tier to Environment and Harness; programmatic callers must do so too.

Super depth is stored privately as uint16 PNG with 0.001 metres per integer. A valid projection means geometric data is available; it does not certify object identity, mirrors or glass. `goto_planner: local` uses a same-floor NavMesh route within 8m only for points originating from observations; arbitrary UV uses straight motion. `goto_planner: straight` disables route planning. Arrival feedback refers only to the model-selected point, and does not submit the task or reveal its score.

```bash
# Physical tool validation; no model calls. Use a new output directory each time.
python -m src.bench.checks.super_tools --gpu 1 \
  --output cache/validation/super-tools-check
```

`locate` returns a single annotated map with temporary purple P1/P2 markers and compact UV results; it does not write memory. Doorway pixels can hit surfaces beyond the opening, so use the projected 2D map to place entrances. No-motion geometric navigation replies do not repeat the current RGB.

## Task notes

`task_memory({"action":"edit","content":"Inspect D02; the left approach was blocked. Try the right approach."})` replaces the entire notebook; the receipt is `{"updated":true}`. `task_memory({"action":"read"})` returns `{"content":"..."}`. Empty content clears it. The text lives in `workspace/memory/task.json`, survives successive requests, and is included in the public checkpoint after compaction. It is model-written working memory, not verified evidence or an evaluation result. Goal changes do not silently erase it; the model updates its plan. `initial_state.task_memory` can seed the text. Basic and super tiers both provide this tool.

