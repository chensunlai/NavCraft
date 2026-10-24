# NavCraft Lite

Python navigation tools, spatial memory, Habitat execution and private benchmark scoring, driven by a local Pi Agent Core. A model receives RGB, continuous map pose, architectural walls, and its own initially empty entity memory.

## Setup

Use the existing Habitat Python environment and build the local Coding Agent SDK:

```bash
git submodule update --init src/agent-core/pi
npm ci --ignore-scripts
npm run build
export NAVCRAFT_PYTHON=/home/chen/.local/anaconda3/envs/vln/bin/python
"$NAVCRAFT_PYTHON" -m src.bench doctor
```

`data/` and `model/` point to local assets. `.env.harness` supplies `NAVCRAFT_TEST_BASE_URL` and `NAVCRAFT_TEST_API_KEY`; neither credentials nor assets enter Git. Model and reasoning settings live in `config/models/`.

## Run

```bash
# Offline semantic wall extraction; no model requests.
"$NAVCRAFT_PYTHON" -m src.bench prepare

# One frozen sample.
"$NAVCRAFT_PYTHON" -m src.bench run --model sol \
  --episode objnav:val:6s7QHgap2fW:chair:1

# Five ObjectNav and five R2R samples per model; dispatch alternates families.
"$NAVCRAFT_PYTHON" -m src.bench pilot --model all
```

Only physical GPUs 0 and 1 are permitted for simulation. The fixed pilot uses DeepSeek xhigh / Qwen27 medium / Sol high, with per-model concurrency 1 / 4 / 2. GOAT uses the same persistent session across subgoals, but is not part of this pilot.

Programmatic callers use `src.harness.open_session(...)` and successive `session.prompt(...)` calls. Tool definitions and execution live in Python; the TS Core is task independent. `submit` ends one request and never returns an evaluation outcome. Native Coding tools can read scene, observations and memory; only scratch is writable. Formal memory changes go through `memory_edit`.

## Boundaries and records

- `src/dataset`: official episode readers and offline semantic wall extraction.
- `src/environment`: fresh observations, panorama, adaptive atomic motion, private pose and microstep path.
- `src/harness`: seven tools, entity memory, map rendering, public state and session lifecycle.
- `src/agent-core`: Pi session, model transport, Coding sandbox, compaction and request ledger.
- `src/bench`: natural requests, scheduling, limits, submission and private SR/SPL/NE.

Each episode records `result.json`, public files under `workspace/`, and private prompts, scoring, actual model/tool events and content-addressed blobs under `private/`. Pi's native session is under `private/pi/sessions/`. Batch `summary.json` reports completion coverage as well as success; infrastructure errors remain explicit. Service outages retry unchanged model decisions without charging normal budgets or outage waiting time. Motion is never automatically replayed.

Walls come from original semantic triangles at floor+1m, not manually built scene maps. Incomplete source labels can leave gaps; blank map space is not a traversability guarantee.

## Checks

```bash
npm run check
npm run test:pi
"$NAVCRAFT_PYTHON" -m pytest -q
"$NAVCRAFT_PYTHON" scripts/check_harness.py local --gpu 0
"$NAVCRAFT_PYTHON" scripts/check_harness.py probe --model sol --gpu 0
"$NAVCRAFT_PYTHON" scripts/check_pilot_geometry.py --help
```

Integration probes and program-assisted physical references are recorded separately from model navigation scores. The Core keeps three recent image rounds and uses native compaction with a short visible summary plus the public factual head and two complete interaction rounds.
