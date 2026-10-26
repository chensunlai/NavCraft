# Navigation task generation

Construct frozen tasks from real Habitat observations, collect teacher demonstrations,
and export the inputs and outputs actually recorded. `episodes/`, `build_maps/` and
the existing map adapter retain their separate responsibilities.

## Run

Activate the existing Habitat Python environment, build Agent Core after changing
its source, then run from the repository root:

```bash
bash scripts/generate_dataset.sh
```

The script composes `config/generation/pilot.yaml` with the existing model,
environment and tool configuration. It uses physical GPU 0, one Sol high worker,
three HM3D training-source scenes and eight task types. The final selected scene
is reserved for development; source-scene identity is retained independently of
its training/development assignment.

Run individual stages to inspect outputs before continuing:

```bash
python -m src.dataset.generation prepare --config config/generation/pilot.yaml
python -m src.dataset.generation collect --config config/generation/pilot.yaml
python -m src.dataset.generation derive --config config/generation/pilot.yaml
python -m src.dataset.generation check --config config/generation/pilot.yaml
python -m src.dataset.generation solve --config config/generation/pilot.yaml
python -m src.dataset.generation export --config config/generation/pilot.yaml
```

## Stages and ownership

| Stage | Output |
|---|---|
| `prepare` | Frozen scene selection, public wall maps, private instance labels and coordinate checks |
| `collect` | Actual script/tool interactions, immutable cuts and independently replayed short approaches |
| `derive` | Public task conditions, private labels, task quotas, teacher selection and `index.html` |
| `check` | Image/reference, mask alignment, source-group split, tool-pair and quota checks |
| `solve` | Independent attempts through `src.bench.subtasks`, existing Harness and native Pi |
| `export` | Verified training conversations, actual decision inputs/targets and deduplicated image assets |

The Dataset layer selects scenes, samples states, injects memory conditions and
creates labels. Bench loads these frozen records, executes and scores them. Both
teacher and student evaluations use the same subtask runner and pure scorer.

## Records

- `SceneRecord` binds a scene/floor to public walls and private instance labels.
- `Snapshot` references a complete idle state, including current frame, active
  goal, public workspace, issued IDs, paired history and private geometric data.
- `TaskSample` specifies a question/request or true continuation, allowed form,
  budgets, public memory/images, private label reference and source group.

Task templates are plain functions in the explicit `TASKS` registry. A template
returns public request fields and a private label. Shared derivation handles
selection, references, memory variants, quotas and provenance. Unsupported task
IDs fail explicitly. Add matching private scoring in Bench when adding a task.

## Pilot tasks

| ID | Form | Expected behavior |
|---|---|---|
| 5a | QA | Point to the requested category or answer `none` |
| 5c | QA | Match a reference crop from a different actual viewpoint |
| 6a | QA | Match a boxed object to an entity ID or answer `new` |
| 6c | QA | Decide whether two records represent one instance |
| 13d | QA | Give map-up relative to the current heading |
| 1b | Decision | Continue navigation and record a newly observed object |
| 1n | Decision | Continue without duplicating an existing instance |
| 14b | Execution | Reach a verified approach region and explicitly submit |

The engineering collection contains ten tasks per ID. Two fixed tasks per ID are
selected for teacher testing, with at most two independent attempts each. This
small collection is for pipeline acceptance, not a cross-scene benchmark claim.

QA keeps the active navigation target, permits existing-evidence queries and
scratch computation, and denies movement or formal-memory edits. Maintenance
starts after a real tool receipt without appending a new user message. Execution
uses the current sliding/turning implementation and actual microstep distance.

## Artifacts and supervision

```text
data/generated/<collection>/
  manifest.json, config.json, scenes.jsonl, snapshots.jsonl, samples.jsonl
  public/scenes/             # geometry-only maps
  private/scenes/            # semantic identities, mesh bounds, provenance
  snapshots/<id>/            # saved workspace and private restore data
  samples/<id>/public/       # question images and initial memory
  samples/<id>/private/      # labels, never mounted into model workspace
  reports/, index.html
  solutions.jsonl            # all teacher outcomes, including failed attempts
  exports/                   # verified training records and hashed assets
```

Teacher attempts live under the `runs/` directory recorded in `teacher_run.json`.
Working captures and locally staged scene assets live in `cache/generation/`.
Resume preserves frozen scenes, selection and attempt identity; changing the
teacher or frozen sample file requires a separate run/collection.

Only independently scored successful **training-split** demonstrations are
exported. Script prefixes, user messages, tool receipts and historical assistant
outputs remain input conditions. Current teacher text/tool calls are targets;
opaque reasoning/signatures are excluded. Each decision stores its actual model
input rather than reconstructing it from the final workspace.

Native semantic AABBs are not assumed accurate: private bounds are decoded from
semantic mesh vertices and checked against real depth/instance pixels. Nearby
approach locations are visually checked and reference actions replayed. New
memory records that cannot be verified are marked for review instead of being
automatically called hallucinations.
