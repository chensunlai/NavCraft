# Task atlas

Read-only website for the 70 task designs in `docs/TaskCatalog.md` and
`docs/Tasks-v2.md`: 67 atomic tasks and three chained tasks, grouped by capability.
Each task shows its input, expected output, entry mode, scoring, dependencies and
an illustrative answer. Eight implemented types additionally show all 80 frozen
real samples, public images/memory, private review labels and actual teacher runs.

```bash
bash scripts/serve_task_catalog.sh
# http://10.82.1.221:8771
```

Use the project Python environment. To build without serving:

```bash
python -m src.viewer.tasks \
  --collection data/generated/spatial-pilot-20261026 \
  --output outputs/task-catalog
```

The output is a self-contained static directory. Serving reuses the existing
public-artifact server, which exposes only this export. It never starts Habitat,
Bench or a model, and does not change the frozen collection or running jobs.

`catalog.json` contains the design descriptions. `examples.py` packages actual
allowlisted images, memory and teacher records from the selected collection.
Images are deduplicated and compressed for display; QA pixel coordinates still
refer to the original 640×480 observations. Planned examples are labeled as
illustrations, while script prefixes, private review labels and teacher output
are shown separately. The UI distinguishes original catalog specifications from
the frozen pilot's actual budgets and scoring.

Questions and instructions display Chinese translations by default, with the
original English in a disclosure. Translations are presentation fields only;
saved model inputs, labels, tool arguments and training records stay unchanged.
