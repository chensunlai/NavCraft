"""Build a standalone website from the task catalog and frozen local samples."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .examples import build_examples


def build(collection, output):
    collection, output = Path(collection).resolve(), Path(output).resolve()
    if output == collection or output.is_relative_to(collection):
        raise ValueError("The viewer output must be separate from the frozen collection")
    source = Path(__file__).parent
    catalog = json.loads((source / "catalog.json").read_text())
    identifiers = [task["id"] for task in catalog["tasks"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Task IDs must be unique")
    output.mkdir(parents=True, exist_ok=True)
    for path in (source / "web").iterdir():
        if path.is_file():
            shutil.copyfile(path, output / path.name)
    examples = build_examples(collection, output)
    catalog["generated_at"] = datetime.now(timezone.utc).isoformat()
    catalog["example_collection"] = collection.name
    for name, value in (("catalog.json", catalog), ("examples.json", examples)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    return {"output": str(output), "task_types": len(identifiers),
            "examples": sum(len(rows) for rows in examples["tasks"].values())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.collection, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
