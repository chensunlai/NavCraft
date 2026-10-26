"""Small on-disk contracts shared by generation and frozen-task evaluation."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

from src.common.io import read_json, write_json


@dataclass
class SceneRecord:
    id: str
    scene_id: str
    dataset: str
    split: str
    floor_height: float
    map_path: str
    instances_path: str
    start_pose: dict
    scene_dataset_config: str | None = None
    capabilities: list[str] = field(default_factory=list)


@dataclass
class Snapshot:
    id: str
    scene_id: str
    directory: str
    group_id: str
    source: str = "script"
    attributes: dict = field(default_factory=dict)


@dataclass
class TaskSample:
    id: str
    task_id: str
    form: str
    entry: str
    snapshot: str
    question: str
    inputs: dict
    limits: dict
    label_file: str
    group_id: str
    split: str = "train"
    attributes: dict = field(default_factory=dict)

    def __post_init__(self):
        from .tasks import TASKS
        if self.task_id not in TASKS:
            raise ValueError(f"Unsupported task: {self.task_id}")
        if self.form not in {"qa", "decision", "execution"}:
            raise ValueError(f"Unknown task form: {self.form}")
        if self.entry not in {"ask", "continue", "prompt"}:
            raise ValueError(f"Unknown task entry: {self.entry}")
        if self.form == "qa" and self.entry != "ask":
            raise ValueError("QA must use ask")


def stable_id(*values):
    raw = json.dumps(values, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def save_records(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            value = asdict(record) if hasattr(record, "__dataclass_fields__") else record
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_records(path, record_type=None):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return [record_type(**row) for row in rows] if record_type else rows


def local_file(root, reference):
    root = Path(root).resolve()
    path = (root / reference).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or external collection reference: {reference}")
    return path


def collection_manifest(root):
    return read_json(Path(root) / "manifest.json")


def update_manifest(root, **values):
    path = Path(root) / "manifest.json"
    previous = read_json(path) if path.exists() else {}
    write_json(path, {**previous, **values})
