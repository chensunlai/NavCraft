"""Public examples, private labels, and form-specific information boundaries."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


COVERAGE = {
    "A01": ("自身定位与定向", ["N1", "M1"], ["simple_qa", "interactive_qa"]),
    "A02": ("短程位姿更新", ["N1"], ["simple_qa"]),
    "A03": ("局部子目标与通路选择", ["N2"], ["simple_qa", "retrieval_qa"]),
    "A04": ("局部动作选择与执行", ["N3"], ["simple_qa", "interactive_ia"]),
    "A05": ("观察→俯视图", ["M1"], ["simple_qa", "retrieval_qa"]),
    "A06": ("俯视图→观察", ["M2"], ["simple_qa", "interactive_qa"]),
    "A07": ("地图目标／路线落实为行动", ["M2", "N2", "N3"], ["interactive_ia"]),
    "A08": ("观察目标经地图对应后选择走法", ["M1", "N2", "N3"], ["retrieval_qa", "interactive_ia"]),
}
FORMS = {"simple_qa": "简单 QA", "retrieval_qa": "查询 QA",
         "interactive_qa": "交互 QA", "interactive_ia": "交互 IA"}
ALLOWED_TOOLS = {
    "simple_qa": (),
    "retrieval_qa": ("read_text", "view_images"),
    "interactive_qa": ("read_text", "view_images", "nav_observe", "nav_step"),
    "interactive_ia": ("read_text", "view_images", "nav_observe", "nav_step"),
}


@dataclass
class AtomicTask:
    id: str
    cluster: str
    form: str
    observation: str
    question: str
    initial_images: list[str]
    resources: dict[str, str]
    anchor: str
    pose_condition: str
    private: dict = field(repr=False)
    scene: str = "QaLdnwvtxbs"

    def public(self):
        value = asdict(self)
        value.pop("private")
        value["capabilities"] = COVERAGE[self.cluster][1]
        value["tools"] = list(ALLOWED_TOOLS[self.form])
        return value

    def save(self, directory: Path):
        write_json(directory / "task.json", self.public())
        write_json(directory / "private" / "labels.json", self.private)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def checked_resource(ref: str, bank: dict[str, str], root: Path) -> Path:
    """Only explicitly provisioned resources, never arbitrary workspace files."""
    if ref not in bank:
        raise ValueError("Unknown resource; use a ref from the initial resource bank")
    path = (root / bank[ref]).resolve()
    if not path.is_relative_to(root.resolve()) or "private" in path.relative_to(root.resolve()).parts:
        raise ValueError("Resource is outside the public sample workspace")
    if not path.is_file():
        raise ValueError("Missing resource")
    return path


def score_answer(answer: dict, labels: dict) -> dict:
    anchored = answer.get("anchor") == labels["anchor"]
    correct = str(answer.get("choice", "")).strip() == str(labels["choice"])
    return {"passed": anchored and correct, "answer_anchor_ok": anchored,
            "choice_ok": correct, "method": labels["label_method"]}
