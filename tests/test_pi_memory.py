"""Public-memory contracts: no simulator dependency and no hidden source fallback."""

import json
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageChops
import pytest

from src.harness import memory


MODULE = Path(memory.__file__)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "workspace"
    (root / "scene/maps").mkdir(parents=True)
    (root / "scene/evidence").mkdir()
    (root / "memory/entities").mkdir(parents=True)
    (root / "observations").mkdir()
    (root / "scratch/images").mkdir(parents=True)
    Image.new("RGB", (400, 200), "white").save(root / "scene/maps/L1.base.png")
    Image.new("RGB", (160, 100), "blue").save(root / "observations/f_001.png")
    put(root / "scene/maps/index.json", {"maps": [{"map_id": "map_L1",
        "base_ref": "scene/maps/L1.base.png", "current_entities_ref": "memory/maps/L1.entities.json",
        "width": 400, "height": 200}]})
    put(root / "memory/maps/L1.entities.json", [{"id": "O01", "map_id": "map_L1", "type": "object",
                                               "geometry": {"box": [0.2, 0.2, 0.4, 0.5]}}])
    (root / "memory/entities/O01.md").write_text("# O01\n- 地图：map_L1 中的 O01\n- 证据：`f_001`\n")
    row = {"id": "f_001", "path": "observations/f_001.png", "source": "observation", "step": 3,
           "view": {"right_degrees": 60, "pitch_degrees": 0}, "derived_from": []}
    (root / "observations/index.jsonl").write_text(json.dumps(row) + "\n")
    put(root / "memory/state.json", {"goal_id": "g1", "at_step": 3, "pose_estimate": {
        "map_id": "map_L1", "position": [0.6, 0.5], "heading_point": [0.6, 0.4],
        "at_frame": "f_001", "status": "tentative"}, "evidence_refs": ["f_001"]})
    return root


def test_delete_entity_never_restores_initial_copy(workspace):
    first = memory.render(workspace, "map_L1", layers=["walls", "object"])
    source = workspace / "memory/maps/L1.entities.json"
    put(workspace / "scene/initial/maps/L1.entities.json", json.loads(source.read_text()))
    put(source, [])
    (workspace / "memory/entities/O01.md").unlink()
    second = memory.render(workspace, "map_L1", layers=["walls", "object"])
    assert memory.check(workspace)["ok"]
    first_image = Image.open(workspace / first["images"][0])
    second_image = Image.open(workspace / second["images"][0])
    assert ImageChops.difference(first_image, second_image).getbbox()
    assert not ImageChops.difference(second_image, Image.new("RGB", second_image.size, "white")).getbbox()
    assert first["data"]["versions"]["memory/maps/L1.entities.json"] != second["data"]["versions"]["memory/maps/L1.entities.json"]


def test_unlocated_entity_is_listed_without_guessing(workspace):
    path = workspace / "memory/maps/L1.entities.json"
    items = json.loads(path.read_text())
    items[0]["geometry"] = None
    put(path, items)
    result = memory.render(workspace, "map_L1", layers=["walls", "object"])
    assert result["data"]["unlocated_ids"] == ["O01"]
    image = Image.open(workspace / result["images"][0])
    assert not ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()


def test_crop_mapping_and_capture_time_survive_reopening(workspace):
    rendered = memory.render(workspace, "map_L1", region=[0.2, 0.1, 0.8, 0.9])
    crop = memory.view(workspace, [rendered["data"]["id"]], crops=[[0.5, 0, 1, 1]])
    info = crop["data"]["images"][0]
    assert info["map_region"] == pytest.approx([0.5, 0.1, 0.8, 0.9])
    assert info["derived_from"] == [rendered["data"]["id"]]
    assert info["source_sha256"] == rendered["data"]["sha256"]
    observation = memory.view(workspace, ["f_001"], crops=[[0, 0, 0.5, 1]])
    reopened = memory.view(workspace, [observation["data"]["images"][0]["id"]])
    assert reopened["data"]["images"][0]["step"] == 3
    assert reopened["data"]["images"][0]["view"]["right_degrees"] == 60
    assert rendered["data"]["pose_at_frame"] == "f_001"


@pytest.mark.parametrize("ref", ["../../secret.png", "/etc/passwd", "memory/state.json", "scratch/escape.png"])
def test_image_access_rejects_private_or_traversal(workspace, tmp_path, ref):
    outside = tmp_path / "private.png"
    Image.new("RGB", (20, 20)).save(outside)
    (workspace / "scratch/escape.png").symlink_to(outside)
    with pytest.raises((memory.MemoryError, FileNotFoundError)):
        memory.view(workspace, [ref])


@pytest.mark.parametrize("readonly_root", ["scene", "observations"])
@pytest.mark.parametrize("link_kind", ["directory", "index"])
@pytest.mark.parametrize("operation", ["crop", "render"])
def test_derived_images_cannot_write_through_links_into_readonly_trees(
        workspace, readonly_root, link_kind, operation):
    before = {p.relative_to(workspace).as_posix(): p.read_bytes()
              for name in ("scene", "observations")
              for p in (workspace / name).rglob("*") if p.is_file()}
    images = workspace / "scratch/images"
    if link_kind == "directory":
        images.rmdir()
        images.symlink_to(workspace / readonly_root, target_is_directory=True)
    else:
        (images / "index.jsonl").symlink_to(workspace / readonly_root / "index.jsonl")
    with pytest.raises(memory.MemoryError, match="outside allowed resource roots"):
        if operation == "crop":
            memory.view(workspace, ["f_001"], crops=[[0, 0, 0.5, 1]])
        else:
            memory.render(workspace, "map_L1")
    after = {p.relative_to(workspace).as_posix(): p.read_bytes()
             for name in ("scene", "observations")
             for p in (workspace / name).rglob("*") if p.is_file()}
    assert after == before


def test_record_image_validates_index_destination_before_writing_image(workspace):
    images = workspace / "scratch/images"
    protected = workspace / "observations/index.jsonl"
    before = protected.read_bytes()
    (images / "index.jsonl").symlink_to(protected)
    with pytest.raises(memory.MemoryError, match="outside allowed resource roots"):
        memory._record_image(workspace, Image.new("RGB", (20, 20)), "crop", {})
    assert protected.read_bytes() == before
    assert not list(images.glob("*.png"))


def test_allowed_root_cannot_be_redefined_by_a_symlink(workspace):
    (workspace / "scratch/images").rmdir()
    (workspace / "scratch").rmdir()
    (workspace / "scratch").symlink_to(workspace / "scene", target_is_directory=True)
    with pytest.raises(memory.MemoryError, match="outside allowed resource roots"):
        memory._record_image(workspace, Image.new("RGB", (20, 20)), "crop", {})
    assert not (workspace / "scene/images").exists()


@pytest.mark.parametrize("mutation", ["coordinate", "duplicate", "missing_card", "dangling_evidence", "dangling_entity", "unknown_frame"])
def test_invalid_current_memory_is_reported_and_never_falls_back(workspace, mutation):
    path = workspace / "memory/maps/L1.entities.json"
    items = json.loads(path.read_text())
    put(workspace / "scene/initial/maps/L1.entities.json", items)
    if mutation == "coordinate":
        items[0]["geometry"]["box"][0] = -1
    elif mutation == "duplicate":
        items.append(dict(items[0]))
    elif mutation == "missing_card":
        (workspace / "memory/entities/O01.md").unlink()
    elif mutation == "dangling_evidence":
        (workspace / "memory/entities/O01.md").write_text("# O01\n- 证据：`not_a_frame`\n")
    elif mutation == "dangling_entity":
        (workspace / "memory/entities/O01.md").write_text("# O01\n- 所属区域：R99\n")
    else:
        state = json.loads((workspace / "memory/state.json").read_text())
        state["pose_estimate"]["at_frame"] = "f_future"
        put(workspace / "memory/state.json", state)
    put(path, items)
    assert not memory.check(workspace)["ok"]
    with pytest.raises(memory.MemoryError):
        memory.render(workspace, "map_L1")


def test_cli_check_runs_without_repository_imports(workspace):
    isolated = workspace / "scene/tools/memory.py"
    isolated.parent.mkdir()
    isolated.write_bytes(MODULE.read_bytes())
    result = subprocess.run([sys.executable, "-B", str(isolated), "check"],
                            input=json.dumps({"workspace": str(workspace)}), text=True,
                            capture_output=True, cwd=workspace)
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(result.stdout)["ok"]


@pytest.mark.parametrize("prefix", ["frame_", "rgb_", "map_", "crop_", "grid_"])
def test_unquoted_evidence_ids_are_checked_even_beside_valid_quotes(workspace, prefix):
    card = workspace / "memory/entities/O01.md"
    card.write_text(f"# O01\n- 证据：`f_001` 与 {prefix}999999\n")
    result = memory.check(workspace)
    assert not result["ok"]
    assert f"unknown evidence {prefix}999999" in result["errors"][0]


@pytest.mark.parametrize("label", ["所属区域", "连接", "关系"])
def test_explicit_custom_entity_references_are_checked_without_parsing_prose(workspace, label):
    path = workspace / "memory/maps/L1.entities.json"
    items = json.loads(path.read_text())
    items.append({"id": "Door_new", "map_id": "map_L1", "type": "portal", "geometry": None})
    put(path, items)
    (workspace / "memory/entities/Door_new.md").write_text("# Door_new\n- 地图：map_L1\n- 证据：未知\n")
    card = workspace / "memory/entities/O01.md"
    card.write_text(f"# O01\n- {label}：`Door_new`，另一侧 unknown hallway 尚未确认\n")
    assert memory.check(workspace)["ok"]
    card.write_text(f"# O01\n- {label}：`Door_missing`，另一侧 unknown hallway 尚未确认\n")
    result = memory.check(workspace)
    assert not result["ok"]
    assert "unknown entity reference Door_missing" in result["errors"][0]


@pytest.fixture
def archive(tmp_path):
    """Tiny authored scene with the same private calibration contract as Qa."""
    root = tmp_path / "archive"
    raw = root / "raw"
    raw.mkdir(parents=True)
    (root / "evidence").mkdir()
    Image.new("RGB", (32, 24), "red").save(root / "evidence/prior.jpg")
    region = {"id": "R01", "name": "卧室", "level": "L1", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
              "description": "窗边的卧室", "evidence_frames": ["E_01"]}
    put(raw / "scene.json", {"levels": [{"id": "L1", "height_m": 0}]})
    put(raw / "regions.json", {"regions": [region]})
    put(raw / "rooms.json", {"rooms": [region]})
    put(raw / "walls.json", {"segments": [{"level": "L1", "start": [0, 0], "end": [1, 0]}]})
    put(raw / "doors.json", {"doors": []})
    put(raw / "objects.json", {"objects": [{"id": "O01", "name": "柜子", "level": "L1", "room": "R01",
        "reference_instance_ids": [999], "evidence_frames": ["E_01"], "primitives": [
            {"type": "rectangle", "center": [0.5, 0.5], "size": [0.2, 0.2]}]}]})
    put(raw / "evidence.json", {"frames": [{"id": "E_01", "rgb": "../evidence/prior.jpg",
                                             "camera_position_xyz": [123, 456, 789], "visible_reference_instances": [999]}]})
    (raw / "floor_plan.svg").write_text('''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
      <defs><clipPath id="p"><rect x="0" y="0" width="100" height="100"/></clipPath></defs>
      <g id="axes_1"><g id="region-R01"><path d="M 10 90 L 90 90 L 90 10 L 10 10"
      clip-path="url(#p)" style="fill: #fff"/></g></g></svg>''')
    return root


def test_prepare_filters_sources_and_does_not_expose_private_metadata(archive, tmp_path):
    output = tmp_path / "new"
    result = memory.prepare(archive, output)
    assert result["data"]["entity_count"] == 2
    assert memory.check(output)["ok"]
    text = "\n".join(p.read_text() for p in output.rglob("*") if p.suffix in {".json", ".jsonl", ".md"})
    for prohibited in ("reference_instance", "camera_position", "visible_reference", "123", "999"):
        assert prohibited not in text
    assert Image.open(output / "scene/maps/L1.base.png").size == (1800, 1440)
    assert not (output / "scene/raw").exists()
    with pytest.raises(memory.MemoryError, match="overwrite"):
        memory.prepare(archive, output)


def test_partial_and_empty_prepare_do_not_reveal_unselected_sources(archive, tmp_path):
    partial = tmp_path / "partial"
    memory.prepare(archive, partial, mode="partial", allowed_ids=["O01"])
    assert memory.check(partial)["ok"]
    assert "所属区域：未知" in (partial / "memory/entities/O01.md").read_text()
    assert not (partial / "memory/entities/R01.md").exists()
    empty = tmp_path / "empty"
    result = memory.prepare(archive, empty, mode="empty")
    assert result["data"]["entity_count"] == 0
    assert result["data"]["evidence_count"] == 0
    assert memory.check(empty)["ok"]


def test_grid_has_sources_but_no_single_map_coordinate_frame(workspace):
    result = memory.view(workspace, ["f_001", "f_001"], layout="grid")
    assert result["data"]["grid"]["map_region"] is None
    assert len(result["images"]) == 1
    assert len(result["data"]["grid"]["sources"]) == 2


def test_png_only_view_does_not_require_or_create_map_geometry(tmp_path):
    workspace = tmp_path / "png-only"
    image = workspace / "scene/sample/task_map.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (120, 80), "white").save(image)
    result = memory.view(workspace, ["scene/sample/task_map.png"])
    assert result["images"] == ["scene/sample/task_map.png"]
    assert result["data"]["images"][0]["source"] == "unindexed"
    assert "map_id" not in result["data"]["images"][0]
    assert not (workspace / "scene/maps/index.json").exists()
    assert not (workspace / "memory").exists()


def test_grid_direction_labels_are_drawn_on_image(workspace, monkeypatch):
    drawn = []
    original = memory.ImageDraw.ImageDraw.text

    def record_text(self, xy, text, *args, **kwargs):
        drawn.append(text)
        return original(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(memory.ImageDraw.ImageDraw, "text", record_text)
    memory.view(workspace, ["f_001"], layout="grid")
    assert any("f_001" in text and "right 60 deg" in text and "pitch 0 deg" in text for text in drawn)
