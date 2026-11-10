"""Freeze model inputs separately from construction evidence."""
import json
from pathlib import Path
import hashlib
import re
import shutil
from PIL import Image


def snapshot_digest(snapshot):
    """Hash the complete snapshot tree independent of its location."""
    snapshot = Path(snapshot)
    state = snapshot / "state.json"
    if snapshot.is_symlink() or not state.is_file() or state.is_symlink():
        raise ValueError("Expected a snapshot directory with a regular state.json")
    digest = hashlib.sha256()
    for path in sorted(snapshot.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in snapshot: {path.relative_to(snapshot)}")
        name = path.relative_to(snapshot).as_posix().encode()
        digest.update(len(name).to_bytes(8, "big") + name)
        if path.is_dir():
            digest.update(b"D")
            continue
        if not path.is_file():
            raise ValueError(f"Nonregular snapshot entry: {path.relative_to(snapshot)}")
        digest.update(b"F" + path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def require_trial_snapshot(root, state, trial):
    if trial.get("invalidated") or not trial.get("start_snapshot_sha256"):
        raise ValueError("Trial lacks a valid start snapshot binding")
    if trial["start_snapshot_sha256"] != snapshot_digest(Path(root) / state["snapshot"]):
        raise ValueError("Trial start snapshot has changed")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def reject(root, kind, candidate, reason, **details):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    record = {**details, "kind": kind, "candidate": candidate, "reason": reason}
    record.setdefault("stage", "construction")
    with (root / "rejections.jsonl").open("a") as out:
        out.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return record


def freeze_case(root, task, label, construction, images):
    root = Path(root).resolve()
    identifier = task["id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identifier):
        raise ValueError(f"Invalid case ID: {identifier!r}")
    directory = root / "cases" / identifier
    if directory.exists():
        raise FileExistsError(directory)
    for field in ("snapshot", "episode"):
        if field not in task or task[field] is None:
            continue
        value = Path(task[field])
        location = (directory / value).resolve()
        if value.is_absolute() or not location.is_relative_to(root):
            raise ValueError(f"Case {field} is outside the collection")
        if field == "snapshot" and not (location / "state.json").is_file():
            raise FileNotFoundError(location / "state.json")
        if field == "episode" and not location.is_file():
            raise FileNotFoundError(location)
    sources = []
    for image in images:
        source = (root / image["path"]).resolve(strict=True)
        if source.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise ValueError(f"Expected a display image: {source.name}")
        with Image.open(source) as loaded:
            if loaded.mode != "RGB":
                raise ValueError(f"Expected RGB evidence: {source.name}")
            loaded.verify()
        sources.append((source, image["caption"]))
    frozen = {**task, "images": [], "label": "private/label.json"}
    try:
        directory.mkdir(parents=True)
        for index, (source, caption) in enumerate(sources, 1):
            relative = f"public/image-{index:02d}{source.suffix.lower()}"
            destination = directory / relative
            destination.parent.mkdir(exist_ok=True)
            with Image.open(source) as loaded:
                rgb = Image.frombytes("RGB", loaded.size, loaded.tobytes())
                rgb.save(destination)
            frozen["images"].append({"path": relative, "caption": caption,
                                     "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
        write_json(directory / "private/label.json", label)
        write_json(directory / "construction.json", construction)
        write_json(directory / "task.json", frozen)
    except Exception:
        shutil.rmtree(directory)
        raise
    return {**frozen, "path": str((directory / "task.json").relative_to(root))}


def current_observation_frames(bundle, state):
    """Return the current view and complete same-pose panoramas already in the cut."""
    from src.harness.spatial.observations import Observations

    keys = state["frame_keys"]
    current = state["current_frame"]
    if current not in keys:
        return []
    frames = bundle["frames"]
    anchor = frames[current]
    groups = {}
    for key in keys:
        row = frames[key]
        group = row.get("group")
        if (isinstance(group, str) and group.startswith("p_") and row.get("step") == state.get("step")
                and row.get("pose", {}).get("position") == anchor.get("pose", {}).get("position")
                and anchor.get("pose", {}).get("position") is not None):
            groups.setdefault(group, []).append(key)
    complete = {key for group in groups.values()
                if len(group) == 6 and {frames[k].get("yaw") for k in group} == set(range(0, 360, 60))
                and all(Observations._valid_pose(frames[k].get("public_pose"))
                        and frames[k]["public_pose"]["pos"] == anchor.get("public_pose", {}).get("pos")
                        for k in group)
                for key in group}
    return [key for key in keys if key == current or key in complete]
