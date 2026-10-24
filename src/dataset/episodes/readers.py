"""Habitat task readers adapted from AstraBench's frozen-sample protocols."""

from pathlib import Path

from src.common.io import read_json

from .base import EpisodeSource, EpisodeSpec


class VLNCEEpisodeSource(EpisodeSource):
    def __init__(self, config: dict):
        super().__init__(config)
        if self.name not in {"r2r", "rxr"}:
            raise ValueError("VLN-CE source requires name r2r or rxr")

    def _read(self, requested: list[str] | None) -> list[EpisodeSpec]:
        role = self.config.get("role", "guide")
        suffix = f"_{role}" if self.name == "rxr" else ""
        path = self.root / self.split / f"{self.split}{suffix}.json.gz"
        document = read_json(path)
        explicit_gt = self.config.get("gt_path")
        truth_path = Path(explicit_gt) if explicit_gt else path.with_name(f"{self.split}{suffix}_gt.json.gz")
        truth = read_json(truth_path) if explicit_gt or truth_path.exists() else {}
        prefix = f"{self.name}:{self.split}" + (f":{role}" if self.name == "rxr" else "")
        episodes = []
        for raw in document["episodes"]:
            source_id = str(raw["episode_id"])
            identifier = f"{prefix}:{source_id}"
            if not self._wanted(identifier, source_id, requested):
                continue
            instruction = raw["instruction"]
            text = instruction["instruction_text"] if isinstance(instruction, dict) else instruction
            if not isinstance(text, str) or not text.strip() or not raw.get("goals"):
                raise ValueError(f"Missing instruction or goals: {path}:{source_id}")
            episodes.append(EpisodeSpec(
                id=identifier, source_id=source_id, benchmark=self.name,
                scene_id=raw["scene_id"], start_position=raw["start_position"],
                start_rotation=raw["start_rotation"], source_file=str(path),
                language=instruction.get("language") if isinstance(instruction, dict) else None,
                goals=[{"modality": "instruction", "instruction": text.strip(),
                        "positions": [goal["position"] for goal in raw["goals"]]}],
                reference_path=raw.get("reference_path", []),
                gt_path=truth.get(source_id, {}).get("locations", []),
            ))
        return episodes


class ObjectNavEpisodeSource(EpisodeSource):
    def __init__(self, config: dict):
        super().__init__(config)
        if self.name != "objnav":
            raise ValueError("ObjectNav source requires name objnav")

    def _read(self, requested: list[str] | None) -> list[EpisodeSpec]:
        episodes = []
        for path in self._shards(requested):
            document = read_json(path)
            for index, raw in enumerate(document["episodes"]):
                original_scene = raw["scene_id"]
                scene = Path(original_scene).name.removesuffix(".basis.glb")
                category = raw["object_category"]
                source_id = str(raw["episode_id"])
                identifier = f"objnav:{self.split}:{scene}:{category}:{index}"
                if not self._wanted(identifier, source_id, requested):
                    continue
                scene_id = original_scene
                if scene_id.startswith("hm3d_v0.2/"):
                    scene_id = "hm3d/" + scene_id[len("hm3d_v0.2/"):]
                key = f"{Path(original_scene).name}_{category}"
                positions = [
                    view["agent_state"]["position"]
                    for goal in document["goals_by_category"][key]
                    for view in goal.get("view_points", [])
                ]
                if not positions:
                    raise ValueError(f"Missing ObjectNav goal viewpoints: {path}:{index}")
                episodes.append(EpisodeSpec(
                    id=identifier, source_id=source_id, benchmark="objnav", scene_id=scene_id,
                    start_position=raw["start_position"], start_rotation=raw["start_rotation"],
                    source_file=str(path), language="en",
                    goals=[{"modality": "object", "positions": positions,
                            "instruction": f"Find a {category.replace('_', ' ')} and approach it, then stop."}],
                ))
        return episodes


class GOATEpisodeSource(EpisodeSource):
    def __init__(self, config: dict):
        super().__init__(config)
        if self.name != "goat":
            raise ValueError("GOAT source requires name goat")
        self.max_description_words = self.config.get("goat_max_description_words", 55)
        if self.max_description_words is not None and (
            type(self.max_description_words) is not int or self.max_description_words < 1
        ):
            raise ValueError("goat_max_description_words must be positive or null")

    def _read(self, requested: list[str] | None) -> list[EpisodeSpec]:
        episodes = []
        for path in self._shards(requested):
            document = read_json(path)
            table = document["goals"]
            shard = path.name.removesuffix(".json.gz")
            for raw in document["episodes"]:
                source_id = str(raw["episode_id"])
                identifier = f"goat:{self.split}:{shard}:{source_id}"
                if not self._wanted(identifier, source_id, requested):
                    continue
                goals, skipped = [], 0
                for index, task in enumerate(raw["tasks"]):
                    category, modality, object_id = task[:3]
                    key = f"{Path(raw['scene_id']).name}_{category}"
                    instances = list(table[key])
                    if not instances:
                        raise ValueError(f"Missing GOAT goal instances: {path}:{key}")
                    for child in instances[0].get("children_object_categories", []) or []:
                        instances.extend(table.get(f"{Path(raw['scene_id']).name}_{child}", []))
                    selected = next((goal for goal in instances if str(goal["object_id"]) == str(object_id)), None)
                    if modality not in {"object", "description", "image"}:
                        raise ValueError(f"Unknown GOAT modality: {modality}")
                    if modality != "object" and selected is None:
                        raise ValueError(f"Missing GOAT instance: {path}:{object_id}")
                    text = (f"Find a {category.replace('_', ' ')} and go next to it."
                            if modality == "object" else (selected or {}).get("lang_desc", ""))
                    if modality == "description":
                        if not text.strip():
                            raise ValueError(f"Empty GOAT description: {path}:{object_id}")
                        # Keep AstraBench's preprocessing definition for protocol parity.
                        if self.max_description_words is not None and len(text.split(" ")) > self.max_description_words:
                            skipped += 1
                            continue
                    candidates = instances if modality == "object" else [selected]
                    positions = [
                        view["agent_state"]["position"]
                        for goal in candidates for view in goal["view_points"]
                    ]
                    if not positions:
                        raise ValueError(f"Missing GOAT viewpoints: {path}:{key}")
                    goal = {
                        "modality": modality, "positions": positions, "source_index": index,
                        "instruction": text if modality != "image" else "Find the instance shown in the target image.",
                    }
                    if modality == "image":
                        if len(task) < 4 or type(task[3]) is not int or not 0 <= task[3] < len(selected["image_goals"]):
                            raise ValueError(f"Invalid GOAT image index: {path}:{task}")
                        camera = selected["image_goals"][task[3]]
                        goal["image_goal"] = {
                            name: camera[name] for name in ("position", "rotation", "hfov", "image_dimensions")
                        }
                    goals.append(goal)
                if not goals:
                    raise ValueError(f"GOAT episode has no goals after preprocessing: {identifier}")
                episodes.append(EpisodeSpec(
                    id=identifier, source_id=source_id, benchmark="goat", scene_id=raw["scene_id"],
                    start_position=raw["start_position"], start_rotation=raw["start_rotation"],
                    goals=goals, source_file=str(path), skipped_goals=skipped,
                ))
        return episodes
