"""Freeze official selections and author explicitly synthetic HM3D route tasks.

Run with ``conda run -n vln python -m src.dataset.prepare_visual``. Geometry
validation uses Habitat; it never queries a model or edits scene-map archives.
"""

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import itertools
import math
import os
from pathlib import Path
import random

import numpy as np
import yaml

from src.common.io import read_json, write_json
from src.dataset.episodes.base import EpisodeSpec
from src.dataset.episodes.readers import GOATEpisodeSource, ObjectNavEpisodeSource

SCENES = {
    "QaLdnwvtxbs": "hm3d/val/00829-QaLdnwvtxbs",
    "kfPV7w3FaU5": "hm3d/train/00000-kfPV7w3FaU5",
    "5cdEh9F2hJL": "hm3d/val/00853-5cdEh9F2hJL",
    "4ok3usBNeis": "hm3d/val/00877-4ok3usBNeis",
}
DEVELOPMENT = ("QaLdnwvtxbs", "kfPV7w3FaU5")
ACCEPTANCE = ("5cdEh9F2hJL", "4ok3usBNeis")


@dataclass(frozen=True)
class Route:
    distance: float
    points: list[list[float]]


def inside(point, polygon):
    x, y = point
    result = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if (a[1] > y) != (b[1] > y):
            if x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]:
                result = not result
    return result


class NavMeshValidator:
    """Private native geometry checks with benchmark-matched agent dimensions."""

    def __init__(self, scene, config):
        from src.environment.simulator import HabitatSimulator

        seed_episode = EpisodeSpec(
            id="prepare",
            source_id="prepare",
            benchmark="r2r",
            scene_id=scene,
            start_position=[0, 0, 0],
            start_rotation=[0, 0, 0, 1],
            goals=[
                {
                    "modality": "instruction",
                    "instruction": "prepare",
                    "positions": [[0, 0, 0]],
                }
            ],
            source_file="offline preparation",
        )
        self.simulator = HabitatSimulator(seed_episode, config)
        self.pathfinder = self.simulator.sim.pathfinder

    def snap(self, point):
        result = np.asarray(
            self.pathfinder.snap_point(np.asarray(point, dtype=np.float32))
        )
        if not np.isfinite(result).all() or math.dist(point, result) > 0.5:
            return None
        return result.astype(float).tolist()

    def path(self, start, end):
        import habitat_sim

        query = habitat_sim.ShortestPath()
        query.requested_start = np.asarray(start, dtype=np.float32)
        query.requested_end = np.asarray(end, dtype=np.float32)
        if not self.pathfinder.find_path(query) or not math.isfinite(
            query.geodesic_distance
        ):
            return None
        return Route(
            float(query.geodesic_distance),
            [np.asarray(p).astype(float).tolist() for p in query.points],
        )

    def goals_reachable(self, start, goals):
        import habitat_sim

        distances = []
        for goal in goals:
            query = habitat_sim.MultiGoalShortestPath()
            query.requested_start = np.asarray(start, dtype=np.float32)
            query.requested_ends = np.asarray(goal["positions"], dtype=np.float32)
            if not self.pathfinder.find_path(query) or not math.isfinite(
                query.geodesic_distance
            ):
                return None
            distances.append(float(query.geodesic_distance))
        return distances

    def start_status(self, start):
        snapped = self.snap(start)
        return {
            "navigable": bool(
                self.pathfinder.is_navigable(np.asarray(start, dtype=np.float32))
            ),
            "snap_displacement_m": (
                math.dist(start, snapped) if snapped is not None else None
            ),
        }

    def close(self):
        self.simulator.close()


class FixtureValidator:
    """Explicitly unvalidated CPU fixture geometry; never an acceptance result."""

    def snap(self, point):
        return list(point)

    def path(self, start, end):
        return Route(math.dist(start, end), [list(start), list(end)])

    def goals_reachable(self, start, goals):
        return [
            min(math.dist(start, point) for point in goal["positions"])
            for goal in goals
        ]

    def start_status(self, start):
        return {"navigable": None, "snap_displacement_m": None}

    def close(self):
        pass


class RoomArchive:
    def __init__(self, directory):
        self.directory = Path(directory)
        raw = self.directory / "raw"
        self.scene = read_json(raw / "scene.json")
        self.scene_id = self.scene["scene_id"]
        self.levels = {row["id"]: row["height_m"] for row in self.scene["levels"]}
        self.rooms = {
            f"{row['level']}:{row['id']}": row
            for row in read_json(raw / "rooms.json")["rooms"]
        }
        self.frames = {
            row["id"]: row for row in read_json(raw / "evidence.json")["frames"]
        }
        self.stairs = read_json(raw / "regions.json").get("stair_connections", [])
        doors = read_json(raw / "doors.json")
        self.adjacency = {key: set() for key in self.rooms}
        for door in doors["doors"]:
            if "closed" in door.get("status", "") or "disconnected" in door.get(
                "status", ""
            ):
                continue
            keys = [f"{door['level']}:{room}" for room in door["connects"]]
            self._connect(keys)
        for opening in doors.get("open_connections", []):
            for level in self.levels:
                self._connect([f"{level}:{room}" for room in opening["rooms"]])
        for stair in self.stairs:
            self._connect(
                [
                    f"{level}:{stair['id']}"
                    for level in (stair["from_level"], stair["to_level"])
                ]
            )

    def _connect(self, keys):
        for first, second in itertools.combinations(
            [key for key in keys if key in self.rooms], 2
        ):
            self.adjacency[first].add(second)
            self.adjacency[second].add(first)

    def graph_distance(self, start, end):
        pending, visited = deque([(start, 0)]), {start}
        while pending:
            node, distance = pending.popleft()
            if node == end:
                return distance
            for next_node in self.adjacency[node] - visited:
                visited.add(next_node)
                pending.append((next_node, distance + 1))
        return 999

    def anchors(self, validator):
        anchors = {}
        for key, room in self.rooms.items():
            if room["kind"] == "stairs":
                continue
            for identifier in room.get("evidence_frames", []):
                frame = self.frames.get(identifier)
                if frame is None:
                    continue
                point = frame.get(
                    "navigable_base_xyz", frame.get("pose", {}).get("position")
                )
                if point is None and "camera_position_xyz" in frame:
                    camera = frame["camera_position_xyz"]
                    point = [camera[0], self.levels[room["level"]], camera[2]]
                if point is None or abs(point[1] - self.levels[room["level"]]) > 0.6:
                    continue
                point = validator.snap(point)
                if point is not None and inside([point[0], -point[2]], room["polygon"]):
                    anchors[key] = point
                    break
        return anchors

    def region(self, point):
        xy = [point[0], -point[2]]
        for stair in self.stairs:
            low, high = sorted(
                [self.levels[stair["from_level"]], self.levels[stair["to_level"]]]
            )
            for level in (stair["from_level"], stair["to_level"]):
                key = f"{level}:{stair['id']}"
                if low - 0.1 <= point[1] <= high + 0.1 and key in self.rooms:
                    if inside(xy, self.rooms[key]["polygon"]):
                        return key
        level = min(self.levels, key=lambda name: abs(self.levels[name] - point[1]))
        for key, room in self.rooms.items():
            if (
                room["level"] == level
                and room["kind"] != "stairs"
                and inside(xy, room["polygon"])
            ):
                return key
        return None

    def route_regions(self, points):
        segments = []
        for first, second in zip(points, points[1:]):
            count = max(1, math.ceil(math.dist(first, second) / 0.25))
            for fraction in np.linspace(0, 1, count + 1):
                point = (
                    np.asarray(first) * (1 - fraction) + np.asarray(second) * fraction
                ).tolist()
                key = self.region(point)
                if key is None:
                    continue
                if segments and segments[-1][0] == key:
                    segments[-1][1].append(point)
                else:
                    segments.append((key, [point]))
        return segments


def instruction_episodes(archive, validator, count, seed, rejections):
    anchors = archive.anchors(validator)
    pairs = list(itertools.permutations(anchors, 2))
    random.Random(seed).shuffle(pairs)
    # Prefer multi-room pairs in the authored graph; native geometry remains decisive.
    pairs.sort(key=lambda pair: archive.graph_distance(*pair) < 2)
    groups = {True: [], False: []}
    for first, last in pairs:
        groups[archive.rooms[first]["level"] != archive.rooms[last]["level"]].append(
            (first, last)
        )
    pairs = list(
        itertools.chain.from_iterable(
            itertools.zip_longest(groups[True], groups[False])
        )
    )
    episodes = []
    for pair in pairs:
        if pair is None:
            continue
        first, last = pair
        start, end = anchors[first], anchors[last]
        route = validator.path(start, end)
        if route is None or route.distance < 5:
            rejections.append(
                {
                    "scene": archive.scene_id,
                    "pair": [first, last],
                    "reason": "unreachable" if route is None else "geodesic_below_5m",
                }
            )
            continue
        regions = archive.route_regions(route.points)
        if len(regions) < 2:
            rejections.append(
                {
                    "scene": archive.scene_id,
                    "pair": [first, last],
                    "reason": "no_verified_room_transition",
                }
            )
            continue
        intermediate = [
            (key, samples) for key, samples in regions if key not in {first, last}
        ]
        names = [archive.rooms[key]["name"] for key, _ in intermediate]
        target_room = archive.rooms[last]
        turn = "右" if len(episodes) % 2 == 0 else "左"
        delta = next(
            (
                np.asarray(point) - start
                for point in route.points[1:]
                if math.dist(start, point) > 0.1
            ),
            np.asarray(end) - start,
        )
        yaw = math.atan2(-float(delta[0]), -float(delta[2])) + (
            math.pi / 2 if turn == "右" else -math.pi / 2
        )
        landmarks = "，依次经过" + "、".join(names) if names else "，沿可见通道前进"
        appearance = target_room.get("description", "").split("。")[0][:90]
        text = (
            f"从{archive.rooms[first]['name']}出发，先向{turn}转{landmarks}，"
            f"最后进入{target_room['name']}。{appearance}。到达该区域后停下。"
        )
        checks = [
            {
                "label": archive.rooms[key]["name"],
                "room_id": archive.rooms[key]["id"],
                "position": samples[len(samples) // 2],
                "radius": 1.5,
                "kind": (
                    "stairs" if archive.rooms[key]["kind"] == "stairs" else "waypoint"
                ),
            }
            for key, samples in intermediate
        ]
        identifier = f"r2r:synthetic_hm3d:{archive.scene_id}:{len(episodes):03d}"
        relative = SCENES.get(archive.scene_id, archive.directory.name)
        episodes.append(
            EpisodeSpec(
                id=identifier,
                source_id=f"manifest:{archive.scene_id}:{len(episodes):03d}",
                benchmark="r2r",
                scene_id=f"{relative}/{archive.scene_id}.basis.glb",
                start_position=start,
                start_rotation=[0, math.sin(yaw / 2), 0, math.cos(yaw / 2)],
                goals=[
                    {"modality": "instruction", "instruction": text, "positions": [end]}
                ],
                source_file=str(archive.directory / "raw/evidence.json"),
                language="zh",
                gt_path=route.points,
                reference_path=route.points,
                route_checks=checks,
                provenance="synthetic_hm3d_instruction",
            )
        )
        if len(episodes) == count:
            return episodes
    raise ValueError(
        f"Only {len(episodes)} valid instruction routes in {archive.scene_id}; requested {count}"
    )


def official_candidates(root, scene, family):
    split = "val" if family == "objnav" else "val_unseen"
    path = Path(root) / split / "content" / f"{scene}.json.gz"
    raw = read_json(path)["episodes"]
    config = {
        "name": family,
        "root": str(root),
        "split": split,
        "goat_max_description_words": None,
    }
    if family == "objnav":
        ids = [
            f"objnav:val:{scene}:{row['object_category']}:{index}"
            for index, row in enumerate(raw)
        ]
        source = ObjectNavEpisodeSource(config)
    else:
        ids = [f"goat:val_unseen:{scene}:{row['episode_id']}" for row in raw]
        source = GOATEpisodeSource(config)
    return source.load(ids)


def balanced_selection(episodes, count, seed):
    rows = list(episodes)
    random.Random(seed).shuffle(rows)
    categories = {}
    for episode in rows:
        category = episode.id.split(":")[-2]
        categories.setdefault(category, []).append(episode)
    result = []
    while len(result) < count and any(categories.values()):
        for category in sorted(categories):
            if categories[category] and len(result) < count:
                result.append(categories[category].pop())
    if len(result) != count:
        raise ValueError(
            f"Cannot freeze {count} tasks; only {len(result)} validated candidates"
        )
    return result


def save_manifest(output, name, episodes, *, validated, seed):
    document = {
        "schema_version": 1,
        "validated": validated,
        "seed": seed,
        "input_setting": "complete_scene_map_and_rgb_initial_pose_only",
        "episodes": [episode.to_dict() for episode in episodes],
    }
    path = output / f"{name}.json"
    write_json(path, document)
    write_json(
        output / f"{name}.selection.json",
        {"episode_ids": [episode.id for episode in episodes]},
    )
    return {
        "count": len(episodes),
        "goals": sum(len(episode.goals) for episode in episodes),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "validated": validated,
    }


def preflight_maps(episodes, map_root):
    from src.common.map import MapCalibration

    calibrations = {}
    results = []
    for episode in episodes:
        scene = Path(episode.scene_id).name.removesuffix(".basis.glb")
        if scene not in calibrations:
            calibrations[scene] = MapCalibration(map_root / SCENES[scene])
        level, origin, _ = calibrations[scene].initial_arrow(
            {
                "position": episode.start_position,
                "rotation_xyzw": episode.start_rotation,
            }
        )
        results.append({"episode_id": episode.id, "level": level, "origin": origin})
    return results


def geometry_summary(checks):
    summary = {}
    for family in ("objnav", "goat", "instruction"):
        for split in ("dev", "test"):
            rows = [
                row
                for row in checks
                if row["family"] == family and row["split"] == split
            ]
            if not rows:
                continue
            values = [row["distances_from_start"][0] for row in rows]
            summary[f"{family}_{split}"] = {
                "count": len(values),
                "first_goal_geodesic_min_m": min(values),
                "first_goal_geodesic_max_m": max(values),
                "first_goal_geodesic_mean_m": sum(values) / len(values),
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("cache/visual-harness-v1"))
    parser.add_argument("--maps", type=Path, default=Path("data/scene_maps"))
    parser.add_argument("--scenes", type=Path, default=Path("data/nas/scene_datasets"))
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace frozen episode manifests",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="CPU fixtures only; manifests marked unvalidated",
    )
    args = parser.parse_args()
    if not args.overwrite and any(args.output.glob("*_*.json")):
        parser.error(
            "Frozen files already exist; use a new --output or explicit --overwrite"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    )
    rejections, summaries, map_checks, geometry_checks = [], {}, [], []
    for family in ("objnav", "goat", "instruction"):
        for split, scenes in (("dev", DEVELOPMENT), ("test", ACCEPTANCE)):
            selected = []
            for scene in scenes:
                if family != "instruction" and scene == "kfPV7w3FaU5":
                    continue
                archive = RoomArchive(args.maps / SCENES[scene])
                env_name = (
                    "objectnav"
                    if family == "objnav"
                    else "r2r" if family == "instruction" else "goat"
                )
                config = yaml.safe_load(
                    Path(f"config/environment/{env_name}_visual.yaml").read_text()
                )["environment"]
                config.update(
                    scenes=str(args.scenes),
                    gpu=args.gpu,
                    depth=False,
                    width=64,
                    height=48,
                    seed=args.seed,
                )
                validator = (
                    FixtureValidator()
                    if args.skip_validation
                    else NavMeshValidator(f"{SCENES[scene]}/{scene}.basis.glb", config)
                )
                try:
                    count = (
                        3
                        if split == "dev" and family != "objnav"
                        else 6 if split == "dev" else 10
                    )
                    if family == "instruction":
                        scene_selected = instruction_episodes(
                            archive, validator, count, args.seed, rejections
                        )
                    else:
                        root = (
                            "data/datasets/objectnav/hm3d/v2"
                            if family == "objnav"
                            else "data/datasets/goat_bench/hm3d/v1"
                        )
                        candidates = official_candidates(root, scene, family)
                        valid = []
                        for episode in candidates:
                            distances = validator.goals_reachable(
                                episode.start_position, episode.goals
                            )
                            if distances is None:
                                rejections.append(
                                    {
                                        "episode_id": episode.id,
                                        "reason": "goal_unreachable_from_start",
                                    }
                                )
                            else:
                                valid.append(episode)
                        scene_selected = (
                            balanced_selection(valid, count, args.seed)
                            if family == "objnav"
                            else valid[:count]
                        )
                        if family == "goat" and len(valid) < count:
                            raise ValueError(
                                f"Cannot preserve {count} full GOAT chains in {scene}; {len(valid)} reachable"
                            )
                    for episode in scene_selected:
                        geometry_checks.append(
                            {
                                "episode_id": episode.id,
                                "family": family,
                                "split": split,
                                "distances_from_start": validator.goals_reachable(
                                    episode.start_position, episode.goals
                                ),
                                "start": validator.start_status(episode.start_position),
                            }
                        )
                    selected.extend(scene_selected)
                except Exception:
                    write_json(args.output / "rejections.json", rejections)
                    raise
                finally:
                    validator.close()
            map_checks.extend(preflight_maps(selected, args.maps))
            write_json(
                args.output / "map-preflight.json", {"passed": map_checks, "failed": []}
            )
            summaries[f"{family}_{split}"] = save_manifest(
                args.output,
                f"{family}_{split}",
                selected,
                validated=not args.skip_validation,
                seed=args.seed,
            )
            write_json(args.output / "rejections.json", rejections)
            write_json(args.output / "summary.json", summaries)
            write_json(
                args.output / "geometry-validation.json",
                {
                    "episodes": geometry_checks,
                    "summary": geometry_summary(geometry_checks),
                },
            )
            print(f"Frozen {family}_{split}: {len(selected)} episodes", flush=True)
    inputs = []
    for relative in SCENES.values():
        inputs.extend(
            (
                args.maps / relative / "raw" / name
                for name in (
                    "scene.json",
                    "rooms.json",
                    "regions.json",
                    "doors.json",
                    "evidence.json",
                    "manifest.json",
                )
            )
        )
    for scene in ("QaLdnwvtxbs", *ACCEPTANCE):
        inputs.append(
            Path(f"data/datasets/objectnav/hm3d/v2/val/content/{scene}.json.gz")
        )
        inputs.append(
            Path(f"data/datasets/goat_bench/hm3d/v1/val_unseen/content/{scene}.json.gz")
        )
    inputs.extend(
        Path(f"config/environment/{name}_visual.yaml")
        for name in ("objectnav", "goat", "r2r")
    )
    write_json(
        args.output / "inputs.json",
        {
            "seed": args.seed,
            "gpu": args.gpu,
            "validated": not args.skip_validation,
            "private_authoring_inputs": [
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in inputs
                if path.is_file()
            ],
        },
    )


if __name__ == "__main__":
    main()
