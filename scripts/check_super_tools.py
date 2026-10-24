"""One real Habitat RGB-D/tool replay; no model requests or benchmark score."""

import argparse
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")

class OfflineClient:
    """Keep the actual application and tools; replace only model transport."""
    def __init__(self, *args):
        pass

    def request(self, op, *args, **kwargs):
        return {"session_id": "offline-tools"} if op == "init" else {}

    def close(self):
        pass

    def drain_events(self, callback):
        pass


def main():
    from src.bench.samples import load_episodes, map_directory
    from src.common.io import write_json
    from src.environment import Environment
    from src.harness.session import open_session

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=1)
    parser.add_argument("--episode", default="objnav:val:LT9Jq6dN3Ea:bed:14")
    parser.add_argument("--output", type=Path, default=REPO / "cache/validation/super-tools-real")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    credentials = output / "offline.env"
    credentials.write_text("NAVCRAFT_TEST_BASE_URL=http://offline.invalid\nNAVCRAFT_TEST_API_KEY=offline\n")
    episode = load_episodes(REPO, [args.episode])[0]
    env = Environment(episode, {"gpu": args.gpu, "tool_tier": "super", "scenes": str(REPO / "data/nas/scene_datasets")},
                      output / "private/environment")
    records = []
    try:
        with open_session(REPO, {"model": "offline", "tool_tier": "super", "credentials": str(credentials)},
                          output, env, map_directory(REPO, episode) / "map.json", client_factory=OfflineClient) as session:
            def call(name, arguments):
                receipt = session._tool(f"offline-{len(records)}", name, arguments)
                value = json.loads(receipt["result"]["content"][0]["text"])
                records.append({"tool": name, "arguments": arguments, "result": value,
                                "images": receipt["result"]["images"]})
                if receipt["result"]["isError"]:
                    raise ValueError(value)
                return value

            pano = call("nav_observe", {"mode": "panorama", "grid": True, "project": True})
            candidate = None
            for view in pano["views"]:
                for x, y in [(x, y) for y in (400, 440, 470, 360, 320) for x in (320, 200, 440, 100, 540)]:
                    point = session.super_tools.geometry.resolve_frame(view["frame"], [x, y])
                    if not point["valid"] or not .7 <= point["distance"] <= 2.5 or not -.05 <= point["height"] <= .15:
                        continue
                    world, _ = session.super_tools.geometry.resolve_frame_point(view["frame"], [x, y])
                    endpoint = env.near_point(world, .2)
                    route = env.shortest_path([endpoint]) if endpoint is not None else []
                    if route and len(route) == 2:
                        candidate = {"frame": view["frame"], "pixel": [x, y]}
                        break
                if candidate:
                    break
            if candidate is None:
                raise ValueError("No visible nearby floor point suitable for the physical replay")
            call("locate", {"source": candidate["frame"], "points": [candidate["pixel"]]})
            call("memory_edit", {"operations": [{"op": "create", "id": "O01", "type": "object", "fields": {
                "name": "Selected floor point", "at": candidate}}]})
            call("memory_read", {"near": {"self": 5}})
            call("map_view", {"view": {"ego": 3}, "overlay": [candidate["frame"]]})
            aligned = call("nav_face", {"target": candidate})
            moved = call("nav_goto", {"target": {"entity": "O01"}, "max_actions": 24})
            checks = {"face_aligned": abs(aligned["bearing_after"]) <= 1,
                      "goto_reached_selected_point": moved["reached"],
                      "actual_actions": env.steps > 0, "actual_path_m": env.path_length,
                      "private_depth_only": not any("depth" in p.name for p in session.workspace.rglob("*")),
                      "no_task_submission": not session._submitted}
            write_json(output / "check.json", {"gpu": args.gpu, "episode": episode.id,
                       "source": "program-assisted tool validation, not a model demonstration",
                       "checks": checks, "records": records,
                       "passed": all(value for value in checks.values())})
            lines = ["# Super tools: physical replay", "", "Program-assisted validation; no model requests or task scores.", ""]
            for record in records:
                lines.extend([f"## {record['tool']}", "", "```json", json.dumps(record["result"], indent=2), "```", ""])
                lines.extend(f"![{record['tool']}](workspace/{ref})\n" for ref in record["images"])
            (output / "README.md").write_text("\n".join(lines))
            if not all(value for value in checks.values()):
                raise ValueError("Physical checks failed; inspect check.json")
            print(output / "README.md")
    finally:
        write_json(output / "records.json", records)
        env.close()


if __name__ == "__main__":
    main()
