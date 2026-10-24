"""Small real-environment acceptance checks, separate from benchmark scores."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.bench.samples import load_episodes, map_directory, read_config
from src.common.io import write_json
from src.environment import Environment
from src.harness.session import PromptLimits, open_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["local", "probe", "compaction"])
    parser.add_argument("--model", choices=["deepseek", "qwen27", "sol"], default="sol")
    parser.add_argument("--gpu", type=int, choices=[0, 1], default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    pilot = read_config(REPO / "config/pilot.yaml")
    episode = load_episodes(REPO, pilot["episodes"][:1])[0]
    output = args.output or REPO / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-lite-{args.mode}-{args.model}")
    output.mkdir(parents=True, exist_ok=False)
    env = Environment(episode, {"scenes": str(REPO / pilot["environment"]["scenes"]), "gpu": args.gpu}, output / "private/environment")
    session = None
    try:
        if args.mode == "local":
            first = env.observe()
            pose_before = env.pose()
            panorama = env.panorama()
            restored = env.pose()
            assert len(panorama) == 6
            assert pose_before == restored, "Panorama did not restore body/camera state"
            checks = []
            for action in ("move_forward", "move_backward", "move_left", "move_right", "turn_left", "turn_right"):
                motion, observation = env.step(action)
                assert observation.frame_id != first.frame_id
                assert abs(motion.auto_yaw_delta_degrees) <= 15.001
                checks.append({"action": action, "blocked": motion.blocked, "collision": motion.collision,
                               "auto_yaw": motion.auto_yaw_delta_degrees, "frame": str(observation.rgb)})
            assert env.steps == 6 and math.isfinite(env.path_length)
            result = {"kind": "integration_check", "model_called": False, "panorama_restored": True,
                      "actions": checks, "path_length": env.path_length, "gpu": args.gpu}
        else:
            config = read_config(REPO / pilot["models"][args.model])
            if args.mode == "compaction":
                config = {**config, "compaction_threshold": 10000}
            session = open_session(REPO, config, output, env, map_directory(REPO, episode) / "map.json")
            question = ("This is a tool integration check. Reopen the current frame with view_images and inspect the full map. "
                        "Use bash to write and run scratch/check.py, importing navlib and printing the current public pose. "
                        "Record one room named 'current area' at your current map position, using the current frame as evidence, "
                        "then read it back with memory_read. Finally submit(answer='integration_ok'). Do not move.")
            first = session.prompt(question, limits=PromptLimits(max_requests=12, max_actions=6, seconds=600))
            results = [first.to_dict()]
            if args.mode == "compaction" and first.submitted:
                # Add realistic user notes to force a threshold crossing without private facts.
                notes = "\n".join(f"Note {i}: Old images remain available by frame ID; uncertain room connections need visual evidence." for i in range(30))
                second = session.prompt("Public working notes:\n" + notes + "\nReopen the current frame and inspect the recorded room, then submit(answer='continued').",
                                        limits=PromptLimits(max_requests=16, max_actions=6, seconds=600))
                results.append(second.to_dict())
            result = {"kind": "integration_check", "model": config["model"], "reasoning_effort": config["reasoning_effort"],
                      "prompts": results, "gpu": args.gpu}
        write_json(output / "check.json", result)
        print(json.dumps({"output": str(output), **result}, ensure_ascii=False), flush=True)
    finally:
        if session is not None:
            session.close()
        env.close()


if __name__ == "__main__":
    main()
