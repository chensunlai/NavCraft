"""Early real-model check: Python owns a tool, Pi owns the agent loop."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import yaml
from PIL import Image

from core_client import CoreClient
import memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/harness-pi/default.yaml")
    args = parser.parse_args()
    config = yaml.safe_load((REPO / args.config).read_text())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = REPO / config["output_dir"] / f"{stamp}-python-coding"
    workspace = directory / "workspace"
    for name in ("scene", "observations", "memory", "scratch"):
        (workspace / name).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (160, 120), "red").save(workspace / "scene/red.png")
    Image.new("RGB", (160, 120), "blue").save(workspace / "scene/blue.png")
    instructions = ("Create scratch/sum.py using native file tools. Run it with bash to compute "
                    "sum([3,5,7]) and write memory/result.json as {\"sum\":15}. Then call "
                    "view_images with scene/red.png and scene/blue.png and identify their colors. "
                    "Read back the saved JSON and finish with a brief factual answer.")
    (workspace / "scene/instructions.md").write_text(instructions)
    spec = {"name": "view_images", "description": "Open existing public images using the Python image tool.",
            "parameters": {"type": "object", "properties": {"refs": {"type": "array", "items": {"type": "string"}}},
                           "required": ["refs"], "additionalProperties": False}}
    calls = []
    events = []

    def tool_handler(call_id, name, arguments):
        if name != "view_images":
            raise ValueError(name)
        result = memory.view(workspace=str(workspace), **arguments)
        calls.append({"call_id": call_id, "name": name, "args": arguments, "result": result})
        return {"result": {"content": [{"type": "text", "text": json.dumps(result["data"])}],
                           "images": result["images"], "details": {"images": result["images"]}}}

    client = CoreClient(REPO, directory / "private")
    print(json.dumps({"directory": str(directory), "phase": "starting"}), flush=True)
    try:
        client.request("init", {"workspace": str(workspace), "directory": str(directory),
                       "budgetFile": str(directory / "private/requests.json"), "maxRequests": 8,
                       "totalLimit": None, "credentialFile": str(REPO / config["credentials"]),
                       "modelId": config["model"], "api": config["api"], "python": config["python"],
                       "systemPrompt": "You are an autonomous agent in a restricted public workspace. Use the native coding tools and the Python-owned image tool. scene/observations are read-only; memory/scratch are writable. Complete and verify the task.",
                       "tools": [spec], "compact": False})
        settled = client.run_prompt("Read scene/instructions.md and complete the task.", [],
                                    tool_handler, 300, on_event=events.append)
        saved = workspace / "memory/result.json"
        result = {"case": "coding", "directory": str(directory), **settled, "python_tool_calls": len(calls),
                  "verification": {"sum": json.loads(saved.read_text()) if saved.exists() else None,
                                   "python_tool_used": bool(calls), "answer": settled.get("answer")}}
        (directory / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        (directory / "private/python_calls.json").write_text(json.dumps(calls, ensure_ascii=False, indent=2))
        with (directory / "private/python_events.jsonl").open("w") as stream:
            for event in events:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
