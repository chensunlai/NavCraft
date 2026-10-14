"""Short real-model check for native Coding and Python-owned tools."""

from datetime import datetime, timezone
import json
from pathlib import Path

from PIL import Image

from .config import core_options, credentials
from .core_client import CoreClient
from .session import _write
from .tools import ToolSet
from .trace import TraceWriter


def run_coding(repo, config, name="coding", max_requests=8, seconds=300,
               directory=None, cancel_event=None):
    repo = Path(repo).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path(directory) if directory else repo / config["output_dir"] / f"{stamp}-coding"
    workspace, private = directory / "workspace", directory / "private"
    for folder in ("scene", "observations", "memory", "scratch"):
        (workspace / folder).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (160, 120), "red").save(workspace / "scene/red.png")
    Image.new("RGB", (160, 120), "blue").save(workspace / "scene/blue.png")
    (workspace / "scene/instructions.md").write_text(
        'Create scratch/sum.py using native file tools. Run it with bash to compute '
        'sum([3,5,7]) and write memory/result.json as {"sum":15}. Then call '
        'view_images with scene/red.png and scene/blue.png and identify their colors. '
        'Read back the JSON and finish with a brief factual answer.')
    trace = TraceWriter(private)
    tools = ToolSet(repo, workspace, cancel_event=cancel_event)
    client = None
    settled = {}
    error = None
    try:
        trace.redact_with(credentials(config, repo)["apiKey"])
        client = CoreClient(repo, private)
        client.request("init", core_options(config, repo, directory, workspace,
                       "You are an autonomous agent in a restricted public workspace. Use native coding tools and the Python-owned image tool to complete and verify the task.",
                       [spec for spec in tools.specs() if spec["name"] == "view_images"], max_requests,
                       compact=False))
        settled = client.run_prompt("Read scene/instructions.md and complete the task.", [],
                                    tools.call, seconds, on_event=trace.accept,
                                    cancel_event=cancel_event)
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    finally:
        if client:
            try:
                client.close()
                client.drain_events(trace.accept)
            except Exception as failure:
                error = error or f"Core cleanup: {failure}"
        saved = workspace / "memory/result.json"
        value = None
        try:
            if saved.exists():
                value = json.loads(saved.read_text())
        except ValueError as failure:
            error = error or f"Invalid generated result: {failure}"
        used = {call["name"] for call in trace.calls if not call["isError"]}
        passed = (not error and settled.get("reason") == "model_finished"
                  and value == {"sum": 15} and {"read", "bash", "view_images"} <= used
                  and bool(settled.get("answer")))
        result = {"case": name, **settled, "model": config["model"], "api": config["api"],
                  "error": error, "verification": {"passed": passed, "sum": value,
                  "python_tool_used": "view_images" in used, "tools_used": sorted(used)},
                  "software_only": False}
        ledger = private / "requests.json"
        if ledger.exists():
            try:
                result["requests"] = len(json.loads(ledger.read_text())["attempts"])
            except (OSError, ValueError, KeyError):
                result["archive_error"] = "Request ledger could not be read"
        result = trace.sanitize(result)
        trace.emit("run_finished", result)
        trace.close()
        _write(directory / "result.json", result)
    return directory
