"""Execute one tool-free model request over a public evidence pack."""
from pathlib import Path
import shutil
import sys

from src.common.io import redact
from src.harness.runtime.config import credentials
from src.harness.runtime.core_client import CoreClient
from src.harness.runtime.trace import TraceWriter


def run_fixed_evidence(repo, config, output, text, images, limits):
    output = Path(output).resolve()
    workspace = output / "workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    private = output / "private"
    auth = credentials(config, repo)
    refs, captions = [], []
    for index, image in enumerate(images, 1):
        source = Path(image["path"])
        ref = workspace / f"image-{index:02d}{source.suffix}"
        shutil.copyfile(source, ref)
        refs.append(str(ref))
        captions.append(f"Image {index}: {image['caption']}")
    client = CoreClient(repo, private)
    trace = TraceWriter(private, (auth["apiKey"],))
    try:
        client.request("init", {
            "workspace": str(workspace), "directory": str(output),
            "budgetFile": str(private / "requests.json"), "maxRequests": limits.max_requests,
            "auth": auth, "modelId": auth["model"], "api": config.get("api", "openai-responses"),
            "python": sys.executable, "thinkingLevel": config.get("reasoning_effort", "high"),
            "contextWindow": config.get("context_window", 32768), "compact": False,
            "compaction": {"reserveTokens": 8192, "keepRecentTokens": 4096}, "coding": False,
            "tools": [], "tracePersist": False,
            "systemPrompt": "Answer the spatial judgment using only the supplied evidence. Return exactly the requested JSON answer.",
        }, timeout=120)
        client.request("configure", {"allowedTools": []})
        prompt = "\n".join([text, *captions])
        trace.emit("prompt_start", {"text": prompt, "images": refs, "allowedTools": []})
        def forbidden(*_args):
            raise RuntimeError("Tools are disabled for judgment tasks")
        result = client.run_prompt(prompt, refs, forbidden, limits.seconds,
                                   on_event=trace.accept, max_turns=limits.max_turns)
        return redact(result, (auth["apiKey"],))
    except Exception as error:
        return {"reason": "runtime_error", "error": redact(str(error), (auth["apiKey"],))}
    finally:
        try:
            client.close()
            client.drain_events(trace.accept)
        finally:
            trace.close()
