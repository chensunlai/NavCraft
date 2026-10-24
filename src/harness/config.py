"""Resolve trusted runtime configuration without exposing credentials to tools."""

from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[2]
PYTHON = "/home/chen/.local/anaconda3/envs/vln/bin/python"


def credentials(config, repo=REPO):
    values = {}
    file = Path(config.get("credentials", ".env.harness"))
    if not file.is_absolute():
        file = Path(repo) / file
    for line in file.read_text().splitlines():
        match = re.fullmatch(r"\s*([A-Z_]+)\s*=\s*(.*?)\s*", line)
        if match:
            values[match[1]] = match[2].strip("\"'")
    result = {"baseUrl": (config.get("base_url") or values.get("NAVCRAFT_TEST_BASE_URL", "")).rstrip("/"),
              "model": config.get("model") or values.get("NAVCRAFT_TEST_MODEL"),
              "apiKey": values.get("NAVCRAFT_TEST_API_KEY")}
    if not all(result.values()):
        raise ValueError("Missing model, API base, or API key")
    return result


def core_options(config, repo, directory, workspace, tools, checkpoint):
    effort = config.get("reasoning_effort", "high")
    if effort not in {"off", "minimal", "low", "medium", "high", "xhigh", "max"}:
        raise ValueError("Unsupported reasoning_effort")
    context = config.get("context_window", 32768)
    threshold = config.get("compaction_threshold", 24576)
    if not 0 < threshold <= context - 8192:
        raise ValueError("Context must reserve at least 8192 output tokens")
    prompt_path = Path(repo) / "config/prompts/navigation.md"
    return {"workspace": str(workspace), "directory": str(directory),
            "budgetFile": str(Path(directory) / "private/requests.json"),
            "maxRequests": 150, "totalLimit": None,
            "auth": credentials(config, repo), "modelId": config["model"],
            "api": config.get("api", "openai-responses"),
            "python": config.get("python", PYTHON), "thinkingLevel": effort,
            "contextWindow": context, "compact": True,
            "compaction": {"reserveTokens": context - threshold, "keepRecentTokens": 4096},
            "systemPrompt": prompt_path.read_text(), "tools": tools,
            "checkpoint": checkpoint, "tracePersist": False,
            "executionPolicy": {"readOnly": ["scene", "observations", "memory"],
                                "readWrite": ["scratch"]}}
