"""Resolve trusted runtime configuration without exposing credentials to tools."""

from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[3]
PYTHON = sys.executable


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


def core_options(config, repo, directory, workspace, tools, selection):
    effort = config.get("reasoning_effort", "high")
    if effort not in {"off", "minimal", "low", "medium", "high", "xhigh", "max"}:
        raise ValueError("Unsupported reasoning_effort")
    context = config.get("context_window", 32768)
    threshold = config.get("compaction_threshold", 24576)
    if not 0 < threshold <= context - 8192:
        raise ValueError("Context must reserve at least 8192 output tokens")
    node_mode = selection.memory_type == "node"
    prompt_path = Path(repo) / "config/prompts" / ("navigation_node.md" if node_mode else "navigation.md")
    system_prompt = prompt_path.read_text()
    if node_mode:
        system_prompt += ("\nSuper geometry is enabled. RGB depth remains private. Image/pixel targets use the displayed image coordinates. A projected point is a surface, not necessarily an object centre or an open passage. RGB-supported targets keep the existing local geometry assistance; node and map-coordinate targets use straight motion. Reached means the selected point, never task success.\n"
                          if selection.geometry else
                          "\nBasic tools are active: move uses steps only. Use the map and historical images to choose actions; there is no RGB projection.\n")
    elif selection.geometry:
        system_prompt += "\nSuper geometry is enabled. RGB depth stays private. Shared image/pixel references use the displayed image's pixels, including registered crops and panoramas. A projected pixel is a surface, not necessarily an object centre or doorway. move(face/approach) acts on your selected point; reached is not task success. Memory fields.location can resolve that same reference. Geometry cannot recognize mirrors or glass.\n"
    else:
        system_prompt += "\nBasic tools are active: move uses steps only. Map-image pixels can be converted to global UV; RGB pixels cannot be projected to the map. Use supported map references or evidence-based UV estimates, and leave uncertain locations null.\n"

    return {"workspace": str(workspace), "directory": str(directory),
            "budgetFile": str(Path(directory) / "private/requests.json"),
            "maxRequests": 150, "totalLimit": None,
            "auth": credentials(config, repo), "modelId": config["model"],
            "api": config.get("api", "openai-responses"),
            "python": config.get("python", PYTHON), "thinkingLevel": effort,
            "contextWindow": context, "compact": True,
            "compaction": {"reserveTokens": context - threshold, "keepRecentTokens": 4096},
            "systemPrompt": system_prompt, "tools": tools,
            "tracePersist": False,
            "executionPolicy": {"readOnly": ["scene", "observations", "memory"],
                                "readWrite": ["scratch"]}}
