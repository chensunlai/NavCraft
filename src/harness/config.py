"""Application configuration; the Core receives only resolved runtime settings."""

from pathlib import Path
import re

import yaml


REPO = Path(__file__).resolve().parents[2]


def load_config(path="config/harness-pi/default.yaml", repo=REPO):
    path = Path(path)
    if not path.is_absolute():
        path = Path(repo) / path
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Harness configuration must be an object")
    return config


def credentials(config, repo=REPO):
    values = {}
    path = Path(config["credentials"])
    if not path.is_absolute():
        path = Path(repo) / path
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"\s*([A-Z_]+)\s*=\s*(.*?)\s*", line)
        if match:
            values[match[1]] = match[2].strip("\"'")
    auth = {"baseUrl": values.get("NAVCRAFT_TEST_BASE_URL", "").rstrip("/"),
            "model": config.get("model") or values.get("NAVCRAFT_TEST_MODEL"),
            "apiKey": values.get("NAVCRAFT_TEST_API_KEY")}
    if not all(auth.values()):
        raise ValueError("Missing model service credentials")
    return auth


def core_options(config, repo, directory, workspace, prompt, tools, max_requests,
                 checkpoint=None, compact=True):
    return {"workspace": str(workspace), "directory": str(directory),
            "budgetFile": str(Path(directory) / "private/requests.json"), "totalLimit": None,
            "maxRequests": max_requests, "auth": credentials(config, repo),
            "modelId": config["model"], "api": config["api"], "python": config["python"],
            "systemPrompt": prompt, "tools": tools, "compact": compact,
            "checkpoint": checkpoint, "tracePersist": False,
            "executionPolicy": {"readOnly": ["scene", "observations"],
                                "readWrite": ["memory", "scratch"]}}
