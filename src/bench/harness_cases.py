"""Named smoke fixtures are evaluation configuration, not Core behavior."""

from pathlib import Path

import yaml


def case_config(repo, name):
    if name == "coding":
        return {"requests": 8, "seconds": 300}
    cases = yaml.safe_load((Path(repo) / "config/harness-pi/smoke.yaml").read_text())["cases"]
    if name not in cases:
        raise ValueError(f"Unknown case {name}; available: coding, {', '.join(cases)}")
    result = dict(cases[name])
    if name == "compaction":
        result.update(requests=16, seconds=600)
    return result
