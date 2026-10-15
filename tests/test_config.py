"""Five explicit YAML roles, repository-relative paths and safe snapshots."""

import json

import pytest
import yaml

from src.common import config as config_module


def files(tmp_path):
    documents = {
        "TASK_CONFIG": {"episodes": {"name": "objnav", "root": "data/objectnav", "selection_file": "data/ids.json"},
                        "task": {"name": "objectnav", "protocol": "fixture-objectnav", "success_distance": 0.1,
                                 "max_steps": 20, "max_subtask_steps": 20}},
        "ENV_CONFIG": {"environment": {"name": "fake", "scenes": "data/scenes"}},
        "MODEL_CONFIG": {"model": {"name": "responses", "base_url": "https://service.test/v1", "model": "vision-model",
                                   "api_key": "test-secret", "timeout_seconds": 60, "max_output_tokens": 100}},
        "HARNESS_CONFIG": {"harness": {"runtime": "agent", "tools": ["nav_observe", "nav_step", "nav_finish"]}},
        "RUN_CONFIG": {"run": {"output_dir": "runs/fixture", "max_requests": 4,
                               "timeout_seconds": 30, "max_context_chars": 10000}},
    }
    variables = {}
    for variable, document in documents.items():
        relative = f"config/{variable.lower()}.yaml"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(document))
        variables[variable] = relative
    return variables, documents


def test_five_roles_resolve_from_repository_root_not_working_directory(tmp_path, monkeypatch):
    variables, _ = files(tmp_path)
    monkeypatch.setattr(config_module, "ROOT", tmp_path)
    elsewhere = tmp_path / "other"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = config_module.load_config(variables)
    assert config.episodes["root"] == str(tmp_path / "data/objectnav")
    assert config.episodes["selection_file"] == str(tmp_path / "data/ids.json")
    assert config.environment["scenes"] == str(tmp_path / "data/scenes")
    assert config.run["output_dir"] == str(tmp_path / "runs/fixture")
    assert set(config.sources) == set(config_module.ROLES)
    assert config.sources["MODEL_CONFIG"] == str(tmp_path / variables["MODEL_CONFIG"])
    assert config.model["base_url"] == "https://service.test/v1"


def test_snapshot_redacts_nested_and_embedded_secrets_without_mutating_config(tmp_path, monkeypatch):
    variables, _ = files(tmp_path)
    monkeypatch.setattr(config_module, "ROOT", tmp_path)
    config = config_module.load_config(variables)
    config.model["note"] = "provider failed test-secret"
    config.harness["metadata"] = {"authorization": "Bearer private", "secret": "other"}
    snapshot = config.snapshot()
    assert snapshot["model"]["api_key"] == "[REDACTED]"
    assert snapshot["model"]["note"] == "provider failed [REDACTED]"
    assert snapshot["harness"]["metadata"] == {"authorization": "[REDACTED]", "secret": "[REDACTED]"}
    assert "test-secret" not in json.dumps(snapshot)
    assert config.model["api_key"] == "test-secret"


def test_missing_role_and_cross_role_section_are_explicit_errors(tmp_path, monkeypatch):
    variables, documents = files(tmp_path)
    monkeypatch.setattr(config_module, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="Missing RUN_CONFIG"):
        config_module.load_config({name: path for name, path in variables.items() if name != "RUN_CONFIG"})
    wrong = {**documents["ENV_CONFIG"], "run": {"max_requests": 999}}
    (tmp_path / variables["ENV_CONFIG"]).write_text(yaml.safe_dump(wrong))
    with pytest.raises(ValueError, match="expected exactly these sections"):
        config_module.load_config(variables)


@pytest.mark.parametrize("role, section, field, value, expected", [
    ("TASK_CONFIG", "task", "name", "goat", "incompatible"),
    ("TASK_CONFIG", "task", "protocol", "", "protocol"),
    ("RUN_CONFIG", "run", "max_requests", 2.5, "integers"),
    ("RUN_CONFIG", "run", "max_context_chars", True, "positive"),
    ("MODEL_CONFIG", "model", "base_url", "", "base_url"),
])
def test_invalid_protocol_and_budgets_fail_during_configuration(tmp_path, monkeypatch, role, section, field, value, expected):
    variables, documents = files(tmp_path)
    monkeypatch.setattr(config_module, "ROOT", tmp_path)
    documents[role][section][field] = value
    (tmp_path / variables[role]).write_text(yaml.safe_dump(documents[role]))
    with pytest.raises(ValueError, match=expected):
        config_module.load_config(variables)
