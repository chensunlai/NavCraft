"""Compose YAML/JSON mappings; later files override earlier ones recursively."""

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


def merge_config(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = (merge_config(result[key], value)
                       if isinstance(result.get(key), dict) and isinstance(value, dict)
                       else deepcopy(value))
    return result


def read_config(paths, _parents=()):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    result = {}
    for name in paths or []:
        path = Path(name).resolve()
        if path in _parents:
            raise ValueError(f"Circular config extends: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config must be a mapping: {path}")
        parents = data.pop("extends", [])
        if isinstance(parents, str):
            parents = [parents]
        inherited = read_config([path.parent / p for p in parents], (*_parents, path))
        result = merge_config(result, merge_config(inherited, data))
    return result


def parse_config_args(parser, argv=None, aliases=None):
    """Config defaults followed by explicit CLI overrides; paths remain repo-relative."""
    parser.add_argument("--config", action="append", type=Path, default=[],
                        help="Repeat to compose configs in order; CLI flags override them")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", action="append", type=Path, default=[])
    files, _ = pre.parse_known_args(argv)
    values = read_config(files.config)
    for source, destination in (aliases or {}).items():
        if source in values:
            values[destination] = values[source]
    for action in parser._actions:
        if action.dest not in values or action.dest in {"config", "help"}:
            continue
        value = values[action.dest]
        if value is None:
            continue
        if action.type:
            value = ([action.type(v) for v in value] if isinstance(value, list)
                     else action.type(value))
        if action.choices and any(v not in action.choices for v in (value if isinstance(value, list) else [value])):
            parser.error(f"Invalid config value for {action.dest}: {value}")
        action.default = value
        action.required = False
    return parser.parse_args(argv)
