"""Validated, atomic batches over the single public spatial-memory document."""

from copy import deepcopy
import json
import math
from pathlib import Path
import re
import os
import tempfile


PREFIXES = {"room": "R", "door": "D", "object": "O"}
COMMON = {"name", "description", "evidence"}
FIELDS = {
    "room": COMMON | {"anchor"},
    "door": COMMON | {"segment", "connects", "passage"},
    "object": COMMON | {"category", "point", "size", "room"},
}
DEFAULTS = {
    "room": {"anchor": None},
    "door": {"segment": None, "connects": [None, None], "passage": "unknown"},
    "object": {"point": None, "room": None, "category": None},
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def observation_index(workspace):
    path = Path(workspace) / "observations/index.jsonl"
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text().splitlines():
        if line.strip():
            item = json.loads(line)
            result[item["frame"]] = item
    return result


def location(entity):
    if entity["type"] == "room":
        return entity.get("anchor")
    if entity["type"] == "object":
        return entity.get("point")
    segment = entity.get("segment")
    return [(segment[0][i] + segment[1][i]) / 2 for i in range(2)] if segment else None


class MemoryStore:
    def __init__(self, workspace, private_dir=None):
        self.workspace = Path(workspace)
        self.path = self.workspace / "memory/entities.json"
        self.used_path = Path(private_dir or self.workspace.parent / "private") / "memory_used_ids.json"
        if not self.path.exists():
            atomic_json(self.path, [])
        initial = self.entities
        if len(initial) > 300 or any(not isinstance(e, dict) for e in initial):
            raise ValueError("Memory snapshot must contain at most 300 entities")
        indexed = {e.get("id"): e for e in initial}
        if len(indexed) != len(initial):
            raise ValueError("Memory snapshot contains duplicate IDs")
        frames = observation_index(self.workspace)
        for entity in initial:
            self._validate(entity, indexed, frames)
        # Snapshot IDs are issued IDs too, including if they are deleted later.
        used = set(json.loads(self.used_path.read_text())) if self.used_path.exists() else set()
        if not set(indexed).issubset(used):
            atomic_json(self.used_path, sorted(used | set(indexed)))

    @property
    def entities(self):
        result = json.loads(self.path.read_text())
        if not isinstance(result, list):
            raise ValueError("memory/entities.json must be an array")
        return result

    def read(self, ids=None, filter=None, limit=30, offset=0):
        try:
            if ids is not None and filter is not None:
                raise ValueError("ids 与 filter 不能同时使用")
            if type(limit) is not int or not 1 <= limit <= 300 or type(offset) is not int or offset < 0:
                raise ValueError("limit 必须为 1–300，offset 必须为非负整数")
            entities = self.entities
            if ids is not None:
                if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
                    raise ValueError("ids 必须为 ID 数组")
                indexed = {e["id"]: e for e in entities}
                frames = observation_index(self.workspace)
                found = []
                for identifier in dict.fromkeys(ids):
                    if identifier in indexed:
                        entity = deepcopy(indexed[identifier])
                        entity["evidence"] = [
                            {"frame": fid, "step": frames[fid].get("step"), "pose": frames[fid].get("pose")}
                            for fid in entity.get("evidence", []) if fid in frames
                        ]
                        found.append(entity)
                return {"entities": found, "missing": [i for i in ids if i not in indexed]}
            if filter is not None:
                if not isinstance(filter, dict) or set(filter) - {"type", "room", "category", "located"}:
                    raise ValueError("filter 只支持 type、room、category、located")
                if "located" in filter and type(filter["located"]) is not bool:
                    raise ValueError("located 必须为布尔值")
                entities = [e for e in entities if all(
                    (location(e) is not None) == value if key == "located"
                    else (e.get(key) == value and (key != "room" or e["type"] == "object"))
                    for key, value in filter.items()
                )]
            page = entities[offset:offset + limit]
            return {
                "entities": [{**{k: e[k] for k in ("id", "type", "name", "category", "room") if k in e},
                              "located": location(e) is not None} for e in page],
                "total": len(entities),
                "next_offset": offset + limit if offset + limit < len(entities) else None,
            }
        except (ValueError, KeyError, TypeError) as exc:
            return {"error": str(exc)}

    def edit(self, operations):
        context = ""
        try:
            if not isinstance(operations, list) or not 1 <= len(operations) <= 20:
                raise ValueError("operations 必须包含 1–20 项")
            current = self.entities
            state = {e["id"]: deepcopy(e) for e in current}
            used = set(json.loads(self.used_path.read_text())) if self.used_path.exists() else set()
            used.update(state)
            changes, touched = [], {}
            for index, operation in enumerate(operations, 1):
                if not isinstance(operation, dict):
                    raise ValueError(f"第 {index} 项必须为对象")
                op = operation.get("op")
                identifier = operation.get("id", operation.get("keep"))
                context = f"第 {index} 项 {op} {identifier}："
                if op == "create":
                    self._keys(operation, {"op", "id", "type", "fields"})
                    kind = operation.get("type")
                    self._id(identifier, kind)
                    if identifier in used:
                        number = next(i for i in range(1, 10000) if f"{PREFIXES[kind]}{i:02d}" not in used)
                        return {"error": context + "ID 已经使用过", "hint": f"可用 ID：{PREFIXES[kind]}{number:02d}"}
                    fields = self._fields(operation.get("fields", {}), kind)
                    state[identifier] = {"id": identifier, "type": kind, **deepcopy(DEFAULTS[kind]),
                                         "description": "", "evidence": [], **fields}
                    used.add(identifier)
                elif op == "update":
                    self._keys(operation, {"op", "id", "fields"})
                    if identifier not in state:
                        raise ValueError("实体不存在")
                    state[identifier].update(self._fields(operation.get("fields", {}), state[identifier]["type"]))
                elif op == "delete":
                    self._keys(operation, {"op", "id"})
                    if identifier not in state:
                        raise ValueError("实体不存在")
                    del state[identifier]
                elif op == "merge":
                    self._keys(operation, {"op", "keep", "remove", "fields"})
                    self._merge(state, operation)
                else:
                    raise ValueError("op 必须为 create、update、delete 或 merge")
                touched[identifier] = context
                changes.append({"op": op, "id": identifier, **({"remove": operation["remove"]} if op == "merge" else {})})
            if len(state) > 300:
                raise ValueError("实体总数不能超过 300")
            frames = observation_index(self.workspace)
            for identifier, entity in state.items():
                context = touched.get(identifier, f"实体 {identifier}：")
                self._validate(entity, state, frames)
            # Reserve IDs first: even an interrupted public replace cannot reuse an issued ID.
            atomic_json(self.used_path, sorted(used))
            atomic_json(self.path, list(state.values()))
            return {"changes": changes}
        except (ValueError, KeyError, TypeError, StopIteration) as exc:
            return {"error": context + str(exc)}

    @staticmethod
    def _keys(value, allowed):
        if set(value) - allowed:
            raise ValueError("未知字段：" + ", ".join(sorted(set(value) - allowed)))

    @staticmethod
    def _id(identifier, kind):
        if kind not in PREFIXES or not isinstance(identifier, str) or not re.fullmatch(PREFIXES[kind] + r"\d{2,4}", identifier):
            raise ValueError("ID 必须与类型前缀一致，后跟 2–4 位数字")

    def _fields(self, fields, kind):
        if not isinstance(fields, dict):
            raise ValueError("fields 必须为对象")
        self._keys(fields, FIELDS[kind])
        fields = deepcopy(fields)
        for key, default in (("description", ""), ("evidence", []), ("connects", [None, None]), ("passage", "unknown")):
            if key in fields and fields[key] is None:
                fields[key] = deepcopy(default)
        return fields

    def _merge(self, state, operation):
        keep, remove = operation.get("keep"), operation.get("remove")
        if keep not in state or not isinstance(remove, list) or not remove or any(not isinstance(i, str) for i in remove):
            raise ValueError("merge 需要存在的 keep 和非空 remove ID 数组")
        if keep in remove or len(set(remove)) != len(remove) or any(i not in state for i in remove):
            raise ValueError("remove 必须互异、存在，且不能包含 keep")
        kind = state[keep]["type"]
        if any(state[i]["type"] != kind for i in remove):
            raise ValueError("只能合并同类型实体")
        fields = self._fields(operation.get("fields", {}), kind)
        sources = [state[keep]] + [state[i] for i in remove]
        result = deepcopy(state[keep])
        for key in ("point", "category", "room", "connects"):
            values = []
            for source in sources:
                value = source.get(key)
                if value is not None and value not in values:
                    values.append(value)
            if len(values) > 1 and key not in fields:
                raise ValueError(f"{key} 有冲突，请在 fields 中指定最终值")
            if values and result.get(key) is None:
                result[key] = values[0]
        result.update(fields)
        evidence = []
        for source in sources + [fields]:
            for fid in source.get("evidence", []):
                if fid not in evidence:
                    evidence.append(fid)
        result["evidence"] = evidence
        state[keep] = result
        for identifier in remove:
            del state[identifier]
        for entity in state.values():
            if entity.get("room") in remove:
                entity["room"] = keep
            if "connects" in entity:
                entity["connects"] = [keep if i in remove else i for i in entity["connects"]]

    def _validate(self, entity, state, frames):
        self._id(entity.get("id"), entity.get("type"))
        self._keys(entity, FIELDS[entity["type"]] | {"id", "type"})
        if not isinstance(entity.get("name"), str) or not entity["name"].strip() or len(entity["name"]) > 40:
            raise ValueError("name 必须为 1–40 字的非空字符串")
        if not isinstance(entity.get("description"), str) or len(entity["description"]) > 300:
            raise ValueError("description 不能超过 300 字")
        evidence = entity.get("evidence")
        if not isinstance(evidence, list) or len(evidence) > 10 or any(not isinstance(i, str) for i in evidence):
            raise ValueError("evidence 必须为不超过 10 个帧号的数组")
        if any(i not in frames for i in evidence):
            raise ValueError("evidence 包含不存在的帧号")
        for key in ("point", "anchor"):
            if entity.get(key) is not None:
                self._point(entity[key])
        if entity.get("segment") is not None:
            if not isinstance(entity["segment"], list) or len(entity["segment"]) != 2:
                raise ValueError("segment 必须为两个端点")
            for point in entity["segment"]:
                self._point(point)
        if entity["type"] == "door":
            if entity.get("passage") not in {"unknown", "seen", "passed", "blocked"}:
                raise ValueError("passage 无效")
            refs = entity.get("connects")
            if not isinstance(refs, list) or len(refs) != 2:
                raise ValueError("connects 必须包含两个房间 ID 或 null")
        else:
            refs = [entity.get("room")] if entity["type"] == "object" else []
        for reference in refs:
            if reference is not None and (not isinstance(reference, str) or reference not in state or state[reference]["type"] != "room"):
                raise ValueError(f"房间 {reference} 不存在")
        if entity["type"] == "object":
            if entity.get("category") is not None and (not isinstance(entity["category"], str) or not entity["category"].strip()):
                raise ValueError("category 必须为字符串或 null")
            if entity.get("size") not in {None, "small", "medium", "large"}:
                raise ValueError("size 无效")

    @staticmethod
    def _point(value):
        if not isinstance(value, list) or len(value) != 2 or any(type(v) not in (float, int) or not math.isfinite(v) or not -.05 <= v <= 1.05 for v in value):
            raise ValueError("坐标必须为 [-0.05, 1.05] 范围内的两个有限数值")
