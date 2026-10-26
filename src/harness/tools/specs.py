"""The seven navigation interfaces; shared references have one schema."""

from src.common.actions import ACTIONS


def definition(name, description, properties, required=()):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(required), "additionalProperties": False}}


def reference():
    pair = {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}}
    return {"oneOf": [
        {"type": "object", "properties": {"entity": {"type": "string"}}, "required": ["entity"], "additionalProperties": False},
        {"type": "object", "properties": {"image": {"type": "string"}, "pixel": pair},
         "required": ["image", "pixel"], "additionalProperties": False},
        {"type": "object", "properties": {"uv": pair}, "required": ["uv"], "additionalProperties": False},
    ]}


def specifications(tool_tier="basic"):
    if tool_tier not in {"basic", "super"}:
        raise ValueError("tool_tier must be basic or super")
    enhanced = tool_tier == "super"
    string = {"type": "string"}
    ids = {"type": "array", "items": string}
    room_id = {"type": ["string", "null"], "pattern": "^R[0-9]{2,4}$",
               "description": "Existing room ID or null; never a room name."}
    point = {"anyOf": [{"type": "null"}, {"type": "array", "minItems": 2, "maxItems": 2,
                       "items": {"type": "number", "minimum": 0, "maximum": 1}}]}
    fields = {"type": "object", "additionalProperties": False, "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 40},
        "description": {"type": ["string", "null"], "maxLength": 300},
        "evidence": {"type": ["array", "null"], "maxItems": 10,
                     "items": {"type": "string", "pattern": "^f_[0-9]{4,}$"},
                     "description": "Original observed frame IDs supporting the record."},
        "anchor": {**point, "description": "Room label position in global UV; null means unknown."},
        "point": {**point, "description": "Door/object reference point in global UV; null means unknown."},
        "location": {**reference(), "description": "Resolve a reference to point/anchor. Do not combine with point/anchor. Image pixels use the displayed image coordinates; RGB projection requires super tools."},
        "connects": {"type": ["array", "null"], "minItems": 2, "maxItems": 2, "items": room_id},
        "passage": {"type": ["string", "null"], "enum": ["unknown", "seen", "passed", "blocked", None],
                    "description": "unknown=inferred, seen=observed, passed=traversed, blocked=confirmed impassable."},
        "category": {"type": ["string", "null"]}, "room": room_id,
        "size": {"type": ["string", "null"], "enum": ["small", "medium", "large", None]},
    }}
    observe = {"mode": {"type": "string", "enum": ["current", "panorama"]}}
    move = {"mode": {"type": "string", "enum": ["steps", "face", "approach"] if enhanced else ["steps"]},
            "actions": {"type": "array", "items": {"type": "string", "enum": list(ACTIONS)}, "minItems": 1, "maxItems": 6},
            "after": {"type": "string", "enum": ["current", "panorama"]}}
    view = {"anyOf": [{"type": "string", "enum": ["explored", "full"]},
                       {"type": "object", "properties": {"around": ids}, "required": ["around"], "additionalProperties": False},
                       *[{"type": "object", "properties": {key: {"type": "number", "exclusiveMinimum": 0}},
                          "required": [key], "additionalProperties": False} for key in ("around_self", "ego")]]}
    maps = {"view": view, "labels": {"type": "string", "enum": ["id", "name", "both", "none"]}, "highlight": ids,
            "project": {"type": "object", "properties": {"source": string, "points": {
                "type": "array", "minItems": 1, "maxItems": 10, "items": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}}}},
                        "required": ["source", "points"], "additionalProperties": False}}
    memory = {"ids": ids, "filter": {"type": "object", "properties": {
                  "type": {"type": "string", "enum": ["room", "door", "object"]},
                  "room": room_id, "category": string, "located": {"type": "boolean"}}, "additionalProperties": False},
              "limit": {"type": "integer", "minimum": 1, "maximum": 300}, "offset": {"type": "integer", "minimum": 0}}
    if enhanced:
        observe["project"] = {"type": "boolean"}
        move.update(target=reference(), stop_distance={"type": "number", "exclusiveMinimum": 0},
                    max_actions={"type": "integer", "minimum": 1, "maximum": 40})
        maps["overlay"] = {"type": "array", "maxItems": 6, "items": string}
        memory["near"] = {"oneOf": [
            {"type": "object", "properties": {"self": {"type": "number", "exclusiveMinimum": 0}}, "required": ["self"], "additionalProperties": False},
            {"type": "object", "properties": {"entity": string, "radius": {"type": "number", "exclusiveMinimum": 0}}, "required": ["entity", "radius"], "additionalProperties": False}]}
    return [
        definition("observe", "Capture current RGB or a six-view panorama. Images are already attached for inspection; reopen only for detail. Panorama labels are right-positive from the capture body heading, which is restored. project=true also attaches a local RGB projection map (super only).", observe),
        definition("move", "steps executes 1-6 actions: translation 0.25m, turn 30deg; sliding may add up to 15deg yaw. face/approach use a selected target (super only); max_actions includes turns. Local planning needs observed provenance, same floor, and a route within 8m; arbitrary map points use straight motion. after selects endpoint RGB (default) or panorama. Feedback reports actual execution; reached refers only to your selected point, never task success. Does not submit.", move, ["mode"]),
        definition("map_view", "Render walls, recorded entities, pose and path, with up to 12 related memory records. view={ego:6} is heading-up; other views are map-up. Full view fits walls with at most 1m padding. project marks selected image pixels; markers do not edit memory. RGB projection requires super tools and hits surfaces, not empty doorway centres. overlay uses existing observation frames (super only). Blank map space is not guaranteed traversable.", maps),
        definition("view_images", "Reopen 1-6 registered images (f_ frame, m_ map, i_ derived). Optional normalized crop [left,top,right,bottom]. Returned images have stable IDs; point selection uses pixels of the displayed image, including crops and grids. No new observation or motion.",
                   {"images": {"type": "array", "minItems": 1, "maxItems": 6, "items": {"type": "object", "properties": {
                       "image": string, "crop": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}},
                       "required": ["image"], "additionalProperties": False}},
                    "layout": {"type": "string", "enum": ["separate", "grid"]}}, ["images"]),
        definition("memory_read", "Read model-written spatial records. ids returns details; queries matching up to 5 entities also return details, otherwise paginated summaries. Detailed results attach at most 2 distinct evidence images; remaining frames can be reopened. near filters in metres and gives current-relative bearings (super only). Records are not verified observations.", memory),
        definition("memory_edit", "Atomically edit 1-20 entities. create requires id,type,fields.name; update requires id,fields; delete requires id; merge requires keep,remove and optional fields. Rooms use anchor; doors point/connects/passage; objects point/category/room/size. All allow description/evidence. location converts a shared reference to stored geometry. Arrays replace; null clears nullable fields. Same-batch room references are allowed. Deleted IDs are never reused. Merge unions evidence and rewrites references; resolve geometry/category/room/connects conflicts explicitly. Bad batches write nothing. Returns actual changed records; validates format/references, not spatial truth.",
                   {"operations": {"type": "array", "minItems": 1, "maxItems": 20,
                    "items": {"type": "object", "properties": {
                        "op": {"type": "string", "enum": ["create", "update", "delete", "merge"]},
                        "id": string, "type": {"type": "string", "enum": ["room", "door", "object"]},
                        "fields": fields, "keep": string, "remove": {**ids, "minItems": 1}},
                        "required": ["op"], "additionalProperties": False}}}, ["operations"]),
        definition("submit", "Submit completion of the current request; include answer for a question. Ends this request, not the session. Acknowledgement contains no evaluation outcome.", {"answer": string}),
    ]
