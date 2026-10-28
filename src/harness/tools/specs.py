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
               "description": "Existing room ID, e.g. R01 or R001 (uppercase R + 2-4 digits), or null; never a room name. Edits may reference rooms created in the same batch."}
    point = {"anyOf": [{"type": "null"}, {"type": "array", "minItems": 2, "maxItems": 2,
                       "items": {"type": "number", "minimum": 0, "maximum": 1}}]}
    fields = {"type": "object", "additionalProperties": False,
              "description": "Entity attributes go here, not beside op/id/type. Omitted update fields are preserved.", "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 40},
        "description": {"type": ["string", "null"], "maxLength": 300},
        "evidence": {"type": ["array", "null"], "maxItems": 10,
                     "items": {"type": "string", "pattern": "^f_[0-9]{4,}$"},
                     "description": "Original scene-observation f_ IDs that show this entity. Target/question images also have f_ IDs but are not evidence; map/grid/crop IDs are not evidence. Arrays replace existing evidence."},
        "anchor": {**point, "description": "Room label position in global UV; null means unknown."},
        "point": {**point, "description": "Door/object reference point in global UV; null means unknown."},
        "location": {**reference(), "description": "Resolve to point/anchor; do not combine with either. entity copies an existing located entity's position, not room membership or a same-batch creation. " +
                     ("Image pixels use displayed coordinates; RGB pixels project to a visible surface." if enhanced else
                      "Basic: image/pixel must select a map image, not RGB; uv is global normalized map position. Omit unknown locations.")},
        "connects": {"type": ["array", "null"], "minItems": 2, "maxItems": 2, "items": room_id},
        "passage": {"type": ["string", "null"], "enum": ["unknown", "seen", "passed", "blocked", None],
                    "description": "unknown=inferred, seen=observed, passed=traversed, blocked=confirmed impassable."},
        "category": {"type": ["string", "null"],
                     "description": "Specific consistent object class, e.g. chair or refrigerator, not furniture/appliance; null if unknown. Category queries match exactly."}, "room": room_id,
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
            "project": {"type": "object", "properties": {"source": {**string,
                "description": "Registered image ID; selected pixels use its displayed coordinates." if enhanced else
                               "Map image ID (m_) or a derived view of a map. RGB projection is unavailable in basic."}, "points": {
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
        definition("observe", "Capture current RGB or a six-view panorama. Images are already attached; reopen only for detail. size is [width,height]; pixel=[x,y] uses the selected image ID's coordinates, not another tile or image. Panorama yaw is right-positive relative to the capture body heading, which is restored. Viewing a tile does not turn the body." +
                   (" project=true also attaches a local RGB projection map." if enhanced else ""), observe),
        definition("move", "steps executes 1-6 actions: translation 0.25m, turn 30deg; sliding may add up to 15deg yaw. after selects endpoint RGB (default) or panorama. Feedback reports actual execution. Does not submit." +
                   (" face/approach use a selected target; max_actions includes turns. Local planning needs observed provenance, same floor, and a route within 8m; arbitrary map points use straight motion. reached refers only to the selected point, never task success." if enhanced else ""), move, ["mode"]),
        definition("map_view", "Render walls, recorded entities, pose and path, with up to 12 related memory records. view={ego:6} is heading-up; other views are map-up. Full view fits walls with at most 1m padding. project marks selected pixels without editing memory. " +
                   ("RGB projection hits surfaces, not empty doorway centres; overlay uses existing observation frames. " if enhanced else
                    "Basic project converts map-image pixels to global UV, not RGB pixels. ") +
                   "Blank map space is not guaranteed traversable.", maps),
        definition("view_images", "Reopen 1-6 registered images (f_ frame, m_ map, i_ derived). Optional normalized crop [left,top,right,bottom]. Returned images have stable IDs; point selection uses pixels of the displayed image, including crops and grids. No new observation or motion.",
                   {"images": {"type": "array", "minItems": 1, "maxItems": 6, "items": {"type": "object", "properties": {
                       "image": string, "crop": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}},
                       "required": ["image"], "additionalProperties": False}},
                    "layout": {"type": "string", "enum": ["separate", "grid"]}}, ["images"]),
        definition("memory_read", "Read model-written spatial records. ids returns details; queries matching up to 5 entities also return details, otherwise paginated summaries. Filters match exactly; if category returns nothing, try type=object to inspect stored names/categories. Detailed results attach at most 2 distinct evidence images; remaining frames can be reopened. Records are not verified observations." +
                   (" near filters in metres and gives current-relative bearings." if enhanced else ""), memory),
        definition("memory_edit", "Atomically edit 1-20 entities. IDs use uppercase R (room), D (door), or O (object) followed by 2-4 digits, e.g. R01/D01/O01. create requires id,type,fields.name; update requires id,fields; delete requires id; merge requires keep,remove and optional fields. Rooms use anchor; doors point/connects/passage; objects point/category/room/size. All allow description/evidence. location converts a shared reference to stored geometry. Arrays replace; null clears nullable fields. Same-batch room references are allowed. Deleted IDs are never reused. Merge unions evidence and rewrites references; resolve geometry/category/room/connects conflicts explicitly. Bad batches write nothing. Position edits attach one preview of saved map points and up to two selected source images. This shows what was saved, not whether the object was identified correctly. Returns actual changed records; validates format/references, not spatial truth.",
                   {"operations": {"type": "array", "minItems": 1, "maxItems": 20,
                    "description": "JSON array of edit objects, not a JSON-encoded string. Put entity attributes inside each operation's fields.",
                    "items": {"type": "object", "properties": {
                        "op": {"type": "string", "enum": ["create", "update", "delete", "merge"]},
                        "id": {**string, "description": "create: choose a never-used ID matching type; update/delete: reuse the exact existing ID. Examples: R01, R001, D01, O01 (letter O). Lowercase IDs and names such as obj_001 are invalid."},
                        "type": {"type": "string", "enum": ["room", "door", "object"]},
                        "fields": fields,
                        "keep": {**string, "description": "merge: exact existing entity ID to retain, e.g. O01."},
                        "remove": {**ids, "minItems": 1, "description": "merge: distinct existing IDs of the same type as keep, excluding keep, e.g. [\"O02\"]."}},
                        "required": ["op"], "additionalProperties": False}}}, ["operations"]),
        definition("submit", "Submit completion of the current request; include answer for a question. Ends this request, not the session. Acknowledgement contains no evaluation outcome.", {"answer": string}),
    ]
