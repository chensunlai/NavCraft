"""Navigation extensions are defined once, in Python."""

ACTIONS = ["move_forward", "move_backward", "move_left", "move_right", "turn_left", "turn_right"]


def definition(name, description, properties, required=()):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(required), "additionalProperties": False}}


def specifications():
    string = {"type": "string"}
    ids = {"type": "array", "items": string}
    room_id = {"type": ["string", "null"], "pattern": "^R[0-9]{2,4}$",
               "description": "An existing room ID such as R01, never a room name; null means unknown."}
    point = {"anyOf": [{"type": "null"}, {"type": "array", "minItems": 2, "maxItems": 2,
                       "items": {"type": "number", "minimum": -.05, "maximum": 1.05}}]}
    fields = {"type": "object", "additionalProperties": False, "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 40},
        "description": {"type": ["string", "null"], "maxLength": 300},
        "evidence": {"type": ["array", "null"], "maxItems": 10,
                     "items": {"type": "string", "pattern": "^f_[0-9]{4,}$"},
                     "description": "Real frame IDs from observations, not filenames or invented evidence."},
        "anchor": {**point, "description": "Room label position [u,v], or null."},
        "point": {**point, "description": "Object approximate centre [u,v], or null."},
        "segment": {"anyOf": [{"type": "null"}, {"type": "array", "minItems": 2, "maxItems": 2,
                                              "items": point["anyOf"][1]}], "description": "Door endpoints [[u1,v1],[u2,v2]], or null."},
        "connects": {"type": ["array", "null"], "minItems": 2, "maxItems": 2, "items": room_id,
                     "description": "Exactly two room IDs/null, e.g. [R01,null]. Never use a room name."},
        "passage": {"type": ["string", "null"], "enum": ["unknown", "seen", "passed", "blocked", None],
                    "description": "unknown=inferred, seen=visually observed, passed=actually traversed, blocked=confirmed impassable."},
        "category": {"type": ["string", "null"]}, "room": room_id,
        "size": {"type": ["string", "null"], "enum": ["small", "medium", "large", None]},
    }}
    return [
        definition("nav_observe", "Capture a fresh current RGB or a six-view panorama. The body heading is restored after panorama; yaw offsets turn right-positive.",
                   {"mode": {"type": "string", "enum": ["current", "panorama"]}}),
        definition("nav_step", "Execute 1–6 atomic actions sequentially. Translate 0.25m or turn 30deg. Translation may slide and auto-turn by at most 15deg per action. The returned pose and RGB describe actual motion; blocked stops the batch.",
                   {"actions": {"type": "array", "items": {"type": "string", "enum": ACTIONS}, "minItems": 1, "maxItems": 6}}, ["actions"]),
        definition("map_view", "Render the fixed map-up wall map, recorded entities, current pose and path. Blank space need not be traversable. Example: view={around_self:6} requests a 6-metre radius.",
                   {"view": {"anyOf": [{"type": "string", "enum": ["explored", "full"]},
                                         {"type": "object", "properties": {"around": ids}, "required": ["around"], "additionalProperties": False},
                                         {"type": "object", "properties": {"around_self": {"type": "number", "exclusiveMinimum": 0}}, "required": ["around_self"], "additionalProperties": False}]},
                    "labels": {"type": "string", "enum": ["id", "name", "both", "none"]}, "highlight": ids}),
        definition("view_images", "Reopen 1–6 real observation or target frames. Optional crop is normalized [left,top,right,bottom]. Does not move or capture new evidence.",
                   {"frames": {"type": "array", "minItems": 1, "maxItems": 6, "items": {
                       "type": "object", "properties": {"frame": string, "crop": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}},
                       "required": ["frame"], "additionalProperties": False}},
                    "layout": {"type": "string", "enum": ["separate", "grid"]}}, ["frames"]),
        definition("memory_read", "Read current spatial memory. ids returns full records with evidence metadata. filter or no selector returns summaries, paginated by limit/offset. Does not infer current-relative direction.",
                   {"ids": ids, "filter": {"type": "object", "properties": {
                       "type": {"type": "string", "enum": ["room", "door", "object"]},
                       "room": room_id, "category": string, "located": {"type": "boolean"}}, "additionalProperties": False},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 300}, "offset": {"type": "integer", "minimum": 0}}),
        definition("memory_edit", "Atomically edit 1–20 entities. create requires id,type,fields with name; update requires id,fields; delete requires id; merge requires keep,remove and optional fields. Room fields: name,anchor; door: name,segment,connects,passage; object: name,category,point,room,size. All allow description,evidence. Only fields valid for that type are allowed. IDs use R/D/O plus 2–4 digits; room and connects reference existing R IDs, including rooms created in this batch. Arrays replace on update; null clears nullable values. Never reuse deleted IDs. Merge same-type entities, union evidence and rewrite references; resolve conflicting point/category/room/connects in fields. A bad batch writes nothing. Validation checks format/references, not spatial truth.",
                   {"operations": {"type": "array", "minItems": 1, "maxItems": 20,
                       "items": {"type": "object", "properties": {
                           "op": {"type": "string", "enum": ["create", "update", "delete", "merge"]},
                           "id": string, "type": {"type": "string", "enum": ["room", "door", "object"]},
                           "fields": fields, "keep": string, "remove": {**ids, "minItems": 1}},
                           "required": ["op"], "additionalProperties": False}}}, ["operations"]),
        definition("submit", "Submit completion of the current request. If the user asked a question, provide answer. This acknowledgement does not reveal an evaluation outcome.", {"answer": string}),
    ]
