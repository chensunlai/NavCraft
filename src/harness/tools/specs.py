"""Navigation extensions are defined once, in Python."""

ACTIONS = ["move_forward", "move_backward", "move_left", "move_right", "turn_left", "turn_right"]


def definition(name, description, properties, required=()):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(required), "additionalProperties": False}}


def specifications():
    string = {"type": "string"}
    ids = {"type": "array", "items": string}
    return [
        definition("nav_observe", "Capture a fresh current RGB or a six-view panorama. The body heading is restored after panorama; yaw offsets turn right-positive.",
                   {"mode": {"type": "string", "enum": ["current", "panorama"]}}),
        definition("nav_step", "Execute 1–6 atomic actions sequentially. Translate 0.25m or turn 30deg. Translation may slide and auto-turn by at most 15deg per action. The returned pose and RGB describe actual motion; blocked stops the batch.",
                   {"actions": {"type": "array", "items": {"type": "string", "enum": ACTIONS}, "minItems": 1, "maxItems": 6}}, ["actions"]),
        definition("map_view", "Render the fixed map-up wall map, recorded entities, current pose and path. Blank space need not be traversable. around_self is a radius in metres.",
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
                       "room": {"type": ["string", "null"]}, "category": string, "located": {"type": "boolean"}}, "additionalProperties": False},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 300}, "offset": {"type": "integer", "minimum": 0}}),
        definition("memory_edit", "Atomically create/update/delete/merge at most 20 spatial entities. Fields: room(name,anchor), door(name,segment,connects,passage), object(name,category,point,room,size); all allow description/evidence. Coordinates normalized; evidence contains real frame IDs. IDs R/D/O plus 2–4 digits, never reuse removed IDs. Arrays replace on update. Merge needs keep/remove/fields and explicit resolution of conflicting point/category/room/connects. Validation checks format/references, not spatial truth.",
                   {"operations": {"type": "array", "minItems": 1, "maxItems": 20,
                       "items": {"type": "object", "properties": {
                           "op": {"type": "string", "enum": ["create", "update", "delete", "merge"]},
                           "id": string, "type": {"type": "string", "enum": ["room", "door", "object"]},
                           "fields": {"type": "object", "additionalProperties": True}, "keep": string, "remove": ids},
                           "required": ["op"], "additionalProperties": False}}}, ["operations"]),
        definition("submit", "Submit completion of the current request. If the user asked a question, provide answer. This acknowledgement does not reveal an evaluation outcome.", {"answer": string}),
    ]
