"""Tool schemas for reading and editing visited-position memory."""


def _object(properties, required=()):
    return {"type": "object", "properties": properties,
            "required": list(required), "additionalProperties": False}


def node_reference():
    pair = {"type": "array", "minItems": 2, "maxItems": 2,
            "items": {"type": "number"}}
    return {"oneOf": [
        _object({"node": {"type": "string", "pattern": "^N[0-9]{3,}$",
                          "description": "Visited anchor. Approach steers directly toward it, not along a recorded route; choose intermediate visible waypoints around obstacles."}}, ["node"]),
        _object({"image": {"type": "string"}, "pixel": pair}, ["image", "pixel"]),
        _object({"uv": pair}, ["uv"]),
    ]}


def memory_specifications():
    node = {"type": "string", "pattern": "^N[0-9]{3,}$"}
    room = {"type": ["string", "null"], "minLength": 1, "pattern": "\\S"}
    room_name = {"type": "string", "minLength": 1, "pattern": "\\S"}
    description = {"type": "string", "minLength": 1, "maxLength": 300, "pattern": "\\S"}
    pagination = {"limit": {"type": "integer", "minimum": 1, "maximum": 300},
                  "offset": {"type": "integer", "minimum": 0}}
    text = {"mode": {"const": "text"}, "source": {"const": "nodes"}, **pagination}
    node_list = _object({**text, "room": room, "query": {"type": "string"}})
    node_detail = _object({**text, "id": node}, ["id"])
    rooms = _object({"mode": {"const": "text"}, "source": {"const": "rooms"},
                     "room": room_name, "query": {"type": "string"}, **pagination}, ["source"])
    observations = _object({"mode": {"const": "text"},
                            "source": {"const": "observations"}, **pagination}, ["source"])
    images = _object({"mode": {"const": "image"},
                      "refs": {"type": "array", "minItems": 1, "maxItems": 2,
                               "items": {"type": "string", "minLength": 1}},
                      "crop": {"type": "array", "minItems": 4, "maxItems": 4,
                               "items": {"type": "number", "minimum": 0, "maximum": 1}}},
                     ["mode", "refs"])
    images["allOf"] = [{"if": {"required": ["crop"]},
                         "then": {"properties": {"refs": {"maxItems": 1}}}}]
    maps = _object({"mode": {"const": "map"},
                    "view": {"type": "string", "enum": ["local", "explored", "ego"]},
                    "radius_m": {"type": "number", "exclusiveMinimum": 0},
                    "room": room_name,
                    "highlight": {"type": "array", "items": node, "uniqueItems": True}}, ["mode"])
    maps["allOf"] = [{"if": {"required": ["view"], "properties": {"view": {"const": "explored"}}},
                       "then": {"not": {"required": ["radius_m"]}}}]
    maps["allOf"].append({"if": {"required": ["room"], "not": {"required": ["view"]}},
                          "then": {"not": {"required": ["radius_m"]}}})

    create = _object({"op": {"const": "create"}, "description": description, "room": room},
                     ["op", "description"])
    update = _object({"op": {"const": "update"}, "id": node, "description": description,
                      "room": room, "refresh": {"type": "boolean"}}, ["op", "id"])
    update["anyOf"] = [{"required": ["description"]}, {"required": ["room"]},
                       {"required": ["refresh"], "properties": {"refresh": {"const": True}}}]
    room_update = _object({"op": {"const": "update"}, "room": room_name,
                           "description": {"type": ["string", "null"], "minLength": 1,
                                           "maxLength": 1000, "pattern": "\\S"}},
                          ["op", "room", "description"])
    delete = _object({"op": {"const": "delete"}, "id": node}, ["op", "id"])
    merge = _object({"op": {"const": "merge"}, "keep": node,
                     "remove": {"type": "array", "minItems": 1, "uniqueItems": True, "items": node},
                     "description": description, "room": room}, ["op", "keep", "remove", "description"])
    return [
        {"name": "memory_read",
         "description": "Read visited-position memory; mode defaults to text. Text source=nodes returns "
                        "a paginated summary or one id's details; room/query filter summaries. query matches "
                        "recorded text, not scene contents. source=rooms lists room descriptions and associated "
                        "node IDs; room/query filter this directory. source=observations lists all captured scene events, "
                        "including unrecorded stops. Text never attaches images. image opens 1-2 node, panorama, "
                        "or registered image refs; a node opens its current panorama. Returned node/anchor_pose identify "
                        "the place; pose is the actual picture's capture pose. Select pixels with the returned image ID "
                        "and size, including panorama grids and crops; frame IDs inside grids only reopen detail. crop=[left,top,right,bottom] "
                        "is normalized and requires one ref. map renders visited nodes and actual paths on a blank "
                        "map with a complete visible node_index and up to five descriptions (highlighted, then nearby); "
                        "room highlights its nodes and returns its description. With room selected, view defaults "
                        "to explored; otherwise local. local/ego radius_m defaults to 6, explored fits recorded "
                        "locations. Reading never captures images or moves.",
         "parameters": {"type": "object", "oneOf": [node_list, node_detail, rooms, observations, images, maps]}},
        {"name": "memory_edit",
         "description": "Atomically apply 1-20 node or room-description edits. create requires description and assigns an N ID; "
                        "room is a stable group name or null. create and update(refresh=true) require an already "
                        "captured panorama at the current pose and step; call observe(mode=panorama) first when "
                        "needed. Editing never captures an image. At most one create or refresh per batch. "
                        "Refresh requires being within 1m of the fixed first anchor; it preserves description "
                        "unless replaced. Text/room corrections work remotely. Omitted room is preserved; null "
                        "clears it. merge requires keep, remove and a final description; resolve differing room "
                        "groups explicitly. Merge keeps the retained node's anchor and panorama. Deleting or "
                        "merging nodes does not delete observation images. Node positions are camera locations, "
                        "not object positions. For a known room, update without id replaces its overall description: "
                        "{op:'update',room:'living_room_1',description:'Sofa facing TV; dining area by the window.'}. "
                        "Room descriptions allow 1000 characters; null removes only the description. Node membership "
                        "comes from node.room; room edits need no panorama. Returned changes acknowledge storage, "
                        "not semantic correctness.",
         "parameters": _object({"operations": {"type": "array", "minItems": 1, "maxItems": 20,
                                                "items": {"oneOf": [create, update, room_update, delete, merge]}}},
                               ["operations"])},
    ]
