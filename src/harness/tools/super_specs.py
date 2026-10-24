"""Optional geometric assistance; schemas remain owned by Python."""

from copy import deepcopy


def extend(tools):
    by_name = {tool["name"]: tool for tool in tools}
    pixel = {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}}
    uv = {"type": "array", "minItems": 2, "maxItems": 2,
          "items": {"type": "number", "minimum": 0, "maximum": 1}}
    string = {"type": "string"}
    target = {"oneOf": [
        {"type": "object", "properties": {"frame": string, "pixel": pixel},
         "required": ["frame", "pixel"], "additionalProperties": False},
        {"type": "object", "properties": {"uv": uv}, "required": ["uv"], "additionalProperties": False},
        {"type": "object", "properties": {"entity": string}, "required": ["entity"], "additionalProperties": False},
    ]}
    by_name["nav_observe"]["parameters"]["properties"]["project"] = {"type": "boolean"}
    by_name["nav_observe"]["description"] += " project=true adds an ego map with RGB-D projection of the captured views."
    props = by_name["map_view"]["parameters"]["properties"]
    props["view"]["anyOf"].append({"type": "object", "properties": {"ego": {
        "type": "number", "exclusiveMinimum": 0}}, "required": ["ego"], "additionalProperties": False})
    props["overlay"] = {"type": "array", "maxItems": 6, "items": string}
    by_name["map_view"]["description"] += " view={ego:6} puts your heading up; overlay takes up to 6 observation frame IDs. Returns view_id and a pixel-to-global-UV affine transform."
    props = by_name["memory_read"]["parameters"]["properties"]
    props["near"] = {"oneOf": [
        {"type": "object", "properties": {"self": {"type": "number", "exclusiveMinimum": 0}},
         "required": ["self"], "additionalProperties": False},
        {"type": "object", "properties": {"entity": string, "radius": {"type": "number", "exclusiveMinimum": 0}},
         "required": ["entity"], "additionalProperties": False}]}
    by_name["memory_read"]["description"] += " near={self:5} or {entity:D02,radius:3} filters by metres. Returned bearing (right-positive) and distance are always relative to the current body. Do not combine near with ids."
    operation = by_name["memory_edit"]["parameters"]["properties"]["operations"]["items"]["properties"]
    at = {"type": "object", "properties": {"frame": string, "pixel": pixel},
          "required": ["frame", "pixel"], "additionalProperties": False}
    operation["fields"]["properties"]["at"] = at
    operation["coordinate_ref"] = string
    # Geometry may be map-view pixels; MemoryStore validates UV after conversion.
    for name in ("point", "anchor"):
        geometry = operation["fields"]["properties"][name]["anyOf"][1]
        geometry["items"] = {"type": "number"}
    by_name["memory_edit"]["description"] += " fields.at={frame:f_0042,pixel:[x,y]} projects the visible surface to point or anchor and adds frame evidence. A pixel inside an open doorway may hit a distant surface; place entrances on the projected 2D map instead. Do not also supply geometry. coordinate_ref=m_0007 interprets point/anchor as pixels in that saved map view, including ego views. Stored memory always uses global UV."
    def tool(name, description, properties, required):
        return {"name": name, "description": description, "parameters": {"type": "object",
                "properties": deepcopy(properties), "required": required, "additionalProperties": False}}
    tools.extend([
        tool("locate", "Project 1-10 original frame or map-view pixels onto the 2D map. Returns P1/P2 labels and global UV plus one map image; invalid points return an error. Purple markers are temporary and do not edit memory. RGB pixels locate the surface at their depth, not empty doorways; use the projected 2D map to place entrances. Geometry cannot identify mirrors, glass or objects.",
             {"source": string, "points": {"type": "array", "minItems": 1, "maxItems": 10, "items": pixel}}, ["source", "points"]),
        tool("nav_face", "Turn in place toward your selected frame pixel, global UV or remembered entity. Uses budgeted turns of at most 30 degrees; reports bearing_after. Does not select targets or submit.",
             {"target": target}, ["target"]),
        tool("nav_goto", "Move toward your selected point, not the benchmark goal. max_actions includes turns. reached/remaining describe this point only. Default stop_distance is 0.2m for height<=0.15m, else 0.6m (height heuristic, not object recognition). Local NavMesh planning requires an observed frame pixel or an entity located with fields.at, a same-floor route within 8m; arbitrary UV uses straight motion. Stops when reached, blocked, limited or outside local scope. Never submits.",
             {"target": target, "stop_distance": {"type": "number", "exclusiveMinimum": 0},
              "max_actions": {"type": "integer", "minimum": 1, "maximum": 40}}, ["target"]),
    ])
