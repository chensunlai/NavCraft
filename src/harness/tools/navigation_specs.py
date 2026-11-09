"""Observation, action and request schemas shared by memory implementations."""

from src.common.actions import ACTIONS
from .schema import definition, reference
from .node_specs import node_reference


def specifications(enhanced=False, *, node=False):
    string = {"type": "string"}
    observe = {"mode": {"type": "string", "enum": ["current", "panorama"]}}
    move = {"mode": {"type": "string", "enum": ["steps", "face", "approach"] if enhanced else ["steps"]},
            "actions": {"type": "array", "items": {"type": "string", "enum": list(ACTIONS)}, "minItems": 1, "maxItems": 6},
            "after": {"type": "string", "enum": ["current", "panorama"]}}
    if enhanced:
        if not node:
            observe["project"] = {"type": "boolean"}
        move.update(target=node_reference() if node else reference(),
                    stop_distance={"type": "number", "exclusiveMinimum": 0},
                    max_actions={"type": "integer", "minimum": 1, "maximum": 40})
    if node:
        observe_tool = definition("observe", "Capture current RGB or a six-view panorama. Images are attached. Select pixels on the returned image ID at its size; f_ IDs inside the panorama reopen single-frame detail with different coordinates. Panorama directions refer to the capture body heading; viewing a tile does not turn the body. A current panorama can be indexed with memory_edit without another capture.",
                             {"mode": {"type": "string", "enum": ["current", "panorama"]}})
    else:
        observe_tool = definition("observe", "Capture current RGB or a six-view panorama. Images are already attached; reopen only for detail. size is [width,height]; pixel=[x,y] uses the selected image ID's coordinates, not another tile or image. Panorama yaw is right-positive relative to the capture body heading, which is restored. Viewing a tile does not turn the body." +
                   (" project=true also attaches a local RGB projection map." if enhanced else ""), observe)
    move_tool = definition("move", "steps executes 1-6 actions: translation 0.25m, turn 30deg; sliding may add up to 15deg yaw. after selects endpoint RGB (default) or panorama. Feedback reports actual execution. Does not submit." +
                   (" face/approach use a selected target; max_actions includes turns. Local planning needs observed provenance, same floor, and a route within 8m; arbitrary map points use straight motion. reached refers only to the selected point, never task success." if enhanced else ""), move, ["mode"])
    if node and enhanced:
        move_tool["description"] += " Node targets resolve to visited anchors and use straight motion, like map coordinates; they do not authorize automatic route planning."
    return [observe_tool, move_tool,
        definition("submit", "Submit completion of the current request; include answer for a question. Ends this request, not the session. Acknowledgement contains no evaluation outcome.", {"answer": string}),
        definition("task_read", "Read the current navigation request and its target images. Images are labelled TARGET with frame IDs and sizes. No earlier requests, evaluation results, new observation or motion.", {})
    ]
