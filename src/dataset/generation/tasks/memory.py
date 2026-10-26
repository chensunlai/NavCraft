"""Maintenance tasks continue ordinary navigation without announcing an edit."""


def memory_task(task_id, context, index):
    return {"form": "decision", "entry": "continue", "question": "", "images": [],
            "limits": {"max_requests": 8, "max_actions": 12, "max_path_length": 3,
                       "max_turns": 6, "seconds": 600},
            "label": {"target_instance": context["target"]["id"],
                      "instances": context["instances"], "transform": context["transform"],
                      "frame_visibility": context["frame_visibility"],
                      "baseline_bindings": context["bindings"],
                      "baseline_ids": [entity["id"] for entity in context["memory"]]}}
