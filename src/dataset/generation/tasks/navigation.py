"""Local approach requests with independently replayed physical references."""


def approach_task(task_id, context, index):
    attributes = context["snapshot"].attributes
    if not attributes.get("reference_success") or not attributes.get("approach_points"):
        raise ValueError("Local execution needs a successful real reference and approach region")
    if attributes["reference_steps"] > 12 or attributes["reference_path_length"] > 2 + 1e-6:
        raise ValueError("Reference exceeds the frozen local-task specification")
    return {"form": "execution", "entry": "prompt", "question":
            f"Go next to the {context['target']['category']} marked in the reference image, then submit.",
            "images": [context["boxed_image"]],
            "limits": {"max_requests": 20, "max_actions": 24, "max_path_length": 4, "seconds": 600},
            "label": {"target_instance": context["target"]["id"],
                      "approach_points": attributes["approach_points"], "approach_tolerance": .35,
                      "reference_actions": attributes["reference_actions"],
                      "reference_motions": attributes.get("reference_motions", []),
                      "reference_endpoint": attributes["reference_endpoint"],
                      "reference_path_length": attributes["reference_path_length"],
                      "reference_steps": attributes["reference_steps"]}}
