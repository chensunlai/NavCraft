"""Questions grounded in actual instance pixels and frozen memory bindings."""

import math


DIRECTIONS = ("front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left")


def recognition_task(task_id, context, index):
    target, frame = context["target"], context["frame"]
    label = {}
    images = []
    if task_id == "5a":
        absent = context.get("absent_category") if index % 2 else None
        category = absent or target["category"]
        question = (f"Is a {category} visible in the current image? If yes, point to any visible instance. "
                    "Submit [x,y] in original 640x480 pixels, or none if absent.")
        label = ({"answer_kind": "none"} if absent else {
            "answer_kind": "pixel", "frame": frame, "mask_file": context["category_mask"]})
    elif task_id == "5c":
        question = ("Find the same physical object shown in the reference image in the current image. "
                    "Submit [x,y] in original 640x480 pixels, or none if absent.")
        images = [context["target_image"]]
        label = {"answer_kind": "pixel", "frame": frame,
                 "mask_file": context["semantic_file"], "instance_id": target["semantic_id"]}
    elif task_id == "6a":
        question = ("Which spatial-memory entity is the object inside the red box? "
                    "Submit its entity ID, or new if it has no record. Other objects outside the box are irrelevant.")
        images = [context["boxed_image"]]
        label = {"accepted_ids": ["new"] if index % 2 else [context["target_memory_id"]]}
    elif task_id == "6c":
        ids = context["comparison_ids"]
        question = f"Do spatial-memory records {ids[0]} and {ids[1]} refer to the same physical object? Submit yes or no."
        label = {"same_instance": index % 2 == 0}
    elif task_id == "13d":
        question = ("Which direction is map-up relative to your current body heading? "
                    "Submit exactly one of front, front-left, left, back-left, back, back-right, right, front-right.")
        bearing = (-context["heading"]) % 360
        direction = DIRECTIONS[int(math.floor((bearing + 22.5) / 45)) % 8]
        label = {"direction": direction}
    else:
        raise ValueError(task_id)
    return {"form": "qa", "entry": "ask", "question": question, "images": images,
            "limits": {"max_requests": 6, "max_actions": 1, "seconds": 600}, "label": label}
