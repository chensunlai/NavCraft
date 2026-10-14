"""Public navigation conditions, separate from private episode annotations."""

import json
from pathlib import Path

SYSTEM_PROMPT = """You are a visual navigation agent. Use the available navigation tools to complete the current goal.
RGB observations, measured depth, world-frame pose and camera calibration are available. Depth values are meters; query nav_depth for pixels in observed frames.
Movement consists only of the listed atomic actions. A batch executes sequentially; inspect its actual feedback before deciding what to do next.
Use nav_finish to declare the CURRENT goal complete. A text claim is not a stop action. In a multi-goal task, nav_finish may return the next goal at the same position; continue until the episode ends.
Do not assume success from the task transition. No hidden target location or success oracle is available. Use visual evidence to decide when to finish.
You may inspect or move as needed; there is no prescribed planning workflow. No structural map or external spatial memory is available in this baseline.
"""


def visual_system_prompt() -> str:
    return Path(__file__).with_name("navigation_guide.md").read_text(encoding="utf-8")


def observation_message(
    data: dict, images: list[Path], roles: dict[str, str] | None = None
) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "input_text", "text": json.dumps(data, ensure_ascii=False)},
            *[
                {
                    "type": "input_image",
                    "image_path": str(path),
                    **(
                        {"image_role": roles[str(path)]}
                        if roles and str(path) in roles
                        else {}
                    ),
                }
                for path in images
            ],
        ],
    }
