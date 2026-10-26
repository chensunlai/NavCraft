"""Explicit task registry; each function makes one public input and private label."""

from .memory import memory_task
from .recognition import recognition_task
from .navigation import approach_task


TASKS = {
    "5a": recognition_task, "5c": recognition_task,
    "6a": recognition_task, "6c": recognition_task, "13d": recognition_task,
    "1b": memory_task, "1n": memory_task, "14b": approach_task,
}

TASK_IDS = tuple(TASKS)
