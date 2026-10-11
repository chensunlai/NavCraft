"""Only the evaluator/runner reads scores; actor tool feedback stays public."""

from src.environment.base import Environment


def evaluate(environment: Environment, reason: str) -> dict:
    environment.finalize(reason)
    return environment.metrics()
