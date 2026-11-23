"""Task semantics: what an episode must contain and how an executed path is scored.

A dataset declares its task. Tasks are independent of rendering and runtime bindings.
"""

from __future__ import annotations

from .metrics import navigation_metrics


class NavigationTask:
    """Goal-conditioned VLN with an explicit goal and optional dense reference locations."""

    name = "navigation"

    def validate(self, episode: dict, reward_mode: str) -> None:
        pass

    def metrics(self, env, episode: dict, positions: list, stopped: bool) -> dict:
        return navigation_metrics(positions, episode.get("_metric_reference_path", []),
                                  episode["goals"][0]["position"], env.distance, stopped=stopped)

    def reference(self, episode: dict) -> dict:
        """The scored goal and the dataset route, for evaluation artifacts only."""
        return {"goal": episode["goals"][0]["position"], "path": episode.get("reference_path")}


class DevelopmentTask:
    """A start and an instruction only; nothing is scored and no goal may be fabricated."""

    name = "development"

    def validate(self, episode: dict, reward_mode: str) -> None:
        if episode.get("goals"):
            raise ValueError("development episodes must not contain scoring goals")
        if reward_mode != "none":
            raise ValueError("development tasks cannot enable goal rewards")

    def metrics(self, env, episode: dict, positions: list, stopped: bool) -> dict:
        return {"evaluation_available": False, "success": None, "spl": None, "ndtw": None,
                "distance_to_goal": None}

    def reference(self, episode: dict) -> dict:
        return {"goal": None, "path": None}


TASKS = {task.name: task for task in (NavigationTask(), DevelopmentTask())}


def task_of(settings, episode: dict):
    """The task of the episode's registered dataset; unregistered episodes are navigation."""
    dataset = settings.simulation.datasets.get(episode.get("_dataset"))
    return TASKS[dataset.task if dataset else NavigationTask.name]
