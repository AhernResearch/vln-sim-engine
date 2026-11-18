"""Task semantics are independent of rendering and runtime bindings."""
from .metrics import navigation_metrics


def metrics(env, episode, positions, stopped=False):
    if episode.get("_task") == "development":
        return {"evaluation_available": False, "success": None, "spl": None, "ndtw": None,
                "distance_to_goal": None}
    return navigation_metrics(positions, episode.get("_metric_reference_path", []),
                              episode["goals"][0]["position"], env.distance, stopped=stopped)
