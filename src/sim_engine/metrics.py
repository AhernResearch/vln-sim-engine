"""VLN navigation metrics, using geodesic distances as in Habitat/VLN-CE.

Success requires an explicit policy STOP. DTW follows the reference-path
recurrence from VLN-CE; thresholds are measured in metres.
"""

import math
from typing import Callable


def navigation_metrics(
    positions: list,
    reference: list,
    goal: list,
    distance: Callable,
    stopped: bool = False,
    threshold: float = 3.0,
) -> dict:
    if not positions:
        raise ValueError("empty executed path")
    errors = [distance(p, goal) for p in positions]
    length = sum(math.dist(a, b) for a, b in zip(positions, positions[1:]))
    shortest = distance(positions[0], goal)
    success = float(stopped and errors[-1] < threshold)
    result = {
        "distance_to_goal": errors[-1],
        "oracle_success": float(min(errors) < threshold),
        "success": success,
        "spl": success * shortest / max(shortest, length, 1e-12),
        "path_length": length,
    }
    if reference:
        previous = [0.0] + [math.inf] * len(reference)
        locations = [p for i, p in enumerate(positions) if i == 0 or p != positions[i - 1]]
        for actual in locations:
            current = [math.inf]
            for i, target in enumerate(reference, 1):
                current.append(
                    math.dist(actual, target) + min(current[-1], previous[i], previous[i - 1])
                )
            previous = current
        result["ndtw"] = math.exp(-previous[-1] / (len(reference) * threshold))
        result["sdtw"] = success * result["ndtw"]
    return result
