"""Deterministic, bounded rejection sampling before observation or movement."""

from .determinism import perturbation
from .storage import RejectedCandidate, write


class ReferenceCache:
    """Cache both feasible and infeasible remaining paths within one scene."""

    def __init__(self):
        self.values = {}

    def get(self, key, build):
        reused = key in self.values
        if not reused:
            self.values[key] = build()
        return self.values[key], reused


def select_pose(candidate, config, evaluate, folder):
    """Only known geometric rejections advance the draw; unexpected errors propagate."""
    limit = config["pose_sampling_attempts"]
    for trial in range(limit):
        draw = (
            {"offset": candidate["offset"], "yaw_delta": candidate["yaw_delta"]}
            if trial == 0
            else perturbation(
                config["seed"],
                f"{candidate['id']}:pose-trial:{trial}",
                config["start_radius_m"],
                config["yaw_degrees"],
            )
        )
        record = {"trial": trial, **draw, "phase": "before_movement"}
        try:
            value = evaluate(draw)
        except RejectedCandidate as error:
            write(
                folder / "pose-trials" / f"{trial:03d}.json",
                dict(record, status="rejected", reason=str(error)),
            )
            continue
        except Exception as error:
            write(
                folder / "pose-trials" / f"{trial:03d}.json",
                dict(record, status="error", reason=repr(error)),
            )
            raise
        write(folder / "pose-trials" / f"{trial:03d}.json", dict(record, status="selected"))
        write(
            folder / "pose-selection.json",
            {
                "status": "selected",
                "attempts": trial + 1,
                "selected_trial": trial,
                "limit": limit,
                "movement_retried": False,
                **draw,
            },
        )
        return value
    write(
        folder / "pose-selection.json",
        {"status": "exhausted", "attempts": limit, "limit": limit, "movement_retried": False},
    )
    raise RejectedCandidate(f"no_safe_pose_within_budget: {limit} checked draws")
