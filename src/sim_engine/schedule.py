"""Geometric SE(2) pose schedule: route polyline -> one pose per 4 Hz row.

Replaces closed-loop path tracking for expert rollouts. A pure-pursuit tracker overshoots
every corner and then oscillates back, which LightNav-0's own rows never do; a fitted
geometric schedule reproduces their per-row turn histogram four times more closely
(total-variation 0.09 vs 0.37 for plain interpolation, see docs/validation-expert.md).

The schedule is built as follows:

* corners sharper than ``corner_deg`` are *in-place* corners: the route is cut there and
  the heading change is spent standing still, ``turn_deg`` per row with the remainder last;
* every remaining piece is Gaussian-smoothed in the position domain and sampled every
  ``step_m`` of arc length with heading = path tangent, so a bend is a bell-shaped run of
  small turns rather than one hard corner;
* a moving row may not turn more than ``cap_deg``; the nearest still-smoothed corner is
  demoted to an in-place corner until none does.

``demote`` exposes the same mechanism to the caller, so a bend whose smoothed arc leaves
the navmesh can fall back to the (zero-deviation) interpolated corner while the rest of
the route stays smooth. Nothing here touches Habitat: it is plane geometry over a route
already expressed in the base frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import Pose, wrap


@dataclass(frozen=True)
class ScheduleProfile:
    """Versioned schedule parameters; recorded with every rollout.

    Defaults are the fitted optimum over 100 R2R val-unseen routes against LightNav-0's
    per-row statistics: sigma 0.6 m, cap 18 deg.
    """

    name: str = "geometric_smooth_v1"
    dt_s: float = 0.25
    step_m: float = 0.15
    turn_deg: float = 18.0
    sigma_m: float = 0.6
    cap_deg: float = 18.0
    corner_deg: float = 135.0
    # Position sampling of the smoothing kernel, and the half-window used to read a tangent.
    dense_m: float = 0.01
    tangent_m: float = 0.05
    # A row below this length is standing still rather than moving.
    moving_m: float = 0.02


def clean(points) -> np.ndarray:
    """Drop repeated points so every segment has a direction."""
    points = np.asarray(points, dtype=float)[:, :2]
    keep = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - keep[-1]) > 1e-3:
            keep.append(point)
    if len(keep) < 2:
        raise ValueError("route has zero length")
    return np.array(keep)


def corner_turns(points: np.ndarray) -> np.ndarray:
    """Signed heading change at each interior vertex (radians)."""
    delta = np.diff(points, axis=0)
    heading = np.arctan2(delta[:, 1], delta[:, 0])
    return np.array([wrap(b - a) for a, b in zip(heading, heading[1:])])


def densify(points: np.ndarray, spacing: float) -> np.ndarray:
    out = [points[0]]
    for start, end in zip(points, points[1:]):
        steps = max(1, math.ceil(float(np.linalg.norm(end - start)) / spacing))
        out.extend(start + (end - start) * t for t in np.arange(1, steps + 1) / steps)
    return np.array(out)


def smooth(points: np.ndarray, sigma: float, spacing: float) -> np.ndarray:
    """Gaussian-smooth dense positions; ends extend linearly so both endpoints are kept."""
    dense = densify(points, spacing)
    if sigma <= 0 or len(dense) < 3:
        return dense
    half = int(math.ceil(3 * sigma / spacing))

    def ray(anchor, direction, offsets):
        unit = direction / np.linalg.norm(direction)
        return anchor + unit * spacing * offsets[:, None]

    padded = np.vstack(
        [
            ray(dense[0], dense[0] - dense[1], np.arange(half, 0, -1)),
            dense,
            ray(dense[-1], dense[-1] - dense[-2], np.arange(1, half + 1)),
        ]
    )
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) * spacing / sigma) ** 2)
    kernel /= kernel.sum()
    out = np.column_stack([np.convolve(padded[:, axis], kernel, "valid") for axis in (0, 1)])
    out[0], out[-1] = dense[0], dense[-1]
    return out


def sample(dense: np.ndarray, profile: ScheduleProfile) -> list[tuple[float, float, float]]:
    """Poses every ``step_m`` of arc length, heading = tangent; the last row takes the remainder."""
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    stops = list(np.arange(profile.step_m, arc[-1], profile.step_m))
    if not stops or arc[-1] - stops[-1] > 1e-3:
        stops.append(float(arc[-1]))

    def at(distance):
        distance = min(max(distance, 0.0), float(arc[-1]))
        return np.array([np.interp(distance, arc, dense[:, axis]) for axis in (0, 1)])

    poses = []
    for distance in [0.0, *stops]:
        x, y = at(distance)
        back, ahead = at(distance - profile.tangent_m), at(distance + profile.tangent_m)
        poses.append((float(x), float(y), math.atan2(*(ahead - back)[::-1])))
    return poses


def turn_rows(x: float, y: float, yaw: float, target: float, step_deg: float):
    """Standing rows that turn ``yaw`` onto ``target``, ``step_deg`` per row, remainder last."""
    rows, delta = [], wrap(target - yaw)
    while abs(delta) > 1e-6:
        step = math.copysign(min(abs(delta), math.radians(step_deg)), delta)
        yaw, delta = wrap(yaw + step), delta - step
        rows.append((x, y, yaw))
    return rows


class PoseSchedule:
    """Pose-per-row schedule over a route, with per-corner fallback to interpolation."""

    def __init__(self, points, start_yaw: float, profile: ScheduleProfile | None = None):
        self.profile = profile or ScheduleProfile()
        self.points = clean(points)
        self.start_yaw = float(start_yaw)
        turns = corner_turns(self.points)
        self.in_place = {
            index
            for index, turn in enumerate(turns, start=1)
            if abs(turn) > math.radians(self.profile.corner_deg)
        }
        self._build()
        while self._demote_over_cap():
            self._build()

    @property
    def poses(self) -> list[Pose]:
        return [Pose(float(x), float(y), float(yaw)) for x, y, yaw in self._poses]

    @property
    def smoothed_corners(self) -> int:
        return len(self.points) - 2 - len(self.in_place)

    def rows(self) -> np.ndarray:
        """Per-row ``(step_m, turn_rad)`` between consecutive poses."""
        delta = np.diff(self._poses, axis=0)
        return np.column_stack(
            [np.linalg.norm(delta[:, :2], axis=1), [wrap(value) for value in delta[:, 2]]]
        )

    def commands(self) -> list[dict]:
        """Velocity label per row, reconstructed from the pose difference over ``dt_s``."""
        return [
            {
                "linear_mps": round(step / self.profile.dt_s, 6),
                "angular_rps": round(turn / self.profile.dt_s, 6),
            }
            for step, turn in self.rows()
        ]

    def demote(self, index: int) -> bool:
        """Make the still-smoothed corner nearest to pose ``index`` an in-place corner.

        Returns False when every corner is already in-place, i.e. the schedule is the plain
        interpolation of the route and cannot fall back any further.
        """
        free = [i for i in range(1, len(self.points) - 1) if i not in self.in_place]
        if not free:
            return False
        here = self._poses[min(max(index, 0), len(self._poses) - 1), :2]
        self.in_place.add(min(free, key=lambda i: float(np.linalg.norm(self.points[i] - here))))
        self._build()
        while self._demote_over_cap():
            self._build()
        return True

    def _build(self) -> None:
        cuts = [0, *sorted(self.in_place), len(self.points) - 1]
        poses = [(self.points[0][0], self.points[0][1], self.start_yaw)]
        for start, end in zip(cuts, cuts[1:]):
            sampled = sample(
                smooth(self.points[start : end + 1], self.profile.sigma_m, self.profile.dense_m),
                self.profile,
            )
            poses += turn_rows(*poses[-1][:2], poses[-1][2], sampled[0][2], self.profile.turn_deg)
            poses += sampled[1:]
        self._poses = np.array(poses, dtype=float)

    def _demote_over_cap(self) -> bool:
        rows = self.rows()
        over = np.flatnonzero(
            (rows[:, 0] > self.profile.moving_m)
            & (np.abs(np.degrees(rows[:, 1])) > self.profile.cap_deg + 1e-9)
        )
        if not over.size:
            return False
        free = [i for i in range(1, len(self.points) - 1) if i not in self.in_place]
        if not free:
            return False
        here = self._poses[over[0] + 1, :2]
        self.in_place.add(min(free, key=lambda i: float(np.linalg.norm(self.points[i] - here))))
        return True
