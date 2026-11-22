"""GPU acceptance of planner-delegated legs on real R2R train scenes (no server, no model).

Usage: python run.py CONFIG OUTPUT EPISODE_ID [EPISODE_ID ...]

For each episode: open a session, observe, pick a safe floor pixel 1.5-3 m ahead in the
best view, walk there (leg 1), observe again and walk a second leg, retrace to leg 1,
then finish and score. Records per-call timings, pose continuity and the public/private
leg split. Points are chosen by this script from the visible floor mask, standing in
for a planner; they are not model decisions.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from sim_engine.backends.habitat import HabitatEnvironment
from sim_engine.config import Settings
from sim_engine.data import EpisodeStore
from sim_engine.navigation.point_goal import intrinsics
from sim_engine.navigation.safe_image import safe_floor_mask
from sim_engine.pool import SessionHost
from sim_engine.session import Session

CAMERA = "visible_88cm_hfov120"


def pick(session, observation, legs, near=1.5, far=3.0):
    """A safe floor pixel at a moderate distance in the view with the most of them."""
    best = None
    for name, view in session.snapshot(observation).views.items():
        depth = view.images["depth"]
        h, w = depth.shape
        k = intrinsics(view.calibration["projections"]["depth"], w, h)
        geometry = session.geometry()
        start = np.asarray(session.env.position())
        mask, goals, _, _ = safe_floor_mask(
            depth, k, view.sensors["depth"], start, geometry.pf, geometry.guard,
            min_distance=near, max_distance=far, horizontal_limit=legs.horizontal_limit_m,
            vertical_limit=legs.vertical_limit_m, same_level=legs.same_level)
        ys, xs = np.nonzero(mask)
        if len(xs) and (best is None or len(xs) > best[0]):
            i = len(xs) // 2
            best = (len(xs), name, [round(xs[i] * 1000 / (w - 1)), round(ys[i] * 1000 / (h - 1))])
    return best


def probes(session, legs):
    """At the start pose: a non-floor pixel, an off-floor pixel near floor, other-level floor."""
    observed = session.observe("probe")["observation_id"]
    result = {"other_level_safe_pixels": {}}
    stairs = None
    for name, view in session.snapshot(observed).views.items():
        depth = view.images["depth"]
        h, w = depth.shape
        k = intrinsics(view.calibration["projections"]["depth"], w, h)
        geometry = session.geometry()
        start = np.asarray(session.env.position())
        mask, goals, _, _ = safe_floor_mask(
            depth, k, view.sensors["depth"], start, geometry.pf, geometry.guard,
            horizontal_limit=legs.horizontal_limit_m, vertical_limit=legs.vertical_limit_m,
            same_level=False)
        other = np.abs(goals[..., 1] - start[1]) > legs.vertical_limit_m
        result["other_level_safe_pixels"][name] = int((mask & other).sum())
        ys, xs = np.nonzero(mask & other)
        if len(xs) and (stairs is None or len(xs) > stairs[0]):
            i = int(np.argmax(np.abs(goals[ys, xs, 1] - start[1])))
            stairs = (len(xs), name, [round(xs[i] * 1000 / (w - 1)), round(ys[i] * 1000 / (h - 1))])
    sky = step(session, "probe-sky", {"kind": "point_goal", "observation_id": observed,
                                      "view": "front", "point": [500, 20]})[0]["leg"]
    result["sky"] = {k: sky[k] for k in ("status", "reason")}
    if stairs:
        height = session.env.position()[1]
        leg = step(session, "probe-level", {"kind": "point_goal", "observation_id": observed,
                                            "view": stairs[1], "point": stairs[2]})[0]["leg"]
        result["other_level_leg"] = {"view": stairs[1], "point": stairs[2], "status": leg["status"],
                                     "reason": leg["reason"],
                                     "height_change_m": session.env.position()[1] - height,
                                     "private_error": session.legs[-1]["private"].get("error")}
    return result


def step(session, key, command):
    started = time.perf_counter()
    reply = session.step({"request_id": key, "expected_event_index": len(session.actions),
                          "command": command})
    return reply, time.perf_counter() - started


def run(settings, store, host, episode_id, out):
    host.session = None
    episode = store.get("r2r_train", episode_id)
    started = time.perf_counter()
    session = Session.open(episode_id, host._env(), episode, CAMERA, settings.cameras[CAMERA], 0,
                           settings, geometry=host._planners)
    record = {"episode_id": episode_id, "scene_id": episode["scene_id"],
              "reset_s": time.perf_counter() - started, "calls": []}
    legs = settings.simulation.legs
    record["probes"] = probes(session, legs)
    for n in (1, 2):
        t = time.perf_counter()
        observed = session.observe(f"o{n}")
        record["calls"].append({"call": "observe", "s": time.perf_counter() - t})
        choice = pick(session, observed["observation_id"], legs)
        if choice is None:
            record["calls"].append({"call": "pick", "result": "no safe floor 1.5-3 m"})
            break
        _, view, point = choice
        before = session.env.position()
        reply, seconds = step(session, f"leg{n}", {"kind": "point_goal", "view": view,
                              "point": point, "observation_id": observed["observation_id"]})
        private = session.legs[-1].get("private", {})
        record["calls"].append({
            "call": "point_goal", "view": view, "point": point, "s": seconds,
            "s_per_action": seconds / max(1, len(reply["executed_actions"])),
            "leg": reply["leg"], "reply_keys": sorted(reply),
            "start_matches_previous_end": math.dist(before, session.positions[reply["leg"]["start_event_index"]]) < 1e-6,
            "snap_px": private.get("grounding", {}).get("snap_px"),
            "private_error": private.get("error") or private.get("planning", {}).get("reason"),
            "goal": private.get("grounding", {}).get("goal"), "start": before,
            "final_goal_distance_m": private.get("final_goal_distance_m")})
    if len(session.legs) >= 2:
        target = session.positions[session.legs[0]["end_event_index"]]
        reply, seconds = step(session, "retrace", {"kind": "retrace", "leg_id": 1})
        record["calls"].append({"call": "retrace", "s": seconds, "leg": reply["leg"],
                                "error_to_leg1_end_m": math.dist(session.env.position(), target)})
    finished = session.metrics()
    session.termination = "policy_stop"
    record["metrics"] = session.metrics()["metrics"]
    record["actions"] = len(finished["actions"])
    record["private_fields_only_in_metrics"] = all("private" not in c.get("leg", {}) for c in record["calls"])
    (out / f"{episode_id}.json").write_text(json.dumps(record, indent=2, default=float))
    print(json.dumps({k: record[k] for k in ("episode_id", "actions", "metrics")}, default=float))
    return record


def main():
    config, out, episodes = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    out.mkdir(parents=True, exist_ok=False)
    settings = Settings.read(config)
    store = EpisodeStore(settings)
    host = SessionHost(None, settings, HabitatEnvironment)
    records = [run(settings, store, host, e, out) for e in episodes]
    (out / "summary.json").write_text(json.dumps({"episodes": records}, indent=2, default=float))


if __name__ == "__main__":
    main()
