"""Pinned Habitat adapter: one Simulator per scene batch, fresh state per candidate."""

import copy
import json
import os
import time
from pathlib import Path

from PIL import Image

from sim_engine.backends.habitat import HabitatEnvironment
from sim_engine.config import Camera, Settings
from sim_engine.core.io import create_json, file_sha256
from sim_engine.navigation.clearance import ClearanceProfile
from sim_engine.navigation.tracking import make_tracker
from sim_engine.planning.geometry import NavMeshContext
from sim_engine.scenes import resolve_scene

from .recording import calibration


def make_backend(worker, manifest, scene):
    return SceneBackend(worker, manifest, scene)


class SceneBackend(NavMeshContext):
    def __init__(self, worker, manifest, scene):

        runtime = copy.deepcopy(manifest["runtime"])
        runtime["settings"]["simulation"]["gpu"] = worker["gpu"]
        self.settings = Settings.model_validate(runtime["settings"])
        camera = Camera.model_validate(runtime["camera"])
        if "train_pilot" not in self.settings.simulation.datasets:
            raise ValueError("frozen simulation requires train_pilot physics")

        self.scene_id = runtime.get("scene_id") or str(
            Path(scene).relative_to(self.settings.simulation.scenes_root))
        self.scene_asset = resolve_scene(self.settings, "train_pilot", self.scene_id)
        super().__init__(
            self.scene_asset.navmesh,
            self.scene_id,
            camera, manifest["config"], ClearanceProfile(**runtime["clearance_profile"]),
        )
        self.scene = scene
        self.tasks = {t["id"]: t for t in manifest["tasks"]}
        self.provenance = {
            "habitat_version": self.hs.__version__, "numpy_version": self.np.__version__,
            "gpu": worker["gpu"], "backend": self.settings.simulation.backend,
            "scene_assets": self.scene_asset.assets, "training_admission": "pending_review",
        }
        self.env = None
        self.loads = self.resets = 0

    def reset_candidate(self, ep):

        if self.env is None:
            self.env = HabitatEnvironment(self.settings)
        before = self.env.scene_loads
        self.env.reset(ep, self.camera, self.config["seed"])
        self.env.enable_depth()
        self.loads += self.env.scene_loads - before
        self.resets += 1
        captured = self.env.capture()
        self.depth_near_far = captured.calibration["depth_near_far"]

    def observe(self, candidate, prepared, folder, *, capture_depth=False):
        started = time.perf_counter()
        self.reset_candidate(prepared["episode"])
        reset_s = time.perf_counter() - started
        images, views = {}, {}
        for label, captured in self.env.observe_views(prepared["yaw"]).items():
            target = folder / (label + ".png")
            Image.fromarray(captured.images["rgb"]).save(target)
            images[label] = {"path": str(target.resolve()), "sha256": file_sha256(target)}
            if capture_depth:
                depth_file = folder / (label + "-depth.npz")
                with depth_file.open("xb") as stream:
                    self.np.savez_compressed(stream, depth_m=self.np.asarray(
                        captured.images["depth"], dtype=self.np.float32))
                views[label] = {**captured.sensors["depth"], "depth_sha256": file_sha256(depth_file)}
        captured = self.env.capture()
        projections = {key: captured.calibration["projections"][key] for key in ("rgb", "depth")}

        create_json(
            folder / "camera.json",
            {**calibration(self.camera.model_dump(), projections, *self.depth_near_far),
             **{k: v for k, v in captured.calibration.items() if k != "projections"}},
        )
        if capture_depth:
            create_json(folder / "view-sensors.json", views)
        create_json(
            folder / "scene-cache.json",
            {
                "scene": self.scene,
                "scene_loads": self.loads,
                "candidate_resets": self.resets,
                "reset_s": reset_s,
                "same_scene_reused": self.resets > 1,
            },
        )
        return {"instruction": self.tasks[candidate["task_id"]]["instruction"], "images": images}

    def rollout(self, candidate, prepared, folder, stop):



        # A new tracker uses the same frozen profile as the preflight.
        tracker = make_tracker(prepared["path"], self.expert_profile)
        (folder / "rgb").mkdir()
        (folder / "depth").mkdir()
        collisions, final, incoming = 0, "budget_exhausted", None
        incoming_collision = False
        with (
            (folder / "commands.jsonl").open("x") as journal,
            (folder / "world_states.jsonl").open("x") as states,
        ):
            for step in range(self.config["max_steps"] + 1):
                captured = self.env.capture()
                observations = captured.images
                image = folder / "rgb" / f"{step:05d}.jpg"
                Image.fromarray(observations["rgb"][:, :, :3]).save(image)
                depth = self.np.asarray(observations["depth"], dtype=self.np.float32)
                depth_valid = bool(
                    self.np.isfinite(depth).all() and not (depth < 0).any() and (depth > 0).any()
                )
                depth_path = folder / "depth" / f"{step:05d}.npz"
                with depth_path.open("xb") as stream:
                    self.np.savez_compressed(stream, depth_m=depth)
                pose = self.env.pose()
                sensors = captured.sensors
                record = {
                    "frame": step,
                    "control_time_s": step * 0.25,
                    "world_position": self.env.position(),
                    "world_rotation_xyzw": captured.rotation_xyzw,
                    "base_pose_se2": [pose.x, pose.y, pose.yaw],
                    "incoming_command": incoming,
                    "sensors": sensors,
                    "collided": bool(incoming_collision),
                    "depth_valid": depth_valid,
                    "rgb_sha256": file_sha256(image),
                    "depth_sha256": file_sha256(depth_path),
                }
                states.write(json.dumps(record) + "\n")
                states.flush()
                if not depth_valid:
                    create_json(
                        folder / f"depth-invalid-{step:05d}.json",
                        {
                            "frame": step,
                            "depth_sha256": record["depth_sha256"],
                            "status": "retained_invalid_raw_depth",
                        },
                    )
                    raise ValueError("invalid depth retained for review")
                if stop.is_set():
                    final = "interrupted_uncertain"
                    break
                command, _ = tracker.command(pose)
                if command is None:
                    final = "expert_stop"
                    break
                if step == self.config["max_steps"]:
                    break
                before = self.env.position()
                journal.write(
                    json.dumps(
                        {"step": step, "phase": "started", "command": command, "position": before}
                    )
                    + "\n"
                )
                journal.flush()
                os.fsync(journal.fileno())
                _, collision = self.env.velocity_step(
                    command["linear_mps"], command["angular_rps"], 0.25, True
                )
                incoming, incoming_collision = command, bool(collision)
                collisions += int(collision)
                journal.write(
                    json.dumps(
                        {
                            "step": step,
                            "phase": "returned",
                            "position": self.env.position(),
                            "collided": bool(collision),
                        }
                    )
                    + "\n"
                )
                journal.flush()
                os.fsync(journal.fileno())
                if collision or not self.guard.edge_clear(before, self.env.position()):
                    final = "collision_or_clearance"
                    break
        distance = self.env.distance(self.env.position(), prepared["requested_goal_position"])
        passed = (
            final == "expert_stop" and collisions == 0 and distance <= self.config["goal_tolerance_m"]
        )
        create_json(
            folder / "terminal.json",
            {
                "receipt_stop": passed,
                "termination": final,
                "frame": step,
                "training_admission": "pending_review",
                "visible_goal_label": None,
                "goal_adjustment_m": prepared["goal_adjustment_m"],
                "endpoint_review_required": prepared["endpoint_review_required"],
                "requested_goal_distance_m": distance,
                "goal_tolerance_m": self.config["goal_tolerance_m"],
            },
        )
        return {
            "passed": passed,
            "termination": final,
            "collision_steps": collisions,
            "final_distance_m": distance,
            "steps": step,
            "source": "expert_not_vla",
        }

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
