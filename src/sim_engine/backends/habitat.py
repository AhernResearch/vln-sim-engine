"""Pinned Habitat backend; constructed and used only inside its worker process."""

from __future__ import annotations

import json
import math
from pathlib import Path

from sim_engine.config import Camera, Physics, Settings
from sim_engine.navigation.pose import Pose
from sim_engine.scenes import resolve_scene

from .contracts import Capture

HABITAT_VERSION = "0.3.1"
GS_VERSION = "0.3.3"


class HabitatEnvironment:
    depth_source = "mesh"

    def __init__(self, settings: Settings):
        import habitat_sim as hs

        self.validate_runtime(hs)
        self.hs, self.settings, self.sim = hs, settings, None
        self.cache_key = None
        self.scene_loads = self.resets = 0
        self._scenes = {}

    def reset(self, episode: dict, camera: Camera, seed: int):
        import magnum as mn
        import numpy as np
        from habitat_sim.utils.common import quat_from_coeffs

        hs = self.hs
        dataset = self.settings.simulation.datasets.get(episode.get("_dataset"))
        self.physics = dataset.physics if dataset else Physics()
        scene_key = (episode.get("_dataset"), episode["scene_id"])
        if scene_key not in self._scenes:
            self._scenes[scene_key] = resolve_scene(self.settings, *scene_key)
        self.scene = self._scenes[scene_key]
        self.camera = camera
        if self.scene.navigation_settings:
            frozen = self.scene.navigation_settings
            for key, value in (("agentRadius", self.physics.agent_radius_m),
                               ("agentHeight", self.physics.agent_height_m)):
                if not math.isclose(frozen[key], value, abs_tol=1e-6):
                    raise ValueError("agent geometry differs from baked NavMesh: " + key)
        key = (
            str(self.scene.render),
            camera.model_dump_json(),
            self.physics.model_dump_json(),
            self.settings.simulation.gpu,
            str(self.scene.navmesh),
        )
        if self.sim is not None and self.cache_key == key:
            self.sim.reset()
            self.sim.seed(seed)
            state = hs.AgentState()
            state.position = np.array(episode["start_position"])
            state.rotation = quat_from_coeffs(episode["start_rotation"])
            self.sim.initialize_agent(0, state)
            self.resets += 1
            return self.sim.get_sensor_observations()["rgb"][:, :, :3]
        self.close()
        config = hs.SimulatorConfiguration()
        config.scene_id = str(self.scene.render)
        config.gpu_device_id = self.settings.simulation.gpu
        config.allow_sliding = self.physics.allow_sliding
        sensor = hs.CameraSensorSpec()
        sensor.uuid, sensor.sensor_type = "rgb", hs.SensorType.COLOR
        sensor.resolution = [camera.height, camera.width]
        # habitat-sim 0.3 bindings require magnum vectors; lists are no longer converted.
        sensor.position = mn.Vector3(0, camera.height_m, 0)
        sensor.orientation = mn.Vector3(-camera.pitch_down_rad, 0, 0)
        sensor.hfov = camera.hfov
        agent = hs.agent.AgentConfiguration()
        agent.height, agent.radius = self.physics.agent_height_m, self.physics.agent_radius_m
        agent.sensor_specifications = [sensor]
        agent.action_space = {
            name: hs.agent.ActionSpec(action, hs.agent.ActuationSpec(amount=amount))
            for name, action, amount in [
                ("MOVE_FORWARD", "move_forward", self.physics.forward_m),
                ("TURN_LEFT", "turn_left", self.physics.turn_degrees),
                ("TURN_RIGHT", "turn_right", self.physics.turn_degrees),
            ]
        }
        self.configure(config)
        self.sim = hs.Simulator(hs.Configuration(config, [agent]))
        if not self.sim.pathfinder.load_nav_mesh(str(self.scene.navmesh)):
            raise RuntimeError("cannot load pinned navigation mesh")
        self._depth_enabled = False
        self._panorama_enabled = False
        if self.settings.simulation.sensors != "rgb":
            self.enable_depth()
        if self.settings.simulation.sensors == "panorama":
            self.enable_panorama()
        self.cache_key = key
        self.scene_loads += 1
        self.resets += 1
        if not self.sim.pathfinder.is_loaded:
            raise RuntimeError("navigation mesh is not loaded")
        self.sim.seed(seed)
        state = hs.AgentState()
        state.position = np.array(episode["start_position"])
        state.rotation = quat_from_coeffs(episode["start_rotation"])
        self.sim.initialize_agent(0, state)
        return self.sim.get_sensor_observations()["rgb"][:, :, :3]

    def pose(self):
        import numpy as np
        from habitat_sim.utils.common import quat_rotate_vector

        state = self.sim.get_agent(0).get_state()
        forward = quat_rotate_vector(state.rotation, np.array([0, 0, -1]))
        return Pose(
            -float(state.position[2]), -float(state.position[0]), math.atan2(-forward[0], -forward[2])
        )

    def position(self):
        return self.sim.get_agent(0).get_state().position.tolist()

    def step(self, action):
        result = self.sim.step(action)
        return result["rgb"][:, :, :3], bool(result["collided"])

    def velocity_step(self, linear_mps, angular_rps, dt, allow_sliding):
        """Integrate a body velocity and snap it to the navmesh (habitat-lab VelocityAction)."""
        import magnum as mn
        import numpy as np

        hs = self.hs
        control = hs.physics.VelocityControl()
        control.controlling_lin_vel = control.controlling_ang_vel = True
        control.lin_vel_is_local = control.ang_vel_is_local = True
        control.linear_velocity = np.array([0.0, 0.0, -linear_mps])
        control.angular_velocity = np.array([0.0, angular_rps, 0.0])
        agent = self.sim.get_agent(0)
        state = agent.get_state()
        current = hs.RigidState(mn.Quaternion(state.rotation.imag, state.rotation.real), state.position)
        goal = control.integrate_transform(dt, current)
        pathfinder = self.sim.pathfinder
        step = pathfinder.try_step if allow_sliding else pathfinder.try_step_no_sliding
        final = step(state.position, goal.translation)
        before = (goal.translation - state.position).dot()
        after = (final - state.position).dot()
        state.position = final
        state.rotation = [*goal.rotation.vector, goal.rotation.scalar]
        state.sensor_states = {}
        agent.set_state(state, False)
        self.sim.step_physics(1.0 / 60.0)
        image = self.sim.get_sensor_observations()["rgb"][:, :, :3]
        return image, bool(after + 1e-5 < before)

    def _shortest(self, start, end):
        path = self.hs.ShortestPath()
        path.requested_start, path.requested_end = start, end
        if not self.sim.pathfinder.find_path(path):
            raise RuntimeError("unreachable geodesic path")
        return path

    def distance(self, start, end):
        return float(self._shortest(start, end).geodesic_distance)

    def snap(self, point):
        return [float(v) for v in self.sim.pathfinder.snap_point(point)]

    def shortest_path(self, start, end):
        """Navmesh geodesic polyline between two snapped points, both ends included."""
        return [[float(v) for v in p] for p in self._shortest(start, end).points]

    def close(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None
        self.cache_key = None

    @staticmethod
    def validate_runtime(hs):
        if hs.__version__ != HABITAT_VERSION:
            raise RuntimeError(f"habitat-sim {HABITAT_VERSION} is required")

    def configure(self, config):
        pass

    def identity(self):
        if not hasattr(self, "_identity"):
            from habitat_sim._ext import habitat_sim_bindings

            from sim_engine.core.io import file_sha256

            bindings = file_sha256(Path(habitat_sim_bindings.__file__))
            self._identity = {"backend": self.settings.simulation.backend,
                              "habitat_version": self.hs.__version__,
                              "bindings_sha256": bindings, "depth_source": self.depth_source,
                              **self.depth_contract(bindings)}
        return self._identity

    def depth_contract(self, bindings_sha256):
        return {"depth_convention": "axial_minus_camera_z"}

    def _add_sensor(self, name, depth=False, yaw=0):
        import magnum as mn

        spec = self.hs.CameraSensorSpec()
        spec.uuid = name
        spec.sensor_type = self.hs.SensorType.DEPTH if depth else self.hs.SensorType.COLOR
        spec.resolution = [self.camera.height, self.camera.width]
        spec.position = mn.Vector3(0, self.camera.height_m, 0)
        spec.orientation = mn.Vector3(-self.camera.pitch_down_rad, yaw, 0)
        spec.hfov = self.camera.hfov
        self.sim.add_sensor(spec, agent_id=0)
        return spec

    def enable_depth(self):
        if not self._depth_enabled:
            self._depth_spec = self._add_sensor("depth", depth=True)
            self._depth_enabled = True

    def enable_panorama(self):
        self.enable_depth()
        if not self._panorama_enabled:
            for label, yaw in (("front", 0), ("right", -math.pi / 2),
                               ("back", math.pi), ("left", math.pi / 2)):
                for kind in ("rgb", "depth"):
                    self._add_sensor(f"{label}_{kind}", kind == "depth", yaw)
            self._panorama_enabled = True

    def capture(self):
        import numpy as np
        from habitat_sim.utils.common import quat_to_coeffs

        state = self.sim.get_agent(0).get_state()
        observations = self.sim.get_sensor_observations()
        # Own buffers; native renderers may reuse their output memory.
        images = {key: np.array(value[:, :, :3] if value.ndim == 3 else value, copy=True)
                  for key, value in observations.items() if key != "collided"}
        sensors, projections = {}, {}
        for key in images:
            sensor = state.sensor_states[key]
            sensors[key] = {"position": np.asarray(sensor.position).tolist(),
                            "rotation_xyzw": quat_to_coeffs(sensor.rotation).tolist()}
            projections[key] = np.asarray(
                self.sim._sensors[key]._sensor_object.render_camera.projection_matrix).tolist()
        calibration = {"projections": projections, **self.identity(),
                       "depth_units": "meters", "geometry_review": self.scene.geometry_review,
                       "depth_convention": self.identity()["depth_convention"]}
        if self._depth_enabled:
            calibration["depth_near_far"] = [float(self._depth_spec.near), float(self._depth_spec.far)]
        return Capture(images, self.position(), quat_to_coeffs(state.rotation).tolist(), sensors,
                       calibration)

    def observe_views(self, yaw):
        import magnum as mn

        self.enable_panorama()
        # Rotate the camera rig relative to the chassis, never the robot state.
        current_yaw = self.pose().yaw
        # pose yaw is CCW around Habitat +Y.
        delta = yaw - current_yaw
        for label, offset in (("front", 0), ("right", -math.pi / 2),
                              ("back", math.pi), ("left", math.pi / 2)):
            for kind in ("rgb", "depth"):
                sensor = self.sim._sensors[f"{label}_{kind}"]._sensor_object
                sensor.node.rotation = (
                    mn.Quaternion.rotation(mn.Rad(delta + offset), mn.Vector3.y_axis())
                    * mn.Quaternion.rotation(mn.Rad(-self.camera.pitch_down_rad), mn.Vector3.x_axis()))
        captured = self.capture()
        return {label: Capture(
            {kind: captured.images[f"{label}_{kind}"] for kind in ("rgb", "depth")},
            captured.position, captured.rotation_xyzw,
            {kind: captured.sensors[f"{label}_{kind}"] for kind in ("rgb", "depth")},
            {**captured.calibration, "projections": {
                kind: captured.calibration["projections"][f"{label}_{kind}"]
                for kind in ("rgb", "depth")}})
            for label in ("front", "right", "back", "left")}


class GaussianEnvironment(HabitatEnvironment):
    depth_source = "gaussian_unvalidated"

    @staticmethod
    def validate_runtime(hs):
        if hs.__version__ != GS_VERSION or not hs.cuda_enabled:
            raise RuntimeError(f"Habitat-GS {GS_VERSION} with CUDA is required")
        if not hasattr(hs.SimulatorConfiguration(), "gaussian_time"):
            raise RuntimeError("pinned Habitat-GS bindings are required")

    def configure(self, config):
        config.enable_physics = False
        config.navmesh_settings = None

    def depth_contract(self, bindings_sha256):
        """Metric depth is admitted only with a passed report for this exact binding."""
        from sim_engine.core.io import file_sha256

        path = self.settings.simulation.depth_validation
        if path is None:
            return {"depth_convention": "unvalidated"}
        report = json.loads(path.read_text())
        if (report.get("bindings_sha256") != bindings_sha256
                or report.get("passed") is not True
                or report.get("depth_convention") != "axial_minus_camera_z"
                or report.get("depth_units") != "meters"):
            raise ValueError("depth validation does not match this runtime")
        return {"depth_convention": report["depth_convention"],
                "depth_validation_sha256": file_sha256(path)}
