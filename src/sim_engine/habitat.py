"""Pinned Habitat backend; constructed and used only inside its worker process."""

from __future__ import annotations

import math

from .config import Camera, Physics, Settings
from .data import scene_path
from .geometry import Pose
from .metrics import navigation_metrics


class HabitatEnvironment:
    def __init__(self, settings: Settings):
        import habitat_sim as hs

        if hs.__version__ != "0.2.4":
            raise RuntimeError("habitat-sim 0.2.4 is required")
        self.hs, self.settings, self.sim = hs, settings, None

    def reset(self, episode: dict, camera: Camera, seed: int):
        import numpy as np
        from habitat_sim.utils.common import quat_from_coeffs

        self.close()
        hs = self.hs
        dataset = self.settings.simulation.datasets.get(episode.get("_dataset"))
        self.physics = dataset.physics if dataset else Physics()
        root = (dataset.scenes_root if dataset else None) or self.settings.simulation.scenes_root
        config = hs.SimulatorConfiguration()
        config.scene_id = str(scene_path(root, episode["scene_id"]))
        config.gpu_device_id = self.settings.simulation.gpu
        config.allow_sliding = self.physics.allow_sliding
        sensor = hs.CameraSensorSpec()
        sensor.uuid, sensor.sensor_type = "rgb", hs.SensorType.COLOR
        sensor.resolution = [camera.height, camera.width]
        sensor.position, sensor.orientation = [0, camera.height_m, 0], [-camera.pitch_down_rad, 0, 0]
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
        self.sim = hs.Simulator(hs.Configuration(config, [agent]))
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

    def distance(self, start, end):
        path = self.hs.ShortestPath()
        path.requested_start, path.requested_end = start, end
        if not self.sim.pathfinder.find_path(path):
            raise RuntimeError("unreachable geodesic path")
        return float(path.geodesic_distance)

    def metrics(self, episode, positions, stopped=False):
        return navigation_metrics(
            positions,
            episode.get("reference_path", []),
            episode["goals"][0]["position"],
            self.distance,
            stopped=stopped,
        )

    def close(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None
