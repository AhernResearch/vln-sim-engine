"""Measure the pinned GS depth convention on two known metric planes."""

import struct
from pathlib import Path

from sim_engine.backends.depth import AXIAL
from sim_engine.backends.runtime import HABITAT_GS
from sim_engine.core.io import create_json, file_sha256

PLANE_DISTANCES_M = (2.0, 4.0)
MAX_ERROR_M = 0.08
FIELDS = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
          "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def _plane(output: Path, distance: float) -> Path:
    import numpy as np

    cloud = output / f"plane-{distance}.gs.ply"
    points = [(x, y) for x in np.linspace(-5, 5, 101) for y in np.linspace(-5, 5, 101)]
    with cloud.open("xb") as stream:
        stream.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
                      + "".join(f"property float {k}\n" for k in FIELDS) + "end_header\n").encode())
        for x, y in points:
            stream.write(struct.pack("<17f", x, y, -distance, 0, 0, 1, 0, 0, 0, 10,
                                     np.log(.08), np.log(.08), np.log(.003), 1, 0, 0, 0))
    stage = output / f"plane-{distance}.stage_config.json"
    create_json(stage, {"render_asset": str(cloud), "up": [0, 1, 0], "front": [0, 0, -1],
                        "norm_quaternion": False})
    return stage


def _measure(hs, stage: Path, distance: float, gpu: int, output: Path) -> dict:
    import magnum as mn
    import numpy as np

    config = hs.SimulatorConfiguration()
    config.scene_id, config.gpu_device_id = str(stage), gpu
    HABITAT_GS.configure(config)
    spec = hs.CameraSensorSpec()
    spec.uuid, spec.sensor_type = "depth", hs.SensorType.DEPTH
    spec.resolution = [64, 64]
    spec.position, spec.orientation = mn.Vector3(0, 0, 0), mn.Vector3(0, 0, 0)
    spec.hfov = 79
    agent = hs.agent.AgentConfiguration()
    agent.sensor_specifications = [spec]
    with hs.Simulator(hs.Configuration(config, [agent])) as sim:
        state = hs.AgentState()
        state.position = np.zeros(3)
        sim.initialize_agent(0, state)
        depth = sim.get_sensor_observations()["depth"]
    with (output / f"plane-{distance}.npz").open("xb") as stream:
        np.savez_compressed(stream, depth_m=depth)
    region = depth[12:52, 12:52]
    error = float(np.max(np.abs(region - distance)))
    return {"distance_m": distance, "median_m": float(np.median(region)),
            "max_absolute_error_m": error, "passed": error < MAX_ERROR_M}


def validate_depth(output: Path, gpu: int) -> dict:
    """Write ``receipt.json`` for ``simulation.depth_validation`` into a new directory."""
    import habitat_sim as hs
    from habitat_sim._ext import habitat_sim_bindings

    HABITAT_GS.validate(hs)
    output.mkdir(parents=True, exist_ok=False)
    samples = [_measure(hs, _plane(output, d), d, gpu, output) for d in PLANE_DISTANCES_M]
    report = {"passed": all(s["passed"] for s in samples), "samples": samples,
              "depth_convention": AXIAL, "depth_units": "meters",
              "bindings_sha256": file_sha256(habitat_sim_bindings.__file__),
              "geometry_accuracy_review": "required"}
    create_json(output / "receipt.json", report)
    return report
