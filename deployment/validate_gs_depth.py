"""Measure the pinned GS depth convention on two known metric planes."""
import argparse
import json
import struct
from pathlib import Path

import habitat_sim as hs
import magnum as mn
import numpy as np

from sim_engine.backends.habitat import GaussianEnvironment
from sim_engine.scan_assets import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=2)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    GaussianEnvironment.validate_runtime(hs)
    samples = []
    for distance in (2.0, 4.0):
        fields = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2',
                  'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
        cloud = args.output / f'plane-{distance}.gs.ply'
        points = [(x, y) for x in np.linspace(-5, 5, 101) for y in np.linspace(-5, 5, 101)]
        with cloud.open('xb') as f:
            f.write(('ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(points))+'\n'
                     + ''.join('property float '+k+'\n' for k in fields)+'end_header\n').encode())
            for x, y in points:
                f.write(struct.pack('<17f', x, y, -distance, 0, 0, 1, 0, 0, 0, 10,
                                    np.log(.08), np.log(.08), np.log(.003), 1, 0, 0, 0))
        stage = args.output / f'plane-{distance}.stage_config.json'
        stage.write_text(json.dumps({'render_asset': str(cloud), 'up': [0,1,0],
                                     'front': [0,0,-1], 'norm_quaternion': False}))
        cfg = hs.SimulatorConfiguration()
        cfg.scene_id = str(stage)
        cfg.gpu_device_id = args.gpu
        cfg.enable_physics = False
        cfg.navmesh_settings = None
        spec = hs.CameraSensorSpec()
        spec.uuid, spec.sensor_type = 'depth', hs.SensorType.DEPTH
        spec.resolution = [64, 64]
        spec.position, spec.orientation = mn.Vector3(0,0,0), mn.Vector3(0,0,0)
        spec.hfov = 79
        agent = hs.agent.AgentConfiguration()
        agent.sensor_specifications = [spec]
        with hs.Simulator(hs.Configuration(cfg, [agent])) as sim:
            state = hs.AgentState()
            state.position = np.zeros(3)
            sim.initialize_agent(0, state)
            depth = sim.get_sensor_observations()['depth']
            region = depth[12:52,12:52]
            error = float(np.max(np.abs(region-distance)))
            samples.append({'distance_m': distance, 'median_m': float(np.median(region)),
                            'max_absolute_error_m': error, 'passed': bool(error < .08)})
            np.savez_compressed(args.output / f'plane-{distance}.npz', depth_m=depth)
    from habitat_sim._ext import habitat_sim_bindings
    report = {'passed': all(s['passed'] for s in samples), 'samples': samples,
              'depth_convention': 'axial_minus_camera_z', 'depth_units': 'meters',
              'bindings_sha256': sha256(habitat_sim_bindings.__file__),
              'geometry_accuracy_review': 'required'}
    (args.output / 'receipt.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
