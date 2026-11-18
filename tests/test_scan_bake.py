import numpy as np
import pytest

from sim_engine.scan_bake import select_start


class ScanPathfinder:
    def snap_point(self, point):
        # Reproduce the native binding's requirement, including JSON integer zero.
        assert isinstance(point, np.ndarray) and point.dtype == np.float32
        return [point[0], -1.0, point[2]]

    def distance_to_closest_obstacle(self, point, radius):
        assert isinstance(point, np.ndarray) and point.dtype == np.float32
        return min(float(point[0]) + 0.01, radius)


def test_start_uses_supported_scan_pose_with_clearance_instead_of_first_boundary():
    poses = [[0.0, 0, 0.0]] * 10 + [[0.5, 0, 0.0]] * 10 + [[1.0, 5, 0.0]]
    start, receipt = select_start(ScanPathfinder(), poses)
    assert start == [0.5, -1.0, 0.0]
    assert receipt['supported_start_candidates'] == 2
    assert receipt['start_clearance_m'] == pytest.approx(0.51)


def test_start_does_not_invent_ground_support_for_unsupported_scan():
    with pytest.raises(RuntimeError, match='supported development start'):
        select_start(ScanPathfinder(), [[0.0, 5, 0.0]])
