import math

import numpy as np
import pytest

from sim_engine.collection.start_pool import allocate, sample


def test_area_allocation_is_exact_with_a_per_scene_floor():
    assert allocate([1, 3], 12, 2) == [4, 8]
    values = allocate([1, 1, 1], 10, 2)
    assert values == [4, 3, 3] and sum(values) == 10
    with pytest.raises(ValueError):
        allocate([1, 2], 3, 2)


def test_start_sampling_excludes_tiny_islands_and_bad_clearance_and_preserves_seed():
    class Pathfinder:
        num_islands = 2

        def seed(self, seed):
            self.i = 0

        def island_area(self, i):
            return [100, 0.5][i]

        def get_random_navigable_point(self):
            self.i += 1
            return np.array([self.i * 0.1, 0.0, 0.0], dtype=np.float32)

        def is_navigable(self, p):
            return True

        def get_island(self, p):
            return 1 if self.i == 1 else 0

        def distance_to_closest_obstacle(self, p, radius):
            return 0.01 if self.i == 2 else 0.3

    a = sample(Pathfinder(), 12, 4)
    b = sample(Pathfinder(), 12, 4)
    assert a == b and len({tuple(x["pose_cell_yaw_bin"]) for x in a}) == 12
    assert all(x["draw"] >= 2 and x["island"] == 0 for x in a)
    assert all(math.isclose(sum(v * v for v in x["rotation"]), 1.0) for x in a)
