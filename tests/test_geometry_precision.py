import math

import pytest

from sim_engine.navigation.pose import Pose, next_action, reached, to_world


@pytest.mark.parametrize('yaw', [-2.8, 0., 1.2574257679851353])
@pytest.mark.parametrize('direction', [-1, 1])
@pytest.mark.parametrize('noise', [0., 3e-8])
def test_resampled_half_turn_is_skipped_before_full_native_turn(yaw, direction, noise):
    origin = Pose(4., -7., yaw)
    midpoint = to_world(origin, [0., 0., direction*(math.pi/12+noise)])
    endpoint = to_world(origin, [0., 0., direction*math.pi/6])
    assert reached(origin, midpoint, .25, 30)
    assert not reached(origin, endpoint, .25, 30)
    assert next_action(origin, endpoint, .25, 30) == ('TURN_LEFT' if direction > 0 else 'TURN_RIGHT')


def test_tolerance_does_not_hide_a_material_turn_or_translation_error():
    origin = Pose(0., 0., 0.)
    assert not reached(origin, Pose(0., 0., math.pi/12+.001), .25, 30)
    assert not reached(origin, Pose(.126, 0., 0.), .25, 30)
