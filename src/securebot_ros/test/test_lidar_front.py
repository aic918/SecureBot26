"""Finding which way the LiDAR faces from a short straight drive (no ROS)."""
import math

import numpy as np
import pytest

from securebot_ros import lidar_front as lf


def room_points(spacing=0.01):
    """Walls of a 6 x 4 m room around (0, 0) plus a 0.4 m pillar: the world's points."""
    pts = []
    for x in np.arange(-3.0, 3.0, spacing):
        pts += [(x, -2.0), (x, 2.0)]
    for y in np.arange(-2.0, 2.0, spacing):
        pts += [(-3.0, y), (3.0, y)]
    for t in np.arange(0.0, 0.4, spacing):              # pillar, front-left of the start
        pts += [(1.2 + t, 0.8), (1.2 + t, 1.2), (1.2, 0.8 + t), (1.6, 0.8 + t)]
    return np.array(pts)


def seen_from(base_x, base_y, base_yaw, laser_yaw, noise=0.0, seed=0):
    """The room in the laser frame of a robot at (base_x, base_y, base_yaw) whose LiDAR is
    turned laser_yaw from base_link (no occlusion: the test room is simple enough)."""
    rng = np.random.default_rng(seed)
    a = base_yaw + laser_yaw
    rel = room_points() - (base_x, base_y)
    c, s = math.cos(-a), math.sin(-a)
    pts = rel @ np.array([[c, s], [-s, c]])
    if noise:
        pts = pts + rng.normal(0.0, noise, pts.shape)
    return pts


@pytest.mark.parametrize('laser_yaw_deg', [0.0, 180.0, 90.0, -35.0])
def test_finds_how_the_lidar_is_turned(laser_yaw_deg):
    yaw = math.radians(laser_yaw_deg)
    before = seen_from(0.0, 0.0, 0.0, yaw, noise=0.01, seed=1)
    after = seen_from(0.3, 0.0, 0.0, yaw, noise=0.01, seed=2)       # 0.3 m forward
    tx, ty, err = lf.estimate_motion(before, after)
    assert math.hypot(tx, ty) == pytest.approx(0.3, abs=0.02)
    assert err < 0.02
    assert math.degrees(lf.wrap(lf.laser_yaw_from_motion(tx, ty) - yaw)) == pytest.approx(0, abs=4)


def test_a_small_turn_during_the_drive_is_taken_from_the_gyro():
    yaw = math.pi
    turn = math.radians(4)                                           # drifted 4 deg left
    before = seen_from(0.0, 0.0, 0.0, yaw)
    after = seen_from(0.3 * math.cos(turn / 2), 0.3 * math.sin(turn / 2), turn, yaw)
    tx, ty, _ = lf.estimate_motion(before, after, rotation=turn)
    assert math.degrees(lf.wrap(lf.laser_yaw_from_motion(tx, ty) - yaw)) == pytest.approx(0, abs=4)


def test_check_combines_out_and_back_and_flags_disagreement():
    yaw, problems = lf.check((-0.30, 0.0, 0.01), (0.30, 0.01, 0.01), driven=0.30)
    assert math.degrees(abs(lf.wrap(yaw - math.pi))) < 3 and problems == []
    _, problems = lf.check((-0.30, 0.0, 0.01), (0.0, 0.30, 0.01), driven=0.30)
    assert any('disagree' in p for p in problems)
    _, problems = lf.check((-0.05, 0.0, 0.01), (0.05, 0.0, 0.01), driven=0.30)
    assert any('wheels' in p for p in problems)
    _, problems = lf.check((-0.30, 0.0, 0.12), (0.30, 0.0, 0.01), driven=0.30)
    assert any('do not match' in p for p in problems)


def test_scan_points_drop_self_hits_and_far_noise():
    pts = lf.scan_points([0.1, 1.0, 20.0, float('inf'), 2.0], 0.0, math.pi / 2)
    assert pts.shape == (2, 2)
    assert pts[0] == pytest.approx((0.0, 1.0), abs=1e-9)       # 1.0 m at +90 deg
    assert pts[1] == pytest.approx((2.0, 0.0), abs=1e-9)       # 2.0 m at 360 deg
    assert lf.nearest_clearance(pts) == pytest.approx(1.0)
