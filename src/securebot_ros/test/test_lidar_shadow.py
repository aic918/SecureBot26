"""Measuring the robot's own LiDAR shadow on a turn on the spot (no ROS)."""
import math

import numpy as np
import pytest
import yaml

from securebot_ros import lidar_shadow as ls

N = 1080
AMIN = -math.pi
AINC = 2 * math.pi / N


def room_range(world_deg):
    """Distance to the walls of a 6 x 4 m room from its centre; a doorway at 40-50 deg (world)
    and a long hallway at 260-280 deg give no return."""
    d = world_deg % 360
    if 40 <= d <= 50 or 260 <= d <= 280:
        return math.inf
    a = math.radians(d)
    c, s = abs(math.cos(a)), abs(math.sin(a))
    return min(3.0 / c if c > 1e-9 else math.inf, 2.0 / s if s > 1e-9 else math.inf)


def scans_of_a_turn(shadow=(170, 190), bracket=(88, 92), step=5):
    out = []
    laser_deg = np.degrees(AMIN + np.arange(N) * AINC)
    for turn in range(0, 360, step):
        r = np.array([room_range(turn + a) for a in laser_deg])
        a360 = laser_deg % 360
        if shadow:
            lo, hi = shadow
            r[(a360 >= lo) & (a360 <= hi)] = math.inf
        if bracket:
            lo, hi = bracket
            r[(laser_deg >= lo) & (laser_deg <= hi)] = 0.30
        out.append((r, AMIN, AINC))
    return out


def test_open_space_is_not_masked_but_the_shadow_and_a_bracket_are():
    seen, near = ls.angle_stats(scans_of_a_turn())
    found = ls.sectors(ls.masked_degrees(seen, near))
    degs = [((a - 180), (b - 180)) for a, b in found]        # index -> laser degrees
    assert len(degs) == 2
    assert any(lo <= 88 and hi >= 91 for lo, hi in degs)       # bracket 88-92 (bin 91)
    assert any(lo >= 168 or hi <= -168 for lo, hi in degs)     # the shadow around 180
    covered = ls.masked_degrees(seen, near)
    assert not any((d + 180) % 360 in covered for d in range(40, 51))   # doorway (world)
    assert not any((d + 180) % 360 in covered for d in range(-100, -79))  # hallway


def test_bounds_cross_180_in_two_parts_with_margin():
    seen, near = ls.angle_stats(scans_of_a_turn(bracket=None))
    bounds = ls.to_radian_bounds(ls.sectors(ls.masked_degrees(seen, near)))
    assert len(bounds) == 2
    degs = sorted((round(math.degrees(lo)), round(math.degrees(hi))) for lo, hi in bounds)
    assert degs[0][0] == -180 and -170 <= degs[0][1] <= -166      # ... -190 + margin
    assert 166 <= degs[1][0] <= 170 and degs[1][1] == 180


def test_no_shadow_means_no_angular_filter():
    seen, near = ls.angle_stats(scans_of_a_turn(shadow=None, bracket=None))
    bounds = ls.to_radian_bounds(ls.sectors(ls.masked_degrees(seen, near)))
    assert bounds == []
    cfg = yaml.safe_load(ls.filters_yaml(bounds, 'test'))['lidar_filters']['ros__parameters']
    assert list(cfg) == ['filter1'] and cfg['filter1']['name'] == 'self_hits'


def test_written_yaml_is_a_valid_filter_chain():
    bounds = [(math.radians(168), math.pi), (-math.pi, math.radians(-167))]
    cfg = yaml.safe_load(ls.filters_yaml(bounds, 'measured test'))['lidar_filters']
    params = cfg['ros__parameters']
    assert [params[k]['type'] for k in ('filter1', 'filter2', 'filter3')] == [
        'laser_filters/LaserScanAngularBoundsFilterInPlace',
        'laser_filters/LaserScanAngularBoundsFilterInPlace',
        'laser_filters/LaserScanRangeFilter']
    assert params['filter1']['params']['upper_angle'] == pytest.approx(math.pi, abs=1e-4)
    assert params['filter3']['params']['lower_replacement_value'] == math.inf


def test_narrow_dropouts_are_ignored():
    deg = {10, 11, 200}                      # 2-wide and 1-wide runs: noise
    assert ls.sectors(deg, min_width=3) == []
    assert ls.sectors({358, 359, 0, 1}, min_width=3) == [(358, 1)]
