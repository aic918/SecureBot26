"""Coverage laps on synthetic maps and the saved Kumar's Lab map (no ROS)."""
import math
import os

import cv2
import numpy as np
import pytest

from securebot_ros import coverage_logic as cl

RES = 0.05
LAB = os.path.expanduser('~/securebot_ws/maps/explore_1026_1922.yaml')


def room(w=4.0, h=3.0, box=None, border=1.0):
    """w x h m free room with walls, unknown all round, origin at the room's corner.
    box=(x0, y0, x1, y1) puts a solid obstacle inside."""
    pad = int(border / RES)
    cols, rows = int(w / RES) + 2 * pad, int(h / RES) + 2 * pad
    free = np.zeros((rows, cols), bool)
    free[pad:rows - pad, pad:cols - pad] = True
    m = cl.Map(free, RES, (-border, -border))
    if box:
        r0, c0 = m.to_cell(box[0], box[3])
        r1, c1 = m.to_cell(box[2], box[1])
        free[r0:r1, c0:c1] = False
    return m


def wall_distance(m, x, y):
    floor = np.pad(m.free.astype(np.uint8), 1)
    dist = cv2.distanceTransform(floor, cv2.DIST_L2, 5)[1:-1, 1:-1] * RES
    return dist[m.to_cell(x, y)]


def test_footprint():
    assert cl.CORNER_RADIUS == pytest.approx(0.356, abs=1e-3)
    assert cl.Params().wall_clearance > cl.CORNER_RADIUS          # can turn anywhere on a lap
    assert cl.Params().lane_spacing < cl.WIDTH                    # laps overlap


def test_empty_room_laps_go_outside_in_and_stay_off_walls():
    m = room()
    p = cl.plan(m, start=(0.5, 0.5))
    assert p.levels == sorted(p.levels) and p.levels[0] == 0
    assert len(set(p.levels)) >= 2
    first = p.loops[0]
    assert first[0] == first[-1]                                  # closed loop
    assert math.dist(first[0], (0.5, 0.5)) < 0.3                  # entered near the robot
    for loop in p.loops:
        for x, y in loop:
            assert wall_distance(m, x, y) >= cl.Params().wall_clearance - 0.08
    assert p.coverage > 0.93                   # 4 x 3 m: the walls' 0.65 m strips are a big share


def test_laps_one_lane_apart():
    m = room(6.0, 6.0)
    p = cl.plan(m, start=(1.0, 1.0))
    lanes = {}
    for loop, lv in zip(p.loops, p.levels):
        lanes.setdefault(lv, []).extend(loop)
    gaps = [min(wall_distance(m, x, y) for x, y in lanes[lv]) for lv in sorted(lanes)]
    assert np.diff(gaps) == pytest.approx([0.45] * (len(gaps) - 1), abs=0.1)


def test_outer_lap_counter_clockwise_and_lap_round_obstacle_clockwise():
    m = room(5.0, 5.0, box=(2.0, 2.0, 3.0, 3.0))
    loops = cl.lap_loops(m, cl.Params())
    outer = [pts for lv, pts in loops if lv == 0]
    assert len(outer) == 2                                       # the walls, the box
    areas = sorted(cl._signed_area(pts) for pts in outer)
    assert areas[0] < 0 < areas[1]
    big = max(outer, key=lambda pts: abs(cl._signed_area(pts)))
    assert cl._signed_area(big) > 0


def test_waypoints_close_enough_for_nav2():
    p = cl.plan(room(8.0, 5.0))
    for loop in p.loops:
        steps = np.linalg.norm(np.diff(np.array(loop), axis=0), axis=1)
        assert steps.max() <= cl.Params().waypoint_spacing + 1e-6


def test_too_narrow_for_the_robot_has_no_laps():
    p = cl.plan(room(0.7, 3.0))                                   # narrower than 2 x clearance
    assert p.loops == [] and p.coverage == 0.0


def test_headings_follow_the_loop():
    loop = [(0, 0), (1, 0), (1, 1), (0, 0)]
    assert cl.headings(loop)[:2] == pytest.approx([0.0, math.pi / 2])
    assert len(cl.headings(loop)) == len(loop)


def test_runner_retries_skips_and_starts_new_rounds():
    m = room(4.0, 4.0)
    r = cl.LapRunner(m, rounds=2, max_failures=2)
    assert r.start((0.5, 0.5))
    n = len(r.plan.loops)
    assert r.finished(False, (0.5, 0.5)) is r.plan.loops[0]      # retried once
    r.finished(False, (0.5, 0.5))                                 # then skipped
    assert r.skipped == 1 and r.index == 1 % n
    while r.round < 1:
        r.finished(True, (0.5, 0.5))
    assert r.active and r.index == 0                              # round 2, planned again
    for _ in range(n - 1):
        assert r.finished(True, (2.0, 2.0)) is not None
    assert r.finished(True, (2.0, 2.0)) is None and not r.active


def test_runner_nothing_to_drive():
    r = cl.LapRunner(room(0.7, 3.0))
    assert not r.start((0.2, 1.0))
    assert r.current() is None and r.finished(True, None) is None


@pytest.mark.skipif(not os.path.exists(LAB), reason="Kumar's Lab map not on this machine")
def test_kumars_lab_map():
    m = cl.load_map(LAB)
    p = cl.plan(m, start=(0.0, 0.0))
    assert len(p.loops) >= 3 and p.coverage > 0.9 and 30 < p.area < 120


def test_legs_cover_the_whole_loop_and_join_up():
    loop = [(float(i), 0.0) for i in range(51)]
    parts = cl.legs(loop, 20)
    assert all(len(p) <= 20 for p in parts)
    assert parts[0][0][0] == loop[0] and parts[-1][-1][0] == loop[-1]
    for a, b in zip(parts, parts[1:]):
        assert a[-1] == b[0]                                   # each leg starts where one ended
    assert cl.legs(loop[:5], 20) == [list(zip(loop[:5], cl.headings(loop[:5])))]
