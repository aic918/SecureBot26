"""Straight-lane coverage on synthetic maps (no ROS)."""
import math

import cv2
import numpy as np
import pytest

from securebot_ros import coverage_logic as cl
from securebot_ros import lane_logic as ll

import test_coverage_logic as rooms        # room(w, h, box) helper

RES = 0.05
LANES = cl.Params(pattern='lanes')


def wall_dist(m):
    padded = np.pad(m.free.astype(np.uint8), 1)
    return cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1] * RES


def at(dist, m, x, y):
    return dist[m.to_cell(x, y)]


def rotated_room(w, h, deg, border=1.0):
    """w x h m room turned `deg` degrees, centred in an unknown margin."""
    size = int((math.hypot(w, h) + 2 * border) / RES)
    free = np.zeros((size, size), np.uint8)
    c = size / 2
    a = math.radians(deg)
    corners = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        x = dx * math.cos(a) - dy * math.sin(a)
        y = dx * math.sin(a) + dy * math.cos(a)
        corners.append((c + x / RES, c - y / RES))           # image rows grow downwards
    cv2.fillPoly(free, [np.array(corners, np.int32)], 1)
    origin = (-size * RES / 2, -size * RES / 2)
    return cl.Map(free.astype(bool), RES, origin)


def lane_angles(paths):
    """Directions (deg, 0..180) of the lanes: the last segment of every path."""
    return [math.degrees(math.atan2(p[-1][1] - p[-2][1], p[-1][0] - p[-2][0])) % 180
            for p in paths]


def test_lanes_run_along_the_long_side():
    m = rooms.room(6.0, 3.0)
    p = cl.plan(m, LANES, start=(0.5, 0.5))
    assert len(p.loops) == 6          # 2.1 m between the side clearances / 0.45 -> 5 gaps, 6 lanes
    for a in lane_angles(p.loops):
        assert min(a, 180 - a) < 3                          # parallel to x


def test_lanes_follow_a_rotated_room():
    m = rotated_room(6.0, 3.0, 30)
    p = cl.plan(m, LANES, start=(0.0, 0.0))
    for a in lane_angles(p.loops):
        assert abs(a - 30) < 6
    assert math.degrees(p.angle) % 180 == pytest.approx(30, abs=6)


def test_side_and_front_clearance():
    m = rooms.room(6.0, 3.0)
    dist = wall_dist(m)
    p = cl.plan(m, LANES, start=(0.5, 0.5))
    for path in p.loops:
        for x, y in path:
            assert at(dist, m, x, y) >= LANES.side_clearance - 0.06
        # a lane's end faces the short wall: 0.60 m to it, not just 0.45
        (x0, _), (x1, _) = path[-2], path[-1]
        end = max(x0, x1) if x1 > x0 else min(x0, x1)
        assert min(end, 6.0 - end) >= LANES.front_clearance - 0.06


def test_lanes_split_round_a_desk_and_hops_stay_clear():
    m = rooms.room(6.0, 4.0, box=(2.5, 1.5, 3.5, 2.5))
    dist = wall_dist(m)
    p = cl.plan(m, LANES, start=(0.5, 0.5))
    beside = 0
    for path in p.loops:
        for a, b in zip(path, path[1:]):          # lanes and hops: never over the desk
            for t in np.linspace(0, 1, 20):
                x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
                assert at(dist, m, x, y) >= LANES.side_clearance - 0.08
        (x0, y0), (x1, y1) = path[-2], path[-1]   # the lane itself
        if min(abs(x0 - 3.0), abs(y0 - 2.0)) < 1.5 and math.dist((x0, y0), (x1, y1)) > 1.0:
            beside += 1
    assert beside                                 # lanes run right beside the desk


def test_thin_obstacles_are_not_filled_in():
    """A desk leg (one 5 cm cell the LiDAR saw) is thinner than fill_holes: it must stay an
    obstacle, while an unseen speck of the same size is filled in as floor."""
    m = rooms.room(6.0, 3.0)
    leg, speck = m.to_cell(2.0, 1.5), m.to_cell(4.0, 1.5)
    m.free[leg] = m.free[speck] = False
    m.wall = np.zeros_like(m.free)
    m.wall[leg] = True
    floor = cl.floor_mask(m, 0.15)
    assert not floor[leg] and floor[speck]
    dist = wall_dist(cl.Map(floor, RES, m.origin))
    p = cl.plan(m, LANES, start=(0.5, 0.5))
    for path in p.loops:
        for a, b in zip(path, path[1:]):
            for t in np.linspace(0, 1, 40):
                x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
                assert at(dist, m, x, y) >= LANES.side_clearance - 0.08


def test_no_straight_hop_through_walls_when_no_way_is_found():
    """Two rooms joined by nothing the robot fits through: lanes in the other room are left
    out instead of being joined by a straight line through the wall."""
    m = rooms.room(7.0, 3.0)
    wall = m.to_cell(3.5, 0.0)[1]
    m.free[:, wall - 1:wall + 2] = False           # a full-height wall at x = 3.5
    space, _, _ = ll.centre_space(m, LANES, (0.5, 0.5))
    far = [((4.5, 1.0), (6.0, 1.0))]               # a lane beyond the wall
    near = [((0.6, 1.0), (2.8, 1.0))]
    paths = ll.chain(m, LANES, near + far, (0.6, 1.0))
    assert len(paths) == 1 and paths[0][-1] == (2.8, 1.0)


def test_lanes_cover_the_room_with_fewer_turns_than_contour_laps():
    m = rooms.room(6.0, 4.0)
    lanes = cl.plan(m, LANES, start=(0.5, 0.5))
    rings = cl.plan(m, cl.Params(wall_clearance=0.6), start=(0.5, 0.5))
    assert lanes.coverage > 0.9
    t_lanes, _ = ll.turn_count([pt for path in lanes.loops for pt in path])
    t_rings, _ = ll.turn_count([pt for path in rings.loops for pt in path])
    assert t_lanes < t_rings


def test_starts_at_the_lane_nearest_the_robot():
    m = rooms.room(6.0, 3.0)
    p = cl.plan(m, LANES, start=(5.2, 2.4))
    first_lane_start = p.loops[0][-2]
    assert math.dist(first_lane_start, (5.2, 2.4)) < 1.0


def test_turn_count():
    square = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    assert ll.turn_count(square)[0] == 3
    assert ll.turn_count([(0, 0), (1, 0.05), (2, 0)])[0] == 0
