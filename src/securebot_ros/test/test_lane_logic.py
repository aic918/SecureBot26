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
    assert len(p.loops) == 3          # 2.0 m between the side clearances / 1.1 -> 2 gaps, 3 lanes
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


def two_rooms(corridor_width, dead_end=False):
    """Two 3 x 3 m rooms joined by a 2 m corridor `corridor_width` wide (or, with dead_end,
    a corridor off the first room that leads nowhere)."""
    w, h = 8.0, 3.0
    free = np.zeros((int(h / RES), int(w / RES)), bool)
    rows = free.shape[0]

    def fill(x0, y0, x1, y1):
        free[rows - int(y1 / RES):rows - int(y0 / RES), int(x0 / RES):int(x1 / RES)] = True
    fill(0.05, 0.05, 3.0, 2.95)
    if not dead_end:
        fill(5.0, 0.05, 7.95, 2.95)
    fill(3.0, 1.5 - corridor_width / 2, 5.0 if not dead_end else 4.8, 1.5 + corridor_width / 2)
    return cl.Map(free, RES, (0.0, 0.0))


def corridor_lanes(paths):
    """Lanes (last stretch of each path) that run inside the corridor (x 3..5)."""
    out = []
    for path in paths:
        (x0, y0), (x1, y1) = path[-2], path[-1]
        if min(x0, x1) < 3.2 and max(x0, x1) > 4.8:
            out.append(((x0, y0), (x1, y1)))
    return out


def test_narrow_aisle_gets_a_lane_through_its_middle():
    m = two_rooms(0.85)
    p = cl.plan(m, LANES, start=(1.5, 1.5))
    lanes = corridor_lanes(p.loops)
    assert len(lanes) == 1
    (x0, y0), (x1, y1) = lanes[0]
    assert abs(y0 - 1.5) < 0.08 and abs(y1 - 1.5) < 0.08      # down the middle
    # it ends in the open rooms, where the robot can turn
    assert min(x0, x1) < 3.0 - LANES.pass_reach + 0.06 and max(x0, x1) > 5.0 + 0.2


def test_too_narrow_or_dead_end_aisles_get_no_lane():
    assert not corridor_lanes(cl.plan(two_rooms(0.6), LANES, start=(1.5, 1.5)).loops)
    assert not corridor_lanes(cl.plan(two_rooms(0.85, dead_end=True), LANES,
                                      start=(1.5, 1.5)).loops)


def test_wall_strips_are_not_aisles():
    assert ll.pass_lanes(rooms.room(6.0, 4.0), LANES, start=(0.5, 0.5)) == []


def test_fewer_lanes_than_before_but_pockets_still_get_one():
    m = rooms.room(6.0, 4.0, box=(2.5, 1.5, 3.5, 2.5))
    wide = cl.plan(m, LANES, start=(0.5, 0.5))
    tight = cl.plan(m, cl.Params(pattern='lanes', check_width=0.45, fill_lanes=False,
                                 pass_lanes=False), start=(0.5, 0.5))
    assert len(wide.loops) < len(tight.loops)
    space = ll.centre_space(m, LANES, (0.5, 0.5))[0]
    seen = cl.swept_floor(m, wide.loops, width=LANES.check_width, closed=False)
    missed = (space & ~seen).sum() * RES ** 2
    assert missed < LANES.min_pocket


def l_corridor():
    """An L of 3 m wide corridors: 20 m along x, then 15 m up along y from its right end."""
    size_x, size_y = int(24 / RES), int(20 / RES)
    free = np.zeros((size_y, size_x), bool)
    free[size_y - int(4 / RES):size_y - int(1 / RES), int(1 / RES):int(21 / RES)] = True
    free[int(1 / RES):size_y - int(1 / RES), int(18 / RES):int(21 / RES)] = True
    return cl.Map(free, RES, (0.0, 0.0))


def test_side_corridor_gets_lanes_along_it():
    """The corridor across the main one is driven along, not crossed by many short lanes."""
    m = l_corridor()
    p = cl.Params(pattern='lanes', check_width=2.0, min_pocket=1.0)
    plan = cl.plan(m, p, (3.0, 2.5))
    assert plan.mixed
    route = [pt for path in plan.loops for pt in path]
    turns, _ = ll.turn_count(route)
    assert turns <= 16, turns         # right-angle hops between lanes (2026-11-07)
    assert plan.coverage > 0.9
    # long lanes up the side corridor
    ups = [(a, b) for path in plan.loops for a, b in zip(path, path[1:])
           if a[0] > 18 and abs(b[1] - a[1]) > 8 and abs(b[0] - a[0]) < 1.0]
    assert ups


def side_aisle_room():
    """8 x 6 m room with a 6 x 2.6 m desk block 0.85 m from the left wall: a side aisle the
    robot fits through but cannot turn in, open at both ends."""
    m = rooms.room(8.0, 6.0)
    r0, c0 = m.to_cell(0.85, 5.0)
    r1, c1 = m.to_cell(3.45, 1.0)
    m.free[r0:r1, c0:c1] = False
    return m


def test_edge_cases_add_lanes_and_leave_the_plan_as_it_was():
    """2026-11-07 (Kumar's Lab side aisles): edge-case lanes come after the plan, which is
    exactly what it was without them."""
    m = side_aisle_room()
    base = cl.plan(m, cl.Params(pattern='lanes', check_width=2.0, min_pocket=1.0,
                                pocket_aisles=False, long_pocket=math.inf), (6.0, 3.0))
    full = cl.plan(m, cl.Params(pattern='lanes', check_width=2.0, min_pocket=1.0),
                   (6.0, 3.0))
    assert full.loops[:len(base.loops)] == base.loops
    assert full.edge == len(full.loops) - len(base.loops)


def test_bent_strip_gets_one_straight_lane():
    """2026-11-08 (Kumar's Lab): a strip whose centre line bends a little got a lane per bend
    (0.7-1.4 m each, with turns between); now one straight lane covers it."""
    m = rooms.room(8.0, 6.0)
    # a strip of floor along the top wall behind a cabinet whose front steps back 0.3 m
    # halfway: the strip's centre line has a jog
    r0, c0 = m.to_cell(1.0, 4.8)
    r1, c1 = m.to_cell(4.0, 4.6)
    m.free[r0:r1, c0:c1] = False
    r0, c0 = m.to_cell(4.0, 4.5)
    r1, c1 = m.to_cell(7.0, 4.3)
    m.free[r0:r1, c0:c1] = False
    p = cl.Params(pattern='lanes', check_width=2.0, min_pocket=1.0)
    lanes = ll.edge_lanes(m, p, (4.0, 2.0), [((0.8, 2.0), (7.2, 2.0))])
    along = [lane for lane in lanes if min(lane[0][1], lane[1][1]) > 4.4]
    assert 1 <= len(along) <= 2
    assert all(math.dist(*lane) > 2.5 for lane in along)       # no short pieces
    assert sum(abs(b[0] - a[0]) for a, b in along) > 5.5        # still the whole strip


def test_wiggly_centre_line_becomes_one_straight_run():
    m = rooms.room(6.0, 3.0)
    space = np.zeros(m.free.shape, bool)
    r0, c0 = m.to_cell(0.4, 2.0)
    r1, c1 = m.to_cell(5.6, 1.0)
    space[r0:r1, c0:c1] = True                                   # a 1 m wide band
    pts = [(0.5 + 0.05 * k, 1.5 + 0.15 * math.sin(k / 6)) for k in range(100)]
    run = ll._straight_runs(space, m, pts)
    assert len(run) == 2 and run[0] == pts[0] and run[-1] == pts[-1]


def test_short_bent_tail_is_absorbed_by_its_long_neighbour():
    space = np.ones((60, 120), bool)
    m = rooms.room(6.0, 3.0)
    segs = [((0.5, 1.5), (4.0, 1.5)), ((4.0, 1.5), (4.6, 1.8))]
    out = ll._absorb_short(space, m, segs, 1.0)
    assert len(out) == 1 and out[0][0] == (0.5, 1.5) and out[0][1][0] > 4.0
    # a tail farther off than the reach stays a lane of its own
    far = [((0.5, 1.5), (4.0, 1.5)), ((4.0, 1.5), (4.0, 2.9))]
    assert len(ll._absorb_short(space, m, far, 1.0)) == 2


def test_long_thin_strip_gets_a_lane():
    m = rooms.room(8.0, 6.0)
    # a 1.2 m wide strip of floor along the top wall, behind a long cabinet
    r0, c0 = m.to_cell(1.0, 4.8)
    r1, c1 = m.to_cell(7.0, 4.6)
    m.free[r0:r1, c0:c1] = False
    p = cl.Params(pattern='lanes', check_width=2.0, min_pocket=1.0)
    q = cl.dataclasses.replace(p, lane_spacing=p.check_width) if hasattr(cl, 'dataclasses') \
        else p
    lanes = ll.edge_lanes(m, q, (4.0, 2.0), [((0.8, 2.0), (7.2, 2.0))])
    along = [lane for lane in lanes if min(lane[0][1], lane[1][1]) > 4.8]
    assert along and max(math.dist(*lane) for lane in along) > 2.0
