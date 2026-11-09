"""Corridor routes (route_logic) and the auto choice on synthetic maps (no ROS)."""
import math

import cv2
import numpy as np

from securebot_ros import coverage_logic as cl
from securebot_ros import lane_logic as ll
from securebot_ros import route_logic as rl

RES = 0.05
P = cl.Params(pattern='corridors', check_width=2.0, min_pocket=1.0)


def corridors(polys, size=(30.0, 30.0), width=3.0):
    """Corridors `width` m wide along polylines [[(x, y), ...]] on a size m map."""
    w, h = int(size[0] / RES), int(size[1] / RES)
    free = np.zeros((h, w), np.uint8)
    for poly in polys:
        pts = np.array([(x / RES, h - y / RES) for x, y in poly], np.int32)
        cv2.polylines(free, [pts], False, 1, thickness=int(width / RES))
    return cl.Map(free.astype(bool), RES, (0.0, 0.0))


def turns(plan):
    route = [q for path in plan.loops for q in path]
    return ll.turn_count(route)[0] if route else 0


def longest_straight(plan, tol_deg=10.0):
    """Longest run along one path whose heading stays within tol_deg."""
    best = 0.0
    for path in plan.loops:
        run, head = 0.0, None
        for a, b in zip(path, path[1:]):
            h = math.atan2(b[1] - a[1], b[0] - a[0])
            if head is not None and abs(math.atan2(math.sin(h - head),
                                                   math.cos(h - head))) > math.radians(tol_deg):
                run = 0.0
            run += math.dist(a, b)
            head = h
            best = max(best, run)
    return best


def test_hallway_is_one_lane_end_to_end():
    m = corridors([[(2.0, 5.0), (38.0, 5.0)]], size=(40.0, 10.0))
    plan = cl.plan(m, P, (3.0, 5.0))
    assert plan.pattern == 'corridors'
    assert len(plan.loops) == 1
    assert turns(plan) == 0
    assert longest_straight(plan) > 30.0


def test_slightly_bent_hallway_has_no_turns():
    """A hallway that bends 4 deg half way (as on a SLAM map) is still one lane, no turns:
    straight lanes at one angle broke into pieces there."""
    a = math.radians(4.0)
    mid = (20.0, 5.0)
    end = (mid[0] + 18.0 * math.cos(a), mid[1] + 18.0 * math.sin(a))
    m = corridors([[(2.0, 5.0), mid, end]], size=(40.0, 10.0))
    plan = cl.plan(m, P, (3.0, 5.0))
    assert len(plan.loops) == 1
    assert turns(plan) == 0


def test_side_corridor_driven_along_not_across():
    m = corridors([[(2.0, 4.0), (24.0, 4.0), (24.0, 26.0)]], size=(28.0, 30.0))
    plan = cl.plan(m, P, (3.0, 4.0))
    assert turns(plan) <= 3
    ups = [(a, b) for path in plan.loops for a, b in zip(path, path[1:])
           if abs(b[1] - a[1]) > 10.0 and abs(b[0] - a[0]) < 1.0]
    assert ups


def test_t_junction_visits_every_arm_and_ends_in_the_longest():
    # stem up from (15, 2) to (15, 12); arms left to (3, 12) (short) and right to (40, 12)
    m = corridors([[(15.0, 2.0), (15.0, 12.0)], [(5.0, 12.0), (40.0, 12.0)]],
                  size=(42.0, 16.0))
    plan = cl.plan(m, P, (15.0, 3.0))
    pts = [q for path in plan.loops for q in path]
    assert min(x for x, _ in pts) < 8.0              # short arm visited
    assert max(x for x, _ in pts) > 36.0             # long arm visited
    end = plan.loops[-1][-1]
    assert end[0] > 30.0                             # ends in the long arm: no way back
    assert turns(plan) <= 6


def test_dead_end_lane_stops_before_the_wall():
    m = corridors([[(2.0, 5.0), (20.0, 5.0)]], size=(24.0, 10.0))
    plan = cl.plan(m, P, (3.0, 5.0))
    far = max(q[0] for path in plan.loops for q in path)
    # the corridor ends at x = 20 + 1.5 (round cap): front clearance 0.6 m before it
    assert far <= 21.5 - P.front_clearance + 0.1


def test_auto_picks_corridors_in_hallways_and_lanes_in_a_room_with_desks():
    hall = corridors([[(2.0, 4.0), (24.0, 4.0), (24.0, 26.0)]], size=(28.0, 30.0))
    assert cl.plan(hall, cl.Params(pattern='auto', check_width=2.0), (3.0, 4.0)).pattern \
        == 'corridors'
    # 10 x 8 m room with three desks in a row: no corridor network
    import test_coverage_logic as rooms
    m = rooms.room(10.0, 8.0)
    for x0 in (2.0, 4.5, 7.0):
        r0, c0 = m.to_cell(x0, 4.6)
        r1, c1 = m.to_cell(x0 + 1.2, 3.4)
        m.free[r0:r1, c0:c1] = False
    plan = cl.plan(m, cl.Params(pattern='auto', check_width=2.0), (1.0, 1.0))
    assert plan.pattern in ('lanes', 'corridors')
    assert plan.coverage > 0.6


def test_route_space_picks_robots_part():
    m = corridors([[(2.0, 5.0), (12.0, 5.0)], [(20.0, 5.0), (30.0, 5.0)]], size=(32.0, 10.0))
    space, _, _ = rl.route_space(m, P, (25.0, 5.0))
    r, c = m.to_cell(25.0, 5.0)
    assert space[r, c]
    r, c = m.to_cell(5.0, 5.0)
    assert not space[r, c]


def test_rest_of_a_bent_lane_and_corners():
    from securebot_ros.laps_node import corners, rest_of
    lane = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    assert rest_of(lane, (5.0, 0.2)) == [(5.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    assert rest_of(lane, (10.1, 9.9)) == []               # less than LEFT_MIN left
    assert rest_of(lane, (5.0, 3.0)) is None              # not on it
    bent = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.5), (20.0, 10.0)]
    assert corners(bent) == [(20.0, 0.5), (20.0, 10.0)]   # the 3 deg bend is no corner


def _angles_ok(plan):
    for path in plan.loops:
        assert len(path) % 2 == 0, path
        for a, b, c in zip(path, path[1:], path[2:]):
            h1 = math.atan2(b[1] - a[1], b[0] - a[0])
            h2 = math.atan2(c[1] - b[1], c[0] - b[0])
            d = abs(math.degrees(math.atan2(math.sin(h2 - h1), math.cos(h2 - h1))))
            assert min(d, abs(d - 90), abs(180 - d)) < 0.5, (a, b, c, d)


def test_waypoints_are_right_angles_and_even():
    """Every waypoint goes straight on or turns 90 / 180 deg; every lane has an even count."""
    for polys, size, start in (
            ([[(2.0, 4.0), (24.0, 4.0), (24.0, 26.0)]], (28.0, 30.0), (3.0, 4.0)),
            ([[(15.0, 2.0), (15.0, 12.0)], [(5.0, 12.0), (40.0, 12.0)]], (42.0, 16.0),
             (15.0, 3.0)),
            ([[(2.0, 5.0), (20.0, 5.0), (20.0 + 18.0 * math.cos(0.07),
                                         5.0 + 18.0 * math.sin(0.07))]], (40.0, 10.0),
             (3.0, 5.0))):
        m = corridors(polys, size=size)
        _angles_ok(cl.plan(m, P, start))


def test_straight_lanes_are_squared_too():
    import test_coverage_logic as rooms
    m = rooms.room(10.0, 8.0, box=(4.0, 3.0, 6.0, 5.0))
    plan = cl.plan(m, cl.Params(pattern='lanes', check_width=2.0), (1.0, 1.0))
    _angles_ok(plan)


def test_square_path_of_a_bent_path_is_clear_with_one_corner():
    m = corridors([[(2.0, 4.0), (24.0, 4.0), (24.0, 26.0)]], size=(28.0, 30.0))
    space, _, _ = rl.route_space(m, P, (3.0, 4.0))
    bent = [(3.0, 4.0), (12.0, 4.3), (23.6, 4.5), (24.3, 15.0), (24.0, 25.0)]
    sq, ok = rl.square_path(m, space, bent, 0.0)
    assert ok
    assert all(ll._segment_clear(space, m, a, b) for a, b in zip(sq, sq[1:]))
    assert len(sq) == 3                                   # start, the corner, the end


def test_way_back_follows_the_lanes_and_skips_detours():
    """Return home: back along the planned lanes (they start at HOME), shortcutting a side
    branch it went into and out of, without Nav2."""
    m = corridors([[(15.0, 2.0), (15.0, 12.0)], [(5.0, 12.0), (40.0, 12.0)]],
                  size=(42.0, 16.0))
    home = (15.0, 3.0)
    plan = cl.plan(m, P, home)
    space, _, _ = rl.route_space(m, P, home)
    way = rl.route_back(plan.loops, (38.0, 12.0), home,
                        clear=lambda a, b: rl._clear(space, m, a, b))
    assert way is not None and way[0] == (38.0, 12.0) and way[-1] == home
    length = sum(math.dist(a, b) for a, b in zip(way, way[1:]))
    assert length < 23.0 + 9.0 + 3.0            # along the arm and down the stem, no detour
    assert min(x for x, _ in way) > 12.0         # did not go into the short west arm
    assert all(ll._segment_clear(space, m, a, b) for a, b in zip(way, way[1:]))


def test_way_back_is_none_far_from_the_lanes():
    m = corridors([[(2.0, 5.0), (38.0, 5.0)]], size=(40.0, 30.0))
    plan = cl.plan(m, P, (3.0, 5.0))
    assert rl.route_back(plan.loops, (20.0, 25.0), (3.0, 5.0)) is None


def test_way_back_ends_straight_at_home_and_is_square():
    """HOME just off the lanes: the way leaves them where it heads for HOME (no hook past
    it), and squared it has right angles only and ends exactly at HOME."""
    m = corridors([[(2.0, 5.0), (38.0, 5.0)]], size=(40.0, 10.0))
    home = (3.0, 5.6)                                     # 0.6 m off the centre lane
    plan = cl.plan(m, P, (3.0, 5.0))
    space, _, turn = rl.route_space(m, P, home)
    way = rl.route_back(plan.loops, (36.0, 5.0), home,
                        clear=lambda a, b: rl._clear(space, m, a, b))
    assert way[-1] == home and min(x for x, _ in way) > 2.5    # no hook past HOME
    sq, ok = rl.square_path_in(m, [turn, space], way, 0.0, exact_end=True)
    assert ok and math.dist(sq[-1], home) < 0.15
    for a, b, c in zip(sq, sq[1:], sq[2:]):
        h1 = math.atan2(b[1] - a[1], b[0] - a[0])
        h2 = math.atan2(c[1] - b[1], c[0] - b[0])
        d = abs(math.degrees(math.atan2(math.sin(h2 - h1), math.cos(h2 - h1))))
        assert min(d, abs(d - 90), abs(180 - d)) < 0.5


def test_start_drives_a_ready_plan_without_planning():
    m = corridors([[(2.0, 5.0), (38.0, 5.0)]], size=(40.0, 10.0))
    ready = cl.plan(m, P, (3.0, 5.0))
    runner = cl.LapRunner(m, P, rounds=1)
    assert runner.start((3.0, 5.0), ready)
    assert runner.plan is ready
