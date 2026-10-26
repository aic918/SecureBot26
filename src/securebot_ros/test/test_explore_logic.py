"""Exploration logic on synthetic maps (no ROS)."""
import math

import numpy as np

from securebot_ros import explore_logic as ex

RES = 0.05
HW = 0.256 + 0.04          # half the robot width plus margin


def room(w=4.0, h=3.0, door=None, unknown_box=None):
    """A w x h m room with 1-cell walls, everything inside free, outside unknown, origin (0, 0)
    at the room corner. door=(x0, x1) opens the top wall there. unknown_box=(x0, y0, x1, y1)
    marks a not-yet-seen patch inside."""
    cols, rows = int(w / RES) + 2, int(h / RES) + 2
    g = np.full((rows, cols), ex.UNKNOWN, dtype=np.int16)
    g[1:-1, 1:-1] = ex.FREE
    g[0, :] = g[-1, :] = ex.OCCUPIED
    g[:, 0] = g[:, -1] = ex.OCCUPIED
    if door:
        g[-1, int(door[0] / RES) + 1:int(door[1] / RES) + 1] = ex.UNKNOWN
    if unknown_box:
        x0, y0, x1, y1 = unknown_box
        g[int(y0 / RES) + 1:int(y1 / RES) + 1, int(x0 / RES) + 1:int(x1 / RES) + 1] = ex.UNKNOWN
    return ex.Grid(g.ravel().tolist(), cols, rows, RES, -RES, -RES)


def test_closed_room_is_fully_covered():
    g = room()
    reach = ex.reachable_free(g, 2.0, 1.5)
    assert reach.sum() > 4000
    assert ex.frontier_clusters(g, reach) == []
    assert ex.coverage(g, reach) == 1.0


def test_open_door_lowers_coverage():
    g = room(door=(1.5, 2.5))                       # 1 m gap in a 14 m perimeter
    reach = ex.reachable_free(g, 2.0, 1.5)
    clusters = ex.frontier_clusters(g, reach)
    assert len(clusters) == 1
    c = ex.coverage(g, reach)
    assert 0.80 < c < 0.95


def test_unseen_patch_is_a_frontier():
    g = room(unknown_box=(2.5, 0.5, 3.5, 1.5))
    reach = ex.reachable_free(g, 1.0, 1.5)
    assert ex.coverage(g, reach) < 0.9


def test_robot_on_unknown_cell_still_finds_free_space():
    g = room(unknown_box=(1.8, 1.3, 2.2, 1.7))      # the LiDAR's blind disc around the robot
    reach = ex.reachable_free(g, 2.0, 1.5)
    assert reach.sum() > 3000


def test_speckle_frontiers_ignored():
    g = room()
    g.cells[30, 30] = ex.UNKNOWN                     # a single unseen cell
    g.unknown = g.cells < 0
    g.free = (g.cells >= 0) & (g.cells <= ex.FREE_MAX)
    reach = ex.reachable_free(g, 2.0, 1.5)
    assert ex.coverage(g, reach) == 1.0


def test_free_distance_to_wall_and_to_unknown():
    g = room(w=4.0, h=3.0)
    d, ends = ex.free_distance(g, 1.0, 1.5, 0.0, HW, 5.0)       # facing +x, wall at 4.0
    assert ends == 'occupied' and 2.8 < d < 3.05
    g = room(unknown_box=(2.5, 0.8, 3.5, 2.2))
    d, ends = ex.free_distance(g, 1.0, 1.5, 0.0, HW, 5.0)
    assert ends == 'unknown' and 1.3 < d < 1.55
    d, ends = ex.free_distance(g, 1.0, 1.5, 0.0, HW, 0.8)
    assert ends == 'max' and d == 0.8


def test_free_distance_respects_robot_width():
    g = room(w=4.0, h=3.0)
    # 0.2 m from the side wall: driving parallel to it, the side of the robot overlaps it
    d, ends = ex.free_distance(g, 1.0, 0.2, 0.0, HW, 2.0)
    assert ends != 'max' and d < 0.5


def test_choose_heading_prefers_unexplored_space():
    g = room(w=4.0, h=4.0, unknown_box=(0.0, 3.0, 4.0, 4.0))   # top quarter unseen
    pick = ex.choose_heading(g, 2.0, 1.0, 0.0, HW, ex.VisitMap())
    assert pick is not None
    heading, leg = pick
    _, ends_in = ex.free_distance(g, 2.0, 1.0, heading, HW, 3.0)
    assert ends_in == 'unknown'                      # the run leads into the unseen part
    assert math.sin(heading) > 0.5                   # i.e. towards +y
    assert 0.5 <= leg <= 1.5


def test_choose_heading_skips_blocked_direction():
    g = room(w=4.0, h=4.0, unknown_box=(0.0, 3.0, 4.0, 4.0))
    heading, _ = ex.choose_heading(g, 2.0, 1.0, 0.0, HW, ex.VisitMap(), blocked=[math.pi / 2])
    assert abs(ex.angle_diff(heading, math.pi / 2)) >= math.radians(40)


def test_choose_heading_avoids_visited_places():
    g = room(w=6.0, h=2.0)
    visits = ex.VisitMap()
    for _ in range(10):
        visits.add(4.0, 1.0)                         # been to the right many times
    heading, _ = ex.choose_heading(g, 3.0, 1.0, 0.0, HW, visits)
    assert abs(ex.angle_diff(heading, math.pi)) < math.radians(60)   # goes left instead


def test_choose_heading_none_when_boxed_in():
    g = room(w=0.9, h=0.9)
    assert ex.choose_heading(g, 0.45, 0.45, 0.0, HW, ex.VisitMap()) is None


def test_rotation_clear():
    g = room(w=4.0, h=3.0)
    assert ex.rotation_clear(g, 2.0, 1.5, 0.4)
    assert not ex.rotation_clear(g, 0.3, 1.5, 0.4)


def test_visits_along_counts_the_swept_corridor():
    v = ex.VisitMap()
    v.add(4.0, 1.0)
    assert v.along(3.0, 1.0, 0.0, 1.5, HW) == 1                       # straight through it
    assert v.along(3.0, 1.0, math.radians(-15), 1.5, HW) == 1         # slightly off still counts
    assert v.along(3.0, 1.0, math.pi, 1.5, HW) == 0                   # the other way
    v.add(3.1, 1.0)                                                   # where the robot stands
    assert v.along(3.0, 1.0, math.pi, 1.5, HW) == 0
