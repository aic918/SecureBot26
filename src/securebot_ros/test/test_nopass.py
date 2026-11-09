"""No-pass lines (no ROS)."""
import math

import numpy as np

from securebot_ros import coverage_logic as cl
from securebot_ros import nopass as npz


def room(w=80, h=60):
    free = np.ones((h, w), dtype=bool)
    free[0, :] = free[-1, :] = free[:, 0] = free[:, -1] = False
    return cl.Map(free, 0.05, (0.0, 0.0), ~free)


def test_line_becomes_a_wall_of_its_width(tmp_path):
    m = room()
    n = npz.NoPass(str(tmp_path / 'r.nopass.json'))
    assert n.add((2.0, 0.0), (2.0, 3.0))
    k = n.mask(m)
    row = k[30]                                   # y = 1.5 m
    cols = np.nonzero(row)[0]
    assert abs((cols.mean() + 0.5) * 0.05 - 2.0) < 0.05
    assert 1 <= len(cols) <= 3                    # ~0.10 m thick
    m2 = n.apply(m)
    assert not m2.free[30, 40] and m2.wall[30, 40]
    assert m.free[30, 40]                         # the room map itself is untouched


def test_lines_are_saved_with_the_room_and_removed_by_click(tmp_path):
    path = str(tmp_path / 'r.nopass.json')
    n = npz.NoPass(path)
    n.add((0.5, 0.5), (2.5, 0.5))
    n.add((1.0, 1.0), (1.0, 2.0))
    assert len(npz.NoPass(path).lines) == 2       # reloaded from the file
    assert n.remove_near(1.5, 0.55) is not None   # a click right on the first line
    assert n.remove_near(3.0, 3.0) is None
    n.undo()
    assert npz.NoPass(path).lines == []
    assert not n.add((1.0, 1.0), (1.02, 1.0))     # a click twice in one place: no line


def test_planner_does_not_cross_a_line():
    """A line across the room: lanes stay on the robot's side of it."""
    m = room(120, 60)                             # 6 x 3 m
    n = npz.NoPass(None)
    n.add((3.0, 0.0), (3.0, 3.0))
    plan = cl.plan(n.apply(m), cl.Params(), start=(1.0, 1.5))
    xs = [x for loop in plan.loops for x, _ in loop]
    assert xs and max(xs) < 3.0
    for loop in plan.loops:
        assert cl.crosses_wall(n.apply(m), loop) is None


def test_follower_sees_the_line_ahead():
    n = npz.NoPass(None)
    n.add((1.0, -1.0), (1.0, 1.0))
    pts = n.points_in_base((0.0, 0.0, 0.0))
    ahead = [x for x, y in pts if abs(y) < 0.2]
    assert ahead and min(ahead) == min(x for x, _ in pts)
    assert abs(min(ahead) - (1.0 - n.width / 2)) < 0.01
    assert n.points_in_base((10.0, 10.0, math.pi)) == []
