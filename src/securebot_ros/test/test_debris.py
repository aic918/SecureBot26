"""Temporary blocks for things the sensors cannot see (no ROS)."""
import pytest
import math

import numpy as np

from securebot_ros import debris as db
from securebot_ros.lap_follower import LapFollower


def test_block_goes_just_ahead_of_the_front_edge():
    d = db.Debris(0.3, front=0.25)
    b = d.add_ahead(1.0, 2.0, math.pi / 2)
    assert math.isclose(b.x, 1.0, abs_tol=1e-9) and math.isclose(b.y, 2.0 + 0.25 + 0.15)
    assert d.add_ahead(1.0, 2.02, math.pi / 2) is None        # same spot: no second block
    assert len(d.blocks) == 1
    assert d.distance(1.0, 2.4) == 0.0
    assert math.isclose(d.distance(1.0, 1.0), 1.25, abs_tol=1e-9)


def test_grid_matches_the_map_and_only_marks_the_block():
    d = db.Debris(0.3)
    d.blocks.append(db.Block(0.0, 0.0, 0.3))
    g = d.grid(0.05, (-1.0, -2.0), 40, 60)
    assert g.shape == (60, 40) and g.dtype == np.int8
    assert set(np.unique(g)) == {0, 100}                      # never -1 (unknown)
    rows, cols = np.nonzero(g)
    assert cols.min() == 17 and cols.max() == 22              # x -0.15..0.15
    assert rows.min() == 37 and rows.max() == 42              # y -0.15..0.15, row 0 = low y


def test_edge_points_are_in_base_link():
    d = db.Debris(0.3)
    d.blocks.append(db.Block(2.0, 0.0, 0.3))
    pts = np.array(d.points_in_base((1.0, 0.0, 0.0)))
    assert np.isclose(pts[:, 0].min(), 0.85) and np.isclose(pts[:, 0].max(), 1.15)
    pts = np.array(d.points_in_base((2.0, -1.0, math.pi / 2)))   # block straight ahead
    assert np.isclose(pts[:, 0].min(), 0.85) and np.abs(pts[:, 1]).max() <= 0.15 + 1e-9
    assert d.points_in_base((20.0, 0.0, 0.0)) == []           # far away: left out


def test_resume_index_is_past_the_block():
    d = db.Debris(0.3)
    d.blocks.append(db.Block(2.0, 0.0, 0.3))
    path = [(0.1 * k, 0.0) for k in range(51)]
    k = db.resume_index(path, 10, d, 0.5)
    assert math.isclose(path[k][0], 2.7, abs_tol=0.051)
    assert db.resume_index(path[:22], 10, d, 0.5) is None      # path ends at the block
    assert db.resume_index([(0, 3), (1, 3)], 0, d, 0.5) == 0  # never near: from the start


def _free_clear(debris, walls=()):
    """clear_at for open floor with `debris`, plus horizontal walls at the given y."""
    def clear_at(x, y):
        d = debris.distance(x, y)
        for wy in walls:
            d = min(d, abs(y - wy))
        return d
    return clear_at


def test_sidestep_passes_close_and_rejoins_soon_after():
    """2026-11-08: Nav2's way round an 8 in box swung 1.1-1.8 m off the lane; the sidestep
    keeps a 10 cm gap beside it and is back on the lane just past it."""
    d = db.Debris()
    block = d.add_ahead(1.0, 0.0, 0.0)
    path = [(0.05 * k, 0.0) for k in range(101)]          # straight lane along x, 5 m
    near = 16                                            # robot backed up to x = 0.8
    k = db.resume_index(path, near, d, 0.2475 + db.REJOIN_GAP)
    way = db.sidestep((0.8, 0.0), path, near, k, block, 0.256, 0.2475, _free_clear(d))
    assert way is not None
    off = max(abs(y) for _, y in way)
    assert math.isclose(off, block.size / 2 + 0.256 + db.SIDESTEP_GAP, abs_tol=1e-6)
    assert off < 0.5                                     # was 1.1-1.8 m with Nav2
    rejoin = way[3]
    assert rejoin[1] == 0.0 and rejoin[0] - (block.x + block.size / 2) < 0.45
    # never closer to the box than the robot's half width plus half the gap
    for p, q in zip(way[:4], way[1:4]):
        for t in [i / 20 for i in range(21)]:
            x, y = p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t
            if math.dist((x, y), way[0]) >= db.START_FREE:
                assert d.distance(x, y) >= 0.256 + db.SIDESTEP_GAP / 2 - 1e-9


def test_sidestep_takes_the_free_side_or_gives_up():
    d = db.Debris()
    block = d.add_ahead(1.0, 0.0, 0.0)
    path = [(0.05 * k, 0.0) for k in range(101)]
    k = db.resume_index(path, 16, d, 0.2475 + db.REJOIN_GAP)
    # a wall 0.5 m to the left: only the right side fits
    way = db.sidestep((0.8, 0.0), path, 16, k, block, 0.256, 0.2475, _free_clear(d, [0.5]))
    assert way is not None and way[1][1] < 0 and way[2][1] < 0
    # walls 0.5 m either side: no room for a sidestep, Nav2 decides
    assert db.sidestep((0.8, 0.0), path, 16, k, block, 0.256, 0.2475,
                       _free_clear(d, [0.5, -0.5])) is None
    assert db.sidestep((0.8, 0.0), path, 16, None, block, 0.256, 0.2475,
                       _free_clear(d)) is None


def test_follower_reports_where_it_drove_without_moving():
    f = LapFollower()
    f.start(0.0, [(0.0, 0.0), (5.0, 0.0)])
    t = 0.0
    while t < 30 and f.stall is None:
        f.step(t, (0.5, 0.0, 0.0), math.inf, math.inf, [])     # pose never changes
        t += 0.1
    assert f.stall == (0.5, 0.0, 0.0, f.p.half_length, 0.0) and f.state == 'backup'


def test_follower_stops_before_a_block():
    d = db.Debris(0.3, front=0.2475)
    d.blocks.append(db.Block(1.5, 0.0, 0.3))
    f = LapFollower()
    f.start(0.0, [(0.0, 0.0), (5.0, 0.0)])
    f.state = 'drive'
    pose = (1.0, 0.0, 0.0)
    f.step(0.0, pose, math.inf, math.inf, d.points_in_base(pose))
    assert f.state == 'blocked'


def test_obstacle_that_stays_is_boxed_where_it_starts():
    f = LapFollower()
    f.start(0.0, [(0.0, 0.0), (5.0, 0.0)])
    f.state = 'drive'
    pts = [(0.45, 0.10)]                        # LiDAR: 0.45 m ahead, 10 cm to the left
    t = 0.0
    while t < 10 and f.stall is None:
        f.step(t, (1.0, 0.0, 0.0), math.inf, math.inf, pts)
        t += 0.1
    assert f.stall is not None
    x, y, yaw, ahead, side = f.stall
    assert math.isclose(ahead, 0.45) and math.isclose(side, 0.10)
    d = db.Debris(0.3)
    b = d.add_ahead(x, y, yaw, ahead, side)
    assert math.isclose(b.x, 1.0 + 0.45 + 0.15) and math.isclose(b.y, 0.10)


def test_add_ahead_with_a_side_offset_and_a_turned_robot():
    d = db.Debris(0.3)
    b = d.add_ahead(0.0, 0.0, math.pi / 2, ahead=0.5, side=0.1)   # facing +y
    assert math.isclose(b.x, -0.1, abs_tol=1e-9) and math.isclose(b.y, 0.65)


def test_default_box_is_8_inches():
    assert db.Debris().size == pytest.approx(8 * 0.0254)


def test_click_removes_the_box_under_it_or_the_nearest():
    d = db.Debris()
    d.blocks += [db.Block(0.0, 0.0, 0.2), db.Block(1.0, 0.0, 0.2)]
    assert d.remove_near(1.05, 0.02).x == 1.0
    assert [b.x for b in d.blocks] == [0.0]
    assert d.remove_near(0.5, 0.0) is None                  # far from any box
    assert d.remove_near(0.2, 0.0).x == 0.0                 # 10 cm from its edge
    assert d.blocks == []
