"""Exploration state machine in a small simulated world (no ROS, no hardware)."""
import math

import numpy as np

from securebot_ros import explore_logic as ex
from securebot_ros.explore_machine import ExploreMachine, FOOT

import test_explore_logic as maps        # synthetic rooms

DT = 0.1
BLIND = math.radians(41.5)               # LiDAR blind wedge (camera tower) each side of ahead


class World:
    """Robot that moves exactly as commanded, with a LiDAR that cannot see straight ahead and
    two forward ultrasonics. The map never fills in (the grid is fixed)."""

    def __init__(self, grid, x, y, yaw, moves=True):
        self.grid, self.x, self.y, self.yaw, self.moves = grid, x, y, yaw, moves
        self.us = None                       # override both ultrasonic ranges
        rows, cols = np.nonzero(grid.occ)
        self.walls = np.array([grid.to_world(r, c) for r, c in zip(rows, cols)])

    def apply(self, cmd):
        if cmd and self.moves:
            lin, ang = cmd
            self.yaw = math.atan2(math.sin(self.yaw + ang * DT), math.cos(self.yaw + ang * DT))
            self.x += math.cos(self.yaw) * lin * DT
            self.y += math.sin(self.yaw) * lin * DT

    def ultrasonic(self):
        if self.us is not None:
            return self.us
        occupied, _ = ex.ray(self.grid, self.x, self.y, self.yaw, 0.0, 4.0, start=0.25)
        return occupied - 0.25 if occupied < 4.0 else math.inf

    def scan(self):
        """Nearest wall cell in each 3 deg slice, outside the blind wedge (robot frame)."""
        dx, dy = self.walls[:, 0] - self.x, self.walls[:, 1] - self.y
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        px, py = c * dx + s * dy, -s * dx + c * dy
        r, a = np.hypot(px, py), np.arctan2(py, px)
        keep = (np.abs(a) >= BLIND) & (r <= 6.0)
        r, px, py, bins = r[keep], px[keep], py[keep], np.floor(np.degrees(a[keep]) / 3.0)
        order = np.lexsort((r, bins))
        first = np.ones(len(order), dtype=bool)
        first[1:] = bins[order][1:] != bins[order][:-1]
        sel = order[first]
        return list(zip(px[sel].tolist(), py[sel].tolist()))

    def wall_distance(self):
        return float(np.min(np.hypot(self.walls[:, 0] - self.x, self.walls[:, 1] - self.y)))

    def run(self, m, seconds, t0=0.0):
        t, states, self.cmds = t0, [], []
        self.max_y, self.closest = self.y, self.wall_distance()
        while t < t0 + seconds:
            r = self.ultrasonic()
            cmd = m.step(t, (self.x, self.y, self.yaw), self.grid, r, r, self.scan())
            self.cmds.append(cmd)
            self.apply(cmd)
            self.max_y = max(self.max_y, self.y)
            self.closest = min(self.closest, self.wall_distance())
            if not states or states[-1] != m.state:
                states.append(m.state)
            t += DT
        return t, states


def test_idle_sends_nothing():
    m = ExploreMachine()
    assert m.step(0.0, (0, 0, 0), None, 1.0, 1.0, []) is None


def test_start_drives_straight_away_without_stopping():
    g = maps.room(w=8.0, h=8.0)
    w, m = World(g, 2.0, 4.0, 0.0), ExploreMachine()
    m.p.coverage_goal = 1.1                              # closed room: keep going
    m.start(0.0)
    _, states = w.run(m, 5.0)
    assert states == ['drive']
    assert all(c[0] > 0.0 for c in w.cmds)               # moving forward every single tick


def test_curves_away_from_a_wall_ahead_while_moving():
    g = maps.room(w=4.0, h=6.0)
    w, m = World(g, 1.0, 3.0, 0.0), ExploreMachine()      # wall 3 m ahead, lots of room aside
    m.p.coverage_goal = 1.1                              # closed room: keep going
    m.start(0.0)
    _, states = w.run(m, 25.0)
    assert states == ['drive'], states                   # never had to stop for it
    assert all(c[0] > 0.0 for c in w.cmds)
    assert any(abs(c[1]) > 0.2 for c in w.cmds)          # it steered
    assert w.closest > m.p.half_length + 0.1             # and kept clear of the walls


def test_continuous_path_through_a_room_reaches_the_unseen_part():
    g = maps.room(w=5.0, h=5.0, unknown_box=(0.0, 3.5, 5.0, 5.0))
    w, m = World(g, 2.5, 1.0, 0.0), ExploreMachine()
    m.start(0.0)
    w.run(m, 90.0)
    moving = sum(1 for c in w.cmds if c and c[0] > 0.0)
    assert moving > 0.8 * len(w.cmds)                    # mostly driving, not stop-and-go
    assert w.max_y > 3.5                                 # it went into the unseen top part
    assert w.closest > m.p.half_length                   # never drove into a wall
    assert m.state not in ('error', 'idle'), m.message   # (map never fills in: no 'done')


def test_closed_room_finishes_once_it_has_driven():
    g = maps.room(w=5.0, h=5.0)
    w, m = World(g, 2.5, 2.5, 0.0), ExploreMachine()
    m.start(0.0)
    w.run(m, 30.0)
    assert m.state == 'done' and 'room scanned' in m.message, (m.state, m.message)
    assert m.coverage >= 0.9 and m.travelled >= m.p.min_travel
    assert m.step(100.0, (w.x, w.y, w.yaw), g, 1.0, 1.0, []) is None   # stays stopped


def test_ultrasonic_within_one_foot_stops_backs_up_turns_and_drives_on():
    g = maps.room(w=6.0, h=6.0)
    w, m = World(g, 3.0, 3.0, 0.0), ExploreMachine()
    m.start(0.0)
    t, _ = w.run(m, 1.0)
    w.us = FOOT - 0.01                                   # something 1 ft ahead
    cmd = m.step(t, (w.x, w.y, w.yaw), g, w.us, w.us, w.scan())
    assert cmd == (0.0, 0.0) and m.state == 'blocked_wait' and 'ultrasonic' in m.message
    assert m.blocked[-1] == w.yaw
    w.us = None
    t, states = w.run(m, 1.5, t0=t + DT)
    assert states == ['blocked_wait', 'backup', 'turn'], states
    assert abs(ex.angle_diff(m.target, m.blocked[-1])) >= m.p.blocked_width - 1e-6
    _, states = w.run(m, 8.0, t0=t)
    assert states[:2] == ['turn', 'drive'], states


def test_exactly_one_foot_counts_as_blocked_but_just_over_does_not():
    m = ExploreMachine()
    none = np.zeros((0, 2))
    assert m._front_blocked(FOOT, 2.0, none)
    assert not m._front_blocked(FOOT + 0.01, 2.0, none)


def test_lidar_point_close_ahead_or_touching_blocks():
    m = ExploreMachine()
    assert m._front_blocked(5.0, 5.0, np.array([[0.4, 0.1]]))     # 15 cm in front of the case
    assert m._front_blocked(5.0, 5.0, np.array([[0.0, 0.30]]))    # beside a wheel
    assert not m._front_blocked(5.0, 5.0, np.array([[1.5, 0.0], [0.0, 1.0], [-1.0, 0.0]]))


def test_not_moving_then_not_turning_is_an_error():
    g = maps.room(w=5.0, h=5.0)
    w, m = World(g, 2.5, 2.5, 0.0, moves=False), ExploreMachine()
    m.start(0.0)
    _, states = w.run(m, 15.0)
    assert 'blocked_wait' in states                      # 'not moving' first
    assert m.state == 'error' and 'not turning' in m.message


def test_boxed_in_is_an_error():
    g = maps.room(w=0.9, h=0.9)
    w, m = World(g, 0.45, 0.45, 0.0), ExploreMachine()
    m.start(0.0)
    w.run(m, 5.0)
    assert m.state == 'error' and 'no open direction' in m.message, (m.state, m.message)


def test_silent_sensors_stop_after_timeout():
    m = ExploreMachine()
    m.start(0.0)
    assert m.step(0.0, (0, 0, 0), None, None, 1.0, []) == (0.0, 0.0)
    m.step(1.0, (0, 0, 0), None, None, 1.0, [])
    assert m.running
    m.step(3.5, (0, 0, 0), None, None, 1.0, [])
    assert m.state == 'error' and 'left ultrasonic' in m.message


def test_stop_button():
    m = ExploreMachine()
    m.start(0.0)
    m.stop(1.0)
    assert m.state == 'idle' and m.step(1.1, (0, 0, 0), None, 1.0, 1.0, []) is None
