"""Exploration state machine in a small simulated world (no ROS, no hardware)."""
import math

from securebot_ros import explore_logic as ex
from securebot_ros.explore_machine import ExploreMachine, FOOT

import test_explore_logic as maps        # synthetic rooms

DT = 0.1


class World:
    """Robot that moves exactly as commanded (speeds scaled down so a burst is a few degrees)."""

    def __init__(self, grid, x, y, yaw, moves=True):
        self.grid, self.x, self.y, self.yaw, self.moves = grid, x, y, yaw, moves
        self.us = None                       # override both ultrasonic ranges

    def apply(self, cmd):
        if cmd and self.moves:
            lin, ang = cmd
            self.yaw = math.atan2(math.sin(self.yaw + ang * DT), math.cos(self.yaw + ang * DT))
            self.x += math.cos(self.yaw) * lin * DT
            self.y += math.sin(self.yaw) * lin * DT

    def ultrasonic(self):
        if self.us is not None:
            return self.us
        d, ends = ex.free_distance(self.grid, self.x, self.y, self.yaw, 0.0, 4.0, start=0.25)
        return d - 0.25 if ends == 'occupied' else math.inf

    def run(self, m, seconds, t0=0.0):
        t, states = t0, []
        self.max_y = self.y
        while t < t0 + seconds:
            r = self.ultrasonic()
            cmd = m.step(t, (self.x, self.y, self.yaw), self.grid, r, r, [])
            self.apply(cmd)
            self.max_y = max(self.max_y, self.y)
            if not states or states[-1] != m.state:
                states.append(m.state)
            t += DT
        return t, states


def test_idle_sends_nothing():
    m = ExploreMachine()
    assert m.step(0.0, (0, 0, 0), None, 1.0, 1.0, []) is None


def test_full_cycle_in_a_room_with_an_unseen_part():
    g = maps.room(w=5.0, h=5.0, unknown_box=(0.0, 3.5, 5.0, 5.0))
    w, m = World(g, 2.5, 1.5, 0.0), ExploreMachine()
    m.start(0.0)
    t, states = w.run(m, 60.0)
    assert states[:4] == ['survey', 'plan', 'turn', 'drive'], states
    assert 'pause' in states and m.legs >= 1
    assert w.max_y > 2.5                     # it drove towards the unseen top part
    assert m.state not in ('error', 'idle'), m.message   # (map never fills in here, so no 'done')


def test_closed_room_finishes_after_one_leg():
    g = maps.room(w=5.0, h=5.0)
    w, m = World(g, 2.5, 2.5, 0.0), ExploreMachine()
    m.start(0.0)
    w.run(m, 90.0)
    assert m.state == 'done' and 'room scanned' in m.message, (m.state, m.message)
    assert m.coverage >= 0.9 and m.legs >= 1
    assert m.step(100.0, (w.x, w.y, w.yaw), g, 1.0, 1.0, []) is None   # stays stopped


def test_ultrasonic_within_one_foot_blocks_backs_up_and_replans():
    g = maps.room(w=5.0, h=5.0, unknown_box=(0.0, 3.5, 5.0, 5.0))
    w, m = World(g, 2.5, 1.5, 0.0), ExploreMachine()
    m.start(0.0)
    t, states = w.run(m, 60.0)
    assert 'drive' in states
    m._enter('drive', t, 'driving')                     # put it back in a leg
    m.target, m.leg, m.start_xy, m.last_xy = w.yaw, 1.5, (w.x, w.y), (w.x, w.y)
    w.us = FOOT - 0.01                                  # something 1 ft ahead
    cmd = m.step(t, (w.x, w.y, w.yaw), g, w.us, w.us, [])
    assert cmd == (0.0, 0.0) and m.state == 'blocked_wait' and 'ultrasonic' in m.message
    assert m.blocked[-1] == w.yaw
    w.us = None
    t, states = w.run(m, 5.0, t0=t + DT)
    assert states[:3] == ['blocked_wait', 'backup', 'pause'] and 'plan' in states, states


def test_exactly_one_foot_counts_as_blocked_but_just_over_does_not():
    m = ExploreMachine()
    assert m._front_blocked(FOOT, 2.0, [])
    assert not m._front_blocked(FOOT + 0.01, 2.0, [])


def test_lidar_point_close_ahead_or_touching_blocks():
    m = ExploreMachine()
    assert m._front_blocked(5.0, 5.0, [(0.4, 0.1)])           # 15 cm in front of the case
    assert m._front_blocked(5.0, 5.0, [(0.0, 0.30)])          # beside a wheel
    assert not m._front_blocked(5.0, 5.0, [(1.5, 0.0), (0.0, 1.0), (-1.0, 0.0)])


def test_not_turning_is_an_error():
    g = maps.room(w=5.0, h=5.0)
    w, m = World(g, 2.5, 2.5, 0.0, moves=False), ExploreMachine()
    m.start(0.0)
    w.run(m, 10.0)
    assert m.state == 'error' and 'not turning' in m.message


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


def test_survey_turns_in_bursts():
    m = ExploreMachine()
    m.start(0.0)
    cmds = [m.step(i * DT, (0, 0, i * 0.05), None, 2.0, 2.0, []) for i in range(14)]
    turning = [c for c in cmds if c[1] != 0.0]
    still = [c for c in cmds if c == (0.0, 0.0)]
    assert turning and still                            # both phases present
    assert all(c[0] == 0.0 for c in cmds)               # never drives forward while surveying
