"""Lap follower in a small simulated world (no ROS, no hardware)."""
import math

import pytest

from securebot_ros import explore_logic as ex
from securebot_ros.lap_follower import densify, LapFollower

DT = 0.1


class World:
    """Robot that moves as commanded; ultrasonics see `wall` (a point) when it is ahead.

    turn_rate: one-speed motors - any turn spins at this rate after a dead time (`dead` s),
    keeps turning for `coast` s after the command stops, and the reported pose lags the real
    one by `lag` s (measured on SecureBot with the gyro: ~0.6 s dead time, ~40 deg/s,
    ~8 deg of coasting)."""

    def __init__(self, x=0.0, y=0.0, yaw=0.0, turn_rate=None, dead=0.0, coast=0.0, lag=0.0):
        self.x, self.y, self.yaw = x, y, yaw
        self.turn_rate, self.dead, self.coast = turn_rate, dead, coast
        self.turning_for = 0.0
        self.coasting = 0.0               # s of coasting left
        self.spin_dir = 0.0
        self.history = [(x, y, yaw)] * (int(round(lag / DT)) + 1)
        self.wall = None
        self.stuck = False

    def sense(self):
        us = math.inf
        if self.wall is not None:
            dx, dy = self.wall[0] - self.x, self.wall[1] - self.y
            if abs(ex.angle_diff(math.atan2(dy, dx), self.yaw)) < math.radians(15):
                us = math.hypot(dx, dy) - 0.2475
        return self.history[0], us, us, []

    def apply(self, cmd):
        lin, ang = cmd if cmd and not self.stuck else (0.0, 0.0)
        if self.turn_rate and (ang or self.coasting) and not lin:
            if ang:
                self.turning_for += DT
                self.spin_dir = math.copysign(1.0, ang)
                spinning = self.turning_for > self.dead
                self.coasting = self.coast if spinning else 0.0
            else:                         # command stopped: the robot coasts on
                self.turning_for = 0.0
                spinning = self.coasting > 0
                self.coasting = max(0.0, self.coasting - DT)
            ang = self.spin_dir * self.turn_rate if spinning else 0.0
        else:
            self.turning_for = self.coasting = 0.0
        self.yaw = ex.angle_diff(self.yaw + ang * DT, 0.0)
        self.x += math.cos(self.yaw) * lin * DT
        self.y += math.sin(self.yaw) * lin * DT
        self.history = self.history[1:] + [(self.x, self.y, self.yaw)]


def run(f, w, seconds, until=None):
    t = 0.0
    while t < seconds and f.running:
        w.apply(f.step(t, *w.sense()))
        t += DT
        if until and until():
            break
    return t


SQUARE = [(0, 0), (3, 0), (3, 2), (0, 2), (0, 0)]


def test_densify():
    pts = densify([(0, 0), (1, 0)], 0.1)
    assert len(pts) == 11 and tuple(pts[-1]) == (1.0, 0.0)


def test_drives_a_square_lap_and_stays_near_it():
    w = World(0.0, 0.0, 0.0)
    f = LapFollower()
    f.start(0.0, SQUARE)
    worst = 0.0
    t = 0.0
    while t < 200 and f.running:
        w.apply(f.step(t, *w.sense()))
        t += DT
        worst = max(worst, min(abs(w.y), abs(w.y - 2), abs(w.x), abs(w.x - 3)))
    assert f.state == 'done', f.message
    assert worst < 0.35                       # never more than 35 cm off the lap
    assert f.remaining < 0.5


def test_starts_facing_the_wrong_way_turns_first():
    w = World(0.0, 0.0, math.pi)              # facing away from the lap
    f = LapFollower()
    f.start(0.0, SQUARE)
    assert f.step(0.0, *w.sense())[0] == 0.0  # turns on the spot, no driving
    run(f, w, 200)
    assert f.state == 'done'


@pytest.mark.parametrize('start_yaw', [math.pi / 2, -math.pi / 2, math.pi, math.radians(30)])
def test_one_speed_spin_lands_the_turn(start_yaw):
    """SecureBot's spin as measured with the gyro: nothing for 0.6 s, then ~40 deg/s whatever
    is asked, ~0.2 s of coasting after the stop, heading ~0.1 s late. The closed-loop turn
    must land within tolerance without swinging back and forth."""
    w = World(0.0, 0.0, start_yaw, turn_rate=math.radians(40), dead=0.6, coast=0.2, lag=0.1)
    f = LapFollower()
    f.start(0.0, SQUARE)
    reversals, last_sign = 0, 0
    t = 0.0
    while t < 30 and f.state != 'drive':
        cmd = f.step(t, *w.sense())
        sign = 0 if not cmd or not cmd[1] else (1 if cmd[1] > 0 else -1)
        if sign and last_sign and sign != last_sign:
            reversals += 1
        last_sign = sign or last_sign
        w.apply(cmd)
        t += DT
    assert f.state == 'drive', f.message
    assert abs(ex.angle_diff(w.yaw, 0.0)) < math.radians(20)
    assert reversals <= 1
    assert t < 15


def test_waits_for_a_person_then_carries_on():
    w = World(0.0, 0.0, 0.0)
    w.wall = (1.0, 0.0)                       # someone 0.5 m in front of the bumper ... closer
    f = LapFollower()
    f.start(0.0, SQUARE)
    run(f, w, 20, until=lambda: f.state == 'blocked')
    assert f.state == 'blocked'
    w.wall = None                             # ... and walks away
    run(f, w, 200)
    assert f.state == 'done' and f.blocks == 0


def test_obstacle_that_stays_is_skipped():
    w = World(0.0, 0.0, 0.0)
    w.wall = (1.5, 0.0)
    f = LapFollower()
    f.start(0.0, SQUARE)
    run(f, w, 30, until=lambda: f.state == 'backup')
    assert f.state == 'backup' and f.blocks == 1
    w.wall = None
    run(f, w, 200)
    assert f.state == 'done'


def test_stuck_robot_fails_the_path():
    w = World(0.0, 0.0, 0.0)
    w.stuck = True                            # wheels spin, robot does not move
    f = LapFollower()
    f.start(0.0, [(0, 0), (5, 0)])
    w.yaw = 0.0
    run(f, w, 120)
    assert f.state in ('failed', 'error')


def test_silent_sensors_stop_with_error():
    f = LapFollower()
    f.start(0.0, SQUARE)
    assert f.step(0.0, None, 1.0, 1.0, []) == (0.0, 0.0)
    f.step(3.0, None, 1.0, 1.0, [])
    assert f.state == 'error' and 'map pose' in f.message


def test_does_not_turn_on_the_spot_next_to_something():
    w = World(0.0, 0.0, math.pi)              # must turn round to start the lap ...
    f = LapFollower()
    f.start(0.0, SQUARE)
    pose, us, _, _ = w.sense()
    chair = [(-0.1, 0.35)]                    # ... but a chair leg is inside the turning circle
    cmds = [f.step(t * DT, pose, us, us, chair) for t in range(10)]
    assert all(c[1] == 0.0 for c in cmds if c)
    assert f.state == 'backup' and 'too close to turn' in f.message
    assert f.index == 0                       # making room does not skip any of the path


def test_wall_ahead_at_a_corner_means_turn_not_wait():
    w = World(1.6, 0.0, 0.0)                  # near the first corner of the square, wall ahead
    f = LapFollower()
    f.start(0.0, SQUARE[1:])                  # path starts at (3, 0) and turns left there
    w.x, f.index = 2.75, 0
    pose, _, _, _ = w.sense()
    f.state = 'drive'
    f.step(0.0, (2.75, 0.0, 0.0), 0.25, 0.25, [])
    assert f.state == 'turn'
