"""Lap follower in a small simulated world (no ROS, no hardware)."""
import math

import pytest

from securebot_ros import explore_logic as ex
from securebot_ros.lap_follower import densify, LapFollower, simplify

DT = 0.1


class World:
    """Robot that moves as commanded; ultrasonics see `wall` (a point) when it is ahead.

    turn_rate: one-speed motors - any turn spins at this rate after a dead time (`dead` s),
    keeps turning for `coast` s after the command stops, and the reported pose lags the real
    one by `lag` s (measured on SecureBot with the gyro: ~0.6 s dead time, ~40 deg/s,
    ~8 deg of coasting)."""

    def __init__(self, x=0.0, y=0.0, yaw=0.0, turn_rate=None, dead=0.0, coast=0.0, lag=0.0,
                 veer=0.0, steer_lag=0.0):
        self.x, self.y, self.yaw = x, y, yaw
        self.veer = veer                  # rad/s the robot turns by itself while driving
        self.steer = [0.0] * (int(round(steer_lag / DT)) + 1)   # wheels answer this late
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
            if lin:
                self.steer = self.steer[1:] + [ang]
                ang = self.steer[0] + self.veer
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


def test_lane_hold_is_gentle_and_cancels_the_veer():
    """A 6 m lane with SecureBot's own veer (~6.5 deg/s at equal power) and wheels that answer
    0.3 s late: the robot stays within a few cm of the line, with small corrections that do
    not swing from one side to the other."""
    w = World(0.0, 0.05, 0.0, veer=math.radians(6.5), steer_lag=0.3, lag=0.1)
    f = LapFollower()
    f.start(0.0, [(0.0, 0.0), (6.0, 0.0)])
    t, offs, steers = 0.0, [], []
    while t < 60 and f.running:
        cmd = f.step(t, *w.sense())
        w.apply(cmd)
        t += DT
        if f.state == 'drive' and w.x > 1.5:      # after it has settled onto the lane
            offs.append(abs(w.y))
            steers.append(cmd[1])
    assert f.state == 'done', f.message
    assert max(offs) < 0.06
    assert max(abs(s) for s in steers) <= f.p.hold_max + 1e-9
    flips = sum(1 for a, b in zip(steers, steers[1:]) if a * b < 0)
    assert flips <= 2                             # not left-right-left


def test_stops_and_turns_at_corners_instead_of_curving():
    w = World(0.0, 0.0, 0.0, turn_rate=math.radians(40), dead=0.6, coast=0.2, lag=0.1)
    f = LapFollower()
    f.start(0.0, [(0, 0), (2, 0), (2, 0.45), (0, 0.45)])     # lane, hop, lane back
    t, turned_at = 0.0, []
    while t < 120 and f.running:
        before = f.state
        w.apply(f.step(t, *w.sense()))
        t += DT
        if before == 'drive' and f.state == 'turn':
            turned_at.append((round(w.x, 2), round(w.y, 2)))
    assert f.state == 'done', f.message
    assert len(turned_at) == 2
    for x, y in turned_at:                        # within ~15 cm of each corner
        assert min(math.dist((x, y), (2, 0)), math.dist((x, y), (2, 0.45))) < 0.15


def test_boxed_in_fails_the_path_instead_of_looping():
    f = LapFollower()
    f.start(0.0, [(0, 0), (-3, 0)])              # must turn round first ...
    near = [(0.2, 0.3), (-0.4, 0.0)]              # ... but too close to turn, wall right behind
    t = 0.0
    while t < 10 and f.running:
        f.step(t, (0.0, 0.0, 0.0), 0.2, 0.2, near)    # and something just ahead too
        t += DT
    assert f.state == 'failed' and 'boxed in' in f.message
    assert t < 3


def test_no_room_behind_creeps_forward_then_turns():
    """Parked with its back to a desk: no room to turn, none to back up, open ahead. It drives
    forward to make room, then turns and carries on."""
    w = World(0.0, 0.0, 0.0)
    f = LapFollower()
    f.start(0.0, [(0, 0), (0, 3)])                # lane to the left: a 90 deg turn first
    t, states = 0.0, set()
    while t < 60 and f.running:
        pose, us, _, _ = w.sense()
        desk = [] if w.x > 0.25 else [(-0.30 - w.x, 0.0), (-0.30 - w.x, 0.15)]   # behind
        w.apply(f.step(t, pose, us, us, desk))
        states.add(f.state)
        t += DT
    assert 'creep' in states
    assert f.state == 'done', f.message


def test_wiggly_transit_path_has_only_its_real_corners():
    """Nav2's path: 5 cm steps with a little noise along an L. Only the L's corner counts."""
    wiggly = [(x / 20, 0.02 * math.sin(x)) for x in range(41)] + \
             [(2.0 + 0.02 * math.sin(y), y / 20) for y in range(1, 31)]
    f = LapFollower()
    f.start(0.0, wiggly)
    assert len(f.corners) == 1
    assert len(simplify(wiggly, 0.08)) == 3


@pytest.mark.parametrize('coast', [0.1, 0.6])
def test_each_corner_is_one_spin_even_with_a_long_coast(coast):
    """Measured 2026-11-04: after a stop the robot free-wheeled ~24 deg (0.6 s at 41 deg/s)
    and the follower, expecting ~10 deg, overshot, then spun back: left-right-left at every
    corner. Now: one spin per corner, with a long coast (free-wheeling) or a short one
    (braking), and the coast it learns matches."""
    w = World(0.0, 0.0, 0.0, turn_rate=math.radians(41), dead=0.3, coast=coast, lag=0.1)
    f = LapFollower()
    f.start(0.0, [(0, 0), (2, 0), (2, 0.45), (-1, 0.45), (-1, 0.9), (2, 0.9)])
    t, last, reversals, spins = 0.0, 0, 0, 0
    while t < 200 and f.running:
        cmd = f.step(t, *w.sense())
        w.apply(cmd)
        t += DT
        sign = 0 if not cmd or cmd[0] or not cmd[1] else (1 if cmd[1] > 0 else -1)
        if sign and sign != last:
            spins += 1
            if last and f.spins > 1:
                reversals += 1
        last = sign or (0 if cmd and cmd[0] else last)
    assert f.state == 'done', f.message
    assert reversals == 0
    assert spins <= 4 + 1                         # 4 corners (+1 if the first one needs a top-up)
    assert abs(f.coast - coast) < 0.2


def test_a_spin_that_does_not_turn_counts_as_blocked():
    """Final run 2026-11-04: told to spin for 17 s, the robot (wedged against a box) did not
    turn. After spin_stall_time with no turning it backs off and skips instead."""
    w = World(0.0, 0.0, 0.0, turn_rate=math.radians(41), dead=0.3, coast=0.3)
    w.stuck = True                                # nothing moves
    f = LapFollower()
    f.start(0.0, [(0, 0), (0, 3)])                # needs a 90 deg turn first
    t = run(f, w, 20, until=lambda: 'not turning' in f.message)
    assert 'not turning' in f.message and t < 6
