"""Lap follower in a small simulated world (no ROS, no hardware)."""
import math

import numpy as np
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
                 veer=0.0, steer_lag=0.0, steer_gain=1.0):
        self.x, self.y, self.yaw = x, y, yaw
        self.steer_gain = steer_gain      # turn rate per unit of steering (real robot ~1.1)
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
                ang = self.steer_gain * self.steer[0] + self.veer
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
    cmds, msgs = [], []
    for t in range(10):
        cmds.append(f.step(t * DT, pose, us, us, chair))
        msgs.append((f.state, f.message))
    assert all(c[1] == 0.0 for c in cmds if c)
    assert ('backup', 'backing up to make room (too close to turn)') in msgs
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
    0.3 s late, from a cold start (no trim learnt yet): the robot stays within ~10 cm of the
    line (the trim learns slowly since 2026-11-06, as faster made it weave with the real
    ~1 s delay; it is kept for the next lanes), with corrections that do not swing from one
    side to the other."""
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
    assert max(offs) < 0.11
    assert max(abs(s) for s in steers) <= f.p.hold_total + 1e-9
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
    near = [(0.2, 0.3), (-0.33, 0.0)]             # ... too close to turn (even tight: 0.36 m)
    #                                               and a wall right behind (within rear_room)
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


class AmclWorld(World):
    """Map pose as AMCL gives it: jumps (gaussian, every 0.25 s) and a heading `crab` off the
    way the robot really travels; the odometry heading is the gyro's (smooth, own zero)."""

    def __init__(self, crab, jump_xy, jump_yaw, seed, **kw):
        import random
        super().__init__(**kw)
        self.crab, self.jxy, self.jyaw = crab, jump_xy, jump_yaw
        self.rng = random.Random(seed)
        self.err, self.t = (0.0, 0.0, 0.0), 0.0

    def sense(self):
        self.t += DT
        if self.t >= 0.25:
            self.t = 0.0
            g = self.rng.gauss
            self.err = (g(0, self.jxy), g(0, self.jxy), g(0, self.jyaw))
        (x, y, yaw), us, _, pts = super().sense()
        ex_, ey, eyaw = self.err
        return (x + ex_, y + ey, yaw - self.crab + eyaw), us, us, pts

    def odom_yaw(self):
        return self.history[0][2] - self.crab + 0.3

    def odom_xy(self):
        """Position in the odom frame: the true one, turned like odom_yaw (map->odom 0.3 rad)."""
        x, y, _ = self.history[0]
        c, s = math.cos(0.3), math.sin(0.3)
        return (c * x - s * y, s * x + c * y)


class OpenFloorWorld(AmclWorld):
    """Open floor: AMCL's position also wanders (a slow random walk of `wander` m, kept
    within ~2x) besides its quick jumps."""

    def __init__(self, wander, *a, **kw):
        super().__init__(*a, **kw)
        self.wander, self.drift, self.t2 = wander, [0.0, 0.0], 0.0

    def sense(self):
        g = self.rng.gauss
        for i in (0, 1):
            self.drift[i] = 0.98 * self.drift[i] + g(0, self.wander * 0.2)
        (x, y, yaw), us, _, pts = super().sense()
        return (x + self.drift[0], y + self.drift[1], yaw), us, us, pts


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_lane_hold_does_not_weave_on_amcl_jumps(seed):
    """AMCL pose jumping ~3 cm / 3 deg and reporting a heading 10 deg off the real direction
    of travel: on odometry heading + learnt crab the robot holds the next lane closely, and
    the steering does not swing left-right-left."""
    w = AmclWorld(math.radians(10), 0.03, math.radians(3), seed, veer=math.radians(6.5),
                  steer_lag=0.3, lag=0.1)
    f = LapFollower()
    for lane in ([(0.0, 0.0), (8.0, 0.0)], [(8.0, 0.0), (0.0, 0.0)]):
        f.start(0.0, lane)
        t, offs, steers = 0.0, [], []
        while t < 120 and f.running:
            pose, us, _, pts = w.sense()
            cmd = f.step(t, pose, us, us, pts, odom_yaw=w.odom_yaw())
            w.apply(cmd)
            t += DT
            if f.state == 'drive' and 2.0 < abs(w.x - lane[0][0]) < 7.0:   # after entry
                offs.append(abs(w.y))
                steers.append(cmd[1])
        assert f.state == 'done', f.message
    assert abs(math.degrees(f.crab) - 10) < 4         # learnt the crab
    assert max(offs) < 0.10                           # second lane
    flips = sum(1 for a, b in zip(steers, steers[1:]) if a * b < 0)
    assert flips <= 3


def back_on_line(seed):
    """(s from the start of driving until within 4 cm of the line for 3 s, furthest swing
    past it) for a robot starting a lane 15 cm off, with AMCL jumps, crab, veer and the
    real robot's steering delay (turns 1.0 s after the steering changes, x1.1)."""
    w = AmclWorld(math.radians(10), 0.03, math.radians(3), seed, veer=math.radians(6.5),
                  steer_lag=1.0, steer_gain=1.1, lag=0.1)
    f = LapFollower()
    f.start(0.0, [(0.0, 0.0), (8.0, 0.0)])
    t = 0.0
    while t < 120 and f.running:
        pose, us, _, pts = w.sense()
        w.apply(f.step(t, pose, us, us, pts, odom_yaw=w.odom_yaw()))
        t += DT
    w.y += 0.15
    f.start(0.0, [(8.0, 0.0), (0.0, 0.0)])
    t, track, t_drive = 0.0, [], None
    while t < 120 and f.running:
        pose, us, _, pts = w.sense()
        w.apply(f.step(t, pose, us, us, pts, odom_yaw=w.odom_yaw()))
        t += DT
        if f.state == 'drive':
            t_drive = t_drive or t
            track.append((t - t_drive, w.y))
    back = next((tt for i, (tt, _) in enumerate(track)
                 if all(abs(y) < 0.04 for _, y in track[i:i + 30])), math.inf)
    return back, -min(y for _, y in track)


def test_gets_back_onto_the_line_quickly():
    """Against AMCL jumps, a 10 deg crab, a 6.5 deg/s veer and the real ~1 s steering delay:
    back within 4 cm in ~11 s on average, no swing past the line of more than 25 cm (the
    gains before 2026-11-06 19:00 never settled with this delay: +-30-40 cm weave)."""
    runs = [back_on_line(seed) for seed in range(1, 7)]
    times = [b for b, _ in runs]
    assert sum(times) / len(times) < 13.0
    assert max(times) < 18.0
    assert max(swing for _, swing in runs) < 0.25


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_no_weave_with_the_real_steering_delay(seed):
    """Run 2026-11-06 18:30: SecureBot turns 1.0 s after the steering changes (x1.1), and the
    lane hold weaved +-20-30 cm every ~6 s. With that delay: a lane started 25 cm off and 10
    deg wrong swings at most 12 cm past the line, then stays within 6 cm of it."""
    w = AmclWorld(math.radians(5), 0.03, math.radians(3), seed, veer=math.radians(3),
                  steer_lag=1.0, steer_gain=1.1, lag=0.1)
    f = LapFollower()
    f.start(0.0, [(0.0, 0.0), (12.0, 0.0)])          # learn the trim and crab first
    t = 0.0
    while t < 200 and f.running:
        pose, us, _, pts = w.sense()
        w.apply(f.step(t, pose, us, us, pts, odom_yaw=w.odom_yaw()))
        t += DT
    w.y, w.yaw = 0.25, math.pi - math.radians(10)
    w.history = [(w.x, w.y, w.yaw)] * len(w.history)
    f.start(0.0, [(12.0, 0.0), (0.0, 0.0)])
    t, track = 0.0, []
    while t < 200 and f.running:
        pose, us, _, pts = w.sense()
        w.apply(f.step(t, pose, us, us, pts, odom_yaw=w.odom_yaw()))
        t += DT
        if f.state == 'drive':
            track.append((t, w.y))
    assert f.state == 'done'
    t0 = track[0][0]
    assert -min(y for _, y in track) < 0.12                        # swing past the line
    assert max(abs(y) for tt, y in track if tt > t0 + 15) < 0.06   # then stays on it


def test_steering_changes_smoothly():
    """No jolts: the lane-hold steering never changes faster than hold_slew."""
    w = AmclWorld(math.radians(10), 0.03, math.radians(3), 1, veer=math.radians(6.5),
                  steer_lag=0.3, lag=0.1)
    f = LapFollower()
    f.start(0.0, [(0.0, 0.0), (8.0, 0.0)])
    t, last, worst = 0.0, None, 0.0
    while t < 120 and f.running:
        pose, us, _, pts = w.sense()
        cmd = f.step(t, pose, us, us, pts, odom_yaw=w.odom_yaw())
        w.apply(cmd)
        t += DT
        if f.state == 'drive' and cmd and cmd[0] > 0:
            if last is not None:
                worst = max(worst, abs(cmd[1] - last))
            last = cmd[1]
        else:
            last = None
    assert worst <= f.p.hold_slew * DT + 1e-9
    assert f.state == 'done'


def test_settles_on_the_gyro_not_on_amcl_jumps():
    """Standing still with AMCL's heading jumping: the turn's settle check uses the gyro, so
    the first spin starts after settle_time, not after settle_max."""
    w = AmclWorld(0.0, 0.03, math.radians(3), 1, turn_rate=math.radians(40), dead=0.3,
                  coast=0.3)
    f = LapFollower()
    f.start(0.0, [(0.0, 0.0), (0.0, 3.0)])                # 90 deg to the left
    t = 0.0
    while t < 5 and f.phase != 'spin':
        pose, us, _, pts = w.sense()
        w.apply(f.step(t, pose, us, us, pts, odom_yaw=w.odom_yaw()))
        t += DT
    assert f.phase == 'spin' and t < f.p.settle_max


def test_corrections_spin_gently_and_step_up_if_they_do_not_move():
    """A small turn is a correction: requested at turn_speed_gentle (the bridge's gentle PWM);
    if the robot does not turn at that power, it goes on at full speed."""
    f = LapFollower()
    f.start(0.0, [(0.0, 0.0), (math.cos(0.35) * 3, math.sin(0.35) * 3)])   # 20 deg left
    t, cmds = 0.0, []
    while t < 4.0:
        cmd = f.step(t, (0.0, 0.0, 0.0), math.inf, math.inf, [], odom_yaw=0.0)   # never turns
        cmds.append((round(t, 1), cmd))
        t += DT
    spins = [c[1] for _, c in cmds if c and c[1]]
    assert spins[0] == pytest.approx(f.p.turn_speed_gentle)
    assert spins[-1] == pytest.approx(f.p.turn_speed)           # stepped up


def test_big_turns_spin_at_full_speed():
    f = LapFollower()
    f.start(0.0, [(0.0, 0.0), (0.0, 3.0)])                     # 90 deg left
    t, spin = 0.0, None
    while t < 2.0 and spin is None:
        cmd = f.step(t, (0.0, 0.0, 0.0), math.inf, math.inf, [], odom_yaw=0.0)
        spin = cmd[1] if cmd and cmd[1] else None
        t += DT
    assert spin == pytest.approx(f.p.turn_speed)


def test_boxed_in_tries_a_tighter_turn_first():
    """Something 0.38 m away (inside the normal 0.40 turn check, outside the 0.356 corners)
    and a wall right behind: turn with the tight radius instead of failing the path."""
    f = LapFollower()
    f.start(0.0, [(0.0, 0.0), (0.0, 3.0)])
    pts = [(0.0, -0.38), (-0.33, 0.20), (0.45, 0.0)]           # beside / behind / ahead
    states, t = [], 0.0
    while t < 3.0:
        cmd = f.step(t, (0.0, 0.0, 0.0), math.inf, math.inf, pts, odom_yaw=0.0)
        states.append(f.state)
        if cmd and cmd[1]:
            break
        t += DT
    assert 'failed' not in states and f.tight and cmd[1] != 0.0


def dist_to_polyline(pts, x, y):
    best = math.inf
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        dx, dy = bx - ax, by - ay
        t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)))
        best = min(best, math.hypot(x - ax - t * dx, y - ay - t * dy))
    return best


def test_square_corners_are_curves_not_spins():
    """90 deg corners with room round them: driven as curves (no stop and spin) that stay
    close to the rounded corner (0.5 m radius cuts ~0.21 m off the corner point)."""
    w = World(0.0, 0.0, 0.0, turn_rate=math.radians(40), dead=0.3, coast=0.2, lag=0.1,
              steer_lag=0.3)
    f = LapFollower()
    f.start(0.0, SQUARE)
    t, spins, states, worst = 0.0, 0, set(), 0.0
    while t < 200 and f.running:
        cmd = f.step(t, *w.sense())
        w.apply(cmd)
        t += DT
        states.add(f.state)
        if cmd and cmd[0] == 0.0 and cmd[1] != 0.0 and t > 5:   # after the first alignment
            spins += 1
        worst = max(worst, dist_to_polyline(SQUARE, w.x, w.y))
    assert f.state == 'done', f.message
    assert 'arc' in states and spins == 0
    assert worst < 0.30


def test_no_curve_when_something_is_beside_the_corner():
    f = LapFollower()
    f.start(0.0, [(0.0, 0.0), (3.0, 0.0), (3.0, 3.0)])
    f.state, f.index = 'drive', 22                          # 0.8 m before the corner
    pts = [(0.35, 0.45)]                                   # base_link: 0.30 m from the curve
    t = 0.0
    while t < 2 and f.state == 'drive':
        f.step(t, (2.55, 0.0, 0.0), math.inf, math.inf, pts)
        t += DT
    assert f.state == 'turn'


def test_curve_is_used_when_clear():
    f = LapFollower()
    f.start(0.0, [(0.0, 0.0), (3.0, 0.0), (3.0, 3.0)])
    f.state, f.index = 'drive', 22
    t = 0.0
    while t < 2 and f.state == 'drive':
        f.step(t, (2.55, 0.0, 0.0), math.inf, math.inf, [])
        t += DT
    assert f.state == 'arc'


def test_sharp_bends_still_stop_and_turn():
    f = LapFollower()
    f.start(0.0, [(0.0, 0.0), (3.0, 0.0), (0.5, 0.3)])      # ~173 deg
    assert f.arcs == {}


def test_wall_right_beside_the_robot_does_not_box_it_in():
    """A wall running along the robot 3 cm from its wheels (inside the 0.40 m turn check):
    no spin, but straight ahead and behind are clear, so it is not 'boxed in' - it creeps
    on, steering slightly away from the wall."""
    f = LapFollower()
    f.start(0.0, [(0.0, 0.0), (0.0, 3.0)])                     # wants to turn 90 deg left
    wall = [(x / 20.0, -0.29) for x in range(-12, 13)]         # along the right side
    states = []
    t = 0.0
    while t < 4.0:
        f.step(t, (0.0, 0.0, 0.0), math.inf, math.inf, wall, odom_yaw=0.0)
        states.append(f.state)
        t += DT
    assert 'failed' not in states
    assert 'backup' in states or 'creep' in states
    assert not f._front_blocked(math.inf, math.inf, np.array(wall))
    assert not f._rear_blocked(np.array(wall))


def test_trim_learns_the_veer_and_keeps_corrections_small():
    """Straight lane, robot veering 6.5 deg/s at equal power: the trim learns the steady
    power difference (~0.11 rad/s) and the steering then stays close to it."""
    w = World(0.0, 0.0, 0.0, veer=math.radians(6.5), steer_lag=0.3, lag=0.1)
    f = LapFollower()
    f.start(0.0, [(0.0, 0.0), (10.0, 0.0)])
    t, late = 0.0, []
    while t < 80 and f.running:
        pose, us, _, pts = w.sense()
        cmd = f.step(t, pose, us, us, pts, odom_yaw=pose[2])
        w.apply(cmd)
        t += DT
        if f.state == 'drive' and w.x > 4.0:
            late.append(cmd[1])
    assert f.state == 'done'
    assert f.hold_i == pytest.approx(-math.radians(6.5), abs=0.02)
    assert max(late) - min(late) < 0.06                        # small corrections only
    assert abs(w.y) < 0.05


def test_slows_down_before_the_end_and_before_a_spin_corner():
    f = LapFollower()
    f.start(0.0, [(0.0, 0.0), (3.0, 0.0), (0.5, 0.3)])      # sharp bend: a spin corner
    f.state = 'drive'
    speeds = {}
    for x in (1.0, 2.7):
        f.index = int(x / f.p.spacing)
        cmd = f.step(0.0, (x, 0.0, 0.0), math.inf, math.inf, [])
        speeds[x] = cmd[0]
    assert speeds[1.0] == f.p.drive_speed
    assert speeds[2.7] == f.p.approach_speed                 # 0.2 m before the corner


def test_back_up_is_short():
    f = LapFollower()
    f.start(0.0, [(0.0, 0.0), (5.0, 0.0)])
    f._enter('backup', 0.0, 'test')
    t, back = 0.0, 0
    while t < 3 and f.state == 'backup':
        cmd = f.step(t, (0.0, 0.0, 0.0), math.inf, math.inf, [])
        back += cmd is not None and cmd[0] < 0
        t += DT
    assert back * DT <= f.p.backup_time + 1e-9


def _drive_with_odom(f, w, path, seconds, rec=None, use_odom=True):
    f.start(0.0, path)
    t = 0.0
    while t < seconds and f.running:
        pose, us, _, pts = w.sense()
        cmd = f.step(t, pose, us, us, pts, odom_yaw=w.odom_yaw(),
                     odom_xy=w.odom_xy() if use_odom else None)
        w.apply(cmd)
        t += DT
        if rec is not None:
            rec.append((f.state, w.x, w.y, w.yaw))
    return f.state


def test_open_floor_lane_hold_steers_on_odometry_not_amcl_wander():
    """Open floor (2026-11-06): AMCL's position wanders several cm; the lane hold follows
    odometry pulled slowly towards AMCL, so the robot drifts off the line less."""
    def rms(use_odom):
        total = []
        for seed in (1, 2, 3):
            w = OpenFloorWorld(0.08, math.radians(5), 0.03, math.radians(3), seed,
                               veer=math.radians(3), steer_lag=0.6, steer_gain=1.2, lag=0.1)
            f = LapFollower()
            _drive_with_odom(f, w, [(0.0, 0.0), (12.0, 0.0)], 200, use_odom=use_odom)
            rec = []
            _drive_with_odom(f, w, [(12.0, 0.0), (0.0, 0.0)], 200, rec, use_odom=use_odom)
            total += [y * y for s, x, y, _ in rec if s == 'drive' and x < 9]
        return (sum(total) / len(total)) ** 0.5
    assert rms(True) < 0.9 * rms(False)


def test_off_the_line_rejoins_ahead_without_stopping_to_spin():
    """Pushed 50 cm off mid-lane: it angles back towards the line ahead (no stop-and-spin,
    never back along the path) and is back within 5 cm in under ~11 s."""
    w = AmclWorld(math.radians(5), 0.03, math.radians(3), 1, veer=math.radians(3),
                  steer_lag=0.6, steer_gain=1.2, lag=0.1)
    f = LapFollower()
    _drive_with_odom(f, w, [(0.0, 0.0), (12.0, 0.0)], 200)
    f.start(0.0, [(12.0, 0.0), (0.0, 0.0)])
    t, pushed, rec = 0.0, None, []
    while t < 200 and f.running:
        pose, us, _, pts = w.sense()
        w.apply(f.step(t, pose, us, us, pts, odom_yaw=w.odom_yaw(), odom_xy=w.odom_xy()))
        t += DT
        if pushed is None and w.x < 9 and f.state == 'drive':
            w.y += 0.5
            w.history = [(w.x, w.y, w.yaw)] * len(w.history)
            pushed = (t, w.x)
        if pushed:
            rec.append((t, f.state, w.x, w.y))
    assert f.state == 'done'
    assert 'turn' not in [s for _, s, _, _ in rec]
    assert max(x for _, _, x, _ in rec) <= pushed[1] + 0.02           # never went back
    back = next(tt for i, (tt, _, _, _) in enumerate(rec)
                if all(abs(y) < 0.05 for _, _, _, y in rec[i:i + 20])) - pushed[0]
    assert back < 11.0


@pytest.mark.parametrize('lag', [0.6, 1.0])
def test_curve_does_not_swing_past_the_exit(lag):
    """2026-11-06: a 100 deg curve swung ~105 deg past its exit (the robot answers the
    steering ~0.6 s late) and drove back along itself. Now it eases off by the predicted
    heading: at most ~15 deg past, no spin."""
    bend = math.radians(100)
    path = [(0.0, 0.0), (3.0, 0.0), (3.0 + 3.0 * math.cos(bend), 3.0 * math.sin(bend))]
    for seed in (1, 3):
        w = AmclWorld(math.radians(5), 0.03, math.radians(3), seed, veer=math.radians(3),
                      steer_lag=lag, steer_gain=1.2, lag=0.1, turn_rate=math.radians(41),
                      dead=0.3, coast=0.3)
        f = LapFollower()
        rec = []
        assert _drive_with_odom(f, w, path, 120, rec) == 'done'
        assert any(s == 'arc' for s, *_ in rec)
        past = max(ex.angle_diff(yaw, bend) for s, x, y, yaw in rec
                   if s in ('arc', 'drive') and x > 2.5)
        assert past < math.radians(20)


def test_slows_down_8_inches_before_a_waypoint():
    """2026-11-07: overshot waypoints in open space. Slow (approach speed, the bridge's gentle
    PWM) for the last 8 in before every waypoint, also ones it drives straight through."""
    w = World(0.0, 0.0, 0.0)
    f = LapFollower()
    f.start(0.0, [(0.0, 0.0), (6.0, 0.0)], waypoints=[(0.0, 0.0), (3.0, 0.0), (6.0, 0.0)])
    t, speeds = 0.0, []
    while t < 80 and f.running:
        pose, us, _, pts = w.sense()
        cmd = f.step(t, pose, us, us, pts, odom_yaw=pose[2])
        w.apply(cmd)
        t += DT
        if f.state == 'drive' and cmd[0] > 0:
            speeds.append((w.x, cmd[0]))
    slow = [x for x, v in speeds if v == f.p.approach_speed]
    assert any(2.75 < x < 3.0 for x in slow)              # before the middle waypoint
    assert all(not (1.0 < x < 2.7) for x in slow)         # full speed well before it
    assert any(3.1 < x < 5.0 for x, v in speeds if v == f.p.drive_speed)   # and after


def test_only_a_small_first_correction_is_halved():
    """2026-11-07: a 20 deg correction turns half its error (the lane hold takes the rest),
    but the follow-up spin of a big turn finishes the turn (halving made one corner three
    spins)."""
    f = LapFollower()
    f.start(0.0, [(0.0, 0.0), (math.cos(0.35) * 3, math.sin(0.35) * 3)])   # 20 deg left
    t = 0.0
    while t < 1.0 and f.spins == 0:
        f.step(t, (0.0, 0.0, 0.0), math.inf, math.inf, [], odom_yaw=0.0)
        t += DT
    assert f.spins == 1 and f.spin_keep == pytest.approx(0.35 * (1 - f.p.gentle_part), abs=0.02)
    f = LapFollower()
    f.start(0.0, [(0.0, 0.0), (0.0, 3.0)])                                 # 90 deg left
    f.spins = 1                                   # as after the first spin of the turn
    f._phase('settle', 0.0)
    t = 0.0
    while t < 2.0 and f.spins < 2:
        f.step(t, (0.0, 0.0, math.radians(60)), math.inf, math.inf, [],
               odom_yaw=math.radians(60))         # 30 deg still to go: a gentle follow-up
        t += DT
    assert f.spins == 2 and f.gentle and f.spin_keep == 0.0
