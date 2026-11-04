"""Follow one lap with SecureBot's one-speed drive (no ROS): decides, every tick, what to send.

SecureBot's motors only start at a high PWM, so every wheel that turns runs at about the same
speed: the robot cannot creep or turn gently, and Nav2's controllers (which expect fine speed
control) overshoot and swing about. This follower drives the way the explore mode does:

  drive: pure pursuit at one speed - steer towards the point `lookahead` m further along the
         path (a curve: the bridge runs the outer wheel, slows the inner one)
  turn:  when that point is more than `turn_start` off the heading (sharp corner, start of a
         lap), stop and turn on the spot, closed loop on the heading (gyro, via odometry):
         turn until the error is down to what the robot coasts after a stop (spin rate x
         `coast_time`), stop, let it settle (`settle_time`), check, and repeat until within
         `turn_tolerance`. Measured on SecureBot: ~0.6 s before it starts turning, ~40 deg/s,
         ~8 deg of coasting, still ~0.4 s after the stop
  blocked (ultrasonic or LiDAR: object within 1 ft ahead): stop and wait `block_wait` s for it
         to go away (a person walking past), else back up, skip `skip` m along the path and
         turn towards it; after `max_blocks` blocks on one path the path counts as failed
  done:  within `goal_tolerance` of the path's end
  error: pose or ultrasonics silent, or the robot does not move / turn when told to
"""
import math

import numpy as np

from securebot_ros import explore_logic as ex
from securebot_ros.explore_machine import FOOT


class Params:
    half_length = 0.2475          # case, centre to front/back edge (m)
    half_width = 0.256            # wheels, centre to outer face (m)
    margin = 0.05
    stop_distance = FOOT          # ultrasonic / LiDAR stop distance from the front edge
    drive_speed = 0.2             # m/s requested (the bridge lifts it to the start power)
    turn_speed = 0.8              # rad/s requested for turning on the spot
    steer_gain = 1.5              # rad/s per rad of heading error to the lookahead point
    max_steer = 0.5               # rad/s while driving
    lookahead = 0.6               # m along the path
    turn_start = math.radians(50)     # heading error that stops the robot to turn on the spot
    turn_tolerance = math.radians(10)
    coast_time = 0.25             # s of spin rate the robot keeps turning after a stop
    min_lead = math.radians(3)    # stop at least this far before the target
    turn_radius = 0.40            # m: LiDAR points this close block turning (corners: 0.356)
    turn_front = 0.12             # m: ultrasonic reading below this blocks turning (front corners)
    settle_time = 0.4             # s stopped before checking the heading
    turn_timeout = 30.0           # s on one turn -> give up on this stretch of path
    spacing = 0.1                 # m between path points after densifying
    search = 1.5                  # m along the path searched for the robot's nearest point
    goal_tolerance = 0.3          # m from the path's end
    block_wait = 3.0              # s to wait for an obstacle to clear
    backup_time = 1.5             # s reversing (the motors need ~0.6 s to get going)
    corner_turn = math.radians(25)    # blocked while the path turns away this much: turn
    max_room_tries = 3            # backing up to make room for a turn, before skipping on
    skip = 1.0                    # m along the path to skip past an obstacle
    max_blocks = 4
    progress_window = 5.0         # s of commanded motion without progress -> blocked / error
    min_progress_drive = 0.05
    sensor_timeout = 2.0


def densify(points, spacing):
    """Path points no more than `spacing` apart (straight lines in between)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / spacing)))
        out.extend(a + (b - a) * (i / n) for i in range(1, n + 1))
    return np.array(out)


class LapFollower:
    def __init__(self, params=None):
        self.p = params or Params()
        self.state, self.message = 'idle', 'waiting'
        self.path = None
        self.index = 0                 # nearest path point reached so far (never goes back)
        self.blocks = 0
        self.room_tries = 0
        self.entered = 0.0
        self.turn_sign = 1.0
        self._reset_timers()
        self._phase('settle', 0.0)

    # ---------------------------------------------------------------- commands
    def start(self, now, path):
        """Follow `path` [(x, y), ...] in the map frame from its start."""
        self.path = densify(path, self.p.spacing)
        self.index, self.blocks, self.room_tries = 0, 0, 0
        self._enter('turn', now, 'turning onto the path')

    def stop(self, now, message='stopped'):
        self._enter('idle', now, message)

    @property
    def running(self):
        return self.state in ('turn', 'drive', 'blocked', 'backup')

    @property
    def remaining(self):
        """Path length left (m), or None."""
        if self.path is None:
            return None
        rest = self.path[self.index:]
        return float(np.sum(np.linalg.norm(np.diff(rest, axis=0), axis=1))) if len(rest) > 1 \
            else 0.0

    # ---------------------------------------------------------------- one control tick
    def step(self, now, pose, us_left, us_right, scan_points):
        """pose (x, y, yaw) in the map or None; ultrasonic ranges in m (inf = nothing) or None
        if silent; scan_points [(x, y)] in base_link. Returns (linear, angular) or None."""
        if not self.running:
            return None
        if pose is None or us_left is None or us_right is None:
            if self.sensors_lost_since is None:
                self.sensors_lost_since = now
            if now - self.sensors_lost_since > self.p.sensor_timeout:
                missing = [n for n, v in (('map pose', pose), ('left ultrasonic', us_left),
                                          ('right ultrasonic', us_right)) if v is None]
                self._enter('error', now, 'no data from ' + ', '.join(missing))
            return (0.0, 0.0)
        self.sensors_lost_since = None
        x, y, yaw = pose
        points = np.asarray(scan_points, dtype=float).reshape(-1, 2)
        self._advance(x, y)
        if math.hypot(*(self.path[-1] - (x, y))) <= self.p.goal_tolerance and \
                self.index >= len(self.path) - 1 - int(self.p.lookahead / self.p.spacing):
            self._enter('done', now, 'path finished')
            return (0.0, 0.0)
        return getattr(self, '_' + self.state)(now, x, y, yaw, us_left, us_right, points)

    # ---------------------------------------------------------------- states
    def _turn(self, now, x, y, yaw, us_left, us_right, points):
        p = self.p
        if now - self.entered > p.turn_timeout:
            return self._give_way(now, 'turn did not finish')
        err = self._heading_error(x, y, yaw)
        rate = 0.0
        if self.last_turn_pose is not None and now > self.last_turn_pose[0]:
            rate = ex.angle_diff(yaw, self.last_turn_pose[1]) / (now - self.last_turn_pose[0])
        self.last_turn_pose = (now, yaw)
        if self.phase == 'settle':
            if now - self.phase_start < p.settle_time:
                return (0.0, 0.0)
            if abs(err) <= p.turn_tolerance:
                self._enter('drive', now, 'following the path')
                return (0.0, 0.0)
            self.turn_sign = 1.0 if err > 0 else -1.0
            self._phase('spin', now)
        elif self.turn_sign * err <= max(p.min_lead, abs(rate) * p.coast_time):
            self._phase('settle', now)          # close enough: the coast does the rest
            return (0.0, 0.0)
        why = self._turn_blocked(us_left, us_right, points)
        if why:
            self.room_tries += 1
            if self.room_tries > p.max_room_tries:
                return self._give_way(now, why)
            self._enter('backup', now, f'backing up to make room ({why})')
            return (0.0, 0.0)
        return (0.0, self.turn_sign * p.turn_speed)

    def _drive(self, now, x, y, yaw, us_left, us_right, points):
        err = self._heading_error(x, y, yaw)
        why = self._front_blocked(us_left, us_right, points)
        if why:
            if abs(err) > self.p.corner_turn and not self._turn_blocked(us_left, us_right, points):
                self._enter('turn', now, 'turning at a corner')    # the wall ahead is the corner
            else:
                self._enter('blocked', now, f'waiting: {why}')
            return (0.0, 0.0)
        if abs(err) > self.p.turn_start:
            self._enter('turn', now, 'turning at a corner')
            return (0.0, 0.0)
        if self.last_xy is None:
            self.last_xy = (x, y)
        if math.hypot(x - self.last_xy[0], y - self.last_xy[1]) >= self.p.min_progress_drive:
            self.last_xy, self.progress_time = (x, y), now
        elif now - self.progress_time > self.p.progress_window:
            return self._give_way(now, 'not moving')
        steer = max(-self.p.max_steer, min(self.p.max_steer, self.p.steer_gain * err))
        return (self.p.drive_speed, steer)

    def _blocked(self, now, x, y, yaw, us_left, us_right, points):
        if not self._front_blocked(us_left, us_right, points):
            self._enter('drive', now, 'following the path')
            return (0.0, 0.0)
        if now - self.entered >= self.p.block_wait:
            return self._give_way(now, 'obstacle stayed')
        return (0.0, 0.0)

    def _backup(self, now, x, y, yaw, us_left, us_right, points):
        if now - self.entered < self.p.backup_time and not self._rear_blocked(points):
            return (-self.p.drive_speed, 0.0)
        self._enter('turn', now, 'going round the obstacle')
        return (0.0, 0.0)

    # ---------------------------------------------------------------- helpers
    def _give_way(self, now, why):
        self.blocks += 1
        if self.blocks > self.p.max_blocks:
            self._enter('failed', now, f'blocked {self.blocks} times ({why})')
            return (0.0, 0.0)
        self.room_tries = 0
        self.index = min(len(self.path) - 1, self.index + int(self.p.skip / self.p.spacing))
        self._enter('backup', now, f'backing up ({why}), skipping {self.p.skip:.1f} m')
        return (0.0, 0.0)

    def _advance(self, x, y):
        """Move `index` to the nearest path point within `search` m ahead of it."""
        end = min(len(self.path), self.index + int(self.p.search / self.p.spacing) + 1)
        seg = self.path[self.index:end]
        self.index += int(np.argmin(np.hypot(seg[:, 0] - x, seg[:, 1] - y)))

    def _heading_error(self, x, y, yaw):
        k = min(len(self.path) - 1, self.index + int(self.p.lookahead / self.p.spacing))
        tx, ty = self.path[k]
        if math.hypot(tx - x, ty - y) < 0.05:
            return 0.0
        return ex.angle_diff(math.atan2(ty - y, tx - x), yaw)

    def _enter(self, state, now, message):
        self.state, self.message, self.entered = state, message, now
        self._reset_timers(now)
        self._phase('settle', now)            # a turn starts by letting the pose settle

    def _phase(self, phase, now):
        self.phase, self.phase_start = phase, now

    def _reset_timers(self, now=0.0):
        self.last_turn_pose = None
        self.progress_time = now
        self.last_xy = None
        self.sensors_lost_since = None

    def _front_blocked(self, us_left, us_right, points):
        if min(us_left, us_right) <= self.p.stop_distance:
            side = 'left' if us_left <= us_right else 'right'
            return f'{side} ultrasonic: object within 1 ft'
        if len(points) == 0:
            return None
        hl, hw = self.p.half_length, self.p.half_width + self.p.margin
        px, py = points[:, 0], points[:, 1]
        if np.any((np.abs(py) <= hw) & (px >= 0.0) & (px <= hl + self.p.stop_distance)):
            return 'LiDAR: object within 1 ft ahead'
        return None

    def _turn_blocked(self, us_left, us_right, points):
        """Why turning on the spot would hit something, or None."""
        if min(us_left, us_right) < self.p.turn_front:
            return 'too close in front to turn'
        if len(points) and np.any(np.hypot(points[:, 0], points[:, 1]) < self.p.turn_radius):
            return 'too close to turn'
        return None

    def _rear_blocked(self, points):
        if len(points) == 0:
            return False
        hl, hw = self.p.half_length, self.p.half_width + self.p.margin
        return bool(np.any((points[:, 0] >= -(hl + 0.25)) & (points[:, 0] <= 0.0)
                           & (np.abs(points[:, 1]) <= hw)))
