"""Follow one lap with SecureBot's one-speed drive (no ROS): decides, every tick, what to send.

SecureBot's motors only start at a high PWM, so every wheel that turns runs at about the same
speed: the robot cannot creep or turn gently, and Nav2's controllers (which expect fine speed
control) overshoot and swing about. This follower drives the way the explore mode does:

  drive: hold the straight stretch of path the robot is on (lane hold): steer to the
         stretch's heading, bent by up to `hold_lat_max` towards the line when the robot is
         off it, with small, smooth corrections (PID on the gyro heading, rate limited): the
         bridge slows the inner wheel a little instead of nearly stopping it. The integral
         learns the robot's own veer (~6.5 deg/s at equal power) and keeps it across paths
  corner: `corner_reach` m before a bend of `corner_angle` or more, stop and turn on the spot
         to the next stretch, instead of curving round it (curves need big wheel differences
         and swing the front towards the walls, where the 1 ft rule stops the robot)
  turn:  at a corner, at the start of a path, or when the path is more than `turn_start` off
         the heading, stop and turn on the spot, closed loop on the heading (gyro, via odometry):
         turn until the error is down to what the robot coasts after a stop (spin rate x the
         coast time, learned from every spin), stop (the firmware brakes), wait until the gyro
         says it has stopped turning, check. One turn is meant to be one spin: after a spin, a
         miss within `spin_tolerance` (or within `no_reverse` the other way) is left to the
         lane hold. Measured on SecureBot (2026-11-04, free-wheeling stop): ~0.3 s before it
         starts turning, ~41 deg/s, ~24 deg of coasting
  blocked (ultrasonic or LiDAR: object within 1 ft ahead): stop and wait `block_wait` s for it
         to go away (a person walking past), else back up, skip `skip` m along the path and
         turn towards it; blocked more than `max_blocks` times on one path: the path failed
  creep: too close to turn and no room to back up: drive forward `creep_time` s if the way
         ahead is clear, then turn (at most `max_creeps` times per path)
  boxed in: blocked ahead and behind as well: the path counts as failed at once
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
    lookahead = 0.6               # m along the path (turn target, big-error check)
    corner_angle = math.radians(20)   # path bend that is a stop-and-turn corner
    corner_reach = 0.10           # m before the corner to stop and turn
    hold_kp = 0.8                 # rad/s per rad of heading error (lane hold)
    hold_ki = 0.25                # rad/s per rad*s: learns the robot's veer
    hold_i_max = 0.12             # rad/s: most the integral may add (veer ~0.11 rad/s)
    hold_kd = 0.3                 # rad/s per rad/s of measured turn rate (damping)
    hold_lat = 1.5                # rad of heading bend per m off the line ...
    hold_lat_max = math.radians(12)   # ... at most
    hold_deadband = math.radians(1.5)
    hold_max = 0.18               # rad/s: biggest correction (inner wheel ~460 vs 650)
    hold_slew = 0.5               # rad/s per s: how fast the correction may change
    turn_start = math.radians(50)     # heading error that stops the robot to turn on the spot
    turn_tolerance = math.radians(10)     # no spin at all if the heading is this close
    spin_tolerance = math.radians(15)     # after a spin: close enough to drive on
    no_reverse = math.radians(25)         # after a spin: never spin back for less than this
    coast_time = 0.35             # s of spin rate the robot keeps turning after a stop (start
    #                               value; learned from every spin, see _learn_coast)
    coast_max = 0.8
    learn_min_rate = math.radians(15)     # rad/s at the stop for a spin to count for learning
    still_rate = math.radians(3)  # rad/s below which the robot counts as no longer turning
    still_time = 0.15             # s it must stay below still_rate
    settle_max = 1.5              # s: check anyway after this long
    spin_stall_time = 3.0         # s spinning with less than spin_stall_angle turned: wedged
    spin_stall_angle = math.radians(5)
    min_lead = math.radians(3)    # stop at least this far before the target
    turn_radius = 0.40            # m: LiDAR points this close block turning (corners: 0.356)
    turn_front = 0.12             # m: ultrasonic reading below this blocks turning (front corners)
    settle_time = 0.25            # s stopped, at least, before checking the heading
    turn_timeout = 30.0           # s on one turn -> give up on this stretch of path
    spacing = 0.1                 # m between path points after densifying
    search = 1.5                  # m along the path searched for the robot's nearest point
    goal_tolerance = 0.3          # m from the path's end
    block_wait = 3.0              # s to wait for an obstacle to clear
    backup_time = 1.5             # s reversing (the motors need ~0.6 s to get going)
    corner_turn = math.radians(25)    # blocked while the path turns away this much: turn
    max_room_tries = 3            # backing up to make room for a turn, before skipping on
    creep_time = 1.5              # s forward when there is no room behind (motors: ~0.6 s dead)
    max_creeps = 2                # per path
    skip = 1.0                    # m along the path to skip past an obstacle
    max_blocks = 1                # skip past an obstacle once; blocked a 2nd time: give up
    progress_window = 5.0         # s of commanded motion without progress -> blocked / error
    min_progress_drive = 0.05
    sensor_timeout = 2.0


# m: wiggles smaller than this are straightened out before following, so Nav2's transit paths
# (many short steps) do not count as corners
STRAIGHTEN = 0.08


def simplify(points, tol):
    """Open polyline with every bend that strays less than `tol` m from a straight line
    removed (Douglas-Peucker); the ends are kept."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < 3 or tol <= 0:
        return pts
    a, b = pts[0], pts[-1]
    d = b - a
    n = math.hypot(*d)
    rel = pts - a
    dist = (np.abs(d[0] * rel[:, 1] - d[1] * rel[:, 0]) / n if n > 1e-9
            else np.hypot(rel[:, 0], rel[:, 1]))
    k = int(np.argmax(dist))
    if dist[k] <= tol:
        return np.array([a, b])
    return np.vstack([simplify(pts[:k + 1], tol)[:-1], simplify(pts[k:], tol)])


def densify(points, spacing):
    """Path points no more than `spacing` apart (straight lines in between)."""
    return _densify(points, spacing)[0]


def _densify(points, spacing):
    """(dense points, the stretch each dense point belongs to, the stretches [(a, b)]).
    Repeated points are dropped; dense point k lies on stretch seg[k] (0 = first)."""
    raw = simplify(np.asarray(points, dtype=float).reshape(-1, 2), STRAIGHTEN)
    pts = [raw[0]]
    for q in raw[1:]:
        if np.linalg.norm(q - pts[-1]) > 1e-6:
            pts.append(q)
    pts = np.array(pts)
    out, seg = [pts[0]], [0]
    for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / spacing)))
        out.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
        seg.extend([i] * n)
    stretches = list(zip(pts[:-1], pts[1:])) or [(pts[0], pts[0])]
    return np.array(out), np.array(seg), stretches


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
        self.hold_i = 0.0              # lane-hold integral (rad/s): kept, it is the robot's veer
        self.corners, self.done_corner, self.creeps = [], -1, 0
        self.coast = self.p.coast_time         # learned coast after a spin (s of spin rate)
        self._reset_timers()
        self._phase('settle', 0.0)

    # ---------------------------------------------------------------- commands
    def start(self, now, path):
        """Follow `path` [(x, y), ...] in the map frame from its start."""
        self.path, self.seg, self.stretches = _densify(path, self.p.spacing)
        heads = [math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in self.stretches]
        # dense index of every bend of corner_angle or more (the first point of the next stretch)
        firsts = {int(s): k for k, s in reversed(list(enumerate(self.seg)))}
        self.corners = [firsts[i + 1] - 1 for i in range(len(heads) - 1)
                        if i + 1 in firsts
                        and abs(ex.angle_diff(heads[i + 1], heads[i])) >= self.p.corner_angle]
        self.index, self.blocks, self.room_tries = 0, 0, 0
        self.done_corner, self.creeps = -1, 0
        self._enter('turn', now, 'turning onto the path')

    def stop(self, now, message='stopped'):
        self._enter('idle', now, message)

    @property
    def running(self):
        return self.state in ('turn', 'drive', 'blocked', 'backup', 'creep')

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
        err = (ex.angle_diff(self.turn_heading, yaw) if self.turn_heading is not None
               else self._heading_error(x, y, yaw))
        rate = 0.0
        if self.last_turn_pose is not None and now > self.last_turn_pose[0]:
            rate = ex.angle_diff(yaw, self.last_turn_pose[1]) / (now - self.last_turn_pose[0])
        self.last_turn_pose = (now, yaw)
        if self.phase == 'settle':
            # wait until the robot has really stopped turning (gyro), not a fixed time: a
            # check while it still coasts sees the target on the wrong side and spins back
            waited = now - self.phase_start
            if abs(rate) > p.still_rate:
                self.still_since = None
            elif self.still_since is None:
                self.still_since = now
            still = self.still_since is not None and now - self.still_since >= p.still_time
            if waited < p.settle_max and not (still and waited >= p.settle_time):
                return (0.0, 0.0)
            self._learn_coast(yaw)
            # one turn = one spin: after a spin, a small miss either way is left to the
            # lane-hold steering, and the robot never spins back for less than no_reverse
            back = self.spins > 0 and (err > 0) != (self.turn_sign > 0)
            if abs(err) <= (p.turn_tolerance if self.spins == 0 else p.spin_tolerance) or \
                    (back and abs(err) <= p.no_reverse):
                self._enter('drive', now, 'following the path')
                return (0.0, 0.0)
            self.turn_sign = 1.0 if err > 0 else -1.0
            self.spins += 1
            self.spin_from = yaw
            self._phase('spin', now)
        elif self.turn_sign * err <= max(p.min_lead, abs(rate) * self.coast):
            self.stopped_at = (yaw, abs(rate))  # close enough: the coast does the rest
            self._phase('settle', now)
            return (0.0, 0.0)
        if self.phase == 'spin' and now - self.phase_start > p.spin_stall_time and \
                abs(ex.angle_diff(yaw, self.spin_from)) < p.spin_stall_angle:
            return self._give_way(now, 'not turning')     # wedged against something
        why = self._turn_blocked(us_left, us_right, points)
        if why:
            self.room_tries += 1
            if self.room_tries > p.max_room_tries:
                return self._give_way(now, why)
            self._enter('backup', now, f'backing up to make room ({why})')
            self.backing_from_turn = True
            return (0.0, 0.0)
        return (0.0, self.turn_sign * p.turn_speed)

    def _learn_coast(self, yaw):
        """After a spin: how long (s of spin rate) the robot kept turning after the stop.
        Feeds the next stop's lead; kept across paths (it depends on floor and battery)."""
        if self.stopped_at is None:
            return
        yaw0, rate = self.stopped_at
        self.stopped_at = None
        if rate < self.p.learn_min_rate:
            return
        went = self.turn_sign * ex.angle_diff(yaw, yaw0)
        seen = max(0.0, min(self.p.coast_max, went / rate))
        self.coast = 0.5 * self.coast + 0.5 * seen

    def _drive(self, now, x, y, yaw, us_left, us_right, points):
        err = self._heading_error(x, y, yaw)
        why = self._front_blocked(us_left, us_right, points)
        if why:
            if abs(err) > self.p.corner_turn and not self._turn_blocked(us_left, us_right, points):
                self._enter('turn', now, 'turning at a corner')    # the wall ahead is the corner
            else:
                self._enter('blocked', now, f'waiting: {why}')
            return (0.0, 0.0)
        corner = self._corner_due(x, y)
        if corner is not None:
            self.done_corner = corner
            self.index = max(self.index, corner)
            a, b = self.stretches[int(self.seg[min(len(self.path) - 1, corner + 1)])]
            self._enter('turn', now, 'turning at a corner')
            self.turn_heading = math.atan2(b[1] - a[1], b[0] - a[0])
            return (0.0, 0.0)
        # far off the path (not just a corner coming up, where the look-ahead point is already
        # round the bend): stop and turn towards it
        if abs(err) > self.p.turn_start and not self._corner_near():
            self._enter('turn', now, 'turning at a corner')
            return (0.0, 0.0)
        if self.last_xy is None:
            self.last_xy = (x, y)
        if math.hypot(x - self.last_xy[0], y - self.last_xy[1]) >= self.p.min_progress_drive:
            self.last_xy, self.progress_time = (x, y), now
        elif now - self.progress_time > self.p.progress_window:
            return self._give_way(now, 'not moving')
        return (self.p.drive_speed, self._hold(now, x, y, yaw))

    def _hold(self, now, x, y, yaw):
        """Lane hold: small, smooth steering along the current straight stretch."""
        p = self.p
        k = min(len(self.path) - 1, self.index + 1)
        a, b = self.stretches[int(self.seg[k])]
        heading = math.atan2(b[1] - a[1], b[0] - a[0])
        off = math.cos(heading) * (y - a[1]) - math.sin(heading) * (x - a[0])  # + = left
        bend = max(-p.hold_lat_max, min(p.hold_lat_max, p.hold_lat * off))
        err = ex.angle_diff(heading - bend, yaw)
        dt, rate = 0.0, 0.0
        if self.hold_last is not None and now > self.hold_last[0]:
            dt = now - self.hold_last[0]
            rate = ex.angle_diff(yaw, self.hold_last[1]) / dt
            rate = 0.5 * rate + 0.5 * self.hold_rate
        self.hold_last, self.hold_rate = (now, yaw), rate
        self.hold_i = max(-p.hold_i_max, min(p.hold_i_max, self.hold_i + p.hold_ki * err * dt))
        e = 0.0 if abs(err) < p.hold_deadband else err - math.copysign(p.hold_deadband, err)
        want = p.hold_kp * e + self.hold_i - p.hold_kd * rate
        want = max(-p.hold_max, min(p.hold_max, want))
        if dt:
            step = p.hold_slew * dt
            want = max(self.hold_steer - step, min(self.hold_steer + step, want))
        self.hold_steer = want
        return want

    def _blocked(self, now, x, y, yaw, us_left, us_right, points):
        if not self._front_blocked(us_left, us_right, points):
            self._enter('drive', now, 'following the path')
            return (0.0, 0.0)
        if now - self.entered >= self.p.block_wait:
            return self._give_way(now, 'obstacle stayed')
        return (0.0, 0.0)

    def _creep(self, now, x, y, yaw, us_left, us_right, points):
        if now - self.entered < self.p.creep_time and \
                not self._front_blocked(us_left, us_right, points):
            return (self.p.drive_speed, 0.0)
        self._enter('turn', now, 'going round the obstacle')
        return (0.0, 0.0)

    def _backup(self, now, x, y, yaw, us_left, us_right, points):
        rear = self._rear_blocked(points)
        if rear and self.backing_from_turn and now - self.entered < 0.3:
            # too close to turn and no room behind: make room ahead if it is clear, else give
            # this path up
            if self.creeps < self.p.max_creeps and \
                    not self._front_blocked(us_left, us_right, points):
                self.creeps += 1
                self._enter('creep', now, 'moving forward to make room (no room behind)')
                return (0.0, 0.0)
            self._enter('failed', now, 'boxed in: no room to turn or back up')
            return (0.0, 0.0)
        if now - self.entered < self.p.backup_time and not rear:
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

    def _corner_due(self, x, y):
        """Dense index of the corner to turn at now, or None: the robot is `corner_reach` or
        less short of the next corner along its stretch, or has passed it (after a skip,
        only the last corner passed counts)."""
        due = None
        for c in self.corners:
            if c <= self.done_corner:
                continue
            if self.index >= c:
                due = c
                continue
            a, b = self.stretches[int(self.seg[c])]
            d = math.hypot(b[0] - a[0], b[1] - a[1])
            ahead = ((b[0] - x) * (b[0] - a[0]) + (b[1] - y) * (b[1] - a[1])) / d if d else 0.0
            if ahead <= self.p.corner_reach and self.index >= c - int(1.0 / self.p.spacing):
                due = c
            break
        return due

    def _corner_near(self):
        """A corner still to turn at lies within the look-ahead distance."""
        reach = int(self.p.lookahead / self.p.spacing) + 1
        return any(self.done_corner < c <= self.index + reach for c in self.corners)

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
        self.spins, self.still_since, self.stopped_at = 0, None, None
        self.spin_from = 0.0
        self.backing_from_turn = False
        self.turn_heading = None           # set for a corner: turn to the next stretch
        self.hold_last, self.hold_rate, self.hold_steer = None, 0.0, 0.0
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
