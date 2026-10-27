"""Room-exploration state machine (no ROS): decides, every tick, what the robot does next.

  idle -> drive (keeps moving, steering towards the most open way ahead, curving away from
                 obstacles the LiDAR, the ultrasonics or the map show)
  drive -> blocked (ultrasonic <= 1 ft, LiDAR point too close, no way ahead, or not moving)
        -> back up -> turn on the spot towards the most open direction -> drive
  drive -> done when the room is >= coverage_goal scanned
  any   -> error (sensors/map silent, robot not turning, nowhere to go) -> stopped

Every tick, candidate directions ahead are scored by how far the robot can go that way (the
nearest of: LiDAR points in the robot's width, occupied map cells, and, straight ahead where
the LiDAR is blind behind the tower, the ultrasonics), with a bonus for unexplored space and
for places not yet driven, and a cost for steering away from the current course. The robot
steers towards the best one without stopping, so it moves in one continuous path.
"""
import math

import numpy as np

from securebot_ros import explore_logic as ex

FOOT = 0.3048


class Params:
    half_length = 0.2475          # case, centre to front/back edge (m)
    half_width = 0.256            # wheels, centre to outer face (m)
    margin = 0.05
    stop_distance = FOOT          # ultrasonic / LiDAR stop distance from the front edge
    drive_speed = 0.2             # m/s requested (the bridge lifts it to the start power)
    turn_speed = 0.8              # rad/s requested for turning on the spot
    steer_gain = 1.2              # rad/s per rad of heading error while driving
    max_steer = 0.5               # rad/s
    steer_span = math.radians(75)     # candidate directions ahead: +/- this ...
    steer_step = math.radians(15)     # ... in steps of this
    look = 2.0                    # m: clearance beyond this counts the same
    side_room = 0.25              # m each side: a wider corridor is scored too, to keep off walls
    min_clearance = 0.15          # m beyond stop_distance a direction ahead needs to be taken
    turn_clearance = 0.6          # m from the front edge a direction needs to turn towards it
    unexplored_bonus = 0.6
    visited_cost = 0.6            # at most, for a way ahead the robot has driven a lot
    turn_cost = 0.3               # per rad away from straight ahead
    switch_cost = 0.2             # per rad away from the direction chosen last tick
    us_cone = math.radians(25)    # ultrasonics count for directions this close to straight ahead
    turn_tolerance = math.radians(15)
    blocked_hold = 0.3
    backup_time = 0.6
    min_progress_turn = math.radians(5)
    min_progress_drive = 0.05
    progress_window = 4.0         # s of commanded motion without progress -> blocked / error
    sensor_timeout = 3.0
    coverage_goal = 0.90
    coverage_period = 2.0         # s between coverage checks while driving
    min_travel = 1.0              # m driven before "room scanned" can end the run
    blocked_memory = 3            # remember this many blocked headings when choosing a turn
    blocked_width = math.radians(40)


class ExploreMachine:
    def __init__(self, params=None):
        self.p = params or Params()
        self.state, self.message = 'idle', 'press Start'
        self.coverage = None
        self.frontiers = []           # last frontier clusters [(x, y), ...] in the map, for RViz
        self.target = None            # heading (map frame) currently steered towards
        self.leg = 0.0                # clearance that way, for the RViz arrow
        self.visits = ex.VisitMap()
        self.blocked = []
        self.travelled = 0.0
        self.avoids = 0
        self._reset_timers()

    # ---------------------------------------------------------------- commands
    def start(self, now):
        self.visits, self.blocked, self.coverage = ex.VisitMap(), [], None
        self.frontiers, self.target, self.travelled, self.avoids = [], None, 0.0, 0
        self.last_pose = None
        self.next_coverage = now + self.p.coverage_period
        self._enter('drive', now, 'exploring')

    def stop(self, now, message='stopped'):
        self._enter('idle', now, message)

    @property
    def running(self):
        return self.state not in ('idle', 'done', 'error')

    # ---------------------------------------------------------------- one control tick
    def step(self, now, pose, grid, us_left, us_right, scan_points):
        """pose (x, y, yaw) in the map or None; grid ex.Grid or None; ultrasonic ranges in m
        (inf = nothing) or None if silent; scan_points [(x, y)] in base_link (robot frame).
        Returns (linear, angular) to send, or None to send nothing (robot stays stopped)."""
        if not self.running:
            return None
        sensors_ok = pose is not None and us_left is not None and us_right is not None
        if not sensors_ok:
            if self.sensors_lost_since is None:
                self.sensors_lost_since = now
            if now - self.sensors_lost_since > self.p.sensor_timeout:
                missing = [n for n, v in (('map pose', pose), ('left ultrasonic', us_left),
                                          ('right ultrasonic', us_right)) if v is None]
                self._enter('error', now, 'no data from ' + ', '.join(missing))
            return (0.0, 0.0)
        self.sensors_lost_since = None
        x, y, yaw = pose
        if self.last_pose is not None:
            self.travelled += math.hypot(x - self.last_pose[0], y - self.last_pose[1])
        self.last_pose = (x, y)
        self.visits.add(x, y)
        points = np.asarray(scan_points, dtype=float).reshape(-1, 2)
        handler = getattr(self, '_' + self.state)
        return handler(now, x, y, yaw, grid, us_left, us_right, points)

    # ---------------------------------------------------------------- states
    def _drive(self, now, x, y, yaw, grid, us_left, us_right, points):
        why = self._front_blocked(us_left, us_right, points)
        if why:
            return self._blocked(now, yaw, why)
        if now >= self.next_coverage and grid is not None:
            self.next_coverage = now + self.p.coverage_period
            if self._update_coverage(now, grid, x, y):
                return (0.0, 0.0)
        # progress watchdog: commanded forward, but the map pose does not move
        if self.last_xy is None:
            self.last_xy = (x, y)
        if math.hypot(x - self.last_xy[0], y - self.last_xy[1]) >= self.p.min_progress_drive:
            self.last_xy, self.progress_time = (x, y), now
        elif now - self.progress_time > self.p.progress_window:
            return self._blocked(now, yaw, 'not moving')

        span, step = self.p.steer_span, self.p.steer_step
        n = int(round(span / step))
        offsets = [k * step for k in range(-n, n + 1)]
        need = self.p.half_length + self.p.stop_distance + self.p.min_clearance
        pick = self._best(offsets, need, x, y, yaw, grid, us_left, us_right, points)
        if pick is None:
            return self._blocked(now, yaw, 'no way ahead')
        offset, self.leg = pick
        self.target = yaw + offset
        steer = max(-self.p.max_steer, min(self.p.max_steer, self.p.steer_gain * offset))
        return (self.p.drive_speed, steer)

    def _blocked_wait(self, now, x, y, yaw, grid, us_left, us_right, points):
        if now - self.entered >= self.p.blocked_hold:
            self._enter('backup', now, 'backing up')
        return (0.0, 0.0)

    def _backup(self, now, x, y, yaw, grid, us_left, us_right, points):
        if now - self.entered < self.p.backup_time and not self._rear_blocked(points):
            return (-self.p.drive_speed, 0.0)
        # choose where to turn: any direction, away from recently blocked ones
        n = int(round(math.pi / self.p.steer_step))
        offsets = [k * self.p.steer_step for k in range(-n + 1, n + 1)]
        offsets = [o for o in offsets
                   if not any(abs(ex.angle_diff(yaw + o, b)) < self.p.blocked_width
                              for b in self.blocked)]
        need = self.p.half_length + self.p.turn_clearance
        pick = self._best(offsets, need, x, y, yaw, grid, math.inf, math.inf, points,
                          prefer_straight=False)
        if pick is None:
            self._enter('error', now, 'no open direction left' + self._coverage_text())
            return (0.0, 0.0)
        offset, self.leg = pick
        self.target = ex.angle_diff(yaw + offset, 0.0)
        self._enter('turn', now, f'turning {math.degrees(offset):+.0f} deg to open space')
        return (0.0, 0.0)

    def _turn(self, now, x, y, yaw, grid, us_left, us_right, points):
        err = ex.angle_diff(self.target, yaw)
        if abs(err) <= self.p.turn_tolerance:
            self._enter('drive', now, 'exploring')
            return (0.0, 0.0)
        if self.turn_ref_yaw is None:
            self.turn_ref_yaw, self.turn_ref_time = yaw, now
        if abs(ex.angle_diff(yaw, self.turn_ref_yaw)) >= self.p.min_progress_turn:
            self.turn_ref_yaw, self.turn_ref_time = yaw, now
        elif now - self.turn_ref_time > self.p.progress_window:
            self._enter('error', now, 'robot is not turning')
            return (0.0, 0.0)
        return (0.0, self.p.turn_speed if err > 0 else -self.p.turn_speed)

    # ---------------------------------------------------------------- helpers
    def _enter(self, state, now, message):
        self.state, self.message, self.entered = state, message, now
        self._reset_timers(now)

    def _reset_timers(self, now=0.0):
        self.progress_time = now
        self.last_xy = None               # progress reference, set on the next drive tick
        self.sensors_lost_since = None
        self.turn_ref_yaw, self.turn_ref_time = None, now

    def _clearance(self, offset, x, y, yaw, grid, us_left, us_right, points, extra=0.0):
        """How far (from the robot's centre) a corridor `extra` wider each side than the robot
        reaches at `offset` from straight ahead, and whether that way leads into unexplored
        space before an obstacle."""
        hw = self.p.half_width + self.p.margin + extra
        clear = min(self.p.look, ex.scan_clearance(points, offset, hw))
        unexplored = False
        if grid is not None:
            occupied, unknown = ex.ray(grid, x, y, yaw + offset, hw, self.p.look)
            clear = min(clear, occupied)
            unexplored = unknown is not None and unknown < occupied
        if abs(offset) <= self.p.us_cone:
            # left sensor sees the left of straight ahead, right sensor the right
            if offset > self.p.us_cone / 2:
                near = us_left
            elif offset < -self.p.us_cone / 2:
                near = us_right
            else:
                near = min(us_left, us_right)
            clear = min(clear, self.p.half_length + near)
        return clear, unexplored

    def _best(self, offsets, need, x, y, yaw, grid, us_left, us_right, points,
              prefer_straight=True):
        """Best direction (offset from straight ahead, clearance) among `offsets` whose
        clearance is at least `need`, or None."""
        p = self.p
        last = ex.angle_diff(self.target, yaw) if self.target is not None and prefer_straight \
            else 0.0
        best, best_score = None, -math.inf
        for o in offsets:
            clear, unexplored = self._clearance(o, x, y, yaw, grid, us_left, us_right, points)
            if clear < need:
                continue
            wide, _ = self._clearance(o, x, y, yaw, grid, us_left, us_right, points, p.side_room)
            visited = self.visits.along(x, y, yaw + o, min(clear, p.look), p.half_width)
            score = (0.5 * (clear + wide)
                     + (p.unexplored_bonus if unexplored else 0.0)
                     - p.visited_cost * min(visited, 30) / 30.0
                     - p.turn_cost * abs(o) * (1.0 if prefer_straight else 0.3)
                     - p.switch_cost * abs(ex.angle_diff(o, last)))
            if score > best_score:
                best, best_score = (o, clear), score
        return best

    def _update_coverage(self, now, grid, x, y):
        """Refresh coverage and frontiers; True (and done) once the room is scanned."""
        reach = ex.reachable_free(grid, x, y)
        self.coverage, clusters = ex.coverage_and_frontiers(grid, reach)
        self.frontiers = [[grid.to_world(r, c) for r, c in cl] for cl in clusters]
        if self.travelled >= self.p.min_travel and self.coverage >= self.p.coverage_goal:
            self._enter('done', now, f'room scanned ({self.coverage:.0%})')
            return True
        return False

    def _coverage_text(self):
        return '' if self.coverage is None else f' ({self.coverage:.0%} scanned)'

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
        if np.any((np.abs(px) <= hl + self.p.margin) & (np.abs(py) <= hw + self.p.margin)):
            return 'LiDAR: object touching the robot'
        return None

    def _rear_blocked(self, points):
        if len(points) == 0:
            return False
        hl, hw = self.p.half_length, self.p.half_width + self.p.margin
        return bool(np.any((points[:, 0] >= -(hl + 0.25)) & (points[:, 0] <= 0.0)
                           & (np.abs(points[:, 1]) <= hw)))

    def _blocked(self, now, yaw, why):
        self.blocked = (self.blocked + [yaw])[-self.p.blocked_memory:]
        self.avoids += 1
        self._enter('blocked_wait', now, f'blocked ({why})')
        return (0.0, 0.0)
