"""Room-exploration state machine (no ROS): decides, every tick, what the robot does next.

  idle -> survey (full turn in bursts, so the LiDAR sees past the tower's blind wedge)
       -> plan -> turn -> drive -> pause -> plan -> ...          until coverage >= goal -> done
  drive -> blocked (ultrasonic <= 1 ft, LiDAR too close, or not moving) -> back up -> plan
  any   -> error (sensors/map silent, robot not turning/moving, no way out) -> stopped

The robot only has one usable motor power (the bridge lifts every command to it), so turns
are done in short bursts with pauses: SLAM and laser odometry keep up, and the yaw is
checked between bursts. Forward legs are short (<= 1.5 m) and re-planned after each one.
"""
import math

from securebot_ros import explore_logic as ex

FOOT = 0.3048


class Params:
    half_length = 0.2475          # case, centre to front/back edge (m)
    half_width = 0.256            # wheels, centre to outer face (m)
    margin = 0.05
    stop_distance = FOOT          # ultrasonic stop distance from the front edge
    lidar_front_box = 0.35        # LiDAR points this far ahead of the front edge (sides) block
    drive_speed = 0.2             # m/s requested (the bridge lifts it to the start power)
    turn_speed = 0.8              # rad/s requested
    turn_on, turn_off = 0.3, 0.4  # burst timing (s)
    turn_tolerance = math.radians(10)
    pause_time = 0.8              # after a leg, for SLAM to catch up
    backup_time = 0.4
    blocked_hold = 0.5
    min_progress_turn = math.radians(5)
    min_progress_drive = 0.05
    progress_window = 4.0         # s of commanded motion without progress -> error / blocked
    sensor_timeout = 3.0
    coverage_goal = 0.90
    blocked_memory = 3            # remember this many blocked headings when planning
    min_leg, max_leg = 0.5, 1.5


class ExploreMachine:
    def __init__(self, params=None):
        self.p = params or Params()
        self.state, self.message = 'idle', 'press Start'
        self.coverage = None
        self.frontiers = []           # last frontier clusters [(x, y), ...] in the map, for RViz
        self.target = None
        self.visits = ex.VisitMap()
        self.blocked = []
        self.legs = 0
        self._reset_timers()

    # ---------------------------------------------------------------- commands
    def start(self, now):
        self.visits, self.blocked, self.legs, self.coverage = ex.VisitMap(), [], 0, None
        self.frontiers, self.target = [], None
        self._enter('survey', now, 'looking all around')
        self.turned = 0.0
        self.last_yaw = None

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
            return (0.0, 0.0)
        self.sensors_lost_since = None
        x, y, yaw = pose
        self.visits.add(x, y)
        handler = getattr(self, '_' + self.state)
        return handler(now, x, y, yaw, grid, us_left, us_right, scan_points)

    # ---------------------------------------------------------------- states
    def _survey(self, now, x, y, yaw, grid, us_left, us_right, scan_points):
        if self.last_yaw is not None:
            self.turned += abs(ex.angle_diff(yaw, self.last_yaw))
        self.last_yaw = yaw
        if self.turned >= 2 * math.pi - self.p.turn_tolerance:
            self._enter('plan', now, 'planning')
            return (0.0, 0.0)
        return self._burst(now, yaw, +1)

    def _plan(self, now, x, y, yaw, grid, us_left, us_right, scan_points):
        if grid is None:
            return (0.0, 0.0)
        reach = ex.reachable_free(grid, x, y)
        self.coverage, clusters = ex.coverage_and_frontiers(grid, reach)
        self.frontiers = [[grid.to_world(r, c) for r, c in cl] for cl in clusters]
        if self.legs >= 1 and self.coverage >= self.p.coverage_goal:
            self._enter('done', now, f'room scanned ({self.coverage:.0%})')
            return (0.0, 0.0)
        pick = ex.choose_heading(grid, x, y, yaw, self.p.half_width + self.p.margin, self.visits,
                                 self.blocked, min_leg=self.p.min_leg, max_leg=self.p.max_leg)
        if pick is None:
            self._enter('error', now, f'no open direction left ({self.coverage:.0%} scanned)')
            return (0.0, 0.0)
        self.target, self.leg = pick
        radius = math.hypot(self.p.half_length, self.p.half_width) + self.p.margin
        if abs(ex.angle_diff(self.target, yaw)) > math.radians(30) and \
                not ex.rotation_clear(grid, x, y, radius):
            self.blocked = (self.blocked + [self.target])[-self.p.blocked_memory:]
            self._enter('backup', now, 'too close to turn, backing up')
            return (0.0, 0.0)
        self._enter('turn', now, f'turning to {math.degrees(self.target):.0f} deg')
        self.last_yaw = yaw
        return (0.0, 0.0)

    def _turn(self, now, x, y, yaw, grid, us_left, us_right, scan_points):
        err = ex.angle_diff(self.target, yaw)
        if abs(err) <= self.p.turn_tolerance:
            self._enter('drive', now, f'driving {self.leg:.1f} m')
            self.start_xy, self.last_xy = (x, y), (x, y)
            return (0.0, 0.0)
        return self._burst(now, yaw, 1 if err > 0 else -1)

    def _drive(self, now, x, y, yaw, grid, us_left, us_right, scan_points):
        why = self._front_blocked(us_left, us_right, scan_points)
        if why:
            return self._blocked(now, yaw, why)
        travelled = math.hypot(x - self.start_xy[0], y - self.start_xy[1])
        if travelled >= self.leg:
            return self._end_leg(now)
        if grid is not None:
            ahead, _ = ex.free_distance(grid, x, y, yaw, self.p.half_width + self.p.margin, 0.8)
            if ahead < self.p.half_length + 0.15:
                return self._end_leg(now)
        # progress watchdog
        if math.hypot(x - self.last_xy[0], y - self.last_xy[1]) >= self.p.min_progress_drive:
            self.last_xy, self.progress_time = (x, y), now
        elif now - self.progress_time > self.p.progress_window:
            return self._blocked(now, yaw, 'not moving')
        steer = max(-0.4, min(0.4, 1.5 * ex.angle_diff(self.target, yaw)))
        return (self.p.drive_speed, steer)

    def _blocked_wait(self, now, x, y, yaw, grid, us_left, us_right, scan_points):
        if now - self.entered >= self.p.blocked_hold:
            self._enter('backup', now, 'backing up')
        return (0.0, 0.0)

    def _backup(self, now, x, y, yaw, grid, us_left, us_right, scan_points):
        if now - self.entered >= self.p.backup_time or self._rear_blocked(scan_points):
            self._enter('pause', now, 'finding a new path')
            return (0.0, 0.0)
        return (-self.p.drive_speed, 0.0)

    def _pause(self, now, x, y, yaw, grid, us_left, us_right, scan_points):
        if now - self.entered >= self.p.pause_time:
            self._enter('plan', now, 'planning')
        return (0.0, 0.0)

    # ---------------------------------------------------------------- helpers
    def _enter(self, state, now, message):
        self.state, self.message, self.entered = state, message, now
        self._reset_timers(now)

    def _reset_timers(self, now=0.0):
        self.progress_time = now
        self.burst_start = now
        self.sensors_lost_since = None
        self.turn_ref_yaw, self.turn_ref_time = None, now

    def _burst(self, now, yaw, direction):
        """Turn in bursts: turn_on s turning, turn_off s still. Errors out if commanded turning
        does not change the yaw (wheels not turning, robot wedged)."""
        if self.turn_ref_yaw is None:
            self.turn_ref_yaw, self.turn_ref_time = yaw, now
        if abs(ex.angle_diff(yaw, self.turn_ref_yaw)) >= self.p.min_progress_turn:
            self.turn_ref_yaw, self.turn_ref_time = yaw, now
        elif now - self.turn_ref_time > self.p.progress_window:
            self._enter('error', now, 'robot is not turning')
            return (0.0, 0.0)
        phase = (now - self.burst_start) % (self.p.turn_on + self.p.turn_off)
        return (0.0, direction * self.p.turn_speed) if phase < self.p.turn_on else (0.0, 0.0)

    def _front_blocked(self, us_left, us_right, scan_points):
        if min(us_left, us_right) <= self.p.stop_distance:
            side = 'left' if us_left <= us_right else 'right'
            return f'{side} ultrasonic: object within 1 ft'
        hl, hw = self.p.half_length, self.p.half_width + self.p.margin
        for px, py in scan_points:
            if abs(py) <= hw and 0.0 <= px <= hl + self.p.lidar_front_box:
                return 'LiDAR: object close ahead'
            if abs(px) <= hl + self.p.margin and abs(py) <= hw + self.p.margin:
                return 'LiDAR: object touching the robot'
        return None

    def _rear_blocked(self, scan_points):
        hl, hw = self.p.half_length, self.p.half_width + self.p.margin
        return any(-(hl + 0.25) <= px <= 0.0 and abs(py) <= hw for px, py in scan_points)

    def _blocked(self, now, yaw, why):
        self.blocked = (self.blocked + [yaw])[-self.p.blocked_memory:]
        self._enter('blocked_wait', now, f'blocked ({why})')
        return (0.0, 0.0)

    def _end_leg(self, now):
        self.legs += 1
        self._enter('pause', now, 'leg done')
        return (0.0, 0.0)
