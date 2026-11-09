"""Follow one lap with SecureBot's one-speed drive (no ROS): decides, every tick, what to send.

SecureBot's motors only start at a high PWM, so every wheel that turns runs at about the same
speed: the robot cannot creep or turn gently, and Nav2's controllers (which expect fine speed
control) overshoot and swing about. This follower drives the way the explore mode does:

  drive: hold the straight stretch of path the robot is on (lane hold): steer to the
         stretch's heading, bent by up to `hold_lat_max` towards the line when the robot is
         off it, with small, smooth corrections (PID on the gyro heading, rate limited): the
         bridge slows the inner wheel a little instead of nearly stopping it. The integral
         learns the robot's own veer (~6.5 deg/s at equal power) and keeps it across paths
  arc:   a bend of up to `arc_max_bend` with room for it is driven as a curve of `arc_radius`
         (pure pursuit along the rounded corner, steering limited and smoothed) instead of a
         stop and spin: fewer corrective movements (2026-11-06). No room (LiDAR within the
         robot's width of the curve), too sharp, or blocked while curving: stop and turn
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
  boxed in: blocked ahead and behind as well: first a turn with less margin (turn_radius_tight,
         still clear of the corners), else the path counts as failed
  gentle: small turns and every spin after a turn's first use turn_speed_gentle, which the
         bridge lifts only to its gentle PWM (~610, not 660): corrections without a jolt; a
         gentle spin that has not got going after gentle_escalate s goes on at full speed
  obstacle: whenever something stops the robot for good (it stayed, or the robot drives but
         does not move), `stall` says where (for the laps node's temporary box)
  done:  within `goal_tolerance` of the path's end
  error: pose or ultrasonics silent, or the robot does not move / turn when told to
"""
import math

import numpy as np

from securebot_ros import explore_logic as ex
from securebot_ros.explore_machine import FOOT


EDGE = 0.02                       # m: obstacle zones start this far inside the body's edge


class Params:
    half_length = 0.2475          # case, centre to front/back edge (m)
    half_width = 0.256            # wheels, centre to outer face (m)
    margin = 0.05
    straight_margin = 0.02        # m beside the wheels that must be clear to drive straight
    stop_distance = FOOT          # ultrasonic / LiDAR stop distance from the front edge
    drive_speed = 0.2             # m/s requested (the bridge lifts it to the start power)
    approach_speed = 0.1          # m/s for the last approach_slow m before a stop: the bridge
    approach_slow = 0.35          # lifts it only to its gentle PWM (~610), less overshoot
    waypoint_slow = 0.2032        # m (8 in): also slow down this far before every waypoint
    #                               (lane / loop point) it drives through or curves round
    turn_speed = 0.8              # rad/s requested for turning on the spot
    turn_speed_gentle = 0.45      # rad/s for corrections (bridge: gentle PWM band)
    gentle_turn = math.radians(30)    # turns smaller than this are corrections
    gentle_escalate = 1.2         # s: a gentle spin not turning by then goes on at full speed
    gentle_part = 0.5             # share of the error a gentle (corrective) spin turns: half
    #                               the time (2026-11-07; was 1.0, the whole error)
    lookahead = 0.6               # m along the path (turn target, big-error check)
    corner_angle = math.radians(20)   # path bend that is a stop-and-turn corner
    corner_reach = 0.10           # m before the corner to stop and turn
    arc_radius = 0.50             # m: corners driven as curves of this radius ...
    arc_max_bend = math.radians(100)  # ... when they bend this much or less
    arc_lookahead = 0.30          # m: pure-pursuit look-ahead on the curve
    arc_w_max = 0.45              # rad/s: most steering on a curve (0.60 nearly stopped the
    #                               inner wheel: the robot pivoted instead of curving)
    arc_delay = 0.6               # s: the robot answers the steering this late (measured on
    #                               2026-11-06): the curve eases off by the heading predicted
    #                               this far ahead, so it does not swing past the exit
    arc_ease = 1.2                # s: steering at most (turn still to do) / arc_ease
    arc_overshoot = math.radians(25)  # swung this far past the exit heading: stop the curve
    #                                   and turn on the spot (2026-11-06 a curve swung ~105 deg
    #                                   past and drove back along itself)
    arc_slew = 1.0                # rad/s per s: how fast the curve's steering may change
    arc_clear = 0.02              # m beyond half the robot's width the curve must be clear
    # lane hold in two layers (2026-11-06): a trim - the steady power difference that makes
    # this robot drive straight, learnt from the gyro's turn rate - plus small corrections:
    # the heading / line error asks for a turn rate of at most hold_rate_max
    hold_kp = 0.5                 # rad/s of turn wanted per rad of heading error ... (0.8
    #                               overcorrected with the ~1 s steering delay, see hold_ki)
    hold_rate_max = 0.08          # ... at most (~4.6 deg/s): corrections stay small ...
    hold_rate_far = 0.12          # ... rising to this when hold_far m or more off the line
    hold_far = 0.30               # (rejoins faster; near the line as gentle as before)
    hold_ki = 0.25                # trim: rad/s per (rad/s of turn-rate error) per s. SecureBot
    #                               turns ~1.0 s after the steering changes (fitted on the
    #                               2026-11-06 18:30 run: gyro rate = 1.1 x steering, 1.0 s
    #                               later); 1.0 made the trim overcorrect, weaving +-20-30 cm
    #                               every ~6 s. 0.25 (and hold_kp 0.5) holds the line, in
    #                               simulation, with 0.6-1.3 s of delay and gain 0.8-1.5
    hold_i_max = 0.15             # rad/s: biggest trim (veer ~0.11 rad/s)
    hold_kr = 0.3                 # rad/s per rad/s of turn-rate error (rate loop)
    hold_lat = 1.0                # rad of heading bend per m off the line ...
    hold_lat_max = math.radians(30)   # ... at most: far off the line it rejoins at up to
    #                                   30 deg (was 12: ~4 cm/s back, 12+ s from 50 cm off);
    #                                   near it the bend is as before (1 deg per 1.7 cm)
    rejoin_angle = math.radians(30)   # off the path: turn towards a point on it far enough
    #                                   ahead to meet it at this angle, never back to the
    #                                   last point reached
    pos_tau = 2.0                 # s: lane hold position = odometry, pulled towards AMCL
    #                               with this time constant (AMCL wanders in open space)
    hold_deadband = math.radians(1.5)
    hold_max = 0.12               # rad/s: biggest correction on top of the learnt veer ...
    hold_total = 0.20             # ... and biggest steering in all (inner wheel ~420 vs 660)
    hold_slew = 0.4               # rad/s per s: how fast the correction may change (gentle:
    #                               0.8 swung the inner wheel 450 -> 660 -> 530 in 0.7 s)
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
    full_spin_early = 0.1         # s: full-speed (high PWM) spins stop this much sooner than
    #                               the coast alone says (2026-11-06: they ran a bit long);
    #                               the lane hold or a gentle spin takes the few deg left
    turn_radius = 0.40            # m: LiDAR points this close block turning (corners: 0.356)
    turn_radius_tight = 0.37      # m: the same when boxed in (no room to back up)
    rear_room = 0.12              # m behind the back edge that must be clear to back up
    turn_front = 0.12             # m: ultrasonic reading below this blocks turning (front corners)
    settle_time = 0.25            # s stopped, at least, before checking the heading
    turn_timeout = 30.0           # s on one turn -> give up on this stretch of path
    spacing = 0.1                 # m between path points after densifying
    search = 1.5                  # m along the path searched for the robot's nearest point
    goal_tolerance = 0.3          # m from the path's end
    block_wait = 3.0              # s to wait for an obstacle to clear
    backup_time = 0.5             # s reversing, less if something is behind (2026-11-06: was
    #                               1.5; the motors need ~0.3-0.6 s to get going)
    corner_turn = math.radians(25)    # blocked while the path turns away this much: turn
    max_room_tries = 3            # backing up to make room for a turn, before skipping on
    creep_time = 1.5              # s forward when there is no room behind (motors: ~0.6 s dead)
    max_creeps = 3                # per path
    creep_steer = 0.15            # rad/s away from the nearest thing beside the robot
    skip = 1.0                    # m along the path to skip past an obstacle
    max_blocks = 1                # skip past an obstacle once; blocked a 2nd time: give up
    progress_window = 5.0         # s of commanded motion without progress -> blocked / error
    # lane hold against AMCL's jumps (2026-11-05): the heading comes from odometry (gyro, smooth)
    # plus a slowly tracked map->odom offset, the offset from the line is smoothed, and the
    # angle between the reported heading and the way the robot really travels (crab) is learnt
    # from its track and steered out
    odom_off_tau = 4.0            # s: time constant of the map->odom heading offset
    lat_tau = 0.4                 # s: time constant of the smoothed offset from the line
    crab_window = 0.4             # m driven per crab sample
    crab_gain = 0.2               # weight of a new crab sample
    crab_max = math.radians(15)
    crab_rate_max = math.radians(4)   # rad/s: turning faster than this spoils a crab sample
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
        self.marks = []                # dense indices of the waypoints (slow down before)
        self.index = 0                 # nearest path point reached so far (never goes back)
        self.blocks = 0
        self.room_tries = 0
        self.entered = 0.0
        self.turn_sign = 1.0
        self.hold_i = 0.0              # lane-hold integral (rad/s): kept, it is the robot's veer
        self.corners, self.done_corner, self.creeps = [], -1, 0
        self.coast = self.p.coast_time         # learned coast after a spin (s of spin rate)
        self.spin_keep = 0.0                   # rad of a gentle spin's error left to the hold
        self.odom_yaw = None
        self.odom_xy = None
        self.crab_n = 0
        self.arcs, self.hold_steer_last = {}, 0.0
        self.crab = 0.0                # learnt: travel direction minus reported heading (rad)
        # (x, y, yaw, ahead, side): robot pose and where the thing that stopped it for good
        # starts (m ahead of the centre, m to the left) - for the laps node's temporary box
        self.stall = None
        self.tight = False             # boxed in: turning with turn_radius_tight
        self._reset_timers()
        self._phase('settle', 0.0)

    # ---------------------------------------------------------------- commands
    def start(self, now, path, waypoints=None):
        """Follow `path` [(x, y), ...] in the map frame from its start. waypoints: points on
        it (default: its own points) the robot slows down for, waypoint_slow m before."""
        self.path, self.seg, self.stretches = _densify(path, self.p.spacing)
        self.marks, k = [], 0
        for wx, wy in (path if waypoints is None else waypoints):
            d = np.hypot(self.path[k:, 0] - wx, self.path[k:, 1] - wy)
            j = int(np.argmin(d))
            if d[j] < 0.15:
                k += j
                self.marks.append(k)
        heads = [math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in self.stretches]
        # dense index of every bend of corner_angle or more (the first point of the next stretch)
        firsts = {int(s): k for k, s in reversed(list(enumerate(self.seg)))}
        self.corners = [firsts[i + 1] - 1 for i in range(len(heads) - 1)
                        if i + 1 in firsts
                        and abs(ex.angle_diff(heads[i + 1], heads[i])) >= self.p.corner_angle]
        self.arcs = {c: a for c in self.corners if (a := self._fillet(c)) is not None}
        self.index, self.blocks, self.room_tries = 0, 0, 0
        self.done_corner, self.creeps = -1, 0
        self._enter('turn', now, 'turning onto the path')

    def stop(self, now, message='stopped'):
        self._enter('idle', now, message)

    @property
    def running(self):
        return self.state in ('turn', 'drive', 'arc', 'blocked', 'backup', 'creep')

    @property
    def remaining(self):
        """Path length left (m), or None."""
        if self.path is None:
            return None
        rest = self.path[self.index:]
        return float(np.sum(np.linalg.norm(np.diff(rest, axis=0), axis=1))) if len(rest) > 1 \
            else 0.0

    # ---------------------------------------------------------------- one control tick
    def step(self, now, pose, us_left, us_right, scan_points, odom_yaw=None, odom_xy=None):
        """pose (x, y, yaw) in the map or None; ultrasonic ranges in m (inf = nothing) or None
        if silent; scan_points [(x, y)] in base_link; odom_yaw: heading in the odom frame (gyro,
        smooth; None = use the map's); odom_xy: position in the odom frame (smooth, drifts
        slowly; None = use the map's). Returns (linear, angular) or None."""
        self.odom_yaw = odom_yaw
        self.odom_xy = odom_xy
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
        # aim the direction the robot will really travel (heading + crab) at the target
        err = (ex.angle_diff(self.turn_heading, yaw + self.crab) if self.turn_heading is not None
               else self._heading_error(x, y, yaw + self.crab))
        # turn rate, stillness and coasting from the gyro (odometry heading) when there is one:
        # AMCL's heading jumps at its updates and at position corrections, which looked like
        # turning and kept every settle waiting for settle_max
        gyro = self.odom_yaw if self.odom_yaw is not None else yaw
        rate = 0.0
        if self.last_turn_pose is not None and now > self.last_turn_pose[0]:
            rate = ex.angle_diff(gyro, self.last_turn_pose[1]) / (now - self.last_turn_pose[0])
        self.last_turn_pose = (now, gyro)
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
            self._learn_coast(gyro)
            # one turn = one spin: after a spin, a small miss either way is left to the
            # lane-hold steering, and the robot never spins back for less than no_reverse
            back = self.spins > 0 and (err > 0) != (self.turn_sign > 0)
            if abs(err) <= (p.turn_tolerance if self.spins == 0 else p.spin_tolerance) or \
                    (back and abs(err) <= p.no_reverse):
                self._enter('drive', now, 'following the path')
                return (0.0, 0.0)
            self.turn_sign = 1.0 if err > 0 else -1.0
            self.spins += 1
            self.spin_from = gyro
            # corrections (a small turn, or any spin after the first) go gently
            self.gentle = self.spins > 1 or abs(err) < p.gentle_turn
            # a small correction (first spin, under gentle_turn) turns only gentle_part of the
            # error: half as long, the lane hold takes the rest (2026-11-07). The follow-up
            # spins of a big turn finish it (halving them made one corner three spins)
            small = self.spins == 1 and abs(err) < p.gentle_turn
            self.spin_keep = (1.0 - p.gentle_part) * abs(err) if small else 0.0
            self._phase('spin', now)
        elif self.turn_sign * err <= self.spin_keep + max(p.min_lead, abs(rate) * (
                self.coast + (0.0 if self.gentle else p.full_spin_early))):
            self.stopped_at = (gyro, abs(rate))  # close enough: the coast does the rest
            self._phase('settle', now)
            return (0.0, 0.0)
        turned = abs(ex.angle_diff(gyro, self.spin_from))
        if self.phase == 'spin' and self.gentle and now - self.phase_start > p.gentle_escalate \
                and turned < p.spin_stall_angle / 2:
            self.gentle = False                # the gentle PWM does not move it: full speed
            self.phase_start = now
        if self.phase == 'spin' and now - self.phase_start > p.spin_stall_time and \
                turned < p.spin_stall_angle:
            return self._give_way(now, 'not turning')     # wedged against something
        why = self._turn_blocked(us_left, us_right, points)
        if why:
            self.room_tries += 1
            if self.room_tries > p.max_room_tries:
                return self._give_way(now, why)
            self._enter('backup', now, f'backing up to make room ({why})')
            self.backing_from_turn = True
            return (0.0, 0.0)
        return (0.0, self.turn_sign * (p.turn_speed_gentle if self.gentle else p.turn_speed))

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
            arc = self.arcs.get(corner)
            if arc is not None and self._arc_clear(arc, x, y, yaw, us_left, us_right, points):
                self._enter('arc', now, 'curving round a corner')
                self.arc, self.arc_k, self.arc_w = arc, 0, self.hold_steer_last
                return (self.p.drive_speed, self.arc_w)
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
            # driving with nothing seen ahead, yet not moving: something too low for the
            # sensors (a bag on the floor) is in the way
            self.stall = (x, y, yaw, self.p.half_length, 0.0)   # touching the front
            return self._give_way(now, 'not moving')
        steer = self._hold(now, x, y, yaw)
        self.hold_steer_last = steer
        # slow down (the bridge's gentle PWM, ~610) for the last bit before a stop - the end
        # of the path or a corner turned on the spot - so the robot overshoots it less
        speed = self.p.approach_speed if self._stop_ahead() < self.p.approach_slow \
            or self._waypoint_near() else self.p.drive_speed
        return (speed, steer)

    def _waypoint_near(self):
        """A waypoint within waypoint_slow m ahead along the path."""
        for k in self.marks:
            if k > self.index:
                return (k - self.index) * self.p.spacing <= self.p.waypoint_slow
        return False

    def _stop_ahead(self):
        """m along the path to where the robot stops next: the path's end, or the next
        corner it turns on the spot (not one it curves round)."""
        p = self.p
        d = (len(self.path) - 1 - self.index) * p.spacing - p.goal_tolerance
        for c in self.corners:
            if c > self.done_corner and c not in self.arcs:
                d = min(d, (c - self.index) * p.spacing - p.corner_reach)
                break
        return d

    def _arc(self, now, x, y, yaw, us_left, us_right, points):
        """Curve round a corner: pure pursuit along the rounded corner's points."""
        p, arc = self.p, self.arc
        if self._front_blocked(us_left, us_right, points) or \
                now - self.entered > arc['time'] * 2.5 + 2.0:
            # something in the way (or the curve is not working out): stop and turn instead
            self.index = max(self.index, arc['corner'])
            self._enter('turn', now, 'turning at a corner')
            self.turn_heading = arc['heading']
            return (0.0, 0.0)
        pts = arc['pts']
        u2 = arc['u2']
        # heading (gyro when there is one) and the turn still to do, as it will be once the
        # steering sent now acts
        gyro_yaw = yaw if self.odom_yaw is None else self.odom_yaw
        rate = 0.0
        if self.arc_last is not None and now > self.arc_last[0]:
            rate = ex.angle_diff(gyro_yaw, self.arc_last[1]) / (now - self.arc_last[0])
        self.arc_last = (now, gyro_yaw)
        sign = arc['sign']
        todo = sign * ex.angle_diff(arc['heading'], yaw + self.crab + rate * p.arc_delay)
        if todo < -p.arc_overshoot:
            self.index = max(self.index, arc['corner'] + 1)
            self._enter('turn', now, 'turning at a corner')
            self.turn_heading = arc['heading']
            return (0.0, 0.0)
        if todo <= math.radians(5) or \
                (x - arc['end'][0]) * u2[0] + (y - arc['end'][1]) * u2[1] >= -0.02:
            self.index = max(self.index, arc['corner'] + 1)       # round: on the next stretch
            self.hold_steer_last = self.arc_w
            self._enter('drive', now, 'following the path')
            self.hold_steer = self.arc_w                         # no jump in the steering
            return (p.drive_speed, self.arc_w)
        d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
        self.arc_k = max(self.arc_k, int(np.argmin(d[self.arc_k:])) + self.arc_k)
        ahead = np.nonzero(d[self.arc_k:] >= p.arc_lookahead)[0]
        tx, ty = pts[self.arc_k + ahead[0]] if len(ahead) else \
            (arc['end'][0] + u2[0] * p.arc_lookahead, arc['end'][1] + u2[1] * p.arc_lookahead)
        alpha = ex.angle_diff(math.atan2(ty - y, tx - x), yaw + self.crab)
        want = 2.0 * p.drive_speed * math.sin(alpha) / p.arc_lookahead
        want = max(-p.arc_w_max, min(p.arc_w_max, want))
        # ease off as the exit heading comes (never steer the other way on a curve)
        cap = max(0.0, todo) / p.arc_ease
        want = sign * min(max(0.0, sign * want), cap)
        dt = now - self.arc_t if self.arc_t is not None and now > self.arc_t else 0.1
        step = p.arc_slew * dt
        self.arc_w = max(self.arc_w - step, min(self.arc_w + step, want))
        self.arc_t = now
        self.index = max(self.index, self._arc_index(arc, x, y))
        if self._waypoint_near():               # same curve, slower: steering scaled too
            return (p.approach_speed, self.arc_w * p.approach_speed / p.drive_speed)
        return (p.drive_speed, self.arc_w)

    def _arc_index(self, arc, x, y):
        """Dense path index level with the robot on a curve (for the waypoint slow-down)."""
        c = arc['corner']
        lo = max(0, c - int(arc['lead'] / self.p.spacing) - 2)
        hi = min(len(self.path), c + int(arc['lead'] / self.p.spacing) + 3)
        seg = self.path[lo:hi]
        return lo + int(np.argmin(np.hypot(seg[:, 0] - x, seg[:, 1] - y)))

    def _fillet(self, c):
        """The rounded corner at dense corner index `c`, or None if it is too sharp or the
        stretches either side are too short for it."""
        p = self.p
        k = min(len(self.path) - 1, c + 1)
        i1, i2 = int(self.seg[c]), int(self.seg[k])
        if i1 == i2:
            return None
        (a1, b1), (a2, b2) = self.stretches[i1], self.stretches[i2]
        l1, l2 = np.linalg.norm(b1 - a1), np.linalg.norm(b2 - a2)
        if l1 < 1e-6 or l2 < 1e-6:
            return None
        u1, u2 = (b1 - a1) / l1, (b2 - a2) / l2
        bend = ex.angle_diff(math.atan2(u2[1], u2[0]), math.atan2(u1[1], u1[0]))
        if abs(bend) > p.arc_max_bend:
            return None
        lead = p.arc_radius * math.tan(abs(bend) / 2)
        # room on both stretches (half the next one: it may round a corner at its end too)
        if lead > l1 - 0.05 or lead > l2 / 2:
            return None
        sign = 1.0 if bend > 0 else -1.0
        start, end = b1 - u1 * lead, b1 + u2 * lead
        centre = start + sign * p.arc_radius * np.array([-u1[1], u1[0]])
        a0 = math.atan2(start[1] - centre[1], start[0] - centre[0])
        n = max(3, int(abs(bend) * p.arc_radius / 0.05))
        ts = a0 + sign * np.linspace(0.0, abs(bend), n)
        pts = centre + p.arc_radius * np.column_stack([np.cos(ts), np.sin(ts)])
        return {'corner': c, 'start': start, 'end': end, 'u2': u2, 'pts': pts, 'sign': sign,
                'lead': lead, 'heading': math.atan2(u2[1], u2[0]),
                'time': abs(bend) * p.arc_radius / p.drive_speed}

    def _arc_clear(self, arc, x, y, yaw, us_left, us_right, points):
        """Nothing the LiDAR sees within the robot's half width (+ arc_clear) of the curve."""
        if min(us_left, us_right) <= self.p.stop_distance:
            return False
        if len(points) == 0:
            return True
        c, s = math.cos(yaw), math.sin(yaw)
        rel = arc['pts'] - (x, y)
        px = c * rel[:, 0] + s * rel[:, 1]                # curve in base_link
        py = -s * rel[:, 0] + c * rel[:, 1]
        reach = self.p.half_width + self.p.margin + self.p.arc_clear
        d = np.hypot(points[:, None, 0] - px[None, :], points[:, None, 1] - py[None, :])
        return bool(d.min() > reach)

    def _hold(self, now, x, y, yaw):
        """Lane hold: small, smooth steering along the current straight stretch.

        AMCL's map pose jumps a few cm / degrees at every update; steering straight on it made
        the robot weave. So the heading is the odometry's (gyro) plus a slowly tracked
        map->odom offset, the offset from the line is smoothed, and the steering aims the
        robot's real direction of travel (heading + learnt crab) along the line."""
        p = self.p
        k = min(len(self.path) - 1, self.index + 1)
        a, b = self.stretches[int(self.seg[k])]
        heading = math.atan2(b[1] - a[1], b[0] - a[0])
        dt = now - self.hold_last[0] if self.hold_last is not None and \
            now > self.hold_last[0] else 0.0
        # smooth heading: odometry + slowly tracked map->odom offset
        if self.odom_yaw is None:
            yaw_s = yaw
        else:
            off_now = ex.angle_diff(yaw, self.odom_yaw)
            if self.odom_off is None:
                self.odom_off = off_now
            elif dt:
                w = min(1.0, dt / p.odom_off_tau)
                self.odom_off += w * ex.angle_diff(off_now, self.odom_off)
            yaw_s = ex.angle_diff(self.odom_yaw + self.odom_off, 0.0)
        # smoothed offset from the line (+ = left), on odometry pulled slowly towards AMCL
        fx, fy = self._fuse(x, y, yaw, dt)
        off = math.cos(heading) * (fy - a[1]) - math.sin(heading) * (fx - a[0])
        if self.lat is None or not dt:
            self.lat = off if self.lat is None else self.lat
        else:
            self.lat += min(1.0, dt / p.lat_tau) * (off - self.lat)
        self._learn_crab(x, y, yaw_s)
        rate = 0.0
        if dt:
            rate = ex.angle_diff(yaw_s, self.hold_last[1]) / dt
            rate = 0.5 * rate + 0.5 * self.hold_rate
        self.hold_last, self.hold_rate = (now, yaw_s), rate
        bend = max(-p.hold_lat_max, min(p.hold_lat_max, p.hold_lat * self.lat))
        err = ex.angle_diff(heading - bend, yaw_s + self.crab)
        e = 0.0 if abs(err) < p.hold_deadband else err - math.copysign(p.hold_deadband, err)
        # small correction: the turn rate wanted, limited
        far = min(1.0, abs(self.lat) / p.hold_far)
        rate_max = p.hold_rate_max + far * (p.hold_rate_far - p.hold_rate_max)
        want_rate = max(-rate_max, min(rate_max, p.hold_kp * e))
        # trim (hold_i): learns, from the gyro, the power difference that gives the wanted
        # turn rate - i.e. cancels the robot's own veer; kept across paths
        if dt:
            trim = self.hold_i + p.hold_ki * (want_rate - rate) * dt
            self.hold_i = max(-p.hold_i_max, min(p.hold_i_max, trim))
        pd = max(-p.hold_max, min(p.hold_max, want_rate + p.hold_kr * (want_rate - rate)))
        want = max(-p.hold_total, min(p.hold_total, self.hold_i + pd))
        step = p.hold_slew * (dt or 0.1)      # first tick too: no jump after a curve
        want = max(self.hold_steer - step, min(self.hold_steer + step, want))
        self.hold_steer = want
        return want

    def _fuse(self, x, y, yaw, dt):
        """Map position for the lane hold: each odometry step (turned into the map frame), with
        AMCL's position blended in over pos_tau s. AMCL jumps and wanders by several cm where
        it sees few walls (open floor); steering on it made the robot chase those errors and
        really drift off. Without odometry: AMCL's position."""
        p = self.p
        if self.odom_xy is None or self.odom_yaw is None:
            return x, y
        ox, oy = self.odom_xy
        if self.fused is None or self.odom_xy_last is None:
            self.fused = (x, y)
        else:
            a = ex.angle_diff(yaw, self.odom_yaw)          # map heading - odom heading
            if self.odom_off is not None:
                a = self.odom_off
            dx, dy = ox - self.odom_xy_last[0], oy - self.odom_xy_last[1]
            c, s = math.cos(a), math.sin(a)
            fx, fy = self.fused[0] + c * dx - s * dy, self.fused[1] + s * dx + c * dy
            w = min(1.0, dt / p.pos_tau) if dt else 0.0
            self.fused = (fx + w * (x - fx), fy + w * (y - fy))
        self.odom_xy_last = (ox, oy)
        return self.fused

    def _learn_crab(self, x, y, yaw_s):
        """Every `crab_window` m of steady driving: the angle between the way the robot went
        and its (smoothed) heading. Kept across paths: it is the robot's (and AMCL's) own."""
        p = self.p
        if self.crab_from is None or abs(self.hold_rate) > p.crab_rate_max:
            self.crab_from = (x, y, yaw_s)
            return
        x0, y0, yaw0 = self.crab_from
        d = math.hypot(x - x0, y - y0)
        if d < p.crab_window:
            return
        self.crab_from = (x, y, yaw_s)
        mean_yaw = yaw0 + 0.5 * ex.angle_diff(yaw_s, yaw0)
        sample = ex.angle_diff(math.atan2(y - y0, x - x0), mean_yaw)
        if abs(sample) > 2 * p.crab_max:              # a pose jump, not a crab
            return
        self.crab_n += 1
        gain = max(p.crab_gain, 1.0 / (self.crab_n + 1))  # first samples count more
        self.crab += gain * (sample - self.crab)
        self.crab = max(-p.crab_max, min(p.crab_max, self.crab))

    def _blocked(self, now, x, y, yaw, us_left, us_right, points):
        if not self._front_blocked(us_left, us_right, points):
            self._enter('drive', now, 'following the path')
            return (0.0, 0.0)
        if now - self.entered >= self.p.block_wait:
            ahead, side = self._obstacle_ahead(us_left, us_right, points)
            self.stall = (x, y, yaw, ahead, side)
            return self._give_way(now, 'obstacle stayed')
        return (0.0, 0.0)

    def _creep(self, now, x, y, yaw, us_left, us_right, points):
        if now - self.entered < self.p.creep_time and \
                not self._front_blocked(us_left, us_right, points):
            # a slight turn away from whatever is closest beside the robot
            steer = 0.0
            if len(points):
                d = np.hypot(points[:, 0], points[:, 1])
                k = int(np.argmin(d))
                if d[k] < self.p.turn_radius and abs(points[k, 1]) > 0.05:
                    steer = -math.copysign(self.p.creep_steer, points[k, 1])
            return (self.p.drive_speed, steer)
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
            if not self.tight and not self._turn_blocked(us_left, us_right, points,
                                                         self.p.turn_radius_tight):
                self._enter('turn', now, 'turning with less room (no room behind)')
                self.tight = True
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
            reach = self.arcs[c]['lead'] if c in self.arcs else self.p.corner_reach
            if ahead <= reach and self.index >= c - int((1.0 + reach) / self.p.spacing):
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
        """Heading error to a point on the path ahead: `lookahead` m past the nearest point
        reached, or further when the robot is off the path, so that it rejoins at no more
        than rejoin_angle - forwards, not back to where it left the path."""
        p = self.p
        nx, ny = self.path[self.index]
        ahead = max(p.lookahead, math.hypot(nx - x, ny - y) / math.tan(p.rejoin_angle))
        k = min(len(self.path) - 1, self.index + int(ahead / p.spacing))
        tx, ty = self.path[k]
        if math.hypot(tx - x, ty - y) < 0.05:
            return 0.0
        return ex.angle_diff(math.atan2(ty - y, tx - x), yaw)

    def _enter(self, state, now, message):
        self.state, self.message, self.entered = state, message, now
        if state in ('drive', 'idle', 'failed', 'done', 'error'):
            self.tight = False
        self._reset_timers(now)
        self._phase('settle', now)            # a turn starts by letting the pose settle
        if state == 'drive':
            # start steering from the learnt veer correction (straight ahead for this robot):
            # ramping up from 0 let the veer pull it off the line first
            self.hold_steer = self.hold_i

    def _phase(self, phase, now):
        self.phase, self.phase_start = phase, now

    def _reset_timers(self, now=0.0):
        self.spins, self.still_since, self.stopped_at = 0, None, None
        self.spin_from = 0.0
        self.gentle = False
        self.backing_from_turn = False
        self.turn_heading = None           # set for a corner: turn to the next stretch
        self.hold_last, self.hold_rate, self.hold_steer = None, 0.0, 0.0
        self.arc, self.arc_k, self.arc_w, self.arc_t = None, 0, 0.0, None
        self.arc_last = None
        self.odom_off, self.lat, self.crab_from = None, None, None
        self.fused, self.odom_xy_last = None, None     # lane-hold position (see _fuse)
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
        hl, hw = self.p.half_length, self.p.half_width + self.p.straight_margin
        px, py = points[:, 0], points[:, 1]
        # from the front edge on: a wall right beside the robot is not in its way when it
        # drives straight (it was, and left the robot "boxed in" next to a wall)
        if np.any((np.abs(py) <= hw) & (px >= hl - EDGE) & (px <= hl + self.p.stop_distance)):
            return 'LiDAR: object within 1 ft ahead'
        return None

    def _turn_blocked(self, us_left, us_right, points, radius=None):
        """Why turning on the spot would hit something, or None."""
        if radius is None:
            radius = self.p.turn_radius_tight if self.tight else self.p.turn_radius
        if min(us_left, us_right) < self.p.turn_front:
            return 'too close in front to turn'
        if len(points) and np.any(np.hypot(points[:, 0], points[:, 1]) < radius):
            return 'too close to turn'
        return None

    def _obstacle_ahead(self, us_left, us_right, points):
        """(m ahead of the centre, m to the left) where the obstacle in front starts."""
        hl, hw = self.p.half_length, self.p.half_width + self.p.straight_margin
        best = (hl + min(us_left, us_right), 0.0) if min(us_left, us_right) <= \
            self.p.stop_distance else None
        if len(points):
            ahead = points[(np.abs(points[:, 1]) <= hw) & (points[:, 0] >= hl - EDGE)
                           & (points[:, 0] <= hl + self.p.stop_distance)]
            if len(ahead):
                k = int(np.argmin(ahead[:, 0]))
                if best is None or ahead[k, 0] < best[0]:
                    best = (float(ahead[k, 0]), float(ahead[k, 1]))
        return best if best is not None else (hl, 0.0)

    def _rear_blocked(self, points):
        if len(points) == 0:
            return False
        hl, hw = self.p.half_length, self.p.half_width + self.p.straight_margin
        # from the back edge on (not beside the robot), as for the front
        return bool(np.any((points[:, 0] >= -(hl + self.p.rear_room))
                           & (points[:, 0] <= -(hl - EDGE))
                           & (np.abs(points[:, 1]) <= hw)))
