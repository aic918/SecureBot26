"""Drive coverage laps round a saved room: ROS wrapper around coverage_logic.LapRunner and
lap_follower.LapFollower.

Plans laps from the saved map (coverage_logic). Each loop is driven by LapFollower, a
turn-then-drive follower made for SecureBot's one-speed motors (Nav2's controllers need fine
speed control the motors do not have). To reach the start of a loop it asks Nav2's planner for
a path round the furniture and follows that first. Its commands go through Nav2's collision
monitor (/cmd_vel_smoothed -> /nav_cmd_vel), so that check and the security gate still have
the last word.

Inputs   TF map -> base_link (AMCL), /scan, /ultrasonic/left, /ultrasonic/right,
         Nav2's compute_path_to_pose action,
         /laps/command  std_msgs/String  start | stop     (the Start / Stop buttons)
Obstacle boxes: whenever something stops the robot for good (an obstacle that stays in front,
or something too low for the LiDAR and ultrasonics: driving with a clear way but not moving),
a debris_size m orange box goes where it starts, for this run only (cleared when the next run
starts, never saved in the map). The follower treats its edges as scan points, Nav2's global
costmap gets it (debris_layer), and after backing up the robot steps round it close by
(debris.sidestep: SIDESTEP_GAP beside it, back on the lane REJOIN_GAP past it) where that is
clear of walls, other boxes, no-pass lines and the live scan, else asks Nav2 for a way round it
to the rest of the lane, so every lane is still driven.
No way found by Nav2: usually the robot itself is the problem (right next to a new box, inside
its safety margin), not the lane: back off and ask again (PLAN_RECOVERIES times) before the
lane counts as failed; STUCK_LANES lanes in a row like that stop the run instead of skipping
every lane in a second.
Boxed in (no room to turn or back up): wait BOXED_WAIT s and try the same lane again, at most
BOXED_TRIES times, then stop - instead of skipping lane after lane without moving.
Position check: whenever the robot stands still (settling after a turn, waiting at an
obstacle; at most every CHECK_PERIOD s) the live scan is matched against the saved map around
AMCL's pose (scan_match); if a clearly better pose is found it is sent to AMCL.
Lost (2026-11-08): when those checks keep matching badly with no better pose nearby, the
robot stops and searches the map for itself (ScanMatcher.relocalize, near AMCL's pose first),
sends a clear answer to AMCL and drives on from there (LOST_* / RELOC_* below).

Outputs  /cmd_vel_smoothed  geometry_msgs/Twist  while running (collision monitor input)
         /initialpose  geometry_msgs/PoseWithCovarianceStamped  position corrections
         /laps/path     nav_msgs/Path  every planned loop, for RViz (latched)
         /nav_cmd_vel   geometry_msgs/Twist  backing up only (straight to the motion manager):
                        the collision monitor's stop zone is in front and stops reversing
                        too, so a robot with something close in front could never back away;
                        the follower checks the LiDAR behind before and while reversing
         /laps/current  nav_msgs/Path  the path being driven now: way to the lane + the lane
                        (latched, empty when stopped)
         /laps/points   visualization_msgs/MarkerArray  the lanes' points and numbers (latched)
         /laps/debris_map  nav_msgs/OccupancyGrid  the run's blocks, room map's size (latched)
         /laps/debris   visualization_msgs/MarkerArray  the blocks as cubes, for RViz (latched)
         /laps/status   std_msgs/String  JSON {state, message, room, round, loop, loops,
                        coverage, length, remaining, skipped}, 2 Hz

Run it with securebot_laps.launch.py.
"""
import json
import math
import os
import threading
import time

import cv2
import numpy as np
import rclpy
import yaml
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped, Twist
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, LaserScan, Range
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from securebot_ros import coverage_logic as cl
from securebot_ros import route_logic as rl
from securebot_ros import debris as db
from securebot_ros import lane_logic as ll
from securebot_ros import nopass as npz
from securebot_ros.home_marker import home_markers, home_path_marker
from securebot_ros.lap_follower import LapFollower
from securebot_ros.scan_match import ScanMatcher

SENSOR_STALE = 1.0        # s: ultrasonic / scan older than this counts as silent
TRANSIT_DIRECT = 0.8      # m: closer than this to a loop's start, drive straight there
CHECK_PERIOD = 3.0        # s between position checks while standing still
# Lost: AMCL can slip metres along a long plain corridor (2026-11-08 hallway10: 3.4 m), more
# than the position check's +-0.4 m reaches. LOST_CHECKS checks in a row matching under
# LOST_SCORE with no correction nearby: stop and search the map for the robot (within
# RELOC_RADIUS m first, then all of it); a pose is taken only with RELOC_MIN of the scan and
# RELOC_MARGIN more than any other place. No clear answer: drive on as it was (a crowd round
# the robot also lowers the match), check again after RELOC_COOLDOWN s, and after
# RELOC_GIVE_UP unclear searches in a row stop the run instead of driving on lost.
LOST_SCORE = 0.55
LOST_CHECKS = 3
RELOC_RADIUS = 6.0
RELOC_MIN = 0.80
RELOC_MARGIN = 0.15
RELOC_COOLDOWN = 20.0
RELOC_GIVE_UP = 3
RELOC_SETTLE = 1.5        # s standing after a new pose was sent (AMCL takes it)
COSTMAP_WAIT = 1.5        # s after publishing a block before asking Nav2 for a way round it
PREVIEW_MOVE = 0.3        # m the robot moves (while idle) before the preview is re-planned
LEFT_MIN = 0.3            # m of a failed lane left: less than this counts as done
MAX_DETOURS = 4           # ways round boxes per lane, then the lane counts as failed
PLAN_RECOVERIES = 2       # Nav2 finds no way from here: back off and ask again this often ...
STUCK_LANES = 3           # ... and if that fails for this many lanes in a row, stop the run
RECOVER_BACK = 0.5        # s reversing (if clear behind) before asking Nav2 again
RECOVER_WAIT = 1.5        # s standing after that (costmap update)
BOXED_WAIT = 5.0          # s to wait before trying a lane again when the robot was boxed in
BOXED_TRIES = 6           # ... this many times (things round it may move), then stop the run


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def corners(path, turn_deg=20.0):
    """The points of `path` where it bends by turn_deg or more, and its end."""
    if len(path) < 3:
        return list(path[-1:])
    out = []
    for a, b, c in zip(path, path[1:], path[2:]):
        h1 = math.atan2(b[1] - a[1], b[0] - a[0])
        h2 = math.atan2(c[1] - b[1], c[0] - b[0])
        if abs(math.atan2(math.sin(h2 - h1), math.cos(h2 - h1))) >= math.radians(turn_deg):
            out.append(b)
    return out + [path[-1]]


def rest_of(lap, xy, off_max=0.5):
    """What is left of the lane `lap` ([(x, y), ...]) from the robot at `xy`: None when the
    robot is not on it (more than off_max m off), [] when less than LEFT_MIN m is left."""
    best = None
    for k, (a, b) in enumerate(zip(lap, lap[1:])):
        length = math.dist(a, b)
        if length < 1e-6:
            continue
        ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        along = min(max((xy[0] - a[0]) * ux + (xy[1] - a[1]) * uy, 0.0), length)
        px, py = a[0] + ux * along, a[1] + uy * along
        off = math.dist(xy, (px, py))
        if best is None or off < best[0]:
            best = (off, k, (px, py))
    if best is None or best[0] > off_max:
        return None
    _, k, proj = best
    if k == 0 and math.dist(proj, lap[0]) < 1e-6:      # not on the lane yet: all of it
        return None
    rest = [proj] + [tuple(q) for q in lap[k + 1:]]
    if sum(math.dist(a, b) for a, b in zip(rest, rest[1:])) < LEFT_MIN:
        return []
    return rest


class LapsNode(Node):
    def __init__(self):
        super().__init__('laps_node')
        defaults = cl.Params()
        self.declare_parameter('map', '')
        self.declare_parameter('room', '')
        self.declare_parameter('rounds', 0)                   # 0 = keep lapping until Stop
        self.declare_parameter('lane_spacing', defaults.lane_spacing)
        self.declare_parameter('wall_clearance', defaults.wall_clearance)
        self.declare_parameter('pattern', 'auto')   # 'auto', 'corridors', 'lanes' or 'contour'
        self.declare_parameter('require_gyro', True)          # refuse Start without /imu
        self.declare_parameter('home', '')                    # 'x,y': fixed HOME, else START
        self.declare_parameter('side_clearance', defaults.side_clearance)
        self.declare_parameter('front_clearance', defaults.front_clearance)
        self.declare_parameter('lane_gap', defaults.check_width)     # m between lanes
        self.declare_parameter('min_pocket', defaults.min_pocket)   # m²: pocket gets a lane
        self.declare_parameter('debris_size', db.DEFAULT_SIZE)
        get = lambda n: self.get_parameter(n).value  # noqa: E731
        params = cl.Params(lane_spacing=float(get('lane_spacing')),
                           wall_clearance=float(get('wall_clearance')),
                           pattern=str(get('pattern')),
                           side_clearance=float(get('side_clearance')),
                           front_clearance=float(get('front_clearance')),
                           check_width=float(get('lane_gap')),
                           min_pocket=float(get('min_pocket')))
        self.unit = 'loop' if params.pattern == 'contour' else 'lane'
        self.room = get('room') or get('map')
        # the room's map, plus the no-pass lines drawn on it (walls for planning and driving)
        self.base_m = cl.load_map(get('map'))
        self.nopass = npz.NoPass(npz.path_for(get('map')))
        self.m = self.nopass.apply(self.base_m)
        self.drawing, self.draw_last = False, None
        # Return home: back to where the last START was pressed (Nav2's way), then stop
        self.home, self.homing, self.home_tries = None, False, 0
        with open(get('map')) as f:
            image = yaml.safe_load(f)['image']
        image = image if os.path.isabs(image) else os.path.join(os.path.dirname(get('map')), image)
        self.matcher = ScanMatcher(cv2.imread(image, cv2.IMREAD_GRAYSCALE), self.m.resolution,
                                   self.m.origin)
        self.last_check, self.corrections = 0.0, 0
        self.low_checks, self.reloc, self.reloc_after, self.reloc_unclear = 0, None, 0.0, 0
        self.runner = cl.LapRunner(self.m, params, rounds=int(get('rounds')))
        self.follower = LapFollower()
        self.debris = db.Debris(float(get('debris_size')), self.follower.p.half_length)
        # clear of a block to rejoin the lane: half the robot's length plus a margin
        # (was + 0.25 until 2026-11-08: the robot gave 8 in boxes too much room)
        self.rejoin_clearance = self.follower.p.half_length + db.REJOIN_GAP
        self.last_block = None
        self.detour, self.detours, self.debris_time = None, 0, 0.0
        self.boxed, self.retry_at = 0, 0.0
        self.plan_tries, self.plan_fail_lanes, self.goal_loop = 0, 0, None
        self.recover_start = 0.0
        self.phase = None                  # 'plan' (asking Nav2 for a way to the loop) | 'drive'
        self.plan_seq = 0                  # planner results of older requests are ignored

        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.planner = ActionClient(self, ComputePathToPose, 'compute_path_to_pose')
        self.us = {'left': (None, 0.0), 'right': (None, 0.0)}
        self.scan_points, self.scan_time = [], 0.0
        self.state, self.message = 'idle', 'Press START'

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.path_pub = self.create_publisher(Path, '/laps/path', latched)
        self.current_pub = self.create_publisher(Path, '/laps/current', latched)
        self.points_pub = self.create_publisher(MarkerArray, '/laps/points', latched)
        self.debris_map_pub = self.create_publisher(OccupancyGrid, '/laps/debris_map', latched)
        self.debris_pub = self.create_publisher(MarkerArray, '/laps/debris', latched)
        self.nopass_pub = self.create_publisher(MarkerArray, '/laps/nopass', latched)
        self.home_pub = self.create_publisher(MarkerArray, '/laps/home', latched)
        self.status_pub = self.create_publisher(String, '/laps/status', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_smoothed', 10)
        self.reverse_pub = self.create_publisher(Twist, '/nav_cmd_vel', 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.create_subscription(String, '/laps/command', self.command_callback, 10)
        self.home = self.fixed_home()      # a fixed HOME shows from the start
        self.publish_home([])
        # RViz 'Publish Point' tool: draws no-pass lines (after 'nopass draw'), else a click
        # on an orange box removes it (this run), or on a no-pass line removes that line
        self.create_subscription(PointStamped, '/clicked_point', self.clicked, 10)
        self.create_subscription(Range, '/ultrasonic/left', lambda m: self.range_cb('left', m), 10)
        self.create_subscription(Range, '/ultrasonic/right',
                                 lambda m: self.range_cb('right', m), 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.imu_time = 0.0
        self.create_subscription(Imu, '/imu', self.imu_callback, 20)
        self.create_timer(0.1, self.tick)
        self.create_timer(0.5, self.publish_status)

        self.publish_debris()
        self.publish_nopass()
        if self.nopass.lines:
            self.get_logger().info(f'{len(self.nopass.lines)} no-pass line(s) from '
                                   f'{self.nopass.path}')
        # shown before Start; re-planned from the robot's pose once it is known (lane 1 starts
        # where the robot is), in the background: planning takes a few seconds
        preview = cl.plan(self.m, params)
        self.publish_path(preview.loops)
        self.preview = preview
        self.params = params
        self.preview_from, self.preview_busy, self.preview_ready = None, False, None
        self.create_timer(2.0, self.refresh_preview)
        self.get_logger().info(
            f'{self.room}: {len(preview.loops)} {self.unit}s '
            f'({preview.pattern or params.pattern}), {preview.length:.0f} m per round, '
            f'{preview.coverage:.0%} of {preview.area:.0f} m² swept. Waiting for Start')

    # ---------------------------------------------------------------- inputs
    def imu_callback(self, msg):
        self.imu_time = time.monotonic()

    def range_cb(self, side, msg):
        self.us[side] = (msg.range, time.monotonic())

    def scan_callback(self, msg):
        try:
            t = self.tf.lookup_transform('base_link', msg.header.frame_id, rclpy.time.Time())
        except Exception:
            return
        tx, ty = t.transform.translation.x, t.transform.translation.y
        yaw = yaw_of(t.transform.rotation)
        c, s = math.cos(yaw), math.sin(yaw)
        points = []
        for k, r in enumerate(msg.ranges):
            if math.isfinite(r) and msg.range_min <= r <= msg.range_max:
                a = msg.angle_min + k * msg.angle_increment
                lx, ly = r * math.cos(a), r * math.sin(a)
                points.append((tx + c * lx - s * ly, ty + s * lx + c * ly))
        self.scan_points, self.scan_time = points, time.monotonic()

    def pose(self, frame='map'):
        try:
            t = self.tf.lookup_transform(frame, 'base_link', rclpy.time.Time())
        except Exception:
            return None
        tr = t.transform
        return tr.translation.x, tr.translation.y, yaw_of(tr.rotation)

    def publish_debris(self):
        """The run's blocks: as a grid the size of the room map (costmap) and as cubes."""
        m = self.m
        rows, cols = m.free.shape
        grid = OccupancyGrid()
        grid.header.frame_id = 'map'
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.info.resolution = m.resolution
        grid.info.width, grid.info.height = cols, rows
        grid.info.origin.position.x, grid.info.origin.position.y = m.origin
        grid.info.origin.orientation.w = 1.0
        g = self.debris.grid(m.resolution, m.origin, cols, rows)
        if self.nopass.lines:                  # no-pass lines: walls for Nav2 too
            g[np.flipud(self.nopass.mask(m))] = 100
        grid.data = g.ravel().tolist()
        self.debris_map_pub.publish(grid)
        markers = MarkerArray()
        wipe = Marker()
        wipe.action = Marker.DELETEALL
        markers.markers.append(wipe)
        for k, b in enumerate(self.debris.blocks):
            mk = Marker()
            mk.header = grid.header
            mk.ns, mk.id, mk.type, mk.action = 'debris', k, Marker.CUBE, Marker.ADD
            mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = b.x, b.y, 0.1
            mk.pose.orientation.w = 1.0
            mk.scale.x = mk.scale.y = b.size
            mk.scale.z = 0.2
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = 1.0, 0.45, 0.0, 0.85
            markers.markers.append(mk)
        self.debris_pub.publish(markers)

    # ---------------------------------------------------------------- helpers
    def stamped(self, x, y, yaw):
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.header.stamp = self.get_clock().now().to_msg()
        p.pose.position.x, p.pose.position.y = x, y
        p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        return p

    def publish_path(self, loops):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        for loop in loops:
            msg.poses += [self.stamped(x, y, yaw) for (x, y), yaw in zip(loop, cl.headings(loop))]
        self.path_pub.publish(msg)
        markers = MarkerArray()
        wipe = Marker()
        wipe.action = Marker.DELETEALL
        markers.markers.append(wipe)
        dots = Marker()
        dots.header = msg.header
        dots.ns, dots.id, dots.type, dots.action = 'lane_points', 0, Marker.SPHERE_LIST, Marker.ADD
        dots.pose.orientation.w = 1.0
        dots.scale.x = dots.scale.y = dots.scale.z = 0.07
        dots.color.r, dots.color.g, dots.color.b, dots.color.a = 1.0, 1.0, 1.0, 1.0
        for k, loop in enumerate(loops):
            for x, y in loop:
                pt = PoseStamped().pose.position
                pt.x, pt.y = float(x), float(y)
                dots.points.append(pt)
            label = Marker()
            label.header = msg.header
            label.ns, label.id = 'lane_numbers', k
            label.type, label.action = Marker.TEXT_VIEW_FACING, Marker.ADD
            label.pose.position.x, label.pose.position.y = float(loop[0][0]), float(loop[0][1])
            label.pose.position.z = 0.15
            label.pose.orientation.w = 1.0
            label.scale.z = 0.18
            label.color.r, label.color.g, label.color.b, label.color.a = 0.0, 0.0, 0.0, 1.0
            label.text = str(k + 1)
            markers.markers.append(label)
        if dots.points:
            markers.markers.append(dots)
        self.points_pub.publish(markers)

    def publish_home(self, path):
        """The HOME mark, and the way home being driven as a magenta line ([] = none)."""
        if self.home is None:
            return
        stamp = self.get_clock().now().to_msg()
        self.home_pub.publish(MarkerArray(markers=home_markers(*self.home, stamp)
                                          + [home_path_marker(path, stamp)]))

    def publish_current(self, path):
        """The path the robot is about to drive ([] = none)."""
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.poses = [self.stamped(float(x), float(y), yaw)
                     for (x, y), yaw in zip(path, cl.headings(path))] if len(path) > 1 else []
        self.current_pub.publish(msg)

    def set_state(self, state, message):
        if (state, message) != (self.state, self.message):
            self.get_logger().info(f'{state}: {message}')
        self.state, self.message = state, message

    def loop_text(self):
        if self.homing:
            return 'Returning home'
        r = self.runner
        return (f'Round {r.round + 1}, {self.unit} {r.number} of {len(r.plan.loops)}'
                + (f' (attempt {r.attempt + 1})' if r.attempt else ''))

    # ---------------------------------------------------------------- commands
    def clicked(self, msg):
        """A click (RViz Publish Point). Drawing: the next corner of a no-pass line. Else, on
        or next to an orange box: remove that box; on a no-pass line: remove that line."""
        if msg.header.frame_id not in ('', 'map'):
            return
        pt = (msg.point.x, msg.point.y)
        if self.drawing:
            if self.draw_last is not None and self.nopass.add(self.draw_last, pt):
                self.get_logger().info(f'No-pass line ({self.draw_last[0]:.2f}, '
                                       f'{self.draw_last[1]:.2f}) - ({pt[0]:.2f}, {pt[1]:.2f})')
                self.nopass_changed()
            self.draw_last = pt
            self.publish_nopass()
            return
        b = self.debris.remove_near(*pt)
        if b is not None:
            self.publish_debris()
            self.get_logger().info(f'Box at ({b.x:.2f}, {b.y:.2f}) removed by click')
            return
        if self.nopass.remove_near(*pt) is not None:
            self.get_logger().info(f'No-pass line at ({pt[0]:.2f}, {pt[1]:.2f}) removed by click')
            self.nopass_changed()
            return
        self.get_logger().info(f'Click at ({pt[0]:.2f}, {pt[1]:.2f}): no box or line there')

    def nopass_command(self, what):
        if what == 'draw':                     # toggles: on, then off (finishes the line)
            self.drawing = not self.drawing
            self.draw_last = None
            self.get_logger().info('No-pass drawing ' + ('on: click the corners of the line '
                                   'in RViz (Publish Point)' if self.drawing else 'off'))
        elif what == 'undo':
            self.nopass.undo()
            self.nopass_changed()
        elif what == 'clear':
            self.nopass.clear()
            self.drawing, self.draw_last = False, None
            self.nopass_changed()
        self.publish_nopass()

    def nopass_changed(self):
        """Lines changed: re-plan on the map with them, tell Nav2 (debris layer)."""
        self.m = self.nopass.apply(self.base_m)
        self.runner.m = self.m                 # used from the next round / Start
        self.preview_from = None               # re-plan the preview now
        self.publish_debris()
        self.publish_nopass()

    def publish_nopass(self):
        markers = MarkerArray()
        wipe = Marker()
        wipe.action = Marker.DELETEALL
        markers.markers.append(wipe)
        stamp = self.get_clock().now().to_msg()
        if self.nopass.lines:
            mk = Marker()
            mk.header.frame_id, mk.header.stamp = 'map', stamp
            mk.ns, mk.id, mk.type, mk.action = 'nopass', 0, Marker.LINE_LIST, Marker.ADD
            mk.pose.orientation.w = 1.0
            mk.scale.x = self.nopass.width
            mk.color.a = 1.0                   # black
            for a, b in self.nopass.lines:
                for x, y in (a, b):
                    p = PoseStamped().pose.position
                    p.x, p.y, p.z = float(x), float(y), 0.05
                    mk.points.append(p)
            markers.markers.append(mk)
        if self.drawing and self.draw_last is not None:
            dot = Marker()
            dot.header.frame_id, dot.header.stamp = 'map', stamp
            dot.ns, dot.id, dot.type, dot.action = 'nopass_start', 0, Marker.SPHERE, Marker.ADD
            dot.pose.position.x, dot.pose.position.y = self.draw_last
            dot.pose.orientation.w = 1.0
            dot.scale.x = dot.scale.y = dot.scale.z = self.nopass.width * 1.5
            dot.color.a = 1.0
            markers.markers.append(dot)
        self.nopass_pub.publish(markers)

    def command_callback(self, msg):
        cmd = msg.data.strip().lower()
        if cmd == 'start':
            self.start()
        elif cmd == 'stop':
            self.stop('Stopped')
        elif cmd == 'home':
            self.go_home()
        elif cmd.startswith('nopass '):
            self.nopass_command(cmd.split(None, 1)[1])

    def fixed_home(self):
        """The 'home' parameter ('x,y') as (x, y), or None (then START sets HOME)."""
        try:
            x, y = (float(v) for v in str(self.get_parameter('home').value).split(','))
            return (x, y)
        except ValueError:
            return None

    def go_home(self):
        """Stop the laps and drive back to where START was pressed; then stop."""
        if self.home is None:
            self.set_state('idle', 'No start point yet: press START first')
            return
        if self.homing:
            return
        self.stop('Returning home')
        self.homing, self.home_tries, self.boxed = True, 1, 0
        self.debris_time, self.detours = 0.0, 0
        self.plan_tries, self.plan_fail_lanes = 0, 0
        self.get_logger().info(f'Returning home to ({self.home[0]:.2f}, {self.home[1]:.2f})')
        self.set_state('running', 'Returning home: heading there')
        self.head_home()

    def head_home(self):
        """Back along the planned lanes (known clear on the map, and the plan starts at HOME);
        Nav2's planner only if the robot is off them, or on the last try."""
        pose = self.pose()
        plan = self.runner.plan or self.preview
        way = None
        if pose is not None and plan is not None and plan.loops and self.home_tries < 3:
            space, _, turn = rl.route_space(self.m, self.params, pose[:2])
            way = rl.route_back(plan.loops, pose[:2], self.home, reach=6.0,
                                clear=lambda a, b: rl._clear(space, self.m, a, b))
            if way is not None and len(way) >= 2:
                # right angles on the plan's grid, like the lanes (odd-angle shortcuts cost
                # 40-65 deg spins on 2026-11-07)
                grid = plan.angle if plan.angle is not None else ll.wall_angle(self.m)
                square, ok = rl.square_path_in(self.m, [turn, space], way, grid,
                                               exact_end=True)
                if ok and len(square) >= 2 and math.dist(square[-1], self.home) < 0.15:
                    way = square
            if way is not None and cl.crosses_wall(self.m, way):
                way = None
        if way is None or len(way) < 2:
            self.get_logger().info('Way home: Nav2 planner')
            self.go_to([self.home])
            return
        length = sum(math.dist(a, b) for a, b in zip(way, way[1:]))
        self.get_logger().info(f'Way home: back along the lanes, {length:.0f} m')
        self.goal_loop = [self.home]
        self.drive(way)

    def home_finished(self, ok, why):
        """The drive home ended: done and stop, or try again (a few times), or give up."""
        self.phase = None
        self.cmd_pub.publish(Twist())
        pose = self.pose()
        if ok or (pose is not None and math.dist(pose[:2], self.home) < 0.3):
            self.homing = False
            self.publish_current([])
            self.publish_home([])
            self.get_logger().info('Back at the start point')
            self.set_state('idle', 'Back at the start point')
            return
        if self.home_tries < 3:
            self.home_tries += 1
            self.get_logger().warn(f'Way home did not work ({why}): trying again')
            if why.startswith('boxed in'):
                self.phase, self.retry_at = 'wait', time.monotonic() + BOXED_WAIT
                return
            self.head_home()
            return
        self.homing = False
        self.publish_current([])
        self.publish_home([])
        self.set_state('error', f'Could not get home: {why}')

    def start(self):
        if self.runner.active:
            return
        self.homing = False
        pose = self.pose()
        if pose is None:
            self.set_state('error', 'Robot position unknown: set "2D Pose Estimate" in RViz '
                                    '(or wait for localization) and press START again')
            return
        if self.get_parameter('require_gyro').value and time.monotonic() - self.imu_time > 2.0:
            self.set_state('error', 'No gyro data (/imu): check the MPU-6050 wiring (VCC, GND, '
                                    'SDA 21, SCL 22) and restart; laps need it to turn')
            return
        if not self.planner.server_is_ready():
            self.set_state('error', 'Nav2 is not ready yet: wait a few seconds and press START')
            return
        self.home = self.fixed_home() or pose[:2]     # Return home comes back here
        self.publish_home([])
        self.debris.clear()            # boxes last for one run: cleared when the next starts
        self.publish_debris()
        self.boxed = 0
        self.low_checks, self.reloc, self.reloc_after, self.reloc_unclear = 0, None, 0.0, 0
        # the plan shown in RViz, if it was made from here (no planning wait at START)
        ready = None
        latest = self.preview_ready if self.preview_ready is not None else self.preview
        if latest is not None and latest.loops and not self.preview_busy and \
                self.preview_from is not None and \
                math.dist(pose[:2], self.preview_from) < PREVIEW_MOVE:
            ready = latest
            self.get_logger().info('Start: driving the plan shown in RViz (no re-planning)')
        if not self.runner.start(pose[:2], ready):
            self.set_state('error', 'No room to drive laps from here: is the robot on the map?')
            return
        self.publish_path(self.runner.plan.loops)
        self.begin_loop()

    def stop(self, message):
        self.reloc, self.low_checks = None, 0
        self.homing = False
        self.runner.stop()
        self.follower.stop(time.monotonic(), message)
        self.phase = None
        self.plan_seq += 1
        self.detour = None
        self.cmd_pub.publish(Twist())
        self.publish_current([])
        self.publish_home([])
        self.set_state('idle', message)

    # ---------------------------------------------------------------- driving
    def begin_loop(self):
        """Get to the current loop's start (via Nav2's planner unless it is close), then lap."""
        loop = self.runner.current()
        self.detours, self.plan_tries = 0, 0
        self.set_state('running', self.loop_text() + ': heading to its start')
        self.go_to(loop)

    def go_to(self, loop):
        """Drive to loop[0] (straight if close, else on Nav2's path) and then along loop."""
        self.goal_loop = loop
        pose = self.pose()
        if pose is not None and math.dist(pose[:2], loop[0]) < TRANSIT_DIRECT \
                and self.debris.distance(*loop[0]) > self.rejoin_clearance:
            self.drive([pose[:2]] + loop)
            return
        goal = ComputePathToPose.Goal()
        if len(loop) > 1:
            yaw = cl.headings(loop)[0]
        else:                                   # a single point (home): arrive facing it
            yaw = math.atan2(loop[0][1] - pose[1], loop[0][0] - pose[0]) if pose else 0.0
        goal.goal = self.stamped(*loop[0], yaw)
        goal.use_start = False
        self.plan_seq += 1
        seq = self.plan_seq
        self.phase = 'plan'
        self.planner.send_goal_async(goal).add_done_callback(
            lambda f: self.plan_accepted(f, seq, loop))

    def plan_accepted(self, future, seq, loop):
        handle = future.result()
        if seq != self.plan_seq:
            return
        if not handle.accepted:
            self.plan_failed('planner refused the way to the loop')
            return
        handle.get_result_async().add_done_callback(lambda f: self.planned(f, seq, loop))

    def planned(self, future, seq, loop):
        if seq != self.plan_seq:
            return
        res = future.result()
        poses = res.result.path.poses if res.status == GoalStatus.STATUS_SUCCEEDED else []
        if not poses:
            self.plan_failed('no way to the loop found')
            return
        transit = [(p.pose.position.x, p.pose.position.y) for p in poses]
        hit = cl.crosses_wall(self.m, transit)
        if hit is not None:
            # never drive through a wall of the room map, whatever the costmap said
            self.get_logger().warn(f'Nav2 way to the loop crosses a wall at ({hit[0]:.2f}, '
                                   f'{hit[1]:.2f}): not taken')
            self.plan_failed('the only way found crosses a wall')
            return
        self.plan_tries, self.plan_fail_lanes = 0, 0
        self.drive(transit + loop)

    def plan_failed(self, why):
        """Nav2 found no way from here: back off and ask again before blaming the lane."""
        self.phase = None
        self.plan_tries += 1
        if self.plan_tries <= PLAN_RECOVERIES:
            self.phase, self.recover_start = 'recover', time.monotonic()
            self.set_state('running', f'{self.loop_text()}: {why}; backing off to try again '
                                      f'({self.plan_tries}/{PLAN_RECOVERIES})')
            return
        self.plan_tries = 0
        if self.homing:                          # no Nav2 way home: try again or give up
            self.home_finished(False, why)
            return
        self.plan_fail_lanes += 1
        if self.plan_fail_lanes >= STUCK_LANES:
            self.runner.stop()
            self.cmd_pub.publish(Twist())
            self.publish_current([])
            self.set_state('error', f'Stopped: Nav2 finds no way from here to '
                                    f'{STUCK_LANES} {self.unit}s in a row (robot stuck next to '
                                    'something?). Move it a little and press START')
            return
        self.loop_finished(False, why)

    def recover(self, now):
        """Back off a little (if the LiDAR sees room behind), stand, then ask Nav2 again."""
        t = now - self.recover_start
        if t < RECOVER_BACK:
            pose = self.pose()
            points = list(self.scan_points) if now - self.scan_time <= SENSOR_STALE else []
            if pose is not None and self.debris.blocks:
                points += self.debris.points_in_base(pose)
            if pose is not None and self.nopass.lines:
                points += self.nopass.points_in_base(pose)
            pts = np.asarray(points, dtype=float).reshape(-1, 2)
            if points and not self.follower._rear_blocked(pts):
                msg = Twist()
                msg.linear.x = -self.follower.p.drive_speed
                self.reverse_pub.publish(msg)
                return
            self.recover_start = now - RECOVER_BACK      # no room behind: just wait
        if t < RECOVER_BACK + RECOVER_WAIT:
            self.cmd_pub.publish(Twist())
            return
        self.go_to(self.goal_loop)

    def drive(self, path):
        self.phase = 'drive'
        self.publish_current(path)
        self.publish_home(path if self.homing else [])
        # slow down before the lane's own corners and its end (not Nav2's way there, not
        # the slight bends of a lane that follows a corridor)
        self.follower.start(time.monotonic(), path, waypoints=corners(self.goal_loop))

    def check_position(self, now, points):
        """While standing still, correct AMCL if the scan fits the map clearly better nearby."""
        f = self.follower
        # standing still: waiting at an obstacle, or stopped before a turn's first spin (a
        # correction between the spins of a turn moved the target and made it spin back)
        still = f.state == 'blocked' or (f.state == 'turn' and f.phase == 'settle'
                                         and f.spins == 0 and now - f.phase_start > 0.25)
        if not still or now - self.last_check < CHECK_PERIOD or len(points) < 50:
            return
        self.last_check = now
        pose = self.pose()
        if pose is None:
            return
        score, better = self.matcher.correct(pose, points)
        if better is None:
            if score >= LOST_SCORE:
                self.low_checks = 0
            elif now >= self.reloc_after:
                self.low_checks += 1
                if self.low_checks >= LOST_CHECKS:
                    self.start_relocalize(pose, points, score)
            return
        self.low_checks = 0
        self.send_pose(better)
        self.corrections += 1
        turn = math.atan2(math.sin(better[2] - pose[2]), math.cos(better[2] - pose[2]))
        self.get_logger().info(
            f'Position corrected by {math.dist(pose[:2], better[:2]) * 100:.0f} cm, '
            f'{abs(math.degrees(turn)):.0f} deg (scan match {score:.0%} -> '
            f'{self.matcher.last_best:.0%})')

    def send_pose(self, pose):
        """Tell AMCL the robot is at `pose` (x, y, yaw in the map)."""
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x, msg.pose.pose.position.y = pose[0], pose[1]
        msg.pose.pose.orientation.z = math.sin(pose[2] / 2)
        msg.pose.pose.orientation.w = math.cos(pose[2] / 2)
        cov = [0.0] * 36
        cov[0] = cov[7] = 0.02
        cov[35] = 0.02
        msg.pose.covariance = cov
        self.pose_pub.publish(msg)

    # ------------------------------------------------------------ lost: find it on the map
    def start_relocalize(self, pose, points, score):
        """Stop and search the map for the robot (in a thread: seconds to a minute)."""
        self.low_checks = 0
        self.detour = None
        self.follower.stop(time.monotonic(), 'finding its position')
        self.cmd_pub.publish(Twist())
        r = {'phase': 'search', 'pose': pose, 'score': score, 'result': None,
             'goal': self.goal_loop, 'homing': self.homing,
             'points': np.asarray(points, dtype=float).reshape(-1, 2).copy()}
        self.reloc, self.phase = r, 'relocalize'
        self.set_state('home' if self.homing else 'running',
                       f'{self.loop_text()}: position lost (scan match {score:.0%}), '
                       'finding it on the map')
        threading.Thread(target=self.relocalize_search, args=(r,), daemon=True).start()

    def relocalize_search(self, r):
        start = time.monotonic()
        clear = lambda best, second: best >= RELOC_MIN and best - second >= RELOC_MARGIN  # noqa
        best, found, second = self.matcher.relocalize(r['points'], around=r['pose'][:2],
                                                      radius=RELOC_RADIUS)
        where = f'within {RELOC_RADIUS:.0f} m'
        if found is None or not clear(best, second):
            best, found, second = self.matcher.relocalize(r['points'])
            where = 'whole map'
        ok = found is not None and clear(best, second)
        r['result'] = (ok, best, found, second, where, time.monotonic() - start)

    def relocalize_step(self, now):
        """While lost: stand still until the search is done, then drive on from there."""
        self.cmd_pub.publish(Twist())
        r = self.reloc
        if r is None:
            self.phase = None
            return
        if r['phase'] == 'search':
            if r['result'] is None:
                return
            ok, best, found, second, where, took = r['result']
            if ok:
                self.send_pose(found)
                self.reloc_unclear = 0
                x, y, a = r['pose']
                turn = math.degrees(abs(math.atan2(math.sin(found[2] - a),
                                                   math.cos(found[2] - a))))
                self.get_logger().warn(
                    f'Position lost and found ({where}, {took:.0f} s): moved '
                    f'{math.hypot(found[0] - x, found[1] - y):.2f} m, {turn:.0f} deg to '
                    f'({found[0]:.2f}, {found[1]:.2f}) (scan match {r["score"]:.0%} -> '
                    f'{best:.0%}, next place {second:.0%})')
                r['phase'], r['until'] = 'settle', now + RELOC_SETTLE
                return
            self.reloc_unclear += 1
            self.get_logger().warn(
                f'Position check low ({r["score"]:.0%}) but no clear place on the map (best '
                f'{best:.0%}, next {second:.0%}, {took:.0f} s): driving on as it was '
                f'({self.reloc_unclear}/{RELOC_GIVE_UP})')
            if self.reloc_unclear >= RELOC_GIVE_UP:
                self.reloc_unclear = 0
                self.stop('Stopped: position lost and not found on the map. Set "2D Pose '
                          'Estimate" in RViz and press START')
                self.set_state('error', self.message)
                return
            self.reloc_after = now + RELOC_COOLDOWN
            r['phase'], r['until'] = 'settle', now
            return
        if now < r['until']:
            return
        self.reloc, self.phase = None, None
        if r['homing']:
            self.head_home()
            return
        loop = r['goal'] or self.runner.current()
        pose = self.pose()
        if loop and pose is not None:
            k = min(range(len(loop)), key=lambda i: math.dist(loop[i], pose[:2]))
            loop = loop[k:] if len(loop) - k >= 2 else loop[-2:]
        self.go_to(loop)

    def refresh_preview(self):
        """While idle: re-plan the preview from where the robot is now, when it has moved."""
        if self.preview_ready is not None and not self.runner.active:
            plan, self.preview_ready = self.preview_ready, None
            self.preview = plan
            self.publish_path(plan.loops)
            self.get_logger().info(f'Plan from the robot: {len(plan.loops)} {self.unit}s '
                                   f'({plan.pattern or self.params.pattern}), '
                                   f'{plan.length:.0f} m, {plan.coverage:.0%} checked')
        if self.runner.active or self.preview_busy:
            return
        pose = self.pose()
        if pose is None or (self.preview_from is not None
                            and math.dist(pose[:2], self.preview_from) < PREVIEW_MOVE):
            return
        self.preview_from, self.preview_busy = pose[:2], True

        def work(start):
            try:
                self.preview_ready = cl.plan(self.m, self.params, start)
            except Exception as e:                     # keep the old preview
                self.get_logger().warn(f'Preview plan failed: {e}')
            finally:
                self.preview_busy = False
        threading.Thread(target=work, args=(pose[:2],), daemon=True).start()

    def tick(self):
        if self.phase == 'relocalize':
            self.relocalize_step(time.monotonic())
            return
        if self.phase == 'wait' and time.monotonic() >= self.retry_at:
            # boxed in: same lane (or the way home) again
            if self.homing:
                self.head_home()
            else:
                self.go_to(self.runner.current())
            return
        if self.phase == 'recover':
            self.recover(time.monotonic())
            return
        if self.phase != 'drive':
            return
        now = time.monotonic()
        fresh = lambda value, stamp: value if now - stamp <= SENSOR_STALE else None  # noqa: E731
        points = self.scan_points if now - self.scan_time <= SENSOR_STALE else []
        self.check_position(now, points)
        pose = self.pose()
        if pose is not None and self.debris.blocks and points:
            points = list(points) + self.debris.points_in_base(pose)
        if pose is not None and self.nopass.lines and points:
            points = list(points) + self.nopass.points_in_base(pose)
        odom = self.pose('odom')
        cmd = self.follower.step(now, pose, fresh(*self.us['left']),
                                 fresh(*self.us['right']), points,
                                 odom_yaw=odom[2] if odom else None,
                                 odom_xy=odom[:2] if odom else None)
        f = self.follower
        if f.stall is not None:
            self.add_debris(now, f)
        if self.detour is not None and f.state != 'backup':
            # backed off the block: stand still until the costmap has it, then go round it
            self.cmd_pub.publish(Twist())
            if now - self.debris_time >= COSTMAP_WAIT:
                self.go_round()
            else:
                self.set_state('running', f'{self.loop_text()}: block ahead, planning round it')
            return
        if cmd is not None:
            msg = Twist()
            msg.linear.x, msg.angular.z = cmd
            if f.state == 'backup' and cmd[0] < 0.0:
                self.reverse_pub.publish(msg)          # past the front-only stop zone
            else:
                self.cmd_pub.publish(msg)
            if f.running:
                self.set_state('running', f'{self.loop_text()}: {f.message}')
            return
        self.cmd_pub.publish(Twist())
        if f.state == 'error':                     # sensors silent: stop everything
            self.runner.stop()
            self.phase = None
            self.set_state('error', f'Stopped: {f.message}')
        else:
            self.loop_finished(f.state == 'done', f.message)

    def add_debris(self, now, f):
        """Something stopped the robot for good: box it (this run only) and go round it."""
        x, y, yaw, ahead, side = f.stall
        f.stall = None
        block = self.debris.add_ahead(x, y, yaw, ahead, side)
        self.last_block = block
        if block is not None:
            self.publish_debris()
            self.get_logger().warn(
                f'Obstacle ({f.message}): {block.size:.2f} m box put at '
                f'({block.x:.2f}, {block.y:.2f}) for this run, going round it')
        self.debris_time = now
        if self.detours < MAX_DETOURS:
            # the rest of the path, from where the robot got stuck (after its back-up)
            self.detour = [tuple(p) for p in f.path[max(0, f.index - 20):]]
            self.detour_from = (x, y)

    def go_round(self):
        """After a block was put down: Nav2's way round it to where the path is clear again."""
        path, (x, y) = self.detour, self.detour_from
        self.detour = None
        self.detours += 1
        self.follower.stop(time.monotonic(), 'going round a block')
        near = min(range(len(path)), key=lambda k: math.dist(path[k], (x, y)))
        k = db.resume_index(path, near, self.debris, self.rejoin_clearance)
        if k is None or len(path) - k < 3:
            # no clear way back onto this lane past the boxes from here: what is left is
            # tried from the lane's other end later (loop_finished)
            self.loop_finished(False, 'rest of the lane blocked from this side')
            return
        self.set_state('running', f'{self.loop_text()}: going round a block')
        # first a tight sidestep (gap SIDESTEP_GAP beside the block) where it is clear; Nav2's
        # way round (inflated costmap: wide swings) only when it is not
        pose = self.pose()
        block = self.last_block or min(self.debris.blocks, default=None,
                                       key=lambda b: math.hypot(b.x - x, b.y - y))
        if pose is not None and block is not None:
            p = self.follower.p
            way = db.sidestep(pose[:2], path, near, k, block, p.half_width, p.half_length,
                              self.clearance_fn(pose))
            if way is not None:
                self.get_logger().info(
                    f'Stepping round the box at ({block.x:.2f}, {block.y:.2f}) with a '
                    f'{db.SIDESTEP_GAP * 100:.0f} cm gap')
                self.goal_loop = path[k:]
                self.drive(way)
                return
        self.go_to(path[k:])

    def clearance_fn(self, pose):
        """clear_at(x, y): m from (x, y) to the nearest map wall, block, no-pass line or point
        of the live scan (for checking a way round a block)."""
        mt = self.matcher
        x0, y0, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        now = time.monotonic()
        scan = np.asarray(self.scan_points if now - self.scan_time <= SENSOR_STALE else [],
                          dtype=float).reshape(-1, 2)
        world = np.column_stack([x0 + c * scan[:, 0] - s * scan[:, 1],
                                 y0 + s * scan[:, 0] + c * scan[:, 1]]) if len(scan) else scan
        lines = list(self.nopass.lines)
        half = self.nopass.width / 2

        def clear_at(x, y):
            r = int(mt.rows - (y - mt.origin[1]) / mt.res)
            col = int((x - mt.origin[0]) / mt.res)
            if not (0 <= r < mt.rows and 0 <= col < mt.cols):
                return 0.0
            d = min(float(mt.dist[r, col]), self.debris.distance(x, y))
            for a, b in lines:
                d = min(d, max(0.0, npz._seg_dist((x, y), a, b) - half))
            if len(world):
                d = min(d, float(np.min(np.hypot(world[:, 0] - x, world[:, 1] - y))))
            return d
        return clear_at

    def loop_finished(self, ok, why):
        if self.homing:
            self.home_finished(ok, why)
            return
        self.phase = None
        if not ok and why.startswith('boxed in'):
            # the robot cannot move from where it is: that is not this lane's fault, so do not
            # skip it (and every lane after it, in seconds): wait, then try the same lane again
            self.boxed += 1
            self.cmd_pub.publish(Twist())
            if self.boxed > BOXED_TRIES:
                self.runner.stop()
                self.publish_current([])
                self.set_state('error', 'Stopped: boxed in (no room to turn or back up). Clear '
                                        'the way and press START')
                return
            self.phase, self.retry_at = 'wait', time.monotonic() + BOXED_WAIT
            self.set_state('running', f'{self.loop_text()}: boxed in, waiting '
                                      f'({self.boxed}/{BOXED_TRIES})')
            return
        self.boxed = 0
        pose = self.pose()
        remaining = None
        if not ok:
            remaining = self.what_is_left(pose)
            if remaining == []:                     # it got (almost) to the end after all
                ok = True
            else:
                self.get_logger().warn(
                    f'{self.unit.capitalize()} {self.runner.number} did not finish: {why}')
        r = self.runner
        number, skipped = r.number, r.skipped
        nxt = r.finished(ok, pose[:2] if pose else None, remaining, len(self.debris.blocks))
        if r.skipped > skipped:
            self.get_logger().warn(f'{self.unit.capitalize()} {number} skipped: every way to '
                                   'it has been tried')
        elif not ok:
            self.get_logger().info(f'{self.unit.capitalize()} {number}: what is left is tried '
                                   'again after the others, from an untried end')
        if nxt is None:
            self.cmd_pub.publish(Twist())
            self.publish_current([])
            self.set_state('done', f'Finished {self.runner.round} round(s)'
                                   + (f', {self.runner.skipped} loop(s) skipped'
                                      if self.runner.skipped else ''))
            return
        if self.runner.index == 0:                  # a new round, planned from here
            self.publish_path(self.runner.plan.loops)
        self.begin_loop()

    def what_is_left(self, pose):
        """Of the current lane: the part from where the robot got to, to its end ([] if less
        than LEFT_MIN m is left; None = all of it: the robot was not on it)."""
        lap = self.runner.lane_part(self.runner.current())
        if self.unit != 'lane' or pose is None or len(lap) < 2:
            return None
        return rest_of(lap, pose[:2])

    def publish_status(self):
        plan = self.runner.plan or self.preview
        rem = self.follower.remaining if self.phase == 'drive' else None
        self.status_pub.publish(String(data=json.dumps({
            'state': 'home' if self.homing and self.state == 'running' else self.state,
            'message': self.message, 'room': self.room,
            'round': self.runner.round + (1 if self.runner.active else 0),
            'loop': self.runner.number if self.runner.active else 0,
            'loops': len(plan.loops), 'coverage': plan.coverage, 'length': plan.length,
            'remaining': rem, 'skipped': self.runner.skipped,
            'corrections': self.corrections, 'nopass': len(self.nopass.lines),
            'drawing': self.drawing})))


def main(args=None):
    rclpy.init(args=args)
    node = LapsNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node.runner.active:
            node.stop('Shutting down')
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
