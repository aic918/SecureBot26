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
Position check: whenever the robot stands still (settling after a turn, waiting at an
obstacle; at most every CHECK_PERIOD s) the live scan is matched against the saved map around
AMCL's pose (scan_match); if a clearly better pose is found it is sent to AMCL.

Outputs  /cmd_vel_smoothed  geometry_msgs/Twist  while running (collision monitor input)
         /initialpose  geometry_msgs/PoseWithCovarianceStamped  position corrections
         /laps/path     nav_msgs/Path  every planned loop, for RViz (latched)
         /laps/status   std_msgs/String  JSON {state, message, room, round, loop, loops,
                        coverage, length, remaining, skipped}, 2 Hz

Run it with securebot_laps.launch.py.
"""
import json
import math
import os
import time

import cv2
import rclpy
import yaml
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, LaserScan, Range
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from securebot_ros import coverage_logic as cl
from securebot_ros.lap_follower import LapFollower
from securebot_ros.scan_match import ScanMatcher

SENSOR_STALE = 1.0        # s: ultrasonic / scan older than this counts as silent
TRANSIT_DIRECT = 0.8      # m: closer than this to a loop's start, drive straight there
CHECK_PERIOD = 3.0        # s between position checks while standing still


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class LapsNode(Node):
    def __init__(self):
        super().__init__('laps_node')
        defaults = cl.Params()
        self.declare_parameter('map', '')
        self.declare_parameter('room', '')
        self.declare_parameter('rounds', 0)                   # 0 = keep lapping until Stop
        self.declare_parameter('lane_spacing', defaults.lane_spacing)
        self.declare_parameter('wall_clearance', defaults.wall_clearance)
        self.declare_parameter('pattern', 'lanes')            # 'lanes' or 'contour'
        self.declare_parameter('require_gyro', True)          # refuse Start without /imu
        self.declare_parameter('side_clearance', defaults.side_clearance)
        self.declare_parameter('front_clearance', defaults.front_clearance)
        get = lambda n: self.get_parameter(n).value  # noqa: E731
        params = cl.Params(lane_spacing=float(get('lane_spacing')),
                           wall_clearance=float(get('wall_clearance')),
                           pattern=str(get('pattern')),
                           side_clearance=float(get('side_clearance')),
                           front_clearance=float(get('front_clearance')))
        self.unit = 'lane' if params.pattern == 'lanes' else 'loop'
        self.room = get('room') or get('map')
        self.m = cl.load_map(get('map'))
        with open(get('map')) as f:
            image = yaml.safe_load(f)['image']
        image = image if os.path.isabs(image) else os.path.join(os.path.dirname(get('map')), image)
        self.matcher = ScanMatcher(cv2.imread(image, cv2.IMREAD_GRAYSCALE), self.m.resolution,
                                   self.m.origin)
        self.last_check, self.corrections = 0.0, 0
        self.runner = cl.LapRunner(self.m, params, rounds=int(get('rounds')))
        self.follower = LapFollower()
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
        self.status_pub = self.create_publisher(String, '/laps/status', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_smoothed', 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.create_subscription(String, '/laps/command', self.command_callback, 10)
        self.create_subscription(Range, '/ultrasonic/left', lambda m: self.range_cb('left', m), 10)
        self.create_subscription(Range, '/ultrasonic/right',
                                 lambda m: self.range_cb('right', m), 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.imu_time = 0.0
        self.create_subscription(Imu, '/imu', self.imu_callback, 20)
        self.create_timer(0.1, self.tick)
        self.create_timer(0.5, self.publish_status)

        preview = cl.plan(self.m, params)          # shown before Start, from no particular spot
        self.publish_path(preview.loops)
        self.preview = preview
        self.get_logger().info(
            f'{self.room}: {len(preview.loops)} loops, {preview.length:.0f} m per round, '
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

    def pose(self):
        try:
            t = self.tf.lookup_transform('map', 'base_link', rclpy.time.Time())
        except Exception:
            return None
        tr = t.transform
        return tr.translation.x, tr.translation.y, yaw_of(tr.rotation)

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

    def set_state(self, state, message):
        if (state, message) != (self.state, self.message):
            self.get_logger().info(f'{state}: {message}')
        self.state, self.message = state, message

    def loop_text(self):
        r = self.runner
        return (f'Round {r.round + 1}, {self.unit} {r.index + 1} of {len(r.plan.loops)}'
                + (' (retrying)' if r.failures else ''))

    # ---------------------------------------------------------------- commands
    def command_callback(self, msg):
        cmd = msg.data.strip().lower()
        if cmd == 'start':
            self.start()
        elif cmd == 'stop':
            self.stop('Stopped')

    def start(self):
        if self.runner.active:
            return
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
        if not self.runner.start(pose[:2]):
            self.set_state('error', 'No room to drive laps from here: is the robot on the map?')
            return
        self.publish_path(self.runner.plan.loops)
        self.begin_loop()

    def stop(self, message):
        self.runner.stop()
        self.follower.stop(time.monotonic(), message)
        self.phase = None
        self.plan_seq += 1
        self.cmd_pub.publish(Twist())
        self.set_state('idle', message)

    # ---------------------------------------------------------------- driving
    def begin_loop(self):
        """Get to the current loop's start (via Nav2's planner unless it is close), then lap."""
        loop = self.runner.current()
        pose = self.pose()
        self.set_state('running', self.loop_text() + ': heading to its start')
        if pose is not None and math.dist(pose[:2], loop[0]) < TRANSIT_DIRECT:
            self.drive([pose[:2]] + loop)
            return
        goal = ComputePathToPose.Goal()
        goal.goal = self.stamped(*loop[0], cl.headings(loop)[0])
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
            self.loop_finished(False, 'planner refused the way to the loop')
            return
        handle.get_result_async().add_done_callback(lambda f: self.planned(f, seq, loop))

    def planned(self, future, seq, loop):
        if seq != self.plan_seq:
            return
        res = future.result()
        poses = res.result.path.poses if res.status == GoalStatus.STATUS_SUCCEEDED else []
        if not poses:
            self.loop_finished(False, 'no way to the loop found')
            return
        transit = [(p.pose.position.x, p.pose.position.y) for p in poses]
        self.drive(transit + loop)

    def drive(self, path):
        self.phase = 'drive'
        self.follower.start(time.monotonic(), path)

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
            return
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x, msg.pose.pose.position.y = better[0], better[1]
        msg.pose.pose.orientation.z = math.sin(better[2] / 2)
        msg.pose.pose.orientation.w = math.cos(better[2] / 2)
        cov = [0.0] * 36
        cov[0] = cov[7] = 0.02
        cov[35] = 0.02
        msg.pose.covariance = cov
        self.pose_pub.publish(msg)
        self.corrections += 1
        turn = math.atan2(math.sin(better[2] - pose[2]), math.cos(better[2] - pose[2]))
        self.get_logger().info(
            f'Position corrected by {math.dist(pose[:2], better[:2]) * 100:.0f} cm, '
            f'{abs(math.degrees(turn)):.0f} deg (scan match was {score:.0%})')

    def tick(self):
        if self.phase != 'drive':
            return
        now = time.monotonic()
        fresh = lambda value, stamp: value if now - stamp <= SENSOR_STALE else None  # noqa: E731
        points = self.scan_points if now - self.scan_time <= SENSOR_STALE else []
        self.check_position(now, points)
        cmd = self.follower.step(now, self.pose(), fresh(*self.us['left']),
                                 fresh(*self.us['right']), points)
        f = self.follower
        if cmd is not None:
            msg = Twist()
            msg.linear.x, msg.angular.z = cmd
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

    def loop_finished(self, ok, why):
        self.phase = None
        if not ok:
            self.get_logger().warn(f'Loop {self.runner.index + 1} did not finish: {why}')
        pose = self.pose()
        nxt = self.runner.finished(ok, pose[:2] if pose else None)
        if nxt is None:
            self.cmd_pub.publish(Twist())
            self.set_state('done', f'Finished {self.runner.round} round(s)'
                                   + (f', {self.runner.skipped} loop(s) skipped'
                                      if self.runner.skipped else ''))
            return
        if self.runner.index == 0:                  # a new round, planned from here
            self.publish_path(self.runner.plan.loops)
        self.begin_loop()

    def publish_status(self):
        plan = self.runner.plan or self.preview
        rem = self.follower.remaining if self.phase == 'drive' else None
        self.status_pub.publish(String(data=json.dumps({
            'state': self.state, 'message': self.message, 'room': self.room,
            'round': self.runner.round + (1 if self.runner.active else 0),
            'loop': self.runner.index + 1 if self.runner.active else 0,
            'loops': len(plan.loops), 'coverage': plan.coverage, 'length': plan.length,
            'remaining': rem, 'skipped': self.runner.skipped,
            'corrections': self.corrections})))


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
