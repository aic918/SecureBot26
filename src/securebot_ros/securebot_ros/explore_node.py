"""Explore the room on its own: ROS wrapper around explore_machine.ExploreMachine.

Inputs   /map (slam_toolbox), TF map -> base_link, /ultrasonic/left, /ultrasonic/right, /scan,
         /explore/command  std_msgs/String  start | stop     (the Start / Stop buttons)
Outputs  /nav_cmd_vel  geometry_msgs/Twist  only while exploring (motion_manager_node gates it)
         /explore/status  std_msgs/String  JSON {state, message, coverage, distance}, 2 Hz
         /explore/markers  visualization_msgs/MarkerArray  frontiers + steering direction, 2 Hz

Run it with securebot_explore.launch.py (SLAM + robot side in keyboard mode + this + buttons).
"""
import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan, Range
from std_msgs.msg import ColorRGBA, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from securebot_ros import explore_logic as ex
from securebot_ros.explore_machine import ExploreMachine

SENSOR_STALE = 1.0        # s: ultrasonic / scan older than this counts as silent


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ExploreNode(Node):
    def __init__(self):
        super().__init__('explore_node')
        self.machine = ExploreMachine()
        self.declare_parameter('coverage_goal', self.machine.p.coverage_goal)
        self.machine.p.coverage_goal = float(self.get_parameter('coverage_goal').value)

        self.grid = None
        self.us = {'left': (None, 0.0), 'right': (None, 0.0)}
        self.scan_points, self.scan_time = [], 0.0
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.was_running = False

        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        self.create_subscription(Range, '/ultrasonic/left', lambda m: self.range_cb('left', m), 10)
        self.create_subscription(Range, '/ultrasonic/right', lambda m: self.range_cb('right', m), 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(String, '/explore/command', self.command_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/nav_cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/explore/status', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/explore/markers', 10)

        self.create_timer(0.1, self.tick)
        self.create_timer(0.5, self.publish_status)
        self.get_logger().info('Explore node ready: waiting for Start')

    # ---------------------------------------------------------------- inputs
    def map_callback(self, msg):
        i = msg.info
        self.grid = ex.Grid(msg.data, i.width, i.height, i.resolution,
                            i.origin.position.x, i.origin.position.y)

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

    def command_callback(self, msg):
        cmd, now = msg.data.strip().lower(), time.monotonic()
        if cmd == 'start':
            self.get_logger().info('Explore: START')
            self.machine.start(now)
        elif cmd == 'stop':
            self.get_logger().info('Explore: STOP')
            self.machine.stop(now, 'stopped by the Stop button')
            self.cmd_pub.publish(Twist())

    def pose(self):
        try:
            t = self.tf.lookup_transform('map', 'base_link', rclpy.time.Time())
        except Exception:
            return None
        age = self.get_clock().now() - rclpy.time.Time.from_msg(t.header.stamp)
        if age.nanoseconds > 1e9 and t.header.stamp.sec != 0:
            return None
        tr = t.transform
        return tr.translation.x, tr.translation.y, yaw_of(tr.rotation)

    # ---------------------------------------------------------------- control loop
    def tick(self):
        now = time.monotonic()
        fresh = lambda value, stamp: value if now - stamp <= SENSOR_STALE else None  # noqa: E731
        us_left = fresh(*self.us['left'])
        us_right = fresh(*self.us['right'])
        points = self.scan_points if now - self.scan_time <= SENSOR_STALE else []
        state_before = self.machine.state
        cmd = self.machine.step(now, self.pose(), self.grid, us_left, us_right, points)
        if cmd is not None:
            msg = Twist()
            msg.linear.x, msg.angular.z = cmd
            self.cmd_pub.publish(msg)
        elif self.was_running:
            self.cmd_pub.publish(Twist())          # one explicit stop when exploring ends
        self.was_running = self.machine.running
        if self.machine.state != state_before:
            self.get_logger().info(f'Explore: {self.machine.state} - {self.machine.message}')

    # ---------------------------------------------------------------- outputs
    def publish_status(self):
        m = self.machine
        self.status_pub.publish(String(data=json.dumps({
            'state': m.state, 'message': m.message,
            'coverage': None if m.coverage is None else round(m.coverage, 3),
            'goal': m.p.coverage_goal, 'distance': round(m.travelled, 1),
            'avoids': m.avoids})))
        self.publish_markers()

    def publish_markers(self):
        stamp = self.get_clock().now().to_msg()
        frontier = Marker()
        frontier.header.frame_id, frontier.header.stamp = 'map', stamp
        frontier.ns, frontier.id, frontier.type = 'frontiers', 0, Marker.POINTS
        frontier.scale.x = frontier.scale.y = 0.05
        frontier.color = ColorRGBA(r=1.0, g=0.3, b=0.1, a=1.0)
        frontier.points = [Point(x=x, y=y, z=0.02) for cl in self.machine.frontiers for x, y in cl]
        markers = [frontier]
        pose = self.pose()
        arrow = Marker()
        arrow.header.frame_id, arrow.header.stamp = 'map', stamp
        arrow.ns, arrow.id, arrow.type = 'heading', 0, Marker.ARROW
        if pose and self.machine.target is not None and self.machine.running:
            x, y, _ = pose
            h, leg = self.machine.target, self.machine.leg
            arrow.points = [Point(x=x, y=y, z=0.1),
                            Point(x=x + math.cos(h) * leg, y=y + math.sin(h) * leg, z=0.1)]
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.05, 0.12, 0.12
            arrow.color = ColorRGBA(r=0.2, g=0.9, b=1.0, a=1.0)
        else:
            arrow.action = Marker.DELETE
        markers.append(arrow)
        self.marker_pub.publish(MarkerArray(markers=markers))


def main(args=None):
    rclpy.init(args=args)
    node = ExploreNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.cmd_pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
