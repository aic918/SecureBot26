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
import os
import subprocess
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Point, PointStamped, Twist
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan, Range
from std_msgs.msg import ColorRGBA, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from securebot_ros import explore_logic as ex
from securebot_ros.explore_machine import ExploreMachine
from securebot_ros import nopass as npz
from securebot_ros.home_marker import home_markers, home_path_marker
from securebot_ros.lap_follower import LapFollower, simplify
from securebot_ros import room_store as rs

SENSOR_STALE = 1.0        # s: ultrasonic / scan older than this counts as silent


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ExploreNode(Node):
    def __init__(self):
        super().__init__('explore_node')
        self.machine = ExploreMachine()
        self.declare_parameter('coverage_goal', self.machine.p.coverage_goal)
        self.machine.p.coverage_goal = float(self.get_parameter('coverage_goal').value)
        # room:=<name>: when the room is scanned, save the map and file it under that name
        self.declare_parameter('room', '')
        self.declare_parameter('maps_dir', rs.DEFAULT_MAPS_DIR)
        self.room = str(self.get_parameter('room').value).strip()
        # nothing to save until a scan has been started (STOP right after a restart saved the
        # whole live map, stains and all, 2026-11-07)
        self.saved_room = True
        # Return home: back to where START was first pressed, on a path planned over the
        # SLAM map, driven by the lap follower; then it stops
        self.home = None
        # home:="x,y" keeps a start point from before (e.g. the node restarted mid-scan)
        self.declare_parameter('home', '')
        try:
            hx, hy = (float(v) for v in str(self.get_parameter('home').value).split(','))
            self.home = (hx, hy)
        except ValueError:
            pass
        self.trip = None                   # see start_trip
        self.home_path = []                # way home being driven (magenta in RViz)
        self.bad_goals = []                # openings it could not reach
        self.follower = LapFollower()
        self.home_tries = 3

        # no-pass lines drawn on the live map (RViz Publish Point): walls for exploring and
        # trips; saved with the room's map when it is saved, so laps keep them
        self.nopass = npz.NoPass(None)
        self.drawing, self.draw_last = False, None
        self.raw_map = None

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
        self.create_subscription(PointStamped, '/clicked_point', self.clicked, 10)
        self.cmd_pub = self.create_publisher(Twist, '/nav_cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/explore/status', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/explore/markers', 10)

        self.create_timer(0.1, self.tick)
        self.create_timer(0.5, self.publish_status)
        self.get_logger().info('Explore node ready: waiting for Start')

    # ---------------------------------------------------------------- inputs
    def map_callback(self, msg):
        i = msg.info
        if self.grid is not None and i.width * i.height < 0.5 * self.grid.width * self.grid.height:
            # SLAM restarted with a fresh map (Rooms: "Clear live map"): its frame is new too
            self.home, self.bad_goals = None, []
            self.nopass.clear()
            self.get_logger().info('Explore: new map from SLAM; home is set at the next START '
                                   'and no-pass lines are cleared')
        self.raw_map = msg
        self.build_grid()

    def build_grid(self):
        """The SLAM map with the no-pass lines drawn in as walls."""
        msg = self.raw_map
        if msg is None:
            return
        i = msg.info
        data = msg.data
        if self.nopass.lines:
            cells = np.array(msg.data, dtype=np.int16).reshape(i.height, i.width)
            px = max(1, int(round(self.nopass.width / i.resolution)))
            to = lambda x, y: (int((x - i.origin.position.x) / i.resolution),  # noqa: E731
                               int((y - i.origin.position.y) / i.resolution))
            for a, b in self.nopass.lines:
                cv2.line(cells, to(*a), to(*b), 100, thickness=px)
            data = cells.ravel()
        self.grid = ex.Grid(data, i.width, i.height, i.resolution,
                            i.origin.position.x, i.origin.position.y)

    def clicked(self, msg):
        """RViz Publish Point: corners of a no-pass line while drawing, else a click on a
        line removes it."""
        if msg.header.frame_id not in ('', 'map'):
            return
        pt = (msg.point.x, msg.point.y)
        if self.drawing:
            if self.draw_last is not None and self.nopass.add(self.draw_last, pt):
                self.get_logger().info(f'Explore: no-pass line ({self.draw_last[0]:.2f}, '
                                       f'{self.draw_last[1]:.2f}) - ({pt[0]:.2f}, {pt[1]:.2f})')
                self.build_grid()
            self.draw_last = pt
        elif self.nopass.remove_near(*pt) is not None:
            self.get_logger().info(f'Explore: no-pass line at ({pt[0]:.2f}, {pt[1]:.2f}) '
                                   'removed by click')
            self.build_grid()

    def nopass_command(self, what):
        if what == 'draw':
            self.drawing, self.draw_last = not self.drawing, None
        elif what == 'undo':
            self.nopass.undo()
        elif what == 'clear':
            self.nopass.clear()
            self.drawing, self.draw_last = False, None
        self.build_grid()

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
            self.end_trip(now, None)
            self.bad_goals = []
            if self.home is None and self.pose() is not None:
                self.home = self.pose()[:2]
                self.get_logger().info(f'Home (start point): ({self.home[0]:.2f}, '
                                       f'{self.home[1]:.2f})')
            self.machine.start(now)
            self.saved_room = False
        elif cmd == 'home':
            self.go_home(now)
        elif cmd.startswith('nopass '):
            self.nopass_command(cmd.split(None, 1)[1])
        elif cmd == 'stop':
            self.get_logger().info('Explore: STOP')
            self.end_trip(now, None)
            self.machine.stop(now, 'stopped by the Stop button')
            self.cmd_pub.publish(Twist())
            if self.room and not self.saved_room:  # you decide the room is done: save it
                self.saved_room = True
                threading.Thread(target=self.save_room, daemon=True).start()

    # ---------------------------------------------------------------- trips
    # A trip drives a path planned over the SLAM map with the lap follower: 'home' (RETURN
    # HOME: back to the start point, then stop) or 'frontier' (driving stopped showing new
    # floor: go to the nearest opening into unseen space, then explore on from there).
    def go_home(self, now):
        if self.home is None:
            self.machine.stop(now, 'No start point yet: press START first')
            return
        self.end_trip(now, None)
        self.machine.stop(now, 'returning home')
        self.cmd_pub.publish(Twist())
        self.get_logger().info('Explore: returning home')
        self.start_trip('home', [self.home])

    def seek_frontier(self, now, pose):
        goals = [g for g in self.machine.open_targets
                 if all(math.dist(g, b) > 1.0 for b in self.bad_goals)]
        self.machine.wants_frontier = False
        if not goals:
            return
        goals.sort(key=lambda g: math.dist(g, pose[:2]))
        self.machine.stop(now, 'heading for unexplored space')
        self.cmd_pub.publish(Twist())
        self.get_logger().info(f'Explore: no new floor lately; heading for unexplored space '
                               f'at ({goals[0][0]:.1f}, {goals[0][1]:.1f})')
        self.start_trip('frontier', goals[:3])

    def start_trip(self, kind, goals):
        self.trip = {'kind': kind, 'goals': goals, 'tries': 0, 'path': None, 'busy': False,
                     'goal': None}
        self.plan_trip()

    def plan_trip(self):
        trip, pose, grid = self.trip, self.pose(), self.grid
        if trip is None:
            return
        if pose is None or grid is None:
            trip['path'] = []
            return
        trip['busy'], trip['tries'] = True, trip['tries'] + 1

        def work():
            path, t0 = None, time.monotonic()
            try:
                for goal in trip['goals']:
                    for clearance in (0.30, 0.24):     # narrow spots: less margin
                        path = ex.plan_path(grid, pose[:2], goal, clearance=clearance)
                        if path:
                            break
                    if path:
                        trip['goal'] = goal
                        path = [tuple(p) for p in simplify(path, 0.08)]
                        break
                    if trip['kind'] == 'frontier':
                        self.bad_goals.append(goal)
            except Exception as e:                          # noqa: BLE001
                self.get_logger().warn(f'Path planning failed: {e}')
                path = None
            length = sum(math.dist(a, b) for a, b in zip(path, path[1:])) if path else 0.0
            self.get_logger().info(
                f'Explore: {trip["kind"]} path ' + (f'{length:.1f} m' if path else 'not found')
                + f' (planned in {time.monotonic() - t0:.1f} s, try {trip["tries"]})')
            trip['path'], trip['busy'] = (path or []), False
            if trip['kind'] == 'home' and path and self.trip is trip:   # not after STOP
                self.home_path = list(path)
        threading.Thread(target=work, daemon=True).start()

    def end_trip(self, now, message):
        if self.trip is None:
            return
        kind, self.trip = self.trip['kind'], None
        self.home_path = []
        self.follower.stop(now)
        self.cmd_pub.publish(Twist())
        if kind == 'frontier' and message is None:
            return
        if kind == 'frontier':
            self.get_logger().info(f'Explore: {message}; exploring on')
            self.machine.resume(now)
        elif message:
            self.machine.stop(now, message)
            self.get_logger().info(f'Explore: {message}')

    def trip_tick(self, now, pose, us_left, us_right, points):
        trip, f = self.trip, self.follower
        if trip['busy']:
            self.cmd_pub.publish(Twist())
            return
        if trip['path'] is not None:               # a new plan: drive it
            path, trip['path'] = trip['path'], None
            if len(path) < 2:
                if trip['tries'] < self.home_tries and trip['kind'] == 'home':
                    self.plan_trip()
                else:
                    self.end_trip(now, 'No way home found on the map' if trip['kind'] == 'home'
                                  else 'no way to the unexplored space')
                return
            f.start(now, path)
        cmd = f.step(now, pose, us_left, us_right, points, odom_yaw=self.odom_yaw())
        if cmd is not None:
            msg = Twist()
            msg.linear.x, msg.angular.z = cmd
            self.cmd_pub.publish(msg)
            return
        home = trip['kind'] == 'home'
        if f.state == 'done':
            self.end_trip(now, 'Back at the start point' if home else 'reached unexplored space')
        elif f.state == 'failed' and trip['tries'] < self.home_tries:
            self.cmd_pub.publish(Twist())
            self.plan_trip()                           # blocked: plan again from here
        else:
            if not home and trip['goal'] is not None:
                self.bad_goals.append(trip['goal'])
            self.end_trip(now, (f'Could not get home: {f.message}' if home
                                else f'could not reach unexplored space ({f.message})'))

    def odom_yaw(self):
        try:
            t = self.tf.lookup_transform('odom', 'base_link', rclpy.time.Time())
        except Exception:
            return None
        return yaw_of(t.transform.rotation)

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
        if self.nopass.lines and points:           # no-pass lines: seen as walls too
            pose = self.pose()
            if pose is not None:
                points = list(points) + self.nopass.points_in_base(pose)
        if self.trip is not None:
            self.trip_tick(now, self.pose(), us_left, us_right, points)
            return
        state_before = self.machine.state
        cmd = self.machine.step(now, self.pose(), self.grid, us_left, us_right, points)
        if cmd is not None:
            msg = Twist()
            msg.linear.x, msg.angular.z = cmd
            self.cmd_pub.publish(msg)
        elif self.was_running:
            self.cmd_pub.publish(Twist())          # one explicit stop when exploring ends
        self.was_running = self.machine.running
        if self.machine.wants_frontier and self.machine.state == 'drive':
            pose = self.pose()
            if pose is not None:
                self.seek_frontier(now, pose)
                return
        if self.machine.state != state_before:
            self.get_logger().info(f'Explore: {self.machine.state} - {self.machine.message}')
            if self.machine.state == 'done' and self.room and not self.saved_room:
                self.saved_room = True
                threading.Thread(target=self.save_room, daemon=True).start()

    def save_room(self):
        """Save SLAM's map (map_saver_cli) and file it as room `room` (new, or its map
        replaced by this newer scan). Laps then: room:="<name>"."""
        log = self.get_logger()
        try:
            name = rs.clean_name(self.room)
            store = rs.RoomStore(os.path.expanduser(str(self.get_parameter('maps_dir').value)))
            stem = rs.stem_for(name)
            os.makedirs(store.maps_dir, exist_ok=True)
            log.info(f'Saving the map as room "{name}" ({stem})...')
            p = subprocess.run(['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f',
                                os.path.join(store.maps_dir, stem), '--ros-args', '-p',
                                'save_map_timeout:=10.0'],
                               capture_output=True, text=True, timeout=60)
            if p.returncode != 0 or not store.map_exists(stem):
                log.error('Map not saved: ' + (p.stdout + p.stderr).strip()[-300:])
                return
            if self.nopass.lines:                  # the lines go with the room's map
                saved = npz.NoPass(npz.path_for(os.path.join(store.maps_dir, stem + '.yaml')),
                                   self.nopass.width)
                saved.lines = list(self.nopass.lines)
                saved.save()
                log.info(f'{len(saved.lines)} no-pass line(s) saved with the map')
            if store.find(name):
                store.set_map(name, stem)
                log.info(f'Room "{name}" now uses the new map {stem}')
            else:
                store.add(name, stem)
                log.info(f'Saved room "{name}" (map {stem}). Laps: ros2 launch securebot_ros '
                         f'securebot_laps.launch.py room:="{name}"')
        except (rs.RoomError, OSError, subprocess.TimeoutExpired) as e:
            log.error(f'Room not saved: {e}')

    # ---------------------------------------------------------------- outputs
    def publish_status(self):
        m = self.machine
        trip = self.trip['kind'] if self.trip else None
        self.status_pub.publish(String(data=json.dumps({
            'state': {'home': 'home', 'frontier': 'seek'}.get(trip, m.state),
            'message': (self.follower.message if self.follower.running else 'planning the way')
            if trip else m.message, 'home': self.home is not None,
            'coverage': None if m.coverage is None else round(m.coverage, 3),
            'goal': m.p.coverage_goal, 'distance': round(m.travelled, 1),
            'avoids': m.avoids, 'nopass': len(self.nopass.lines),
            'drawing': self.drawing})))
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
        if self.home is not None:
            markers += home_markers(*self.home, stamp)
        markers.append(home_path_marker(self.home_path, stamp))
        lines = Marker()
        lines.header.frame_id, lines.header.stamp = 'map', stamp
        lines.ns, lines.id, lines.type = 'nopass', 0, Marker.LINE_LIST
        if self.nopass.lines:
            lines.pose.orientation.w = 1.0
            lines.scale.x = self.nopass.width
            lines.color = ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0)       # black
            lines.points = [Point(x=float(x), y=float(y), z=0.05)
                            for a, b in self.nopass.lines for x, y in (a, b)]
        else:
            lines.action = Marker.DELETE
        dot = Marker()
        dot.header.frame_id, dot.header.stamp = 'map', stamp
        dot.ns, dot.id, dot.type = 'nopass_start', 0, Marker.SPHERE
        if self.drawing and self.draw_last is not None:
            dot.pose.position.x, dot.pose.position.y = self.draw_last
            dot.pose.orientation.w = 1.0
            dot.scale.x = dot.scale.y = dot.scale.z = self.nopass.width * 1.5
            dot.color = ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0)
        else:
            dot.action = Marker.DELETE
        markers += [lines, dot]
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
