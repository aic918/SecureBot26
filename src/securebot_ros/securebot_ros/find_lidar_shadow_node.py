"""Measure where the robot blinds its own LiDAR (camera tower, brackets) on one turn on the spot.

  ros2 launch securebot_ros securebot_find_shadow.launch.py              # measure, print
  ros2 launch securebot_ros securebot_find_shadow.launch.py save:=true   # ... and save it

The launch starts the LiDAR, the ESP32 bridge and the motion manager; this node then:
  1. checks that nothing is within `clearance` m (a turn on the spot sweeps ~0.36 m; returns
     closer than lidar_shadow.SELF_RANGE are ignored here, they may be the robot itself),
  2. turns the robot once on the spot (gyro) while collecting /scan_raw (the driver's output,
     before any filter),
  3. finds what stayed blind, or always hit the robot, all the way round (lidar_shadow), and
  4. prints it next to the current mask; with save:=true writes config/lidar_filters.yaml
     (restart the LiDAR / SLAM / laps launches to use it).
Open space (doorways, long hallways) is not mistaken for a shadow: it sweeps past every angle
during the turn. A turn that takes too long stops the robot and the test.
"""
import math
import os
import time

import rclpy
import yaml
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan

from securebot_ros import lidar_front as lf
from securebot_ros import lidar_shadow as ls

FILTERS_FILE = os.path.expanduser('~/securebot_ws/src/securebot_ros/config/lidar_filters.yaml')


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def current_bounds(path=FILTERS_FILE):
    """Angular filters in the current lidar_filters.yaml as [(lower, upper)] radians."""
    try:
        with open(path) as f:
            params = yaml.safe_load(f)['lidar_filters']['ros__parameters']
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        return None
    return [(float(v['params']['lower_angle']), float(v['params']['upper_angle']))
            for k, v in sorted(params.items())
            if isinstance(v, dict) and v.get('type', '').endswith('AngularBoundsFilterInPlace')]


class FindShadow(Node):
    def __init__(self):
        super().__init__('find_lidar_shadow')
        for name, value in (('turn_speed', 0.6), ('clearance', 0.5), ('save', False),
                            ('extra_turn_deg', 20.0)):
            self.declare_parameter(name, value)
        self.p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.scans = []
        self.collect = False
        self.last_scan = None
        self.yaw = None
        self.turned = 0.0
        self.cmd_pub = self.create_publisher(Twist, '/nav_cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan_raw', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Imu, '/imu', self.on_imu, 20)

    def on_scan(self, msg):
        self.last_scan = msg
        if self.collect:
            self.scans.append((list(msg.ranges), msg.angle_min, msg.angle_increment))

    def on_imu(self, msg):
        yaw = yaw_of(msg.orientation)
        if self.yaw is not None:
            self.turned += lf.wrap(yaw - self.yaw)
        self.yaw = yaw

    def spin_for(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.02)

    def stop(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.cmd_pub.publish(Twist())
            self.spin_for(0.05)

    def run(self):
        log = self.get_logger()
        end = time.monotonic() + 20.0
        while (self.last_scan is None or self.yaw is None) and time.monotonic() < end:
            self.spin_for(0.2)
        if self.last_scan is None or self.yaw is None:
            missing = [n for n, v in (('LiDAR /scan_raw', self.last_scan), ('gyro /imu', self.yaw))
                       if v is None]
            log.error('no data from ' + ', '.join(missing))
            return 1
        m = self.last_scan
        pts = lf.scan_points(m.ranges, m.angle_min, m.angle_increment, min_range=ls.SELF_RANGE)
        near = lf.nearest_clearance(pts)
        if near < self.p('clearance'):
            log.error(f'Something is {near:.2f} m from the LiDAR: put the robot where nothing is '
                      f'within {self.p("clearance")} m, so it can turn on the spot')
            return 1
        target = 2 * math.pi + math.radians(self.p('extra_turn_deg'))
        log.info(f'Turning once on the spot ({math.degrees(target):.0f} deg), keep clear')
        self.stop(1.0)
        self.turned, self.scans, self.collect = 0.0, [], True
        msg = Twist()
        msg.angular.z = float(self.p('turn_speed'))
        deadline = time.monotonic() + 2.0 * target / self.p('turn_speed') + 5.0
        try:
            while abs(self.turned) < target:
                if time.monotonic() > deadline:
                    log.error(f'The turn did not finish in time ({math.degrees(self.turned):.0f} '
                              'deg): blocked, or the bridge is off')
                    return 1
                self.cmd_pub.publish(msg)
                self.spin_for(0.05)
        finally:
            self.collect = False
            self.stop(1.0)
        seen, near_share = ls.angle_stats(self.scans)
        found = ls.sectors(ls.masked_degrees(seen, near_share))
        bounds = ls.to_radian_bounds(found)
        log.info(f'{len(self.scans)} scans over {math.degrees(abs(self.turned)):.0f} deg')
        log.info(f'MEASURED robot shadow (laser frame, incl. {ls.MARGIN} deg margin): '
                 f'{ls.describe(bounds)}')
        old = current_bounds()
        log.info(f'configured now: {"unreadable" if old is None else ls.describe(old)}')
        if self.p('save'):
            note = (f'measured {time.strftime("%Y-%m-%d %H:%M")} by find_lidar_shadow: '
                    f'{len(self.scans)} scans over one turn')
            with open(FILTERS_FILE, 'w') as f:
                f.write(ls.filters_yaml(bounds, note))
            log.info(f'Saved to {FILTERS_FILE}: restart the LiDAR / SLAM / laps launches to '
                     'use it')
        return 0


def main(args=None):
    rclpy.init(args=args)
    node = FindShadow()
    code = 1
    try:
        code = node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop(0.5)
        node.destroy_node()
        rclpy.try_shutdown()
    raise SystemExit(code)


if __name__ == '__main__':
    main()
