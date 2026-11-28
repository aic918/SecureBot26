"""Measure which way the LiDAR faces on the robot (laser_yaw) on a short straight drive.

  ros2 launch securebot_ros securebot_find_front.launch.py              # measure, print
  ros2 launch securebot_ros securebot_find_front.launch.py save:=true   # ... and save it

The launch starts the LiDAR, the ESP32 bridge and the motion manager; this node then:
  1. checks that nothing is within `clearance` m of the LiDAR in any direction (the front is
     not known yet, so all round),
  2. scans, drives `distance` m forward by the wheel encoders, scans, drives back, scans,
  3. aligns the scans (lidar_front.estimate_motion) to see which way the LiDAR moved in its own
     frame, and from that the laser_yaw (lidar_front.laser_yaw_from_motion),
  4. prints it next to the current setting, and with save:=true writes it to
     src/securebot_ros/config/lidar_mount.yaml (securebot_lidar.launch.py reads it; restart
     any running launch to use it).
"Forward" is what the motors and encoders call forward: fix the motor wiring first if that is
not the robot's front.

It uses /scan_raw (the driver's own output, before any filter or TF), /wheel_ticks and /imu
(the gyro's turn during the drive), and drives through /nav_cmd_vel. Anything within
`stop_range` m during a drive, or a drive that takes too long, stops the robot and the test.
"""
import math
import os
import re
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Int64MultiArray

from securebot_ros import lidar_front as lf

MOUNT_FILE = os.path.expanduser('~/securebot_ws/src/securebot_ros/config/lidar_mount.yaml')


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def save_mount(yaw, note, path=MOUNT_FILE):
    """Replace the laser_yaw line of lidar_mount.yaml, keeping its comments."""
    with open(path) as f:
        text = f.read()
    line = f'laser_yaw: {yaw:.5f}   # {note}'
    text, n = re.subn(r'^laser_yaw:.*$', line, text, flags=re.M)
    if n == 0:
        text = text.rstrip('\n') + '\n' + line + '\n'
    with open(path, 'w') as f:
        f.write(text)


class FindFront(Node):
    def __init__(self):
        super().__init__('find_lidar_front')
        for name, value in (('distance', 0.3), ('speed', 0.15), ('clearance', 0.7),
                            ('stop_range', 0.35), ('ticks_per_meter', 1190.0), ('save', False),
                            ('scans', 8), ('current_yaw', float('nan'))):
            self.declare_parameter(name, value)
        self.p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.scan = None
        self.scan_count = 0
        self.ticks = None
        self.yaw = None
        self.cmd_pub = self.create_publisher(Twist, '/nav_cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan_raw', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Int64MultiArray, '/wheel_ticks', self.on_ticks, 20)
        self.create_subscription(Imu, '/imu', self.on_imu, 20)

    # ---------------------------------------------------------------------------- inputs
    def on_scan(self, msg):
        self.scan = lf.scan_points(msg.ranges, msg.angle_min, msg.angle_increment)
        self.scan_count += 1

    def on_ticks(self, msg):
        if len(msg.data) == 2:
            self.ticks = (msg.data[0], msg.data[1])

    def on_imu(self, msg):
        self.yaw = yaw_of(msg.orientation)

    def spin_for(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.02)

    # ---------------------------------------------------------------------------- steps
    def wait_inputs(self, timeout=20.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.spin_for(0.2)
            if self.scan is not None and self.ticks is not None and self.yaw is not None:
                return None
        missing = [n for n, v in (('LiDAR /scan_raw', self.scan), ('/wheel_ticks', self.ticks),
                                  ('gyro /imu', self.yaw)) if v is None]
        return 'no data from ' + ', '.join(missing)

    def collect(self):
        """Points of several scans, the robot standing still."""
        self.stop(1.0)
        pts, seen = [], self.scan_count
        while self.scan_count < seen + int(self.p('scans')):
            self.spin_for(0.05)
            if self.scan is not None and self.scan_count > seen + len(pts):
                pts.append(self.scan)
        return np.vstack(pts)

    def stop(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.cmd_pub.publish(Twist())
            self.spin_for(0.05)

    def drive(self, sign):
        """Drive `distance` m forward (sign 1) or back (-1) by the encoders. Returns the
        distance driven (m) or raises RuntimeError."""
        tpm = float(self.p('ticks_per_meter'))
        start = sum(self.ticks) / 2.0
        deadline = time.monotonic() + 3.0 * self.p('distance') / self.p('speed') + 3.0
        msg = Twist()
        msg.linear.x = sign * float(self.p('speed'))
        try:
            while True:
                driven = (sum(self.ticks) / 2.0 - start) / tpm
                if driven * sign >= self.p('distance'):
                    return abs(driven)
                if lf.nearest_clearance(self.scan) < self.p('stop_range'):
                    raise RuntimeError('something came within '
                                       f'{self.p("stop_range"):.2f} m of the LiDAR')
                if time.monotonic() > deadline:
                    raise RuntimeError(f'the wheels did not cover {self.p("distance")} m in time '
                                       f'({abs(driven):.2f} m): blocked, or the bridge is off')
                self.cmd_pub.publish(msg)
                self.spin_for(0.05)
        finally:
            self.stop(1.5)

    def run(self):
        log = self.get_logger()
        problem = self.wait_inputs()
        if problem:
            log.error(problem)
            return 1
        near = lf.nearest_clearance(self.scan)
        if near < self.p('clearance'):
            log.error(f'Something is {near:.2f} m from the LiDAR: put the robot where nothing is '
                      f'within {self.p("clearance")} m all round, with walls in view')
            return 1
        log.info(f'Measuring: {self.p("distance")} m forward and back, keep clear')
        try:
            a, yaw_a = self.collect(), self.yaw
            out = self.drive(1)
            b, yaw_b = self.collect(), self.yaw
            back = self.drive(-1)
            c, yaw_c = self.collect(), self.yaw
        except RuntimeError as e:
            log.error(f'Stopped: {e}')
            return 1
        forward = lf.estimate_motion(a, b, rotation=lf.wrap(yaw_b - yaw_a))
        backward = lf.estimate_motion(b, c, rotation=lf.wrap(yaw_c - yaw_b))
        yaw, problems = lf.check(forward, backward, driven=(out + back) / 2.0)
        current = float(self.p('current_yaw'))
        log.info(f'Drive out: LiDAR moved ({forward[0]:+.3f}, {forward[1]:+.3f}) m in its frame, '
                 f'match {forward[2] * 100:.1f} cm; back: ({backward[0]:+.3f}, '
                 f'{backward[1]:+.3f}) m, match {backward[2] * 100:.1f} cm; wheels '
                 f'{out:.3f} / {back:.3f} m')
        log.info(f'MEASURED laser_yaw = {yaw:.3f} rad ({math.degrees(yaw):.1f} deg); '
                 f'configured: {current:.3f} rad ({math.degrees(current):.1f} deg)')
        for p in problems:
            log.warn(p)
        if problems:
            log.error('Measurement not trusted: nothing saved. Try a spot with more walls in '
                      'view.')
            return 1
        if self.p('save'):
            save_mount(yaw, f'measured {time.strftime("%Y-%m-%d %H:%M")} by find_lidar_front')
            log.info(f'Saved to {MOUNT_FILE}: restart the LiDAR / SLAM / laps launches to use it')
        elif not math.isnan(current) and abs(lf.wrap(yaw - current)) > math.radians(10):
            log.warn('That differs from the configured value: run again with save:=true')
        return 0


def main(args=None):
    rclpy.init(args=args)
    node = FindFront()
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
