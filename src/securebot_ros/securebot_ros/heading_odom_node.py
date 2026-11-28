"""Odometry from the wheel encoders (rf2o LiDAR scan matching while they are silent) with the
gyro's heading: ROS wrapper around heading_odom.HeadingOdom.

Inputs   /odom_laser  nav_msgs/Odometry  rf2o, run with publish_tf off
         /imu         sensor_msgs/Imu    esp32_serial_bridge (MPU-6050 yaw and yaw rate)
         /wheel_ticks std_msgs/Int64MultiArray  esp32_serial_bridge ([left, right] counts)
Parameter ticks_per_meter: encoder counts per metre of wheel travel (0 = ignore the encoders).
  1190, measured 2026-11-22 with securebot_find_front.launch.py distance:=0.6: four 0.6 m legs
  (out and back, twice, speed control on), the LiDAR's movement from whole-scan alignment
  against the wheels, agreed within 1.5 %. (A first try against one wall's distance gave 1245,
  measured while the LiDAR's orientation was set wrong; the spin estimate was 1150.)
Outputs  /odom        nav_msgs/Odometry  odom -> base_link
         TF odom -> base_link, on every rf2o and every gyro update (so the heading is fresh
         while turning)
"""
import math
import time

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Int64MultiArray
from tf2_ros import TransformBroadcaster

from securebot_ros.heading_odom import HeadingOdom


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class HeadingOdomNode(Node):
    def __init__(self):
        super().__init__('heading_odom')
        self.declare_parameter('spin_gate', 0.6)
        self.declare_parameter('ticks_per_meter', 1190.0)
        tpm = float(self.get_parameter('ticks_per_meter').value)
        self.odom = HeadingOdom(spin_gate=float(self.get_parameter('spin_gate').value),
                                ticks_per_meter=tpm)
        self.vx = 0.0
        self.tf = TransformBroadcaster(self)
        self.pub = self.create_publisher(Odometry, '/odom', 10)
        self.create_subscription(Odometry, '/odom_laser', self.laser_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 20)
        self.create_subscription(Int64MultiArray, '/wheel_ticks', self.wheels_callback, 20)
        self.warned = False
        source = (f'wheel encoders ({tpm:g} ticks/m; rf2o while they are silent)' if tpm > 0
                  else 'rf2o')
        self.get_logger().info(f'Odometry: {source} distance + gyro heading -> /odom, '
                               'odom->base_link')

    def imu_callback(self, msg):
        now = time.monotonic()
        self.odom.update_imu(now, yaw_of(msg.orientation), msg.angular_velocity.z)
        self.publish()

    def wheels_callback(self, msg):
        if len(msg.data) == 2:
            self.odom.update_wheels(time.monotonic(), msg.data[0], msg.data[1])
            self.publish()

    def laser_callback(self, msg):
        now = time.monotonic()
        p = msg.pose.pose
        self.odom.update_laser(now, p.position.x, p.position.y, yaw_of(p.orientation))
        self.vx = msg.twist.twist.linear.x
        if not self.odom.imu_ok(now) and not self.warned:
            self.get_logger().warn('No gyro data on /imu: using rf2o heading for now')
        self.warned = not self.odom.imu_ok(now)
        self.publish()

    def publish(self):
        o = self.odom
        stamp = self.get_clock().now().to_msg()
        qz, qw = math.sin(o.yaw / 2.0), math.cos(o.yaw / 2.0)
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, 'odom', 'base_link'
        t.transform.translation.x, t.transform.translation.y = o.x, o.y
        t.transform.rotation.z, t.transform.rotation.w = qz, qw
        self.tf.sendTransform(t)
        m = Odometry()
        m.header.stamp, m.header.frame_id, m.child_frame_id = stamp, 'odom', 'base_link'
        m.pose.pose.position.x, m.pose.pose.position.y = o.x, o.y
        m.pose.pose.orientation.z, m.pose.pose.orientation.w = qz, qw
        m.twist.twist.linear.x, m.twist.twist.angular.z = self.vx, o.imu_rate
        self.pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = HeadingOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
