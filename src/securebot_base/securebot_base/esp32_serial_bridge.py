#!/usr/bin/env python3


import serial
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range


class ESP32SerialBridge(Node):
    def __init__(self):
        super().__init__('esp32_serial_bridge')

        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('max_pwm', 150)
        self.declare_parameter('max_linear_mps', 0.35)
        self.declare_parameter('max_angular_rps', 1.5)
        self.declare_parameter('wheel_base_m', 0.22)

        self.port = self.get_parameter('port').value
        self.baudrate = self.get_parameter('baudrate').value
        self.max_pwm = int(self.get_parameter('max_pwm').value)
        self.max_linear_mps = float(self.get_parameter('max_linear_mps').value)
        self.max_angular_rps = float(self.get_parameter('max_angular_rps').value)
        self.wheel_base_m = float(self.get_parameter('wheel_base_m').value)

        self.serial_port = serial.Serial(self.port, self.baudrate, timeout=0.05)

        self.cmd_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        self.left_pub = self.create_publisher(Range, '/ultrasonic/left', 10)
        self.right_pub = self.create_publisher(Range, '/ultrasonic/right', 10)

        self.reader_thread = threading.Thread(target=self.read_serial_loop, daemon=True)
        self.reader_thread.start()

        self.get_logger().info(f'ESP32 bridge running on {self.port} at {self.baudrate} baud')

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def cmd_vel_callback(self, msg):
        linear = self.clamp(msg.linear.x, -self.max_linear_mps, self.max_linear_mps)
        angular = self.clamp(msg.angular.z, -self.max_angular_rps, self.max_angular_rps)

        left_speed = linear - (angular * self.wheel_base_m / 2.0)
        right_speed = linear + (angular * self.wheel_base_m / 2.0)

        left_pwm = int((left_speed / self.max_linear_mps) * self.max_pwm)
        right_pwm = int((right_speed / self.max_linear_mps) * self.max_pwm)

        left_pwm = self.clamp(left_pwm, -self.max_pwm, self.max_pwm)
        right_pwm = self.clamp(right_pwm, -self.max_pwm, self.max_pwm)

        command = f'M,{left_pwm},{right_pwm}\n'
        self.serial_port.write(command.encode('utf-8'))

    def publish_range(self, publisher, distance_ft, frame_id):
        msg = Range()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.radiation_type = Range.ULTRASOUND
        msg.field_of_view = 0.26
        msg.min_range = 0.02
        msg.max_range = 4.0
        # Firmware reports -1 for "no reading"; REP-117 says out-of-range is +inf.
        msg.range = distance_ft * 0.3048 if distance_ft > 0 else float('inf')
        publisher.publish(msg)

    def read_serial_loop(self):
        while rclpy.ok():
            try:
                line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()

                if not line:
                    continue

                parts = line.split(',')

                # Expected from ESP32:
                # U,left_distance_ft,right_distance_ft
                if parts[0] == 'U' and len(parts) == 3:
                    left_ft = float(parts[1])
                    right_ft = float(parts[2])

                    self.publish_range(self.left_pub, left_ft, 'left_ultrasonic')
                    self.publish_range(self.right_pub, right_ft, 'right_ultrasonic')

            except Exception as e:
                self.get_logger().warn(f'Serial read error: {e}')

    def stop_motors(self):
        try:
            self.serial_port.write(b'M,0,0\n')
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = ESP32SerialBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_motors()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
