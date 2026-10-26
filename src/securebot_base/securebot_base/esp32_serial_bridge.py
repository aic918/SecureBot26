#!/usr/bin/env python3
"""The only program that talks to the ESP32 (firmware 2.0 protocol, see firmware/esp32/).

Inputs
  /esp32/mode  std_msgs/String  stop | patrol | drive   (published by motion_manager_node)
  /cmd_vel     geometry_msgs/Twist                       used only in drive mode
Outputs
  /ultrasonic/left, /ultrasonic/right  sensor_msgs/Range  (no echo -> +inf, REP-117)
  /esp32/status  std_msgs/String  the firmware's `S,<mode>,<l>,<r>,<avoid>` line, without "S,"

Safety: the current command is resent every `heartbeat_period` (the firmware stops on its own
after 250 ms without one). A mode or /cmd_vel older than `command_timeout` counts as "stop",
so if the motion manager or the planner dies, the robot stops.
"""
import threading
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range
from std_msgs.msg import String
import serial

MODES = ('stop', 'patrol', 'drive')


def clamp(value, low, high):
    return max(low, min(high, value))


def twist_to_pwm(linear, angular, max_linear_mps, max_angular_rps, wheel_base_m, max_pwm,
                 min_pwm=0):
    """Differential-drive Twist -> (left_pwm, right_pwm), each clamped to +/-max_pwm.
    Below min_pwm the motors only hum, so a command whose faster wheel is under it is scaled
    up (both wheels together, keeping their ratio and so the direction of a curve)."""
    linear = clamp(linear, -max_linear_mps, max_linear_mps)
    angular = clamp(angular, -max_angular_rps, max_angular_rps)
    left = (linear - angular * wheel_base_m / 2.0) / max_linear_mps * max_pwm
    right = (linear + angular * wheel_base_m / 2.0) / max_linear_mps * max_pwm

    fastest = max(abs(left), abs(right))
    if 0 < fastest < min_pwm:
        left, right = left * min_pwm / fastest, right * min_pwm / fastest

    to_pwm = lambda v: int(clamp(round(v), -max_pwm, max_pwm))  # noqa: E731
    return to_pwm(left), to_pwm(right)


def choose_command(mode, mode_age, cmd, cmd_age, timeout, pwm_args):
    """What to send to the ESP32 now. Stale mode -> stop; drive with a stale /cmd_vel -> M,0,0."""
    if mode not in MODES or mode_age > timeout or mode == 'stop':
        return 'stop'
    if mode == 'patrol':
        return 'patrol'
    if cmd is None or cmd_age > timeout:
        return 'M,0,0'
    left, right = twist_to_pwm(cmd[0], cmd[1], *pwm_args)
    return f'M,{left},{right}'


def parse_line(line):
    """('U', left_ft, right_ft) | ('S', 'drive,90,90,0') | ('I'/'E', text) | None."""
    parts = line.split(',')
    if parts[0] == 'U' and len(parts) == 3:
        try:
            return 'U', float(parts[1]), float(parts[2])
        except ValueError:
            return None
    if parts[0] in ('S', 'I', 'E') and len(parts) > 1:
        return parts[0], line[2:]
    return None


class ESP32SerialBridge(Node):
    def __init__(self):
        super().__init__('esp32_serial_bridge')

        # Default: the ESP32's USB socket (path 2.3). /dev/ttyUSBn numbers can swap with the
        # LiDAR, which uses the same CP2102 adapter with the same serial number.
        self.declare_parameter('port', '/dev/serial/by-path/platform-3610000.usb-usb-0:2.3:1.0-port0')
        self.declare_parameter('baudrate', 115200)
        # PWM is 10-bit duty (0-1023) at 10 kHz. The robot needs ~650 to start from standstill
        # (tuned with tools/motor_tune.py), so 650 is both the floor for any wheel that should
        # turn and the ceiling: for now the robot drives at one speed and steers/stops.
        self.declare_parameter('max_pwm', 650)
        self.declare_parameter('min_pwm', 650)          # deadband floor for M,l,r (0 = off)
        self.declare_parameter('max_linear_mps', 0.35)
        self.declare_parameter('max_angular_rps', 1.5)
        # Drive wheels 72 mm wide, 368 mm apart inside edge to inside edge -> 440 mm centre to
        # centre (measured 2026-10-26). Wheel diameter 192 mm (used once encoders give odometry).
        self.declare_parameter('wheel_base_m', 0.44)
        self.declare_parameter('patrol_speed', 650)     # PWM the firmware's own patrol uses
        self.declare_parameter('heartbeat_period', 0.1)     # firmware watchdog is 250 ms
        self.declare_parameter('command_timeout', 0.5)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.port = p('port')
        self.pwm_args = (float(p('max_linear_mps')), float(p('max_angular_rps')),
                         float(p('wheel_base_m')), int(p('max_pwm')), int(p('min_pwm')))
        self.patrol_speed = int(p('patrol_speed'))
        self.timeout = float(p('command_timeout'))

        self.serial_port = serial.Serial(self.port, int(p('baudrate')), timeout=0.05)
        self.drain_startup_backlog()
        self.write_line('stop')
        self.write_line(f'speed {self.patrol_speed}')

        self.mode, self.mode_time = 'stop', 0.0
        self.cmd, self.cmd_time = None, 0.0
        self.last_sent = None

        self.create_subscription(String, '/esp32/mode', self.mode_callback, 10)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.left_pub = self.create_publisher(Range, '/ultrasonic/left', 10)
        self.right_pub = self.create_publisher(Range, '/ultrasonic/right', 10)
        self.status_pub = self.create_publisher(String, '/esp32/status', 10)
        self.create_timer(float(p('heartbeat_period')), self.send_current_command)

        self.reader_thread = threading.Thread(target=self.read_serial_loop, daemon=True)
        self.reader_thread.start()

        self.get_logger().info(f'ESP32 bridge on {self.port}, patrol speed {self.patrol_speed}')

    def drain_startup_backlog(self):
        """Opening the port resets the ESP32, and the USB driver hands over everything the
        board printed while no program had the port open. Wait for the reset, then drop it."""
        time.sleep(2.0)
        self.serial_port.reset_input_buffer()
        end = time.monotonic() + 0.3
        while time.monotonic() < end:
            self.serial_port.read(4096)

    def write_line(self, text):
        try:
            self.serial_port.write((text + '\n').encode())
        except serial.SerialException as e:
            self.get_logger().error(f'Serial write failed: {e}', throttle_duration_sec=2.0)

    def mode_callback(self, msg):
        self.mode, self.mode_time = msg.data.strip().lower(), time.monotonic()

    def cmd_vel_callback(self, msg):
        self.cmd, self.cmd_time = (msg.linear.x, msg.angular.z), time.monotonic()

    def send_current_command(self):
        now = time.monotonic()
        command = choose_command(self.mode, now - self.mode_time, self.cmd, now - self.cmd_time,
                                 self.timeout, self.pwm_args)
        self.write_line(command)          # every period: this is the watchdog heartbeat
        if command != self.last_sent:
            self.get_logger().info(f'ESP32 <- {command}')
            self.last_sent = command

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
            except Exception as e:
                self.get_logger().warn(f'Serial read error: {e}', throttle_duration_sec=2.0)
                time.sleep(0.5)
                continue
            parsed = parse_line(line) if line else None
            if parsed is None:
                continue
            if parsed[0] == 'U':
                self.publish_range(self.left_pub, parsed[1], 'left_ultrasonic')
                self.publish_range(self.right_pub, parsed[2], 'right_ultrasonic')
            elif parsed[0] == 'S':
                self.status_pub.publish(String(data=parsed[1]))
            elif parsed[0] == 'E':
                self.get_logger().warn(f'ESP32: {parsed[1]}')
            else:
                self.get_logger().info(f'ESP32: {parsed[1]}')

    def stop_motors(self):
        self.write_line('stop')


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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
