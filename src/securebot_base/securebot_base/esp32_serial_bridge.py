#!/usr/bin/env python3
"""The only program that talks to the ESP32 (firmware 2.2 protocol, see firmware/esp32/).

Inputs
  /esp32/mode  std_msgs/String  stop | patrol | drive   (published by motion_manager_node)
  /cmd_vel     geometry_msgs/Twist                       used only in drive mode: sent as
               wheel speeds (V, speed_control:=true, default) or PWM (M)
  /tamper/command  std_msgs/String  on | off | clear (clear = the alarm was acknowledged)
Outputs
  /ultrasonic/left, /ultrasonic/right  sensor_msgs/Range  (no echo -> +inf, REP-117)
  /esp32/status  std_msgs/String  the firmware's `S,<mode>,<l>,<r>,<avoid>` line, without "S,"
  /imu           sensor_msgs/Imu  heading from the MPU-6050 gyro (`G,<yaw_deg>,<rate_dps>`):
                 yaw (integrated since the ESP32 started, so it drifts slowly) and yaw rate,
                 in base_link; imu_yaw_sign:=-1.0 if the board is mounted upside down
  /wheel_ticks   std_msgs/Int64MultiArray  [left, right] wheel encoder counts since the ESP32
                 started (`W,<left>,<right>`, + = that wheel forward); they restart at 0 when
                 the ESP32 restarts
  /battery/raw   sensor_msgs/BatteryState  pack voltage (V) and current (A, + = discharging)
                 from the INA226 (`B,<mV>,<mA>`), 2 Hz; guard_node turns it into a percentage
  /tamper        std_msgs/String  off | armed | tripped,<tilt|lift> | unavailable (`T,...`)

Tamper detection is armed when the bridge starts (tamper_detect:=true, the default): from then
on, lifting or tipping the robot stops the motors in the ESP32 until /tamper/command "clear".

Safety: the current command is resent every `heartbeat_period` (the firmware stops on its own
after 250 ms without one). A mode or /cmd_vel older than `command_timeout` counts as "stop",
so if the motion manager or the planner dies, the robot stops.

The ESP32 is restarted (EN pulsed through the USB adapter, boot pin at "run") when the bridge
starts and whenever it has sent nothing for `silence_reset` seconds: once it was found running
but silent, and opening the port does not reliably restart it.
"""
import math
import threading
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import BatteryState, Imu, Range
from std_msgs.msg import Int64MultiArray, String
import serial

MODES = ('stop', 'patrol', 'drive')


def clamp(value, low, high):
    return max(low, min(high, value))


def twist_to_pwm(linear, angular, max_linear_mps, max_angular_rps, wheel_base_m, max_pwm,
                 min_pwm=0, gentle_pwm=0, gentle_below=0):
    """Differential-drive Twist -> (left_pwm, right_pwm), each clamped to +/-max_pwm.
    Below min_pwm the motors only hum, so a command whose faster wheel is under it is scaled
    up (both wheels together, keeping their ratio and so the direction of a curve).
    Gentle band: a slow request (faster wheel under `gentle_below` before lifting, e.g. the lap
    follower's small correction spins) is lifted only to `gentle_pwm`, so corrections do not
    jolt the robot at full power."""
    linear = clamp(linear, -max_linear_mps, max_linear_mps)
    angular = clamp(angular, -max_angular_rps, max_angular_rps)
    left = (linear - angular * wheel_base_m / 2.0) / max_linear_mps * max_pwm
    right = (linear + angular * wheel_base_m / 2.0) / max_linear_mps * max_pwm

    fastest = max(abs(left), abs(right))
    floor = gentle_pwm if gentle_pwm and 0 < fastest < gentle_below else min_pwm
    if 0 < fastest < floor:
        left, right = left * floor / fastest, right * floor / fastest

    to_pwm = lambda v: int(clamp(round(v), -max_pwm, max_pwm))  # noqa: E731
    return to_pwm(left), to_pwm(right)


def twist_to_tps(linear, angular, max_linear_mps, max_angular_rps, wheel_base_m,
                 ticks_per_meter, min_tps=0):
    """Differential-drive Twist -> (left, right) wheel speeds in encoder ticks/s, for the
    firmware's speed loop (V command). A wheel that should turn but would get under min_tps is
    lifted to it (the loop cannot hold a crawl), keeping its sign."""
    linear = clamp(linear, -max_linear_mps, max_linear_mps)
    angular = clamp(angular, -max_angular_rps, max_angular_rps)
    out = []
    for v in (linear - angular * wheel_base_m / 2.0, linear + angular * wheel_base_m / 2.0):
        t = v * ticks_per_meter
        if 0 < abs(t) < min_tps:
            t = math.copysign(min_tps, t)
        out.append(int(round(t)))
    return out[0], out[1]


def choose_command(mode, mode_age, cmd, cmd_age, timeout, pwm_args, speed_args=None):
    """What to send to the ESP32 now. Stale mode -> stop; drive with a stale /cmd_vel -> M,0,0
    (V,0,0 with speed control). With speed_args (twist_to_tps after the twist) the wheels get
    speeds (V,l,r) for the firmware's speed loop instead of PWM (M,l,r)."""
    if mode not in MODES or mode_age > timeout or mode == 'stop':
        return 'stop'
    if mode == 'patrol':
        return 'patrol'
    if cmd is None or cmd_age > timeout:
        return 'V,0,0' if speed_args else 'M,0,0'
    if speed_args:
        left, right = twist_to_tps(cmd[0], cmd[1], *speed_args)
        return f'V,{left},{right}'
    left, right = twist_to_pwm(cmd[0], cmd[1], *pwm_args)
    return f'M,{left},{right}'


def parse_line(line):
    """('U', left_ft, right_ft) | ('G', yaw_deg, rate_dps) | ('W', left_ticks, right_ticks) |
    ('S', 'drive,90,90,0') | ('B', volts, amps) | ('T', 'tripped,tilt') | ('I'/'E', text) |
    None."""
    parts = line.split(',')
    if parts[0] == 'B' and len(parts) == 3:
        try:
            return 'B', int(parts[1]) / 1000.0, int(parts[2]) / 1000.0
        except ValueError:
            return None
    if parts[0] == 'W' and len(parts) == 3:
        try:
            return 'W', int(parts[1]), int(parts[2])
        except ValueError:
            return None
    if parts[0] in ('U', 'G') and len(parts) == 3:
        try:
            return parts[0], float(parts[1]), float(parts[2])
        except ValueError:
            return None
    if parts[0] in ('S', 'T', 'I', 'E') and len(parts) > 1:
        return parts[0], line[2:]
    return None


def imu_message(yaw, rate):
    """sensor_msgs/Imu in base_link with only the heading filled in: yaw (rad) as orientation,
    yaw rate (rad/s); roll, pitch and acceleration are unknown (huge / -1 covariance)."""
    msg = Imu()
    msg.header.frame_id = 'base_link'
    msg.orientation.z, msg.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
    msg.orientation_covariance = [1e6, 0.0, 0.0, 0.0, 1e6, 0.0, 0.0, 0.0, 0.01]
    msg.angular_velocity.z = rate
    msg.angular_velocity_covariance = [1e6, 0.0, 0.0, 0.0, 1e6, 0.0, 0.0, 0.0, 1e-4]
    msg.linear_acceleration_covariance[0] = -1.0
    return msg


class ESP32SerialBridge(Node):
    def __init__(self):
        super().__init__('esp32_serial_bridge')

        # Default: the ESP32's USB socket (path 2.4). /dev/ttyUSBn numbers can swap with the
        # LiDAR, which uses the same CP2102 adapter with the same serial number.
        self.declare_parameter('port', '/dev/serial/by-path/platform-3610000.usb-usb-0:2.4:1.0-port0')
        self.declare_parameter('baudrate', 115200)
        # PWM is 10-bit duty (0-1023) at 10 kHz, on a DRV8262 driver since 2026-11-22. Measured
        # by spinning in place (gyro): nothing below ~550, ~13 deg/s at 560, ~45 at 575-580,
        # ~65 at 600, ~90 at 620, ~118 at 655. So 575 is the floor for any wheel that should turn
        # (a 0.8 rad/s turn lands there) and 655 (the firmware's cap) the ceiling.
        self.declare_parameter('max_pwm', 655)
        self.declare_parameter('min_pwm', 575)          # deadband floor for M,l,r (0 = off)
        # slow requests (correction spins) are lifted only to gentle_pwm (0 = off), see
        # twist_to_pwm; 300 sits above a 0.45 rad/s correction spin (~185) and below a 0.8 rad/s
        # turn (~330) or driving at 0.2 m/s (~375). 565 spins ~20 deg/s.
        self.declare_parameter('gentle_pwm', 565)
        self.declare_parameter('gentle_below', 300)
        self.declare_parameter('max_linear_mps', 0.6)     # laps 0.56 m/s (1.25 mph), 2026-11-22
        self.declare_parameter('max_angular_rps', 1.5)
        # Drive wheels 72 mm wide, 368 mm apart inside edge to inside edge -> 440 mm centre to
        # centre (measured 2026-10-26). Wheel diameter 192 mm (used once encoders give odometry).
        self.declare_parameter('wheel_base_m', 0.44)
        self.declare_parameter('patrol_speed', 655)     # PWM the firmware's own patrol uses
        self.declare_parameter('heartbeat_period', 0.1)     # firmware watchdog is 250 ms
        self.declare_parameter('command_timeout', 0.5)
        self.declare_parameter('silence_reset', 3.0)    # s without a line -> restart the ESP32
        self.declare_parameter('imu_yaw_sign', 1.0)     # -1.0: IMU board mounted upside down
        self.declare_parameter('tamper_detect', True)   # arm tamper detection at start
        # Wheel speed control (firmware V command, 2026-11-22): /cmd_vel becomes wheel speeds in
        # encoder ticks/s and the ESP32 holds each wheel at its speed with the encoders (the two
        # motors differ by up to 50 % at the same PWM). false: open-loop PWM (M command) with
        # the min/max/gentle_pwm settings above. ticks_per_meter: same as heading_odom_node's.
        self.declare_parameter('speed_control', True)
        self.declare_parameter('ticks_per_meter', 1190.0)   # measured, see heading_odom_node
        self.declare_parameter('min_wheel_tps', 40)     # slowest wheel speed asked for (~3.5 cm/s)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.port = p('port')
        self.pwm_args = (float(p('max_linear_mps')), float(p('max_angular_rps')),
                         float(p('wheel_base_m')), int(p('max_pwm')), int(p('min_pwm')),
                         int(p('gentle_pwm')), int(p('gentle_below')))
        self.speed_args = (float(p('max_linear_mps')), float(p('max_angular_rps')),
                           float(p('wheel_base_m')), float(p('ticks_per_meter')),
                           int(p('min_wheel_tps'))) if p('speed_control') else None
        self.patrol_speed = int(p('patrol_speed'))
        self.timeout = float(p('command_timeout'))
        self.silence_reset = float(p('silence_reset'))
        self.imu_sign = float(p('imu_yaw_sign'))
        self.tamper_detect = bool(p('tamper_detect'))

        self.serial_port = serial.Serial(self.port, int(p('baudrate')), timeout=0.05)
        if self.restart_esp32(wait=True):
            self.get_logger().info('ESP32 restarted and ready')
        else:
            self.get_logger().warn('ESP32 did not report ready after a restart; '
                                   'is it powered and running the SecureBot firmware?')
        self.send_settings()
        self.last_rx = time.monotonic()
        self.last_reset = time.monotonic()

        self.mode, self.mode_time = 'stop', 0.0
        self.cmd, self.cmd_time = None, 0.0
        self.last_sent = None

        self.create_subscription(String, '/esp32/mode', self.mode_callback, 10)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.create_subscription(String, '/tamper/command', self.tamper_callback, 10)
        self.battery_pub = self.create_publisher(BatteryState, '/battery/raw', 10)
        self.tamper_pub = self.create_publisher(String, '/tamper', 10)
        self.left_pub = self.create_publisher(Range, '/ultrasonic/left', 10)
        self.right_pub = self.create_publisher(Range, '/ultrasonic/right', 10)
        self.status_pub = self.create_publisher(String, '/esp32/status', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu', 20)
        self.ticks_pub = self.create_publisher(Int64MultiArray, '/wheel_ticks', 20)
        self.create_timer(float(p('heartbeat_period')), self.send_current_command)
        self.create_timer(1.0, self.check_silence)

        self.reader_thread = threading.Thread(target=self.read_serial_loop, daemon=True)
        self.reader_thread.start()

        self.get_logger().info(f'ESP32 bridge on {self.port}, patrol speed {self.patrol_speed}')

    def restart_esp32(self, wait=False):
        """Restart the ESP32 into its program, as esptool does: RTS pulls EN low while DTR
        keeps IO0 high. Drops what the USB driver buffered before (it replays old lines when
        the port opens). With wait, reads until the firmware's ready line (only before the
        reader thread runs), logs the firmware's boot messages (e.g. whether the IMU answered)
        and returns whether the ready line came."""
        s = self.serial_port
        s.dtr = False
        s.rts = True
        time.sleep(0.1)
        s.reset_input_buffer()
        s.rts = False
        if not wait:
            return None
        buf, end = b'', time.monotonic() + 4.0      # boot + ~1.3 s gyro calibration
        while time.monotonic() < end and b'I,SecureBot' not in buf:
            buf += s.read(256)
        for raw in buf.decode('utf-8', errors='ignore').splitlines():
            parsed = parse_line(raw.strip())
            if parsed and parsed[0] == 'E':
                self.get_logger().warn(f'ESP32: {parsed[1]}')
            elif parsed and parsed[0] == 'I' and not parsed[1].startswith('SecureBot'):
                self.get_logger().info(f'ESP32: {parsed[1]}')
        return b'I,SecureBot' in buf

    def send_settings(self):
        self.write_line('stop')
        self.write_line(f'speed {self.patrol_speed}')
        self.write_line('tamper on' if self.tamper_detect else 'tamper off')

    def check_silence(self):
        now = time.monotonic()
        silent = now - self.last_rx
        if silent > self.silence_reset and now - self.last_reset > self.silence_reset:
            self.get_logger().error(f'No data from the ESP32 for {silent:.0f} s; restarting it')
            self.last_reset = now
            self.restart_esp32()

    def write_line(self, text):
        try:
            self.serial_port.write((text + '\n').encode())
        except serial.SerialException as e:
            self.get_logger().error(f'Serial write failed: {e}', throttle_duration_sec=2.0)

    def mode_callback(self, msg):
        self.mode, self.mode_time = msg.data.strip().lower(), time.monotonic()

    def tamper_callback(self, msg):
        cmd = msg.data.strip().lower()
        if cmd in ('on', 'off', 'clear'):
            self.write_line(f'tamper {cmd}')
            self.get_logger().info(f'ESP32 <- tamper {cmd}')

    def publish_battery(self, volts, amps):
        msg = BatteryState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.voltage, msg.current = float(volts), float(amps)
        msg.percentage = float('nan')
        msg.present = True
        self.battery_pub.publish(msg)

    def cmd_vel_callback(self, msg):
        self.cmd, self.cmd_time = (msg.linear.x, msg.angular.z), time.monotonic()

    def send_current_command(self):
        now = time.monotonic()
        command = choose_command(self.mode, now - self.mode_time, self.cmd, now - self.cmd_time,
                                 self.timeout, self.pwm_args, self.speed_args)
        self.write_line(command)          # every period: this is the watchdog heartbeat
        if command != self.last_sent:
            self.get_logger().info(f'ESP32 <- {command}')
            self.last_sent = command

    def publish_imu(self, yaw_deg, rate_dps):
        msg = imu_message(self.imu_sign * math.radians(yaw_deg),
                          self.imu_sign * math.radians(rate_dps))
        msg.header.stamp = self.get_clock().now().to_msg()
        self.imu_pub.publish(msg)

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
            self.last_rx = time.monotonic()
            if parsed[0] == 'U':
                self.publish_range(self.left_pub, parsed[1], 'left_ultrasonic')
                self.publish_range(self.right_pub, parsed[2], 'right_ultrasonic')
            elif parsed[0] == 'G':
                self.publish_imu(parsed[1], parsed[2])
            elif parsed[0] == 'W':
                self.ticks_pub.publish(Int64MultiArray(data=[parsed[1], parsed[2]]))
            elif parsed[0] == 'S':
                self.status_pub.publish(String(data=parsed[1]))
            elif parsed[0] == 'B':
                self.publish_battery(parsed[1], parsed[2])
            elif parsed[0] == 'T':
                self.tamper_pub.publish(String(data=parsed[1]))
            elif parsed[0] == 'E':
                self.get_logger().warn(f'ESP32: {parsed[1]}')
            else:
                self.get_logger().info(f'ESP32: {parsed[1]}')
                if parsed[1].startswith('SecureBot'):      # it restarted: settings are gone
                    self.send_settings()

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
