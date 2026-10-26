"""Final gate between the security decision and the wheels.

Inputs
  /security/motion  std_msgs/String  stop | patrol    (securebot_live_node, every camera frame)
  /nav_cmd_vel      geometry_msgs/Twist                (Nav2 / teleop later; optional)
Outputs (to esp32_serial_bridge)
  /esp32/mode  std_msgs/String  stop | patrol | drive
  /cmd_vel     geometry_msgs/Twist   (only meaningful in drive mode)

Rules, in order:
  1. security says stop, or has said nothing for `security_timeout` -> stop
     (skipped when require_security is false: keyboard driving without the camera)
  2. a /nav_cmd_vel newer than `nav_timeout`                        -> drive with it
  3. otherwise                                                      -> idle_action: patrol (the
     firmware's own drive-forward + ultrasonic avoidance, the default) or stop (keyboard
     driving and Nav2, where the robot must not wander off on its own)
"""
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist


def decide(security, security_age, nav_age, security_timeout, nav_timeout,
           require_security=True, idle_action='patrol'):
    """-> 'stop' | 'drive' | 'patrol'. Ages are seconds since the last message (inf if never)."""
    if require_security and (security != 'patrol' or security_age > security_timeout):
        return 'stop'
    if nav_age <= nav_timeout:
        return 'drive'
    return 'patrol' if idle_action == 'patrol' else 'stop'


class MotionManagerNode(Node):
    def __init__(self):
        super().__init__('motion_manager_node')

        self.declare_parameter('security_timeout', 1.0)
        self.declare_parameter('nav_timeout', 0.5)
        self.declare_parameter('require_security', True)
        self.declare_parameter('idle_action', 'patrol')     # patrol | stop
        self.security_timeout = float(self.get_parameter('security_timeout').value)
        self.nav_timeout = float(self.get_parameter('nav_timeout').value)
        self.require_security = bool(self.get_parameter('require_security').value)
        self.idle_action = self.get_parameter('idle_action').value
        if self.idle_action not in ('patrol', 'stop'):
            raise ValueError(f'idle_action must be patrol or stop, not {self.idle_action!r}')

        self.security, self.security_time = 'stop', None
        self.nav_cmd, self.nav_time = Twist(), None
        self.last_mode = None

        self.create_subscription(String, '/security/motion', self.security_callback, 10)
        self.create_subscription(Twist, '/nav_cmd_vel', self.nav_cmd_callback, 10)
        self.mode_pub = self.create_publisher(String, '/esp32/mode', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.timer = self.create_timer(0.05, self.publish_final_cmd)

        self.get_logger().info(
            'Motion manager: /security/motion + /nav_cmd_vel -> /esp32/mode + /cmd_vel '
            f'(require_security={self.require_security}, idle_action={self.idle_action})')

    def security_callback(self, msg):
        self.security, self.security_time = msg.data.strip().lower(), time.monotonic()

    def nav_cmd_callback(self, msg):
        self.nav_cmd, self.nav_time = msg, time.monotonic()

    def publish_final_cmd(self):
        now = time.monotonic()
        age = lambda t: float('inf') if t is None else now - t  # noqa: E731
        mode = decide(self.security, age(self.security_time), age(self.nav_time),
                      self.security_timeout, self.nav_timeout,
                      self.require_security, self.idle_action)

        self.mode_pub.publish(String(data=mode))
        self.cmd_pub.publish(self.nav_cmd if mode == 'drive' else Twist())

        if mode != self.last_mode:
            self.get_logger().info(f'Motion: {mode}')
            self.last_mode = mode


def main(args=None):
    rclpy.init(args=args)
    node = MotionManagerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
