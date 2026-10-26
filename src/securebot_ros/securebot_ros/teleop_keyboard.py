"""Drive the robot from the keyboard, hold-to-move, for mapping.

  ros2 launch securebot_ros securebot_teleop.launch.py      # terminal 1: bridge + motion manager
  ros2 run securebot_ros teleop_keyboard                    # terminal 2: this, needs the keyboard

Publishes /nav_cmd_vel at 10 Hz. A key keeps the robot moving only while it is held (the
terminal's key repeat); `hold_time` after the last key press the command drops to zero, and
the motion manager / bridge stop the wheels if this program dies.

  w forward     s backward     a turn left     d turn right
  q forward+left   e forward+right   space stop now
  + / -  faster / slower (within max_linear / max_angular)     x or Ctrl+C quit
"""
import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# key -> (linear sign, angular sign); angular + is counter-clockwise = left
KEYS = {
    'w': (1, 0), 's': (-1, 0), 'a': (0, 1), 'd': (0, -1),
    'q': (1, 1), 'e': (1, -1), ' ': (0, 0),
}
HELP = ('w/s forward/back  a/d turn  q/e curve  space stop  +/- speed  x quit   '
        '(hold the key; let go to stop)')


class TeleopState:
    """Current command from key presses; goes to zero `hold_time` after the last one."""

    def __init__(self, linear, angular, max_linear, max_angular, hold_time):
        self.linear, self.angular = linear, angular
        self.max_linear, self.max_angular = max_linear, max_angular
        self.hold_time = hold_time
        self.direction = (0, 0)
        self.last_key_time = None

    def key(self, ch, now):
        """Handle one key. Returns False for quit."""
        if ch in ('x', '\x03'):                     # x or Ctrl+C
            self.direction = (0, 0)
            return False
        if ch in KEYS:
            self.direction, self.last_key_time = KEYS[ch], now
        elif ch in ('+', '='):
            self.scale(1.25)
        elif ch in ('-', '_'):
            self.scale(0.8)
        return True

    def scale(self, factor):
        self.linear = min(self.max_linear, self.linear * factor)
        self.angular = min(self.max_angular, self.angular * factor)

    def command(self, now):
        """(linear m/s, angular rad/s) to send now."""
        if self.last_key_time is None or now - self.last_key_time > self.hold_time:
            return 0.0, 0.0
        return self.direction[0] * self.linear, self.direction[1] * self.angular


class TeleopKeyboard(Node):
    def __init__(self):
        super().__init__('teleop_keyboard')
        self.declare_parameter('linear', 0.15)          # m/s, mapping pace for rf2o
        self.declare_parameter('angular', 0.6)          # rad/s
        self.declare_parameter('max_linear', 0.25)
        self.declare_parameter('max_angular', 1.0)
        # Longer than the terminal's key-repeat delay (~0.5 s), so a held key does not stutter.
        self.declare_parameter('hold_time', 0.7)
        p = lambda name: float(self.get_parameter(name).value)  # noqa: E731
        self.state = TeleopState(p('linear'), p('angular'), p('max_linear'), p('max_angular'),
                                 p('hold_time'))
        self.pub = self.create_publisher(Twist, '/nav_cmd_vel', 10)

    def publish(self):
        linear, angular = self.state.command(time.monotonic())
        msg = Twist()
        msg.linear.x, msg.angular.z = linear, angular
        self.pub.publish(msg)
        return linear, angular


def read_key(timeout):
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.read(1) if ready else None


def main(args=None):
    if not sys.stdin.isatty():
        sys.exit('teleop_keyboard needs a terminal: run it with `ros2 run`, not from a launch file.')
    rclpy.init(args=args)
    node = TeleopKeyboard()
    saved = termios.tcgetattr(sys.stdin)
    print(HELP)
    try:
        tty.setcbreak(sys.stdin.fileno())
        next_pub = 0.0
        while rclpy.ok():
            ch = read_key(0.05)
            if ch is not None and not node.state.key(ch, time.monotonic()):
                break
            now = time.monotonic()
            if now >= next_pub:
                linear, angular = node.publish()
                s = node.state
                print(f'\r  sending {linear:+.2f} m/s {angular:+.2f} rad/s   '
                      f'(speed {s.linear:.2f} m/s, turn {s.angular:.2f} rad/s)   ', end='',
                      flush=True)
                next_pub = now + 0.1
            rclpy.spin_once(node, timeout_sec=0)
    finally:
        node.state.direction = (0, 0)
        node.state.last_key_time = None
        for _ in range(3):                          # make sure a stop goes out
            node.publish()
            time.sleep(0.05)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, saved)
        print('\nstopped.')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
