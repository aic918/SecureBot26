"""SecureBot guard: battery percentage, return home on a low battery, stop when tampered with.

Inputs
  /battery/raw    sensor_msgs/BatteryState  pack voltage and current (esp32_serial_bridge,
                                            from the INA226)
  /tamper         std_msgs/String           off | armed | tripped,<reason> | unavailable
  /laps/status, /explore/status  std_msgs/String (JSON)  whether the robot is out on its job
Outputs
  /power/status   std_msgs/String (JSON)    pct, volts, amps, low, return_home_pct, cells
  /battery        sensor_msgs/BatteryState  the same as a standard message
  /laps/command, /explore/command  'home' while the battery is low and the robot is on its
                  job; 'stop' when it is tampered with (the ESP32 has already stopped the
                  motors itself and ignores driving until the alert is acknowledged)

Settings (return-home %, 6 or 7 cells, pack resistance): ~/securebot_ws/config/power.json,
written by the web panel and re-read here when it changes. Logic: power_logic.py.
"""
import json
import os
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from std_msgs.msg import String

from securebot_ros import power_logic as pl

STATUS_STALE_S = 3.0      # s: a laps / explore status older than this = that node is not running


class GuardNode(Node):
    def __init__(self):
        super().__init__('securebot_guard')
        self.declare_parameter('settings_file', pl.SETTINGS_FILE)
        self.settings_file = os.path.expanduser(str(self.get_parameter('settings_file').value))
        self.settings_mtime = None
        self.battery = pl.BatteryMonitor(pl.load_settings(self.settings_file))
        self.guard = pl.Guard()
        self.tamper_state, self.tamper_reason = 'unavailable', ''
        self.states = {}                  # kind -> (state, monotonic time)
        self.cmd_pub = {k: self.create_publisher(String, f'/{k}/command', 10)
                        for k in ('laps', 'explore')}
        self.status_pub = self.create_publisher(String, '/power/status', 10)
        self.battery_pub = self.create_publisher(BatteryState, '/battery', 10)
        self.create_subscription(BatteryState, '/battery/raw', self.on_battery, 10)
        self.create_subscription(String, '/tamper', self.on_tamper, 10)
        for kind in ('laps', 'explore'):
            self.create_subscription(String, f'/{kind}/status',
                                     lambda m, k=kind: self.on_status(k, m), 10)
        self.create_timer(1.0, self.tick)
        self.get_logger().info(
            f'Guard up: return home at {self.battery.settings["return_home_pct"]} % '
            f'({self.battery.settings["cells"]}S pack), stop when tampered with')

    def reload_settings(self):
        try:
            mtime = os.path.getmtime(self.settings_file)
        except OSError:
            mtime = None
        if mtime == self.settings_mtime:
            return
        self.settings_mtime = mtime
        new = pl.load_settings(self.settings_file)
        if new != self.battery.settings:
            self.battery.settings = new
            self.battery.low, self.battery.under_since = False, None    # judge again
            self.get_logger().info(f'Power settings: return home at {new["return_home_pct"]} '
                                   f'%, {new["cells"]}S pack')

    def on_battery(self, msg):
        if self.battery.update(time.monotonic(), float(msg.voltage), float(msg.current)):
            self.get_logger().warn(f'Battery low ({self.battery.pct:.0f} %, '
                                   f'{msg.voltage:.2f} V): returning home')
            self.act()

    def on_tamper(self, msg):
        parsed = pl.parse_tamper(msg.data)
        if parsed is None:
            return
        state, reason = parsed
        if state == 'tripped' and self.tamper_state != 'tripped':
            self.get_logger().error(f'TAMPER: robot {pl.tamper_text(reason)}; motors stopped')
        elif state != 'tripped' and self.tamper_state == 'tripped':
            self.get_logger().info(f'Tamper alert cleared ({state})')
        self.tamper_state, self.tamper_reason = state, reason
        self.act()

    def on_status(self, kind, msg):
        try:
            state = json.loads(msg.data).get('state')
        except (ValueError, AttributeError):
            return
        self.states[kind] = (state, time.monotonic())

    def act(self):
        now = time.monotonic()
        states = {k: (s if now - t < STATUS_STALE_S else None)
                  for k, (s, t) in self.states.items()}
        for kind, cmd in self.guard.commands(now, self.battery.low, self.tamper_state, states):
            for _ in range(3):        # a busy node can miss one message
                self.cmd_pub[kind].publish(String(data=cmd))
            self.get_logger().info(f'{kind} <- {cmd}')

    def tick(self):
        self.reload_settings()
        now = time.monotonic()
        status = self.battery.status(now)
        status['tamper'] = {'state': self.tamper_state, 'reason': self.tamper_reason}
        self.status_pub.publish(String(data=json.dumps(status)))
        if status['pct'] is not None:
            msg = BatteryState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.voltage, msg.current = float(status['volts']), -float(status['amps'])
            msg.percentage = float(status['pct']) / 100.0
            msg.capacity = msg.design_capacity = 3.2
            msg.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LION
            msg.present = True
            self.battery_pub.publish(msg)
        self.act()


def main(args=None):
    rclpy.init(args=args)
    node = GuardNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
