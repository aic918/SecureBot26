"""SecureBot web panel: run laps from a browser instead of terminal commands and windows.

  ros2 run securebot_ros web_panel                 # then open http://<robot>:8080
  ros2 run securebot_ros web_panel --port 8080 --host 0.0.0.0

Power the robot software on for a room, Start / Stop / Go home, re-localize, watch the robot
on the map and tap an orange box to remove it. The panel itself is light (no Nav2, no
motors): leave it running; it starts and stops the laps launch (web_logic.StackManager).
Anyone who can reach the port can drive the robot and enroll faces (add themselves as
authorized): use --host 127.0.0.1 off a trusted network.

This file is the ROS side (RosBridge: /laps/command, /clicked_point out; /laps/status,
/laps/path, /laps/current, /laps/debris, /laps/home, /esp32/status, TF map -> base_link in);
the rest is in web_logic.py.
"""
import argparse
import json
import math
import os
import threading
import time

import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray

from securebot_ros import web_logic as wl


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class RosBridge(Node):
    def __init__(self):
        super().__init__('web_panel')
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.lock = threading.Lock()
        self.reset()
        self.cmd_pub = self.create_publisher(String, '/laps/command', 10)
        self.click_pub = self.create_publisher(PointStamped, '/clicked_point', 10)
        self.create_subscription(String, '/laps/status', self.on_status, 10)
        self.create_subscription(String, '/esp32/status', self.on_esp32, 10)
        self.create_subscription(Path, '/laps/path', lambda m: self.on_path('path', m), latched)
        self.create_subscription(Path, '/laps/current', lambda m: self.on_path('current', m),
                                 latched)
        self.create_subscription(MarkerArray, '/laps/debris', self.on_debris, latched)
        self.create_subscription(MarkerArray, '/laps/home', self.on_home, latched)
        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, self)
        self.create_timer(0.2, self.update_pose)

    def reset(self):
        with self.lock:
            self.status, self.status_time = None, None
            self.pose = self.home = self.esp32 = None
            self.path, self.current, self.boxes = [], [], []

    # in --------------------------------------------------------------------------------------
    def on_status(self, msg):
        try:
            status = json.loads(msg.data)
        except ValueError:
            return
        with self.lock:
            self.status, self.status_time = status, time.monotonic()

    def on_esp32(self, msg):
        with self.lock:
            self.esp32 = msg.data

    def on_path(self, which, msg):
        pts = wl.thin_path([(p.pose.position.x, p.pose.position.y) for p in msg.poses])
        with self.lock:
            setattr(self, which, pts)

    def on_debris(self, msg):
        with self.lock:
            self.boxes = wl.boxes_from_markers(msg.markers)

    def on_home(self, msg):
        home = wl.home_from_markers(msg.markers)
        if home is not None:
            with self.lock:
                self.home = home

    def update_pose(self):
        try:
            t = self.tf.lookup_transform('map', 'base_link', rclpy.time.Time())
        except Exception:
            return
        tr = t.transform.translation
        with self.lock:
            self.pose = [round(tr.x, 3), round(tr.y, 3), round(yaw_of(t.transform.rotation), 3)]

    # out -------------------------------------------------------------------------------------
    def send_command(self, cmd):
        msg = String(data=cmd)
        for _ in range(3):              # a busy laps node can miss a single message
            self.cmd_pub.publish(msg)
            time.sleep(0.05)
        self.get_logger().info(f'command: {cmd}')

    def click(self, x, y):
        msg = PointStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x, msg.point.y = float(x), float(y)
        self.click_pub.publish(msg)

    def snapshot(self):
        with self.lock:
            age = None if self.status_time is None else time.monotonic() - self.status_time
            return {'status': self.status, 'status_age': age, 'pose': self.pose,
                    'path': self.path, 'current': self.current, 'boxes': self.boxes,
                    'home': self.home, 'esp32': self.esp32}


def static_dir():
    try:
        from ament_index_python.packages import get_package_share_directory
        d = os.path.join(get_package_share_directory('securebot_ros'), 'web')
        if os.path.exists(os.path.join(d, 'index.html')):
            return d
    except Exception:
        pass
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'web')


def main(args=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8080)
    opts, ros_args = ap.parse_known_args(args)
    rclpy.init(args=ros_args)
    bridge = RosBridge()
    app = wl.PanelApp(bridge)
    server = wl.make_server(app, opts.host, opts.port, static_dir())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    bridge.get_logger().info(f'SecureBot panel on http://{opts.host}:{opts.port}')
    try:
        rclpy.spin(bridge)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        server.shutdown()
        app.faces.shutdown()
        if app.stack.state not in ('off', 'crashed'):
            bridge.get_logger().info('Shutting the robot software down')
            app.stack.stop()
        bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
