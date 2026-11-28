"""SecureBot web panel: run laps from a browser instead of terminal commands and windows.

  ros2 run securebot_ros web_panel                 # then open http://<robot>:8080
  ros2 run securebot_ros web_panel --port 8080 --host 0.0.0.0

Start (powering the robot software on for the active room if needed) / Stop / Return home,
re-localize, watch the robot on the map or the camera, tap an orange box to remove it; manage
rooms (rename, delete, no-pass lines, map a new one by exploring) and enrolled faces. The
panel itself is light (no Nav2, no motors): leave it running; it starts and stops the laps or
explore launch (web_logic.StackManager).
Anyone who can reach the port can drive the robot and enroll faces (add themselves as
authorized): use --host 127.0.0.1 off a trusted network.

This file is the ROS side (RosBridge: /laps/command, /explore/command, /clicked_point,
/tamper/command out; /laps/status, /explore/status, /laps/path, /laps/current, /laps/debris,
/laps/home, /map (while exploring), /esp32/status, /power/status (guard_node), /tamper,
/face_status and /recognized_name (live face node), TF map -> base_link in); the rest is in
web_logic.py.
"""
import argparse
import json
import math
import os
import threading
import time

import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray

from securebot_ros import power_logic as pl
from securebot_ros import web_logic as wl

FRESH_S = 3.0           # s: older guard / tamper / face messages = that node is not running


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class RosBridge(Node):
    def __init__(self):
        super().__init__('web_panel')
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.lock = threading.Lock()
        self.reset()
        self.cmd_pub = {'laps': self.create_publisher(String, '/laps/command', 10),
                        'explore': self.create_publisher(String, '/explore/command', 10)}
        self.click_pub = self.create_publisher(PointStamped, '/clicked_point', 10)
        self.tamper_pub = self.create_publisher(String, '/tamper/command', 10)
        self.tamper_hook = None           # fn(state, reason) per /tamper message (PanelApp)
        self.create_subscription(String, '/power/status', self.on_power, 10)
        self.create_subscription(String, '/tamper', self.on_tamper, 10)
        self.create_subscription(String, '/face_status', lambda m: self.on_face('status', m), 10)
        self.create_subscription(String, '/recognized_name',
                                 lambda m: self.on_face('names', m), 10)
        self.create_subscription(String, '/laps/status', lambda m: self.on_status('laps', m), 10)
        self.create_subscription(String, '/explore/status',
                                 lambda m: self.on_status('explore', m), 10)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, latched)
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
            self.status = {}              # kind -> (status dict, monotonic time)
            self.pose = self.home = self.esp32 = None
            self.path, self.current, self.boxes = [], [], []
            self.grid, self.grid_version, self.png = None, 0, (0, None)
            self.power_status = self.tamper_status = None     # (value, monotonic time)
            self.face = {}                                     # 'status'/'names' -> (text, t)

    # in --------------------------------------------------------------------------------------
    def on_status(self, kind, msg):
        try:
            status = json.loads(msg.data)
        except ValueError:
            return
        with self.lock:
            self.status[kind] = (status, time.monotonic())

    def on_map(self, msg):
        info = msg.info
        with self.lock:
            self.grid = msg
            self.grid_version += 1
            self.grid_meta = {'resolution': float(info.resolution),
                              'origin': [float(info.origin.position.x),
                                         float(info.origin.position.y)],
                              'width': int(info.width), 'height': int(info.height),
                              'version': self.grid_version}

    def live_png(self):
        """The newest /map as PNG (converted once per new map)."""
        with self.lock:
            grid, version = self.grid, self.grid_version
            if grid is None:
                return None
            if self.png[0] == version:
                return self.png[1]
        png = wl.occupancy_png(grid.data, grid.info.width, grid.info.height)
        with self.lock:
            self.png = (version, png)
        return png

    def on_power(self, msg):
        try:
            status = json.loads(msg.data)
        except ValueError:
            return
        with self.lock:
            self.power_status = (status, time.monotonic())

    def on_tamper(self, msg):
        parsed = pl.parse_tamper(msg.data)
        if parsed is None:
            return
        with self.lock:
            self.tamper_status = (parsed, time.monotonic())
        if self.tamper_hook:
            self.tamper_hook(*parsed)

    def on_face(self, which, msg):
        with self.lock:
            self.face[which] = (msg.data, time.monotonic())

    def power(self):
        with self.lock:
            v = self.power_status
        return v[0] if v and time.monotonic() - v[1] < FRESH_S else None

    def tamper(self):
        with self.lock:
            v = self.tamper_status
        return v[0] if v and time.monotonic() - v[1] < FRESH_S else None

    def faces(self):
        """{'status': AUTHORIZED | UNAUTHORIZED | SCANNING | NO_PERSON, 'names': ...} while the
        live face node runs, else None."""
        now = time.monotonic()
        with self.lock:
            fresh = {k: v for k, (v, t) in self.face.items() if now - t < FRESH_S}
        return {'status': fresh['status'], 'names': fresh.get('names', '')} \
            if 'status' in fresh else None

    def tamper_command(self, cmd):
        for _ in range(3):
            self.tamper_pub.publish(String(data=cmd))
            time.sleep(0.05)
        self.get_logger().info(f'tamper: {cmd}')

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
    def send_command(self, cmd, target='laps'):
        msg = String(data=cmd)
        for _ in range(3):              # a busy node can miss a single message
            self.cmd_pub[target].publish(msg)
            time.sleep(0.05)
        self.get_logger().info(f'command to {target}: {cmd}')

    def click(self, x, y):
        msg = PointStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x, msg.point.y = float(x), float(y)
        self.click_pub.publish(msg)

    def snapshot(self, kind='laps'):
        with self.lock:
            status, t = self.status.get(kind, (None, None))
            age = None if t is None else time.monotonic() - t
            return {'status': status, 'status_age': age, 'pose': self.pose,
                    'path': self.path, 'current': self.current, 'boxes': self.boxes,
                    'home': self.home, 'esp32': self.esp32,
                    'live_map': self.grid_meta if kind == 'explore' and self.grid else None}


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
    bridge.tamper_hook = app.tamper_event
    server = wl.make_server(app, opts.host, opts.port, static_dir())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log = bridge.get_logger()
    pruned = wl.prune_logs()
    if pruned:
        log.info(f'Deleted {pruned} old logs from log/ '
                 f'(keeping the {wl.KEEP_LOGS} newest of each)')
    log.info('SecureBot panel is up. Open one of these in a browser (not 0.0.0.0):')
    for url in wl.local_panel_urls(opts.host, opts.port):
        log.info(f'  {url}')
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
