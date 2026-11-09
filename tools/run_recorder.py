#!/usr/bin/env python3
"""Record a laps run in detail, for analysing it afterwards (run it beside the laps launch).

  python3 tools/run_recorder.py --name 20261107_1950        # Ctrl+C (or SIGINT) to finish

Writes to log/laps_runs/ (or --dir):
  track_<name>.csv    t, map x/y/yaw (TF map->base_link), odom yaw, path id, laps state,
                      loop, message: ~20 Hz
  imu_cmd_<name>.csv  t, kind, a, b: imu (yaw, yaw rate), cmd (/cmd_vel_smoothed: speed,
                      turn rate), out (/nav_cmd_vel), bridge (/cmd_vel), as they arrive
  paths_<name>.jsonl  every path the robot was given (/laps/current), numbered
  events_<name>.log   every /laps/status change (state + message)
"""
import argparse
import json
import math
import os
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--name', default=time.strftime('%Y%m%d_%H%M'))
    ap.add_argument('--dir', default=os.path.expanduser('~/securebot_ws/log/laps_runs'))
    ap.add_argument('--rate', type=float, default=20.0, help='track rows per second')
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    path = lambda kind, ext: os.path.join(args.dir, f'{kind}_{args.name}.{ext}')  # noqa: E731

    rclpy.init()
    n = Node('run_recorder')
    tf = Buffer()
    TransformListener(tf, n)
    track = open(path('track', 'csv'), 'w')
    fast = open(path('imu_cmd', 'csv'), 'w')
    paths = open(path('paths', 'jsonl'), 'w')
    events = open(path('events', 'log'), 'w')
    track.write('t,x,y,yaw,odom_yaw,path_id,state,loop,message\n')
    fast.write('t,kind,a,b\n')
    st = {'s': {}, 'path': 0, 'last': None}

    def status(m):
        s = json.loads(m.data)
        st['s'] = s
        key = (s.get('state'), s.get('message'))
        if key != st['last']:
            st['last'] = key
            events.write(f"{time.time():.3f} {s.get('state', '')}: {s.get('message', '')}\n")
            events.flush()

    def got_path(m):
        st['path'] += 1
        paths.write(json.dumps({'t': time.time(), 'id': st['path'],
                                'pts': [(p.pose.position.x, p.pose.position.y)
                                        for p in m.poses]}) + '\n')
        paths.flush()

    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
    n.create_subscription(String, '/laps/status', status, 10)
    n.create_subscription(Path, '/laps/current', got_path, latched)
    n.create_subscription(Imu, '/imu', lambda m: fast.write(
        f'{time.time():.3f},imu,{yaw_of(m.orientation):.4f},{m.angular_velocity.z:.4f}\n'), 50)
    for topic, kind in (('/cmd_vel_smoothed', 'cmd'), ('/nav_cmd_vel', 'out'),
                        ('/cmd_vel', 'bridge')):
        n.create_subscription(Twist, topic, lambda m, k=kind: fast.write(
            f'{time.time():.3f},{k},{m.linear.x:.3f},{m.angular.z:.4f}\n'), 50)
    print(f'Recording to {args.dir} as *_{args.name}.*  (Ctrl+C to stop)', flush=True)
    period, next_row, last_flush = 1.0 / args.rate, 0.0, 0.0
    try:
        while rclpy.ok():
            rclpy.spin_once(n, timeout_sec=0.02)
            now = time.time()
            if now - last_flush > 1.0:
                fast.flush()
                track.flush()
                last_flush = now
            if now < next_row:
                continue
            next_row = now + period
            try:
                t = tf.lookup_transform('map', 'base_link', rclpy.time.Time()).transform
                o = tf.lookup_transform('odom', 'base_link', rclpy.time.Time()).transform
            except Exception:                               # noqa: BLE001 (no TF yet)
                continue
            s = st['s']
            msg = str(s.get('message', '')).replace('"', "'")
            track.write(f"{now:.2f},{t.translation.x:.3f},{t.translation.y:.3f},"
                        f"{yaw_of(t.rotation):.4f},{yaw_of(o.rotation):.4f},{st['path']},"
                        f"{s.get('state', '')},{s.get('loop', '')},\"{msg}\"\n")
    except KeyboardInterrupt:
        pass
    finally:
        for f in (track, fast, paths, events):
            f.close()
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
