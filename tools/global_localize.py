#!/usr/bin/env python3
"""Find the robot on a saved map from one LiDAR scan, and tell AMCL (/initialpose).

  python3 tools/global_localize.py --map maps/explore_1026_1922.yaml          # find and send
  python3 tools/global_localize.py --map ... --dry-run                         # just print
  python3 tools/global_localize.py --map ... --wait --tries 5        # at startup

--wait first waits until AMCL is up (subscribed to /initialpose, plus 5 s to activate; it
ignores /initialpose before that) and then takes a scan from after that moment; the laps
launch runs it this way (auto_localize:=true). --tries retries with a fresh scan when the
match is weak or ambiguous (e.g. someone standing in front of the LiDAR).

Scores every free spot of the map (every --step m, every --yaw-step deg) with scan_match's
score (share of scan points landing on a wall), keeps the best few places, refines them, and
sends the best pose if it matches at least --min-score of the scan
(ScanMatcher.global_search: laps_node uses the same search when the robot gets lost). Run
it with the laps / nav stack up (it needs /scan and TF base_link <- laser), robot standing
still.
"""
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

from securebot_ros.scan_match import ScanMatcher


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Grab(Node):
    def __init__(self):
        super().__init__('global_localize')
        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, self)
        self.points = None
        self.create_subscription(LaserScan, '/scan', self.cb, 10)
        self.pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

    def cb(self, msg):
        try:
            t = self.tf.lookup_transform('base_link', msg.header.frame_id, rclpy.time.Time())
        except Exception:
            return
        tx, ty = t.transform.translation.x, t.transform.translation.y
        yaw = yaw_of(t.transform.rotation)
        c, s = math.cos(yaw), math.sin(yaw)
        pts = []
        for k, r in enumerate(msg.ranges):
            if math.isfinite(r) and msg.range_min <= r <= msg.range_max:
                a = msg.angle_min + k * msg.angle_increment
                lx, ly = r * math.cos(a), r * math.sin(a)
                pts.append((tx + c * lx - s * ly, ty + s * lx + c * ly))
        self.points = np.array(pts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--map', required=True)
    ap.add_argument('--step', type=float, default=0.10)
    ap.add_argument('--yaw-step', type=float, default=5.0)
    ap.add_argument('--min-score', type=float, default=0.70)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--wait', action='store_true',
                    help='wait until AMCL is active (TF map -> base_link) before scanning')
    ap.add_argument('--tries', type=int, default=1,
                    help='attempts, each with a fresh scan, until a clear match')
    ap.add_argument('--min-margin', type=float, default=0.15,
                    help='best score must beat the next distinct place by this much')
    args = ap.parse_args()

    with open(args.map) as f:
        meta = yaml.safe_load(f)
    image = meta['image']
    if not os.path.isabs(image):
        image = os.path.join(os.path.dirname(os.path.abspath(args.map)), image)
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    res, origin = float(meta['resolution']), tuple(float(v) for v in meta['origin'][:2])
    matcher = ScanMatcher(img, res, origin)

    # free spots, the robot's centre >= 0.25 m from a wall
    free_xy = matcher.spots(args.step)

    rclpy.init()
    node = Grab()
    if args.wait:
        # AMCL subscribes to /initialpose when configured and is activated ~1 s later; it
        # drops poses that arrive before then, so give it a few seconds more
        print('global_localize: waiting for AMCL (/initialpose subscriber)', flush=True)
        while rclpy.ok() and node.count_subscribers('/initialpose') == 0:
            rclpy.spin_once(node, timeout_sec=0.2)
        end = time.monotonic() + 5.0
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.2)
        node.points = None                     # only scans from after AMCL came up

    for attempt in range(1, max(1, args.tries) + 1):
        node.points = None
        while rclpy.ok() and node.points is None:
            rclpy.spin_once(node, timeout_sec=0.2)
        pts = node.points
        if len(pts) > 300:
            pts = pts[np.linspace(0, len(pts) - 1, 300).astype(int)]
        print(f'global_localize: attempt {attempt}/{args.tries}: {len(free_xy)} spots x '
              f'{int(360 / args.yaw_step)} headings, {len(pts)} scan points', flush=True)
        # the search laps_node also uses when the robot is lost (ScanMatcher.global_search)
        refined = matcher.global_search(pts, free_xy, math.radians(args.yaw_step))
        for sc, (x, y, a) in refined[:5]:
            print(f'  score {sc:.0%}  x {x:6.2f}  y {y:6.2f}  '
                  f'yaw {math.degrees(a) % 360:6.1f} deg')
        best, (x, y, a) = refined[0]
        runner_up = next((sc for sc, p in refined[1:]
                          if math.hypot(p[0] - x, p[1] - y) > 0.5), 0.0)
        print(f'best {best:.0%}, next distinct place {runner_up:.0%}', flush=True)
        if best >= args.min_score and best - runner_up >= args.min_margin:
            break
        print('No clear match' + (': trying again with a new scan'
                                  if attempt < args.tries else ''), flush=True)
        time.sleep(2.0)
    else:
        print('global_localize: not sending a pose; set it with "2D Pose Estimate" in RViz',
              flush=True)
        sys.exit(1)
    if args.dry_run:
        return
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = 'map'
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
    msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = math.sin(a / 2), math.cos(a / 2)
    cov = [0.0] * 36
    cov[0] = cov[7] = 0.02
    cov[35] = 0.02
    msg.pose.covariance = cov
    for _ in range(5):
        msg.header.stamp = node.get_clock().now().to_msg()
        node.pub.publish(msg)
        end = time.monotonic() + 0.4
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
    print(f'global_localize: sent /initialpose x {x:.2f} y {y:.2f} '
          f'yaw {math.degrees(a) % 360:.1f} deg (score {best:.0%})', flush=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
