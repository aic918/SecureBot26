#!/usr/bin/env python3
"""Find the robot on a saved map from one LiDAR scan, and tell AMCL (/initialpose).

  python3 tools/global_localize.py --map maps/explore_1026_1922.yaml          # find and send
  python3 tools/global_localize.py --map ... --dry-run                         # just print
  python3 tools/global_localize.py --map ... --wait --tries 5        # at startup
  python3 tools/global_localize.py --map ... --near 0,0              # parked at the start point
  python3 tools/global_localize.py --map ... --near "-3.1,2.4;0,0"   # last run's end, then start

--wait first waits until AMCL is up (subscribed to /initialpose, plus 5 s to activate; it
ignores /initialpose before that) and then takes a scan from after that moment; the laps
launch runs it this way (auto_localize:=true). --tries retries with a fresh scan when the
match is weak or ambiguous (e.g. someone standing in front of the LiDAR).

Scores every free spot of the map (every --step m, every --yaw-step deg) with scan_match's
score (share of scan points landing on a wall), keeps the best few places, refines them, and
sends the best pose if it matches at least --min-score of the scan and beats the next distinct
place by --min-margin (ScanMatcher.global_search: laps_node uses the same search when the robot
gets lost). Run it with the laps / nav stack up (it needs /scan and TF base_link <- laser),
robot standing still.

--turn: when a first, still scan is not clear (a plain wall, or the same spot facing the other
way scoring as high), the robot turns once on the spot in 8 stops and the still scans, turned
back by the gyro, are matched as one all-round view: far more distinctive than one scan. Only
when 0.45 m is clear all round; the robot ends up facing about where it started.

--near x,y (e.g. where the last run in this room ended, or the room's start point 0,0, where
exploring began; several as x,y;x,y) searches the spots within --near-radius m of each
first. That takes seconds instead of minutes on a big map, and since the place is then known,
only --near-margin (smaller) is asked of the next distinct place: in a
hallway the same spot facing the other way always scores high (2026-11-22, Student Services:
97-100 % against 88-89 %, five scans in a row, so the 0.15 margin never passed). If the near
search is not convincing, the whole map is searched as before.
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
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan
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
        self.yaw = None                    # gyro heading (rad), for --turn
        self.create_subscription(LaserScan, '/scan', self.cb, 10)
        self.create_subscription(Imu, '/imu', self.imu_cb, 20)
        self.pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.cmd = self.create_publisher(Twist, '/nav_cmd_vel', 10)

    def imu_cb(self, msg):
        self.yaw = yaw_of(msg.orientation)

    def spin_for(self, seconds):
        end = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

    def fresh_points(self, timeout=3.0):
        self.points = None
        end = time.monotonic() + timeout
        while rclpy.ok() and self.points is None and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        return self.points

    def stop(self, seconds):
        end = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < end:
            self.cmd.publish(Twist())
            self.spin_for(0.05)

    def turn_scan(self, steps=8, step=math.radians(45), rate=0.6, clearance=0.45,
                  self_range=0.35):
        """Look all round by turning on the spot in `steps` stops: each stop a still scan,
        turned back by the gyro's angle into the frame of the first. Returns (points, turned)
        or (None, reason). Only if nothing is within `clearance` m (closer than `self_range`
        counts as the robot itself)."""
        pts = self.fresh_points()
        if pts is None or self.yaw is None:
            return None, 'no scan or no gyro'
        d = np.hypot(pts[:, 0], pts[:, 1])
        d = d[d >= self_range]
        if len(d) and d.min() < clearance:
            return None, f'something {d.min():.2f} m away: no room to turn'
        merged, turned = [], 0.0
        for k in range(steps):
            if k:
                start, last = self.yaw, self.yaw
                done, end = 0.0, time.monotonic() + 4.0
                msg = Twist()
                msg.angular.z = rate
                while abs(done) < step and time.monotonic() < end:
                    self.cmd.publish(msg)
                    self.spin_for(0.05)
                    done += math.atan2(math.sin(self.yaw - last), math.cos(self.yaw - last))
                    last = self.yaw
                self.stop(0.6)
                turned += math.atan2(math.sin(self.yaw - start), math.cos(self.yaw - start))
                if abs(done) < math.radians(10):
                    return None, 'the robot did not turn'
            pts = self.fresh_points()
            if pts is None:
                continue
            a = turned
            c, s = math.cos(a), math.sin(a)
            merged.append(np.column_stack([c * pts[:, 0] - s * pts[:, 1],
                                           s * pts[:, 0] + c * pts[:, 1]]))
        self.stop(0.3)
        if not merged:
            return None, 'no scans'
        return np.vstack(merged), turned

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
    ap.add_argument('--near', default='',
                    help='x,y[;x,y...]: search within --near-radius of these points first, '
                         'in this order')
    ap.add_argument('--near-radius', type=float, default=3.0)
    ap.add_argument('--near-margin', type=float, default=0.05,
                    help='margin over the next distinct place for a match near --near')
    ap.add_argument('--turn', action='store_true',
                    help='if the first scan is not clear: turn once on the spot in 8 stops '
                         'and match the all-round view (needs 0.45 m clear)')
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
    near_sets = []                     # [(label, spots)] searched before the whole map
    for item in [v for v in args.near.split(';') if v.strip()]:
        try:
            nx, ny = (float(v) for v in item.split(','))
        except ValueError:
            print(f'global_localize: --near {item!r} is not x,y; skipped', flush=True)
            continue
        d = np.hypot(free_xy[:, 0] - nx, free_xy[:, 1] - ny)
        spots = free_xy[d <= args.near_radius]
        if len(spots):
            near_sets.append((f'near {nx:.2f},{ny:.2f}', spots))
        else:
            print(f'global_localize: no free spot within {args.near_radius} m of {item}',
                  flush=True)

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

    turn_offset = 0.0                   # turned since the matched view (--turn)
    turn_tried = False
    for attempt in range(1, max(1, args.tries) + 1):
        node.points = None
        while rclpy.ok() and node.points is None:
            rclpy.spin_once(node, timeout_sec=0.2)
        pts = node.points
        turn_offset = 0.0
        if args.turn and attempt > 1 and not turn_tried:
            turn_tried = True
            print('global_localize: one scan was not enough: turning on the spot to look all '
                  'round', flush=True)
            view, info = node.turn_scan()
            if view is None:
                print(f'global_localize: no turn ({info}); going on with single scans',
                      flush=True)
            else:
                pts, turn_offset = view, info
                print(f'global_localize: all-round view from {math.degrees(info):.0f} deg of '
                      f'turning, {len(view)} points', flush=True)
        if len(pts) > 300:
            pts = pts[np.linspace(0, len(pts) - 1, 300).astype(int)]
        found = False
        searches = [(w, sp, args.near_margin) for w, sp in near_sets]
        for where, spots, margin in searches + [('everywhere', free_xy, args.min_margin)]:
            print(f'global_localize: attempt {attempt}/{args.tries}, {where}: {len(spots)} spots'
                  f' x {int(360 / args.yaw_step)} headings, {len(pts)} scan points', flush=True)
            # the search laps_node also uses when the robot is lost (ScanMatcher.global_search)
            refined = matcher.global_search(pts, spots, math.radians(args.yaw_step))
            for sc, (x, y, a) in refined[:5]:
                print(f'  score {sc:.0%}  x {x:6.2f}  y {y:6.2f}  '
                      f'yaw {math.degrees(a) % 360:6.1f} deg')
            best, (x, y, a) = refined[0]
            runner_up = next((sc for sc, p in refined[1:]
                              if math.hypot(p[0] - x, p[1] - y) > 0.5), 0.0)
            print(f'best {best:.0%}, next distinct place {runner_up:.0%} '
                  f'(needs {margin:.0%} more)', flush=True)
            if best >= args.min_score and best - runner_up >= margin:
                found = True
                break
        if found:
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
    a += turn_offset                    # the match is for the view's frame (before the turn)
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
