"""Check and correct the robot's map pose by matching the live LiDAR scan to the saved map (no
ROS).

AMCL drifts while SecureBot drives (scan-matching odometry, a busy CPU, a room with repeated
desk rows). When the robot stands still, `correct()` scores the scan against the map around
the current pose (share of scan points within `hit` of a wall) and returns a better pose if
there is a clearly better one, for the laps node to send to AMCL as /initialpose.
"""
import math

import cv2
import numpy as np


class ScanMatcher:
    def __init__(self, image, resolution, origin, hit=0.10):
        """image: the map image (0 = wall, as map_saver writes it), row 0 = top."""
        walls = (image < 100).astype(np.uint8)
        self.dist = cv2.distanceTransform(1 - walls, cv2.DIST_L2, 5) * resolution
        self.res, self.origin, self.hit = resolution, origin, hit
        self.rows, self.cols = self.dist.shape

    def score(self, pose, points):
        """Share (0..1) of `points` (Nx2, robot frame) that land on a wall from `pose`."""
        x, y, yaw = pose
        if len(points) == 0:
            return 0.0
        c, s = math.cos(yaw), math.sin(yaw)
        wx = x + c * points[:, 0] - s * points[:, 1]
        wy = y + s * points[:, 0] + c * points[:, 1]
        cols = ((wx - self.origin[0]) / self.res).astype(int)
        rows = (self.rows - (wy - self.origin[1]) / self.res).astype(int)
        inside = (rows >= 0) & (rows < self.rows) & (cols >= 0) & (cols < self.cols)
        d = np.full(len(wx), np.inf)
        d[inside] = self.dist[rows[inside], cols[inside]]
        return float(np.mean(d < self.hit))

    def search(self, pose, points, span=0.4, step=0.1, yaw_span=math.radians(20),
               yaw_step=math.radians(5)):
        """Best (score, pose) on a grid around `pose`, then refined finer."""
        x0, y0, a0 = pose
        offs = np.arange(-span, span + 1e-9, step)
        yaws = np.arange(-yaw_span, yaw_span + 1e-9, yaw_step)
        best = max((self.score((x0 + dx, y0 + dy, a0 + da), points), (x0 + dx, y0 + dy, a0 + da))
                   for dx in offs for dy in offs for da in yaws)
        for _ in range(2):
            sc, (x, y, a) = best
            best = max((self.score((x + dx, y + dy, a + da), points), (x + dx, y + dy, a + da))
                       for dx in (-0.04, -0.02, 0, 0.02, 0.04)
                       for dy in (-0.04, -0.02, 0, 0.02, 0.04)
                       for da in np.radians((-2, -1, 0, 1, 2)))
        return best

    def correct(self, pose, points, min_score=0.75, min_gain=0.15, max_points=400):
        """(current score, better pose or None). A better pose is returned only if it matches
        at least `min_score` of the scan and `min_gain` more than the current pose."""
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(pts) > max_points:
            pts = pts[np.linspace(0, len(pts) - 1, max_points).astype(int)]
        now = self.score(pose, pts)
        if now >= min_score + min_gain:
            return now, None
        best, better = self.search(pose, pts)
        if best >= min_score and best - now >= min_gain:
            return now, (float(better[0]), float(better[1]), math.atan2(math.sin(better[2]),
                                                                        math.cos(better[2])))
        return now, None
