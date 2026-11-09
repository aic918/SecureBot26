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
        self.free = image > 250
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

    last_best = 0.0                # score of the pose correct() last found

    # ------------------------------------------------------------ finding the robot anywhere
    def spots(self, step=0.10, min_wall=0.25, around=None, radius=None):
        """World (x, y) of free map cells every `step` m, at least `min_wall` m from a wall,
        within `radius` m of `around` if given."""
        every = max(1, int(round(step / self.res)))
        rows, cols = np.nonzero(self.free[::every, ::every])
        rows, cols = rows * every, cols * every
        keep = self.dist[rows, cols] >= min_wall
        rows, cols = rows[keep], cols[keep]
        xy = np.column_stack([self.origin[0] + (cols + 0.5) * self.res,
                              self.origin[1] + (self.rows - rows - 0.5) * self.res])
        if around is not None and radius is not None:
            xy = xy[np.hypot(xy[:, 0] - around[0], xy[:, 1] - around[1]) <= radius]
        return xy

    def global_search(self, points, spots, yaw_step=math.radians(5.0), keep=10):
        """Best `keep` (score, (x, y, yaw)) over every spot and heading, refined, best first
        (vectorised: tools/global_localize.py until 2026-11-08)."""
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(pts) > 300:
            pts = pts[np.linspace(0, len(pts) - 1, 300).astype(int)]
        if not len(spots) or not len(pts):
            return []
        results = []
        for a in np.arange(0.0, 2 * math.pi, yaw_step):
            c, s = math.cos(a), math.sin(a)
            rx = c * pts[:, 0] - s * pts[:, 1]
            ry = s * pts[:, 0] + c * pts[:, 1]
            for chunk in np.array_split(spots, max(1, len(spots) // 2000)):
                wx = chunk[:, :1] + rx[None, :]
                wy = chunk[:, 1:] + ry[None, :]
                cols = ((wx - self.origin[0]) / self.res).astype(int)
                rows = (self.rows - (wy - self.origin[1]) / self.res).astype(int)
                inside = (rows >= 0) & (rows < self.rows) & (cols >= 0) & (cols < self.cols)
                d = np.full(wx.shape, np.inf)
                d[inside] = self.dist[rows[inside], cols[inside]]
                sc = np.mean(d < self.hit, axis=1)
                for k in np.argsort(sc)[-keep:]:
                    results.append((float(sc[k]), (float(chunk[k, 0]), float(chunk[k, 1]), a)))
        results.sort(key=lambda r: -r[0])
        # the best few *places* (0.5 m apart), so the runner-up is a real alternative and an
        # ambiguous match (a long plain corridor) shows as one
        chosen = []
        for sc, p in results:
            if all(math.hypot(p[0] - q[0], p[1] - q[1]) > 0.5 for _, q in chosen):
                chosen.append((sc, p))
                if len(chosen) == keep:
                    break
        refined = [self.search(p, pts, span=0.1, step=0.05, yaw_span=math.radians(5),
                               yaw_step=math.radians(2.5)) for _, p in chosen]
        return sorted(((float(sc), (float(x), float(y), math.atan2(math.sin(a), math.cos(a))))
                       for sc, (x, y, a) in refined), key=lambda r: -r[0])

    def relocalize(self, points, around=None, radius=None, step=0.10,
                   yaw_step=math.radians(5.0)):
        """(best score, best pose, best score of a place over 0.5 m from it) of a search over
        the map (within `radius` m of `around` if given); (0, None, 0) if nothing to search."""
        found = self.global_search(points, self.spots(step, around=around, radius=radius),
                                   yaw_step)
        if not found:
            return 0.0, None, 0.0
        best, pose = found[0]
        runner_up = next((sc for sc, p in found[1:]
                          if math.hypot(p[0] - pose[0], p[1] - pose[1]) > 0.5), 0.0)
        return best, pose, runner_up

    def correct(self, pose, points, min_score=0.75, min_gain=0.15, max_points=400,
                big_shift=0.20, big_turn=math.radians(5), big_score=0.85):
        """(current score, better pose or None). A better pose is returned only if it matches
        at least `min_score` of the scan and `min_gain` more than the current pose; a big
        correction (over big_shift m or big_turn) needs big_score (2026-11-07: 25-50 cm jumps
        in a long, plain hallway). The better pose's score is left in self.last_best."""
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(pts) > max_points:
            pts = pts[np.linspace(0, len(pts) - 1, max_points).astype(int)]
        now = self.score(pose, pts)
        if now >= min_score + min_gain:
            return now, None
        best, better = self.search(pose, pts)
        self.last_best = best
        shift = math.hypot(better[0] - pose[0], better[1] - pose[1])
        turn = abs(math.atan2(math.sin(better[2] - pose[2]), math.cos(better[2] - pose[2])))
        if (shift > big_shift or turn > big_turn) and best < big_score:
            return now, None
        if best >= min_score and best - now >= min_gain:
            return now, (float(better[0]), float(better[1]), math.atan2(math.sin(better[2]),
                                                                        math.cos(better[2])))
        return now, None
