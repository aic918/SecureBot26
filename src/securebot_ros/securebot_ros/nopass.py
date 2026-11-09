"""No-pass lines the user draws on a room's map (no ROS).

A no-pass line is a thick line (`width` m, black in RViz) the robot treats as a wall: the lane
planner keeps off it and does not cover what lies only beyond it, Nav2 routes round it, and
the lap follower stops before it. Lines are drawn with clicks in RViz (laps node) and belong
to the room: saved next to its map as <map stem>.nopass.json, so they last across runs.
"""
import json
import math
import os
from dataclasses import replace

import cv2
import numpy as np

DEFAULT_WIDTH = 0.10        # m: line thickness (a wall this thick)
EDGE_SPACING = 0.05         # m between points given to the follower


def path_for(map_yaml):
    """Where the lines of the room whose map is `map_yaml` are kept."""
    return os.path.splitext(map_yaml)[0] + '.nopass.json'


class NoPass:
    def __init__(self, path=None, width=DEFAULT_WIDTH):
        self.path, self.width = path, width
        self.lines = []               # [((x1, y1), (x2, y2))]
        if path and os.path.exists(path):
            self.load()

    def load(self):
        try:
            with open(self.path) as f:
                data = json.load(f)
            self.width = float(data.get('width', self.width))
            self.lines = [(tuple(map(float, a)), tuple(map(float, b)))
                          for a, b in data.get('lines', [])]
        except (OSError, ValueError, TypeError):
            self.lines = []

    def save(self):
        if not self.path:
            return
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'width': self.width, 'lines': [[list(a), list(b)] for a, b in self.lines]},
                      f, indent=1)
        os.replace(tmp, self.path)

    def add(self, a, b):
        """Add the line a-b (ignored if shorter than its width). Returns True if added."""
        if math.dist(a, b) < self.width:
            return False
        self.lines.append((tuple(a), tuple(b)))
        self.save()
        return True

    def undo(self):
        if self.lines:
            self.lines.pop()
            self.save()

    def clear(self):
        self.lines = []
        self.save()

    def remove_near(self, x, y, reach=0.15):
        """Remove the line nearest (x, y) if it passes within `reach` m. Returns it or None."""
        best, best_d = None, reach + self.width / 2
        for line in self.lines:
            d = _seg_dist((x, y), *line)
            if d <= best_d:
                best, best_d = line, d
        if best is not None:
            self.lines.remove(best)
            self.save()
        return best

    def mask(self, m):
        """bool mask over map `m` (row 0 = top, as coverage_logic.Map): cells under a line."""
        out = np.zeros(m.free.shape, dtype=np.uint8)
        px = max(1, int(round(self.width / m.resolution)))
        for a, b in self.lines:
            ra, ca = m.to_cell(*a)
            rb, cb = m.to_cell(*b)
            cv2.line(out, (ca, ra), (cb, rb), 1, thickness=px)
        return out.astype(bool)

    def apply(self, m):
        """Copy of map `m` with the lines as walls."""
        if not self.lines:
            return m
        k = self.mask(m)
        wall = (m.wall if m.wall is not None else ~m.free) | k
        return replace(m, free=m.free & ~k, wall=wall)

    def points_in_base(self, pose, reach=3.0):
        """Points along the lines' edges within `reach` m, in base_link, for pose (x, y, yaw)."""
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        out = []
        h = self.width / 2
        for a, b in self.lines:
            if _seg_dist((x, y), a, b) > reach:
                continue
            length = math.dist(a, b)
            ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
            n = max(1, int(math.ceil(length / EDGE_SPACING)))
            for k in range(n + 1):
                t = length * k / n
                for side in (-h, h):
                    wx = a[0] + ux * t - uy * side
                    wy = a[1] + uy * t + ux * side
                    if math.hypot(wx - x, wy - y) <= reach:
                        dx, dy = wx - x, wy - y
                        out.append((c * dx + s * dy, -s * dx + c * dy))
        return out


def _seg_dist(p, a, b):
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 < 1e-12 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / l2))
    return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)
