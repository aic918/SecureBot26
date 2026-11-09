"""Temporary blocks for things on the floor the sensors cannot see (no ROS).

Whenever something stops the robot for good - an obstacle that stays in front of it, or a
bag lower than the LiDAR and under the ultrasonics' cone (the robot drives with a clear way
but does not move) - a square block of `size` m (orange in RViz) is put where it starts, and
the robot goes round it to the rest of its path. Blocks last for the run only
(shown until the next run starts): they never go into the saved map. They are
  - a grid matching the room map, for Nav2's global costmap (an extra static layer), so the
    planner routes round them,
  - points round their edges, added to the scan the follower checks, so it stops before them,
  - cubes in RViz.
"""
import math
from dataclasses import dataclass

import numpy as np

DEFAULT_SIZE = 0.2032       # m: side of a block (8 in)
EDGE_SPACING = 0.05         # m between edge points given to the follower


@dataclass
class Block:
    x: float
    y: float
    size: float


class Debris:
    def __init__(self, size=DEFAULT_SIZE, front=0.2475):
        self.size, self.front = size, front
        self.blocks = []

    def clear(self):
        self.blocks = []

    def add_ahead(self, x, y, yaw, ahead=None, side=0.0):
        """Block whose near side is `ahead` m in front of the robot centre (x, y facing yaw;
        default: touching the front edge) and `side` m to the left. Returns the new block, or
        None if one is already there."""
        d = (self.front if ahead is None else ahead) + self.size / 2
        c, s = math.cos(yaw), math.sin(yaw)
        bx, by = x + d * c - side * s, y + d * s + side * c
        for b in self.blocks:
            if math.hypot(b.x - bx, b.y - by) < 0.5 * (b.size + self.size) * 0.75:
                return None
        block = Block(bx, by, self.size)
        self.blocks.append(block)
        return block

    def remove_near(self, x, y, reach=0.15):
        """Remove the block under (x, y), or else the nearest within `reach` m of its edge (a
        click in RViz). Returns the removed block, or None."""
        best, best_d = None, reach
        for b in self.blocks:
            dx = max(0.0, abs(x - b.x) - b.size / 2)
            dy = max(0.0, abs(y - b.y) - b.size / 2)
            d = math.hypot(dx, dy)
            if d <= best_d:
                best, best_d = b, d
        if best is not None:
            self.blocks.remove(best)
        return best

    def distance(self, x, y):
        """m from (x, y) to the nearest block's edge (0 inside one; inf with none)."""
        best = math.inf
        for b in self.blocks:
            dx = max(0.0, abs(x - b.x) - b.size / 2)
            dy = max(0.0, abs(y - b.y) - b.size / 2)
            best = min(best, math.hypot(dx, dy))
        return best

    def points_in_base(self, pose, reach=3.0):
        """Edge points of blocks within `reach` m, in base_link [(x, y)] for pose (x, y, yaw)."""
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        out = []
        for b in self.blocks:
            if math.hypot(b.x - x, b.y - y) > reach + b.size:
                continue
            n = max(1, int(math.ceil(b.size / EDGE_SPACING)))
            h = b.size / 2
            ts = np.linspace(-h, h, n + 1)
            edge = [(t, -h) for t in ts] + [(t, h) for t in ts] + \
                [(-h, t) for t in ts[1:-1]] + [(h, t) for t in ts[1:-1]]
            for ex_, ey in edge:
                dx, dy = b.x + ex_ - x, b.y + ey - y
                out.append((c * dx + s * dy, -s * dx + c * dy))
        return out

    def grid(self, resolution, origin, width, height):
        """int8 [height, width] occupancy (row 0 = lowest y, as nav_msgs/OccupancyGrid):
        100 inside a block, 0 elsewhere (0 not -1, so the costmap layer never marks unknown)."""
        g = np.zeros((height, width), dtype=np.int8)
        for b in self.blocks:
            h = b.size / 2
            c0 = max(0, int(math.floor((b.x - h - origin[0]) / resolution)))
            c1 = min(width, int(math.ceil((b.x + h - origin[0]) / resolution)))
            r0 = max(0, int(math.floor((b.y - h - origin[1]) / resolution)))
            r1 = min(height, int(math.ceil((b.y + h - origin[1]) / resolution)))
            g[r0:r1, c0:c1] = 100
        return g


SIDESTEP_GAP = 0.10         # m: gap between the robot's side and a block when stepping round it
TURN_RADIUS = 0.356         # m: what a turn on the spot sweeps (centre to a case corner)
REJOIN_GAP = 0.12           # m: past a block's edge, beyond the robot's half length, to rejoin
#                             (half_length 0.2475 + 0.12 > TURN_RADIUS: turns there clear it)
START_FREE = 0.15           # m from the robot's spot not checked (it is already there)


def sidestep(start, path, near, k, block, half_width, half_length, clear_at,
             gap=SIDESTEP_GAP, step=0.05):
    """A tight way round `block` (2026-11-08: Nav2's way round an 8 in box swung up to 1.8 m
    off the lane): from `start` straight to a point beside the block, past it parallel to the
    lane, then back onto the lane at path[k] (resume_index). `gap` m between the robot's side
    and the block. Tries the left, then the right; each must keep the robot clear everywhere
    (clear_at(x, y) -> m from the robot centre to the nearest wall, other block or scan point;
    half_width + gap on the way, TURN_RADIUS + 0.02 where it turns on the spot).
    Returns [start, beside_before, beside_after] + path[k:], or None (then ask Nav2)."""
    if k is None or k >= len(path):
        return None
    pts = np.asarray(path, float)
    b = np.array([block.x, block.y])
    kb = int(np.argmin(np.hypot(*(pts[near:k + 1] - b).T))) + near
    a0, a1 = pts[max(near, kb - 5)], pts[min(len(pts) - 1, max(kb + 5, kb + 1))]
    u = a1 - a0
    if np.hypot(*u) < 1e-6:
        return None
    u = u / np.hypot(*u)
    n = np.array([-u[1], u[0]])
    h = block.size / 2
    s = h + half_width + gap                       # lane to robot centre, beside the block
    reach = h + half_length + gap                  # along the lane, before / after the block
    best = None
    for side in (1.0, -1.0):
        before = b - reach * u + side * s * n
        after = b + reach * u + side * s * n
        way = [np.asarray(start, float), before, after, pts[k]]
        ok = True
        for p, q in zip(way, way[1:]):
            m_ = max(2, int(np.hypot(*(q - p)) / step) + 1)
            for t in np.linspace(0.0, 1.0, m_):
                x, y = p + (q - p) * t
                if np.hypot(x - way[0][0], y - way[0][1]) < START_FREE:
                    continue
                if clear_at(x, y) < half_width + gap / 2:
                    ok = False
                    break
            if not ok:
                break
        if ok and all(clear_at(*c) >= TURN_RADIUS + 0.02 for c in way[1:3]) \
                and clear_at(*way[3]) >= TURN_RADIUS:
            room = min(clear_at(*before), clear_at(*after))
            if best is None or room > best[0]:
                best = (room, [tuple(map(float, c)) for c in way[:3]])
    if best is None:
        return None
    return best[1] + [tuple(map(float, q)) for q in path[k:]]


def resume_index(path, start, debris, clearance):
    """First index from `start` on whose point is at least `clearance` m from every block,
    after the path has come near one; None if the rest of the path stays near blocks."""
    near = False
    for k in range(start, len(path)):
        d = debris.distance(*path[k])
        if d < clearance:
            near = True
        elif near:
            return k
    return None if near else start
