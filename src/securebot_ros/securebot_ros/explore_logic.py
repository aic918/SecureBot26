"""Pure logic for room exploration (no ROS): map geometry, room coverage, choosing a heading.

The map is slam_toolbox's occupancy grid: -1 unknown, 0 free, 100 occupied (values in
between are treated by threshold). Cells are indexed [row, col] = [y, x].
"""
import math
from collections import deque

import numpy as np

UNKNOWN, FREE, OCCUPIED = -1, 0, 100
FREE_MAX = 25           # <= this is free
OCC_MIN = 65            # >= this is occupied


class Grid:
    def __init__(self, data, width, height, resolution, origin_x, origin_y):
        self.cells = np.asarray(data, dtype=np.int16).reshape(height, width)
        self.width, self.height, self.res = width, height, resolution
        self.ox, self.oy = origin_x, origin_y
        self.free = (self.cells >= 0) & (self.cells <= FREE_MAX)
        self.occ = self.cells >= OCC_MIN
        self.unknown = self.cells < 0

    def to_cell(self, x, y):
        return int(math.floor((y - self.oy) / self.res)), int(math.floor((x - self.ox) / self.res))

    def to_world(self, row, col):
        return self.ox + (col + 0.5) * self.res, self.oy + (row + 0.5) * self.res

    def inside(self, row, col):
        return 0 <= row < self.height and 0 <= col < self.width

    def state(self, x, y):
        """'free' | 'occupied' | 'unknown' at a world point (outside the grid = unknown)."""
        r, c = self.to_cell(x, y)
        if not self.inside(r, c):
            return 'unknown'
        if self.occ[r, c]:
            return 'occupied'
        return 'free' if self.free[r, c] else 'unknown'


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = NEIGHBORS4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))


def reachable_free(grid, x, y, search_radius=0.5):
    """Boolean mask of free cells connected to the robot (4-connected flood fill). The robot's
    own cell can be unknown (the LiDAR's blind spot), so start from the nearest free cell."""
    mask = np.zeros_like(grid.free)
    r0, c0 = grid.to_cell(x, y)
    reach = int(search_radius / grid.res)
    start = None
    for d in range(reach + 1):
        for dr in range(-d, d + 1):
            for dc in range(-d, d + 1):
                r, c = r0 + dr, c0 + dc
                if max(abs(dr), abs(dc)) == d and grid.inside(r, c) and grid.free[r, c]:
                    start = (r, c)
                    break
            if start:
                break
        if start:
            break
    if start is None:
        return mask
    queue = deque([start])
    mask[start] = True
    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if grid.inside(nr, nc) and grid.free[nr, nc] and not mask[nr, nc]:
                mask[nr, nc] = True
                queue.append((nr, nc))
    return mask


def _touches(mask, rows, cols, other):
    """For each (row, col), whether any 4-neighbour is True in `other`."""
    hits = np.zeros(len(rows), dtype=bool)
    h, w = mask.shape
    for dr, dc in NEIGHBORS4:
        nr, nc = rows + dr, cols + dc
        ok = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
        hits[ok] |= other[nr[ok], nc[ok]]
    return hits


def frontier_clusters(grid, reachable, min_cells=6):
    """Frontier = reachable free cell next to unknown. Returns a list of clusters (lists of
    (row, col)), 8-connected, ignoring clusters smaller than min_cells (scan speckle: one
    unseen cell leaves a 4-cell ring)."""
    rows, cols = np.nonzero(reachable)
    is_frontier = _touches(reachable, rows, cols, grid.unknown)
    cells = set(zip(rows[is_frontier].tolist(), cols[is_frontier].tolist()))
    clusters = []
    while cells:
        seed = cells.pop()
        cluster, queue = [seed], deque([seed])
        while queue:
            r, c = queue.popleft()
            for dr, dc in NEIGHBORS8:
                n = (r + dr, c + dc)
                if n in cells:
                    cells.remove(n)
                    cluster.append(n)
                    queue.append(n)
        if len(cluster) >= min_cells:
            clusters.append(cluster)
    return clusters


def coverage_and_frontiers(grid, reachable, min_frontier_cells=6):
    """How enclosed the explored space is: wall cells / (wall cells + frontier cells) along the
    edge of the free space the robot can reach (1.0 = every edge is a wall: room fully seen),
    and the frontier clusters themselves."""
    occ_rows, occ_cols = np.nonzero(grid.occ)
    walls = int(_touches(grid.occ, occ_rows, occ_cols, reachable).sum())
    clusters = frontier_clusters(grid, reachable, min_frontier_cells)
    frontier = sum(len(c) for c in clusters)
    if walls + frontier == 0:
        return 0.0, clusters
    return walls / float(walls + frontier), clusters


def coverage(grid, reachable, min_frontier_cells=6):
    return coverage_and_frontiers(grid, reachable, min_frontier_cells)[0]


def free_distance(grid, x, y, heading, half_width, max_dist, start=0.25, step=None):
    """How far the robot (half_width each side of its centre line) can drive along `heading`
    through known-free cells, measured from its centre. The check starts `start` ahead (the
    front edge): the robot's own cells are never seen by the LiDAR (20 cm minimum range).
    Returns (distance, ends_in) with ends_in 'occupied', 'unknown' or 'max'."""
    step = step or grid.res / 2.0
    ch, sh = math.cos(heading), math.sin(heading)
    offsets = (-half_width, 0.0, half_width)
    d = start
    while d <= max_dist:
        for o in offsets:
            px, py = x + ch * d - sh * o, y + sh * d + ch * o
            st = grid.state(px, py)
            if st != 'free':
                return max(0.0, d - step), st
        d += step
    return max_dist, 'max'


def rotation_clear(grid, x, y, radius):
    """True if no occupied cell lies within `radius` of (x, y): the robot can spin in place."""
    r0, c0 = grid.to_cell(x, y)
    n = int(math.ceil(radius / grid.res))
    rr = int(radius / grid.res)
    for dr in range(-n, n + 1):
        for dc in range(-n, n + 1):
            if dr * dr + dc * dc <= rr * rr and grid.inside(r0 + dr, c0 + dc) \
                    and grid.occ[r0 + dr, c0 + dc]:
                return False
    return True


class VisitMap:
    """Coarse record of where the robot has been, to prefer new places."""

    def __init__(self, cell=0.5):
        self.cell, self.counts = cell, {}

    def key(self, x, y):
        return int(math.floor(x / self.cell)), int(math.floor(y / self.cell))

    def add(self, x, y):
        k = self.key(x, y)
        self.counts[k] = self.counts.get(k, 0) + 1

    def count(self, x, y):
        return self.counts.get(self.key(x, y), 0)

    def along(self, x, y, heading, length, half_width, start=0.3):
        """Visits in the distinct cells the robot's width sweeps over on a leg (from `start`
        ahead, so the cell it stands in does not count)."""
        ch, sh = math.cos(heading), math.sin(heading)
        keys, d = set(), start
        while d <= length:
            for o in (-half_width, 0.0, half_width):
                keys.add(self.key(x + ch * d - sh * o, y + sh * d + ch * o))
            d += self.cell / 2.0
        return sum(self.counts.get(k, 0) for k in keys)


def angle_diff(a, b):
    """Signed smallest difference a - b, in (-pi, pi]."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


def ray(grid, x, y, heading, half_width, max_dist, start=0.25, step=None):
    """Like free_distance, but looks through unexplored cells: the robot may drive into space
    the map has not seen yet (the ultrasonics and LiDAR guard it there). Returns
    (distance to the first occupied cell, or max_dist; distance to the first unknown cell, or
    None), both from the robot's centre."""
    step = step or grid.res / 2.0
    ch, sh = math.cos(heading), math.sin(heading)
    offsets = (-half_width, 0.0, half_width)
    first_unknown, d = None, start
    while d <= max_dist:
        for o in offsets:
            st = grid.state(x + ch * d - sh * o, y + sh * d + ch * o)
            if st == 'occupied':
                return max(0.0, d - step), first_unknown
            if st == 'unknown' and first_unknown is None:
                first_unknown = d
        d += step
    return max_dist, first_unknown


def scan_clearance(points, heading, half_width):
    """Distance from the robot's centre along `heading` (robot frame, rad) to the nearest LiDAR
    point inside the robot's width; inf if none. points: (N, 2) array in base_link."""
    if len(points) == 0:
        return math.inf
    c, s = math.cos(heading), math.sin(heading)
    u = points[:, 0] * c + points[:, 1] * s
    v = -points[:, 0] * s + points[:, 1] * c
    ahead = (u > 0.0) & (np.abs(v) <= half_width)
    return float(u[ahead].min()) if ahead.any() else math.inf
