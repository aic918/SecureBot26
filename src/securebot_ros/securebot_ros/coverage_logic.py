"""Coverage laps on a saved map (no ROS): plan loops that sweep the whole scanned floor.

Two patterns (Params.pattern): 'contour' (below) and 'lanes', straight parallel lanes with a
direction-dependent wall offset (lane_logic).

Contour-parallel coverage, the "laps" pattern used by cleaning and patrol robots (and one of
the patterns in coverage planners such as Fields2Cover / opennav_coverage):

  1. free space = the map's free cells (small unseen specks inside the room filled in);
     unknown counts as blocked, so the robot only goes where the LiDAR has seen floor
  2. distance transform: how far every free cell is from the nearest wall / obstacle
  3. lap k runs along the line where that distance = wall_clearance + k * lane_spacing, so the
     first lap hugs the walls (and goes round furniture) and each next lap is one lane further
     in, until the middle of the room is reached
  4. laps are ordered outside-in, each entered at the point closest to where the robot is,
     and simplified to waypoints for Nav2

wall_clearance keeps the whole body off the walls, also while it turns (the footprint's
corner radius, 495 x 512 mm with the wheels, is 0.356 m) and leaves the front bumper 1 ft from
a wall the robot faces at a corner: closer, the firmware and the collision monitor stop it.
lane_spacing is a bit less than the robot's width so neighbouring laps overlap and leave no
stripes.
"""
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import yaml

LENGTH = 0.495                     # m, case front to back
WIDTH = 0.512                      # m, across the drive wheels (the case alone is 0.360)
CORNER_RADIUS = math.hypot(LENGTH / 2, WIDTH / 2)   # 0.356 m: what a turn on the spot sweeps


@dataclass
class Params:
    wall_clearance: float = 0.65   # m centre to walls: front 1 ft + 10 cm off a wall it faces
    lane_spacing: float = 0.45     # m between laps (robot width 0.512 - ~6 cm overlap)
    waypoint_spacing: float = 1.5  # m: longest gap between waypoints on a lap
    simplify: float = 0.10         # m: straighten wiggles smaller than this (fewer waypoints)
    min_loop: float = 1.0          # m: loops shorter than this are skipped
    fill_holes: float = 0.15       # m: unseen gaps narrower than this inside count as free
    pattern: str = 'contour'       # 'contour' laps or straight 'lanes'
    side_clearance: float = 0.50   # lanes: centre to anything beside it (turn on the spot)
    front_clearance: float = 0.60  # lanes: centre to anything ahead at a lane end (1 ft stop)
    lane_angle: float = None       # lanes: radians, None = the angle with the fewest lanes
    min_lane: float = 1.0          # lanes: shorter ones cost two turns for little floor: dropped


@dataclass
class Map:
    free: np.ndarray               # bool [rows, cols], row 0 = top of the image (max y)
    resolution: float
    origin: tuple                  # (x, y) of the bottom-left pixel corner
    wall: np.ndarray = None        # bool: cells the LiDAR saw as occupied (None = all not free)

    def to_world(self, row, col):
        return (self.origin[0] + (col + 0.5) * self.resolution,
                self.origin[1] + (self.free.shape[0] - row - 0.5) * self.resolution)

    def to_cell(self, x, y):
        return (int(self.free.shape[0] - (y - self.origin[1]) / self.resolution),
                int((x - self.origin[0]) / self.resolution))


@dataclass
class Plan:
    loops: list                    # [[(x, y), ...]] closed (last point == first), in drive order
    levels: list                   # lap level of each loop (0 = along the walls)
    coverage: float = 0.0          # share of the reachable floor the body sweeps
    length: float = 0.0            # m, all loops, without the moves between them
    area: float = 0.0              # m², reachable floor
    swept: np.ndarray = field(default=None, repr=False)     # bool mask of swept floor
    angle: float = None            # lanes: the lane angle used (radians)


def load_map(yaml_path):
    """Map from a map_saver .yaml + image. Free = lighter than free_thresh, and in trinary maps
    not the 205 gray that marks unknown."""
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    image = meta['image']
    if not os.path.isabs(image):
        image = os.path.join(os.path.dirname(yaml_path), image)
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise OSError(f'cannot read map image {image}')
    p = img.astype(np.float32) / 255.0
    occ = p if meta.get('negate', 0) else 1.0 - p
    free = occ < float(meta.get('free_thresh', 0.25))
    if meta.get('mode', 'trinary') == 'trinary':
        free &= img != 205
    wall = occ > float(meta.get('occupied_thresh', 0.65))
    origin = meta.get('origin', [0.0, 0.0, 0.0])
    return Map(free, float(meta['resolution']), (float(origin[0]), float(origin[1])), wall)


def _disk(radius_px):
    r = max(int(round(radius_px)), 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def floor_mask(m, fill_holes):
    """Free cells, with small unseen gaps that free space surrounds filled in. Only unseen
    cells are filled: anything the LiDAR saw as occupied (a desk or chair leg is thinner than
    `fill_holes`) stays an obstacle."""
    free = m.free.astype(np.uint8)
    closed = cv2.morphologyEx(free, cv2.MORPH_CLOSE, _disk(fill_holes / m.resolution / 2))
    wall = m.wall if m.wall is not None else ~m.free
    return closed.astype(bool) & ~wall


def reachable(m, p, start=None):
    """(floor the robot can reach, distance-to-obstacle in m). Reachable = the connected part
    of the space where the robot centre fits (>= wall_clearance from everything) that holds
    `start`, or the largest one; the floor is that space grown by half the robot's width,
    i.e. every cell the body can pass over (the strip right along the walls it cannot reach
    without scraping them is left out)."""
    floor = floor_mask(m, p.fill_holes)
    padded = np.pad(floor.astype(np.uint8), 1)               # the image edge is a wall too
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1] * m.resolution
    cspace = (dist >= p.wall_clearance).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cspace, connectivity=8)
    if n <= 1:
        return np.zeros_like(floor), dist
    label = 0
    if start is not None:
        r, c = m.to_cell(*start)
        if 0 <= r < labels.shape[0] and 0 <= c < labels.shape[1]:
            label = labels[r, c]
    if label == 0:
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    centre_ok = labels == label
    body = cv2.dilate(centre_ok.astype(np.uint8), _disk(WIDTH / 2 / m.resolution))
    return (body.astype(bool) & floor), np.where(centre_ok, dist, 0.0)


def _signed_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _resample(pts, step):
    """Closed polyline with extra points so no segment is longer than step."""
    out = []
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        out.extend(a + (b - a) * (i / n) for i in range(n))
    return np.array(out)


def _loop_length(pts):
    return float(np.sum(np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)))


def lap_loops(m, p, start=None):
    """[(level, closed polyline in world m as an Nx2 array)], every loop of every lap level.
    Outer edges run counter-clockwise and edges round furniture clockwise, so the nearest wall
    is always on the robot's right."""
    _, centre_dist = reachable(m, p, start)
    loops, level = [], 0
    while True:
        d = p.wall_clearance + level * p.lane_spacing
        region = (centre_dist >= d).astype(np.uint8)
        if not region.any():
            break
        contours, hierarchy = cv2.findContours(region, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for i, cnt in enumerate(contours):
            cnt = cv2.approxPolyDP(cnt, p.simplify / m.resolution, True).reshape(-1, 2)
            if len(cnt) < 3:
                continue
            pts = np.array([m.to_world(r, c) for c, r in cnt])
            if _loop_length(pts) < p.min_loop:
                continue
            is_hole = hierarchy[0][i][3] >= 0
            ccw = _signed_area(pts) > 0
            if ccw == is_hole:
                pts = pts[::-1]
            loops.append((level, _resample(pts, p.waypoint_spacing)))
        level += 1
    return loops


def order_loops(loops, start):
    """Drive order: outer laps first; within a lap level the nearest loop next, each rotated to
    begin at the point nearest the robot and closed back on itself."""
    here = np.array(start if start is not None else loops[0][1][0] if loops else (0.0, 0.0))
    out, levels = [], []
    for level in sorted({lv for lv, _ in loops}):
        todo = [pts for lv, pts in loops if lv == level]
        while todo:
            gaps = [np.min(np.linalg.norm(pts - here, axis=1)) for pts in todo]
            pts = todo.pop(int(np.argmin(gaps)))
            k = int(np.argmin(np.linalg.norm(pts - here, axis=1)))
            pts = np.vstack([np.roll(pts, -k, axis=0), pts[k:k + 1]])
            out.append([(float(x), float(y)) for x, y in pts])
            levels.append(level)
            here = pts[-1]
    return out, levels


def swept_floor(m, loops, width=WIDTH, closed=True):
    """Cells the body passes over while driving the loops (open paths with closed=False)."""
    mask = np.zeros(m.free.shape, np.uint8)
    thick = max(1, int(round(width / m.resolution)))
    for loop in loops:
        cells = np.array([m.to_cell(x, y)[::-1] for x, y in loop], np.int32)   # (col, row)
        cv2.polylines(mask, [cells], closed, 1, thickness=thick)
    return mask.astype(bool)


def lane_plan(m, p, start=None):
    """Straight-lane plan: one path per lane (the way there + the lane), in drive order.
    Without p.lane_angle, plans angles around the room's main wall direction and the one
    across it, and keeps the cheapest (estimated time + unswept floor)."""
    from securebot_ros import lane_logic as ll
    space, _, floor = ll.centre_space(m, p, start)
    body = cv2.dilate(space.astype(np.uint8), _disk(WIDTH / 2 / m.resolution)).astype(bool)
    floor = body & floor
    if p.lane_angle is not None:
        angles = [p.lane_angle]
    else:
        # the result swings with a degree or two (where lanes clip desk corners): try every
        # degree within 3 of the walls and of the direction across them
        wall = ll.wall_angle(m)
        angles = [(wall + base + math.radians(d)) % math.pi
                  for base in (0.0, math.pi / 2) for d in range(-3, 4)]
    best = None
    for angle in angles:
        lane_list, _ = ll.lanes(m, p, start, angle)
        paths = ll.chain(m, p, lane_list, start) if lane_list else []
        swept = swept_floor(m, paths, closed=False) & floor
        route = [pt for path in paths for pt in path]
        turns, degrees = ll.turn_count(route) if route else (0, 0.0)
        length = sum(math.dist(a, b) for a, b in zip(route, route[1:]))
        uncovered = float((floor & ~swept).sum()) * m.resolution ** 2
        cost = (length / ll.DRIVE_MPS + turns * ll.LANE_END_S / 2
                + ll.UNCOVERED_S_PER_M2 * uncovered)
        if best is None or cost < best[0]:
            best = (cost, angle, paths, swept, length)
    _, angle, paths, swept, length = best
    return Plan(loops=paths, levels=[0] * len(paths), swept=swept,
                area=float(floor.sum()) * m.resolution ** 2,
                coverage=float(swept.sum()) / max(int(floor.sum()), 1),
                length=length, angle=angle)


def plan(m, p=None, start=None):
    """Coverage plan from `start` (x, y in the map frame; None = from the first loop)."""
    p = p or Params()
    if p.pattern == 'lanes':
        return lane_plan(m, p, start)
    loops, levels = order_loops(lap_loops(m, p, start), start)
    floor, _ = reachable(m, p, start)
    swept = swept_floor(m, loops) & floor
    area = float(floor.sum()) * m.resolution ** 2
    return Plan(loops=loops, levels=levels, swept=swept, area=area,
                coverage=float(swept.sum()) / max(int(floor.sum()), 1),
                length=sum(_loop_length(np.array(lp[:-1])) for lp in loops))


def headings(loop):
    """Yaw at every waypoint: towards the next one (the last copies the one before)."""
    yaws = [math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(loop, loop[1:])]
    return yaws + yaws[-1:]


def legs(loop, max_poses):
    """Split a loop into consecutive legs of at most max_poses ((x, y), yaw) waypoints; each
    leg starts where the one before ended, so together they drive the whole loop."""
    poses = list(zip(loop, headings(loop)))
    step = max(1, max_poses - 1)
    return [poses[i:i + max_poses] for i in range(0, max(len(poses) - 1, 1), step)]


class LapRunner:
    """Which loop to drive next. start() plans from the robot's position, then next_loop() /
    finished() walk the loops; after the last loop of a lap round it plans again from where
    the robot is and starts the next round, until `rounds` are done (0 = keep going)."""

    def __init__(self, m, params=None, rounds=0, max_failures=2):
        self.m, self.p = m, params or Params()
        self.rounds, self.max_failures = rounds, max_failures
        self.plan = None
        self.index = 0                 # loop being driven
        self.round = 0                 # completed rounds
        self.failures = 0              # in a row, on the current loop
        self.skipped = 0
        self.active = False

    def start(self, pose):
        self.round, self.skipped, self.active = 0, 0, True
        self._replan(pose)
        if not self.plan.loops:
            self.active = False
        return self.active

    def stop(self):
        self.active = False

    def _replan(self, pose):
        self.plan = plan(self.m, self.p, pose)
        self.index, self.failures = 0, 0

    def current(self):
        return self.plan.loops[self.index] if self.active else None

    def finished(self, ok, pose):
        """Report the current loop's result; returns the next loop or None when all are done.
        A failed loop is tried again (max_failures in a row), then skipped."""
        if not self.active:
            return None
        if not ok:
            self.failures += 1
            if self.failures < self.max_failures:
                return self.current()
            self.skipped += 1
        self.index, self.failures = self.index + 1, 0
        if self.index >= len(self.plan.loops):
            self.round += 1
            if self.rounds and self.round >= self.rounds:
                self.active = False
                return None
            self._replan(pose)
        return self.current()
