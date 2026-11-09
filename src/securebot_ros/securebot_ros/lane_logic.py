"""Straight-lane coverage on a saved map (no ROS): the "lawnmower" / boustrophedon pattern.

Parallel straight lanes cross the room, one robot-width apart; the robot drives a lane, turns,
hops sideways to the next one and drives it back. Fewer and gentler turns than the contour
laps, which matters for SecureBot: every turn on the spot costs ~0.6 s before the wheels move,
spins at ~40 deg/s and is where localization drifts most.

The wall offset depends on direction, because the robot needs different room sideways and
ahead:
  side_clearance  (0.50 m)  centre to anything beside it: enough to turn on the spot anywhere
                            (corner radius 0.356 m, plus 10 cm over the follower's 0.40 m
                            turning check). Lanes and the hops between them stay in this space.
  front_clearance (0.60 m)  centre to anything straight ahead where a lane ends: half the
                            robot's length 0.2475 + the 1 ft stop 0.305 + 5 cm, so the robot
                            stops at a lane end before the 1 ft rule stops it. Lanes are
                            trimmed back until they have this much free floor ahead.

Lanes are `check_width` (1.1 m) apart: the LiDAR sees far past the robot's sides, so a lane
checks more floor than the robot's own width and the lanes need not overlap.

Pockets (fill lanes): wider lanes can miss smaller bits of open floor (between desks, round a
corner). Every pocket of turning space no lane passes within check_width / 2 gets a lane
along its long side, so narrower routes are still checked.

Narrow aisles (pass lanes): an aisle too tight to turn in (under 2 x side_clearance) but wide
enough to drive straight through (centre >= pass_clearance from both sides) gets one straight
lane down its middle, from open space to open space, so the robot turns only where it has room.
Dead-end aisles are left out: the robot could not turn round in them.

Steps: the space where the robot centre fits (side_clearance) is cut into lanes at the angle
that needs the fewest lanes (or `lane_angle`); lanes split around furniture; each lane end is
trimmed for front_clearance; lanes are chained greedily (nearest lane end next), joined
directly when the straight hop is clear, else by the shortest way through the free space.
"""
import heapq
import math

import cv2
import numpy as np


def centre_space(m, p, start=None):
    """(bool mask of where the robot centre may be, distance-to-obstacle in m) for the
    side clearance: the connected part holding `start`, or the largest."""
    from securebot_ros import coverage_logic as cl          # avoid a circular import
    floor = cl.floor_mask(m, p.fill_holes)
    padded = np.pad(floor.astype(np.uint8), 1)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1] * m.resolution
    ok = (dist >= p.side_clearance).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ok, connectivity=8)
    if n <= 1:
        return np.zeros(ok.shape, bool), dist, floor
    label = 0
    if start is not None:
        r, c = m.to_cell(*start)
        if 0 <= r < labels.shape[0] and 0 <= c < labels.shape[1]:
            label = labels[r, c]
    if label == 0:
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    space = labels == label
    if getattr(p, 'pass_lanes', False):
        # open space beyond a narrow passage the robot can drive through counts too
        reach = (dist >= p.pass_clearance - p.pass_slack).astype(np.uint8)
        _, plabels = cv2.connectedComponents(reach, connectivity=8)
        joined = set(np.unique(plabels[space])) - {0}
        linked = set(np.unique(labels[np.isin(plabels, list(joined))])) - {0}
        space = np.isin(labels, list(linked))
    return space, dist, floor


def _runs(space, m, angle, spacing):
    """Raw lanes: runs of `space` along lines at world `angle`, at most `spacing` apart and
    spread evenly so the outermost lanes sit on the edge of `space` (as close to the walls as
    the side clearance allows), as world ((x0, y0), (x1, y1)) in the direction of `angle`."""
    h, w = space.shape
    # image rows grow downwards, so a world angle is the same angle clockwise in the image
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), -math.degrees(angle), 1.0)
    # rotate onto a canvas big enough for any angle so nothing is cut off
    diag = int(math.ceil(math.hypot(h, w))) + 2
    rot[0, 2] += (diag - w) / 2
    rot[1, 2] += (diag - h) / 2
    inv = cv2.invertAffineTransform(rot)
    rotated = cv2.warpAffine(space.astype(np.uint8), rot, (diag, diag), flags=cv2.INTER_NEAREST)
    # a slanted edge turns into a staircase in the rotated grid: bridge gaps of a few cells
    # along each row (real obstacles leave gaps of 2 x side_clearance or more) and keep the
    # outermost lanes one cell inside the edge
    rotated = cv2.morphologyEx(rotated, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
    rows = np.nonzero(rotated.any(axis=1))[0]
    lanes = []
    if len(rows) < 3:
        return lanes
    first, last = rows[0] + 1, rows[-1] - 1
    span = last - first
    count = int(math.ceil(span / (spacing / m.resolution))) + 1
    step = span / (count - 1) if count > 1 else 0.0
    for k in range(count):
        r = first + k * step
        row = rotated[int(round(r))]
        edges = np.diff(np.concatenate([[0], row, [0]]))
        for c0, c1 in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
            ends = []
            for c in (c0 + 0.5, c1 - 1.5):
                x, y = inv @ np.array([c, r, 1.0])
                ends.append(m.to_world(y, x))
            (ax, ay), (bx, by) = ends
            if (bx - ax) * math.cos(angle) + (by - ay) * math.sin(angle) < 0:
                ends = ends[::-1]
            lanes.append(tuple(ends))
    return lanes


def _free_ahead(floor, m, x, y, heading, half_width, max_dist):
    """Free floor straight ahead of (x, y) along `heading`, `half_width` wide (m)."""
    c, s = math.cos(heading), math.sin(heading)
    for d in np.arange(0.0, max_dist, m.resolution):
        for off in np.linspace(-half_width, half_width, 5):
            px, py = x + c * d - s * off, y + s * d + c * off
            r, col = m.to_cell(px, py)
            if not (0 <= r < floor.shape[0] and 0 <= col < floor.shape[1]) or not floor[r, col]:
                return d
    return max_dist


def _trim(lane, floor, m, front, half_width, min_len):
    """Shorten both ends until each has `front` m of free floor ahead; None if too short."""
    a, b = np.array(lane[0], float), np.array(lane[1], float)
    length = np.linalg.norm(b - a)
    if length < 1e-6:
        return None
    u = (b - a) / length
    heading = math.atan2(u[1], u[0])
    ahead_b = _free_ahead(floor, m, *b, heading, half_width, front + 0.05)
    ahead_a = _free_ahead(floor, m, *a, heading + math.pi, half_width, front + 0.05)
    b = b - u * max(0.0, front - ahead_b)
    a = a + u * max(0.0, front - ahead_a)
    if np.dot(b - a, u) < min_len:
        return None
    return (tuple(a), tuple(b))


def lanes(m, p, start=None, angle=None):
    """Trimmed lanes at `angle` (radians; None = best_angle())."""
    from securebot_ros import coverage_logic as cl
    space, _, floor = centre_space(m, p, start)
    if angle is None:
        angle = best_angle(space, m, p)
    out = []
    for lane in _runs(space, m, angle, p.lane_spacing):
        t = _trim(lane, floor, m, p.front_clearance, cl.WIDTH / 2, p.min_lane)
        if t is not None:
            out.append(t)
    return out, angle


def _run_length(space, angle):
    """Per cell of `space`: length (cells) of the straight run of `space` through it along
    world `angle` (0 outside `space`)."""
    h, w = space.shape
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), -math.degrees(angle), 1.0)
    diag = int(math.ceil(math.hypot(h, w))) + 2
    rot[0, 2] += (diag - w) / 2
    rot[1, 2] += (diag - h) / 2
    rotated = cv2.warpAffine(space.astype(np.uint8), rot, (diag, diag), flags=cv2.INTER_NEAREST)
    rotated = cv2.morphologyEx(rotated, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
    out = np.zeros(rotated.shape, np.float32)
    for r in np.nonzero(rotated.any(axis=1))[0]:
        edges = np.diff(np.concatenate([[0], rotated[r], [0]]))
        for c0, c1 in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
            out[r, c0:c1] = c1 - c0
    back = cv2.warpAffine(out, cv2.invertAffineTransform(rot), (w, h), flags=cv2.INTER_NEAREST)
    return np.where(space, back, 0.0)


def split_by_direction(space, m, angle, smooth=1.5, min_area=1.0):
    """Cut `space` into parts whose lanes run along `angle` and parts whose lanes run across
    it (angle + 90 deg): every spot goes with the direction the free space runs further in
    (a corridor: along it), smoothed over `smooth` m. Returns [(mask, lane angle)], one per
    connected part of at least `min_area` m² (smaller bits are left to the fill lanes)."""
    along = _run_length(space, angle)
    across = _run_length(space, angle + math.pi / 2)
    pref = ((along >= across) & space).astype(np.uint8) * 255
    k = max(3, int(round(smooth / m.resolution)) | 1)
    pref = cv2.medianBlur(pref, k) > 127
    out = []
    for mask, a in ((space & pref, angle), (space & ~pref, (angle + math.pi / 2) % math.pi)):
        n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            part = labels == i
            if part.sum() * m.resolution ** 2 >= min_area:
                out.append((part, a))
    return out


def region_lanes(m, p, start=None, angle=0.0):
    """Trimmed lanes with a direction per part of the floor (split_by_direction): along each
    corridor, so a corridor across the main one is not crossed by many short lanes."""
    from securebot_ros import coverage_logic as cl
    space, _, floor = centre_space(m, p, start)
    out = []
    for part, a in split_by_direction(space, m, angle):
        for lane in _runs(part, m, a, p.lane_spacing):
            t = _trim(lane, floor, m, p.front_clearance, cl.WIDTH / 2, p.min_lane)
            if t is not None:
                out.append(t)
    return out


def wall_angle(m):
    """Direction of the room's main walls (radians, 0..pi): the most common direction of the
    straight edges of the free floor, weighted by their length."""
    from securebot_ros import coverage_logic as cl
    floor = cl.floor_mask(m, 0.15).astype(np.uint8) * 255
    edges = cv2.Canny(floor, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, minLineLength=int(0.6 / m.resolution),
                            maxLineGap=3)
    if lines is None:
        return 0.0
    votes = np.zeros(90)
    for x0, y0, x1, y1 in lines[:, 0]:
        deg = math.degrees(math.atan2(-(y1 - y0), x1 - x0)) % 90     # image y points down
        votes[int(round(deg)) % 90] += math.hypot(x1 - x0, y1 - y0)
    smooth = np.convolve(np.concatenate([votes[-3:], votes, votes[:3]]), np.ones(7), 'valid')
    return math.radians(int(np.argmax(smooth)))


DRIVE_MPS = 0.18          # measured lapping speed
LANE_END_S = 6.5          # two turns on the spot at a lane end: 2 x (0.6 + 1.0 s settle/spin)
UNCOVERED_S_PER_M2 = 30.0  # how much unswept floor is worth in seconds when picking an angle
LANE_S = 30.0             # extra cost of each lane (getting to it, lining up): fewer lanes


def best_angle(space, m, p):
    """Lane angle (0..180 deg, 5 deg steps) with the lowest estimated cost: driving time
    (lane length / speed + LANE_END_S per lane) plus UNCOVERED_S_PER_M2 for every m² of
    reachable floor the lanes would leave unswept."""
    from securebot_ros import coverage_logic as cl
    floor = cl.floor_mask(m, p.fill_holes)
    half = cl._disk(cl.WIDTH / 2 / m.resolution)
    body = cv2.dilate(space.astype(np.uint8), half).astype(bool) & floor
    thick = max(1, int(round(cl.WIDTH / m.resolution)))
    best = None
    for deg in range(0, 180, 5):
        runs = [r for r in _runs(space, m, math.radians(deg), p.lane_spacing)
                if math.dist(*r) >= p.min_lane]
        swept = np.zeros(space.shape, np.uint8)
        for a, b in runs:
            ra, ca = m.to_cell(*a)
            rb, cb = m.to_cell(*b)
            cv2.line(swept, (ca, ra), (cb, rb), 1, thick)
        uncovered = float((body & ~swept.astype(bool)).sum()) * m.resolution ** 2
        cost = (sum(math.dist(*r) for r in runs) / DRIVE_MPS + LANE_END_S * len(runs)
                + UNCOVERED_S_PER_M2 * uncovered)
        if best is None or cost < best[0]:
            best = (cost, deg)
    return math.radians(best[1])


def _segment_clear(space, m, a, b):
    """The straight a-b stays in `space`: the same points as stepping along it one cell at a
    time (Map.to_cell), checked at once (2026-11-07: was ~30 s of a hallway plan)."""
    n = max(2, int(math.dist(a, b) / m.resolution) + 1)
    t = np.linspace(0.0, 1.0, n)
    x = a[0] + (b[0] - a[0]) * t
    y = a[1] + (b[1] - a[1]) * t
    rows = np.trunc(space.shape[0] - (y - m.origin[1]) / m.resolution).astype(np.int64)
    cols = np.trunc((x - m.origin[0]) / m.resolution).astype(np.int64)
    if rows.min() < 0 or cols.min() < 0 or rows.max() >= space.shape[0] \
            or cols.max() >= space.shape[1]:
        # off the map somewhere: clear only up to there, as stepping would find
        inside = (rows >= 0) & (cols >= 0) & (rows < space.shape[0]) & (cols < space.shape[1])
        return False if not inside.all() else bool(space[rows, cols].all())
    return bool(space[rows, cols].all())


def _snap(space, cell, radius=4):
    """`cell`, or the nearest cell of `space` within `radius` cells (lane ends can sit a cell
    outside it after rotating), or None."""
    r0, c0 = cell
    rows, cols = space.shape
    best = None
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r, c = r0 + dr, c0 + dc
            if 0 <= r < rows and 0 <= c < cols and space[r, c]:
                d = dr * dr + dc * dc
                if best is None or d < best[0]:
                    best = (d, (r, c))
    return None if best is None or best[0] > radius * radius else best[1]


def _shortest_way(space, m, a, b):
    """Waypoints from a to b through `space` (A* on the grid, then straightened), or None."""
    rows, cols = space.shape
    start, goal = _snap(space, m.to_cell(*a)), _snap(space, m.to_cell(*b))
    if start is None or goal is None:
        return None
    # A* over a padded flat copy of the grid (cell (r, c) -> (r + 1) * w + c + 1, which sorts
    # like (r, c), so ties break as before): same search, same path, ~4x faster than tuples
    # and per-cell numpy lookups (2026-11-07: was ~80 s of a hallway plan)
    w = cols + 2
    free = np.pad(space, 1).astype(np.uint8).tobytes()
    s_i = (start[0] + 1) * w + start[1] + 1
    g_i = (goal[0] + 1) * w + goal[1] + 1
    gr, gc = goal[0] + 1, goal[1] + 1
    openq = [(0.0, s_i)]
    came, cost = {s_i: -1}, {s_i: 0.0}
    steps = [(w, 1.0), (-w, 1.0), (1, 1.0), (-1, 1.0),
             (w + 1, 1.414), (w - 1, 1.414), (-w + 1, 1.414), (-w - 1, 1.414)]
    hypot, push, pop, inf = math.hypot, heapq.heappush, heapq.heappop, math.inf
    while openq:
        _, cur = pop(openq)
        if cur == g_i:
            break
        base = cost[cur]
        for d, step in steps:
            nxt = cur + d
            if free[nxt]:
                g = base + step
                if g < cost.get(nxt, inf):
                    cost[nxt], came[nxt] = g, cur
                    r, c = divmod(nxt, w)
                    push(openq, (g + hypot(r - gr, c - gc), nxt))
    if g_i not in came:
        return None
    cells, cur = [], g_i
    while cur != -1:
        r, c = divmod(cur, w)
        cells.append((r - 1, c - 1))
        cur = came[cur]
    pts = [m.to_world(r, c) for r, c in cells[::-1]]
    # string-pull: keep only the points needed for straight, clear hops (from each kept point
    # the farthest one it reaches straight; all candidates checked at once)
    out, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = _farthest_clear(space, m, pts, i)
        out.append(pts[j])
        i = j
    return out


def _farthest_clear(space, m, pts, i):
    """The largest j > i with the straight pts[i]-pts[j] clear (_segment_clear), else i + 1:
    every j checked in one go, the same answer as trying j = last, last - 1, ... in turn."""
    js = np.arange(i + 2, len(pts))
    if not len(js):
        return i + 1
    a = np.asarray(pts[i], float)
    b = np.asarray([pts[j] for j in js], float)
    n = np.maximum(2, (np.hypot(*(b - a).T) / m.resolution).astype(np.int64) + 1)
    seg = np.repeat(np.arange(len(js)), n)
    starts = np.concatenate([[0], np.cumsum(n)[:-1]])
    k = np.arange(seg.size) - np.repeat(starts, n)
    t = k / np.repeat(n - 1, n)
    x = a[0] + (b[seg, 0] - a[0]) * t
    y = a[1] + (b[seg, 1] - a[1]) * t
    rows = np.trunc(space.shape[0] - (y - m.origin[1]) / m.resolution).astype(np.int64)
    cols = np.trunc((x - m.origin[0]) / m.resolution).astype(np.int64)
    inside = (rows >= 0) & (cols >= 0) & (rows < space.shape[0]) & (cols < space.shape[1])
    ok = inside.copy()
    ok[inside] = space[rows[inside], cols[inside]]
    clear = np.logical_and.reduceat(ok, starts)
    good = np.nonzero(clear)[0]
    return int(js[good[-1]]) if len(good) else i + 1


def fill_lanes(m, p, space, floor, lane_list, rounds=3):
    """Lanes for pockets of `space` that no lane in `lane_list` passes within check_width / 2:
    one lane through each pocket along its long side (as far as `space` goes), trimmed for the
    front clearance."""
    from securebot_ros import coverage_logic as cl
    out = []
    thick = max(1, int(round(p.check_width / m.resolution)))
    for _ in range(rounds):
        seen = np.zeros(space.shape, np.uint8)
        for a, b in lane_list + out:
            ra, ca = m.to_cell(*a)
            rb, cb = m.to_cell(*b)
            cv2.line(seen, (ca, ra), (cb, rb), 1, thick)
        pockets = (space & ~seen.astype(bool)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(pockets, connectivity=8)
        added = 0
        for k in range(1, n):
            rows, cols = np.nonzero(labels == k)
            if stats[k, cv2.CC_STAT_AREA] * m.resolution ** 2 < p.min_pocket:
                continue
            xy = np.array([m.to_world(r, c) for r, c in zip(rows, cols)], np.float32)
            vx, vy, x0, y0 = cv2.fitLine(xy, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
            lane = _along(m, space, np.array([x0, y0], float), np.array([vx, vy], float))
            if lane is None:
                continue
            lane = _trim(lane, floor, m, p.front_clearance, cl.WIDTH / 2, p.min_lane)
            if lane is None or any(_same_lane(lane, q) for q in lane_list + out):
                continue
            out.append(lane)
            added += 1
        if not added:
            break
    return out


def _pocket_length(rows, cols, m):
    """Length (m) of a pocket's longest side (its smallest enclosing rectangle)."""
    if len(rows) < 3:
        return 0.0
    (_, _), (w, h), _ = cv2.minAreaRect(np.column_stack([cols, rows]).astype(np.float32))
    return max(w, h) * m.resolution


def _along(m, space, mid, u, max_dist=12.0):
    """The stretch of the line through `mid` along `u` that stays in `space`, or None."""
    r, c = m.to_cell(*mid)
    if not (0 <= r < space.shape[0] and 0 <= c < space.shape[1]) or not space[r, c]:
        return None
    ends = []
    for sign in (-1.0, 1.0):
        d = 0.0
        while d < max_dist:
            q = mid + sign * (d + m.resolution) * u
            r, c = m.to_cell(*q)
            if not (0 <= r < space.shape[0] and 0 <= c < space.shape[1]) or not space[r, c]:
                break
            d += m.resolution
        ends.append(tuple(map(float, mid + sign * d * u)))
    return ends[0], ends[1]


def _skeleton(mask):
    """One-cell-wide centre lines of a bool mask (morphological skeleton)."""
    img = mask.astype(np.uint8)
    if hasattr(cv2, 'ximgproc'):
        return cv2.ximgproc.thinning(img * 255) > 0
    skel = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while img.any():
        opened = cv2.dilate(cv2.erode(img, kernel), kernel)
        skel |= img & ~opened
        img = cv2.erode(img, kernel)
    return skel.astype(bool)


def pass_lanes(m, p, start=None):
    """Straight lanes through narrow aisles, each ending `pass_reach` m into the turning
    space at both ends (see the module notes). World ((x0, y0), (x1, y1)) list.

    Every piece of narrow centre line is an aisle candidate: a straight line fitted through
    it, walked out both ways."""
    space, dist, floor = centre_space(m, p, start)
    passable = dist >= p.pass_clearance
    # aisles are where the centre line of the drivable floor is too narrow to turn (a strip
    # along a wall is never on a centre line); bits shorter than min_aisle are noise
    narrow = _skeleton(passable) & (dist < p.side_clearance)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(narrow.astype(np.uint8), 8)
    keep = np.zeros(narrow.shape, bool)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] * m.resolution >= p.min_aisle:
            keep |= labels == k
    # walking out an aisle may come this much closer to the sides (still clear of the
    # follower's own side check, half_width + margin = 0.306 m)
    walkable = dist >= p.pass_clearance - p.pass_slack
    out = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] * m.resolution < p.min_aisle:
            continue
        rows, cols = np.nonzero(labels == k)
        xy = np.array([m.to_world(r, c) for r, c in zip(rows, cols)], np.float32)
        vx, vy, x0, y0 = cv2.fitLine(xy, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        mid, u = np.array([x0, y0], float), np.array([vx, vy], float)
        lane = _through(m, walkable, space, mid, u, p.pass_reach)
        if lane is None and getattr(p, 'pocket_aisles', False):
            # edge case (2026-11-07, Kumar's Lab side aisles): an aisle that ends in a small
            # pocket where the robot can still turn (any turning space, less than pass_reach
            # of it) gets its lane too; a true dead end still does not
            lane = _through(m, walkable, space, mid, u, m.resolution)
        if lane is None or math.dist(*lane) < p.min_lane:
            continue
        if any(_same_lane(lane, q) for q in out):   # the same aisle found twice: keep one
            continue
        out.append(lane)
    return out


def _same_lane(a, b, tol=0.35):
    return (math.dist(a[0], b[0]) < tol and math.dist(a[1], b[1]) < tol) or \
        (math.dist(a[0], b[1]) < tol and math.dist(a[1], b[0]) < tol)


def _through(m, passable, space, mid, u, reach):
    """From `mid` both ways along `u`: walk while passable until `reach` m inside `space`.
    The lane (a, b), or None if either way hits something first (a dead end)."""
    ends = []
    for sign in (-1.0, 1.0):
        inside, d, end = 0.0, 0.0, None
        while d < 8.0:
            q = mid + sign * d * u
            r, c = m.to_cell(*q)
            if not (0 <= r < passable.shape[0] and 0 <= c < passable.shape[1]) \
                    or not passable[r, c]:
                break
            if space[r, c]:
                inside += m.resolution
                if inside >= reach:
                    end = q
                    break
            else:
                inside = 0.0
            d += m.resolution
        if end is None:
            return None
        ends.append(tuple(map(float, end)))
    return ends[0], ends[1]


def chain(m, p, lane_list, start):
    """Drive order: from `start`, always the nearest unvisited lane end next. Returns paths
    [[(x, y), ...]], one per lane: the way there (from the previous lane's end) + the lane."""
    space, dist, _ = centre_space(m, p, start)
    passable = (dist >= p.pass_clearance - p.pass_slack) if getattr(p, 'pass_lanes', False) \
        else None
    todo = list(lane_list)
    here = tuple(start) if start is not None else (todo[0][0] if todo else (0.0, 0.0))
    paths = []
    while todo:
        _, i, k = min((math.dist(here, end), i, k)
                      for i, lane in enumerate(todo) for k, end in enumerate(lane))
        a, b = (todo[i][0], todo[i][1]) if k == 0 else (todo[i][1], todo[i][0])
        todo.pop(i)
        if math.dist(here, a) < 0.05 or _segment_clear(space, m, here, a):
            way = [here]
        else:
            way = _shortest_way(space, m, here, a)
            if way is None and passable is not None:      # through a narrow passage
                way = _shortest_way(space | passable, m, here, a)
            if way is None and not paths:    # from the robot's own spot: the laps node
                way = [here, a]              # gets there by the Nav2 planner
            elif way is None:                # between lanes, no safe way: leave the lane out
                continue
            way = [here] + way[1:-1]
        paths.append([tuple(map(float, q)) for q in way + [a, b]])
        here = b
    return paths


def turn_count(path, turn_deg=45.0, merge=0.6):
    """(turns on the spot, total degrees turned) along a path: heading changes of `turn_deg`
    or more within `merge` m count as one stop-and-turn."""
    pts = np.asarray(path, float)
    seg = np.diff(pts, axis=0)
    seg = seg[np.hypot(seg[:, 0], seg[:, 1]) > 0.02]
    if len(seg) < 2:
        return 0, 0.0
    head = np.arctan2(seg[:, 1], seg[:, 0])
    d = np.degrees(np.abs(np.arctan2(np.sin(np.diff(head)), np.cos(np.diff(head)))))
    lengths = np.hypot(seg[1:, 0], seg[1:, 1])
    turns, acc, acc_len = 0, 0.0, 0.0
    for a, ln in zip(d, lengths):
        acc += a
        acc_len += ln
        if acc >= turn_deg:
            turns, acc, acc_len = turns + 1, 0.0, 0.0
        elif acc_len > merge:
            acc, acc_len = 0.0, 0.0
    return turns, float(d.sum())


# ------------------------------------------------------------------ edge cases (side aisles)
def edge_lanes(m, p, start, lane_list):
    """Extra lanes for floor the plan's lanes (`lane_list`, chosen without these) leave out,
    though the robot fits there (2026-11-07, Kumar's Lab side aisles). World 2-point lanes:
      - an aisle too narrow to turn in that ends in a small pocket where it can (less than
        pass_reach of turning space): a lane through it (p.pocket_aisles);
      - a long, thin strip of turning space no lane comes near (along a wall, crossed too
        briefly by the lanes): lanes along its centre line, if it is at least p.long_pocket
        m long (whatever its area)."""
    import dataclasses
    from securebot_ros import coverage_logic as cl
    out = []
    if getattr(p, 'pocket_aisles', False) and p.pass_lanes:
        for lane in pass_lanes(m, dataclasses.replace(p, pocket_aisles=True), start):
            if not any(_same_lane(lane, q) for q in lane_list + out):
                out.append(lane)
    long_pocket = getattr(p, 'long_pocket', math.inf)
    if not math.isfinite(long_pocket):
        return out
    space, _, floor = centre_space(m, p, start)
    seen = np.zeros(space.shape, np.uint8)
    thick = max(1, int(round(p.check_width / m.resolution)))
    for a, b in lane_list + out:
        ra, ca = m.to_cell(*a)
        rb, cb = m.to_cell(*b)
        cv2.line(seen, (ca, ra), (cb, rb), 1, thick)
    pockets = (space & ~seen.astype(bool)).astype(np.uint8)
    n, labels, _, _ = cv2.connectedComponentsWithStats(pockets, connectivity=8)
    for k in range(1, n):
        rows, cols = np.nonzero(labels == k)
        if _pocket_length(rows, cols, m) < long_pocket:
            continue
        # as few straight lanes as fit (2026-11-08, Kumar's Lab: the centre line of an L-shaped
        # strip, straightened within 0.2 m, still bent 4 times and each bend made a lane of
        # its own, 0.7-1.4 m long, driven with turns between them): from the dense centre
        # line, the longest straight runs that stay where the robot centre may go
        line = _straight_runs(space, m, _centre_line(labels == k, m, tol=0.0))
        if len(line) < 2 or sum(math.dist(a, b) for a, b in zip(line, line[1:])) < long_pocket:
            continue
        segs = [(a, b) for a, b in zip(line, line[1:]) if math.dist(a, b) >= 0.3]
        segs = _absorb_short(space, m, segs, p.check_width / 2)
        if not segs:
            continue
        first = _trim(segs[0], floor, m, p.front_clearance, cl.WIDTH / 2, 0.3)
        last = _trim(segs[-1], floor, m, p.front_clearance, cl.WIDTH / 2, 0.3) \
            if len(segs) > 1 else first
        tail = [last] if last and len(segs) > 1 else []
        segs = ([first] if first else []) + segs[1:-1] + tail
        out.extend(tuple(map(tuple, seg)) for seg in segs if seg)
    return out


def _straight_runs(space, m, pts):
    """The points of `pts` (a dense line in `space`) where straight runs meet: from each, the
    farthest point the straight line to which stays in `space`."""
    if len(pts) < 2:
        return list(pts)
    out, i = [pts[0]], 0
    while i < len(pts) - 1:
        i = _farthest_clear(space, m, pts, i)
        out.append(pts[i])
    return out


def _absorb_short(space, m, segs, reach, max_ext=1.5, step=0.05):
    """Drop a run whose floor a neighbouring longer run checks anyway (every point of it within
    `reach` m of that run) once the longer run is extended straight towards it, as far as it
    stays in `space` (at most max_ext m): a strip's short bent tail is no lane of its own.
    Runs stay in their order along the line."""
    def pdist(q, a, b):
        a, b, q = np.asarray(a, float), np.asarray(b, float), np.asarray(q, float)
        ab = b - a
        t = np.clip(np.dot(q - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
        return float(np.hypot(*(q - a - t * ab)))

    def extend(a, b, toward):
        """b moved on along a->b while clear and while it brings it closer to `toward`."""
        u = (np.asarray(b, float) - a) / max(math.dist(a, b), 1e-9)
        end = np.asarray(b, float)
        for d in np.arange(step, max_ext + step / 2, step):
            q = np.asarray(b, float) + u * d
            if math.dist(q, toward) >= math.dist(end, toward) \
                    or not _segment_clear(space, m, end, q):
                break
            end = q
        return tuple(map(float, end))

    segs = [tuple(map(tuple, s)) for s in segs]
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, s in enumerate(segs):
            for j in (i - 1, i + 1):
                if not 0 <= j < len(segs):
                    continue
                k = segs[j]
                if math.dist(*k) <= math.dist(*s):
                    continue
                far = max(s, key=lambda q: min(math.dist(q, k[0]), math.dist(q, k[1])))
                if math.dist(k[1], far) <= math.dist(k[0], far):
                    grown = (k[0], extend(k[0], k[1], far))
                else:
                    grown = (extend(k[1], k[0], far), k[1])
                n = max(2, int(math.dist(*s) / step) + 1)
                pts = [np.asarray(s[0]) + (np.asarray(s[1]) - s[0]) * t
                       for t in np.linspace(0.0, 1.0, n)]
                if all(pdist(q, *grown) <= reach for q in pts):
                    segs[j] = grown
                    del segs[i]
                    changed = True
                    break
            if changed:
                break
    return segs


def _centre_line(mask, m, tol=0.20):
    """The longest path along the skeleton of `mask`, straightened within `tol` m: world
    points from one end to the other."""
    from collections import deque
    skel = _skeleton(mask)
    cells = set(zip(*np.nonzero(skel)))
    if not cells:
        return []

    def far(src):
        prev, q, last = {src: None}, deque([src]), src
        while q:
            cur = q.popleft()
            last = cur
            r, c = cur
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nb = (r + dr, c + dc)
                    if nb in cells and nb not in prev:
                        prev[nb] = cur
                        q.append(nb)
        return last, prev
    a, _ = far(next(iter(cells)))
    b, prev = far(a)
    path, cur = [], b
    while cur is not None:
        path.append(m.to_world(*cur))
        cur = prev[cur]
    from securebot_ros.route_logic import _plain_douglas
    return [tuple(map(float, q)) for q in _plain_douglas(path, tol)]
