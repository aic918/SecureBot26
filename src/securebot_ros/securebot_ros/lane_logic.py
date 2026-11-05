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
    return labels == label, dist, floor


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
    n = max(2, int(math.dist(a, b) / m.resolution) + 1)
    for t in np.linspace(0.0, 1.0, n):
        r, c = m.to_cell(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        if not (0 <= r < space.shape[0] and 0 <= c < space.shape[1]) or not space[r, c]:
            return False
    return True


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
    openq = [(0.0, start)]
    came, cost = {start: None}, {start: 0.0}
    steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    while openq:
        _, cur = heapq.heappop(openq)
        if cur == goal:
            break
        for dr, dc, w in steps:
            nxt = (cur[0] + dr, cur[1] + dc)
            if 0 <= nxt[0] < rows and 0 <= nxt[1] < cols and space[nxt]:
                g = cost[cur] + w
                if g < cost.get(nxt, math.inf):
                    cost[nxt], came[nxt] = g, cur
                    h = math.hypot(nxt[0] - goal[0], nxt[1] - goal[1])
                    heapq.heappush(openq, (g + h, nxt))
    if goal not in came:
        return None
    cells, cur = [], goal
    while cur is not None:
        cells.append(cur)
        cur = came[cur]
    pts = [m.to_world(r, c) for r, c in cells[::-1]]
    # string-pull: keep only the points needed for straight, clear hops
    out, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _segment_clear(space, m, pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def chain(m, p, lane_list, start):
    """Drive order: from `start`, always the nearest unvisited lane end next. Returns paths
    [[(x, y), ...]], one per lane: the way there (from the previous lane's end) + the lane."""
    space, _, _ = centre_space(m, p, start)
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
