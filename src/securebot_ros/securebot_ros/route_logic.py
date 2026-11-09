"""Corridor routes on a saved map (no ROS): lanes that follow the corridors, few turns.

Straight lanes at one angle (lane_logic) suit a box-shaped room, but in a building's hallways
they break up: a hallway that bends a little on the map leaves a straight line, so its lanes
are cut into pieces, each with its own turns, and a side corridor gets crossed by many short
lanes. Here the route follows the corridors instead:

1. Where the robot centre can be: `pass_clearance` from anything (narrow aisles included);
   specks of unseen floor up to `route_holes` m² are filled first (they only add knots).
2. Centre lines: the skeleton of that space as a graph (junctions, dead ends and the corridor
   pieces between them). Short dead-end spurs (door alcoves, room corners: shorter than the
   corridor's half-width + `route_spur` m) and dead ends too narrow to turn round in are cut.
3. Lanes: each corridor piece gets k lanes that follow its centre line, side by side and at
   most `check_width` apart across the turning space (k = 1 in a hallway up to ~3 m wide: one
   lane straight down the middle). Lanes are straightened (`route_simplify`) where that stays
   clear, so a slightly bent hallway is one lane with a few gentle bends, not many turns.
4. Order: a depth-first walk over the corridor graph from the robot. Side branches go first
   and the longest branch last (the robot ends there and never drives back along it). A
   branch the robot comes back from gets an even number of lanes (there on one side, back on
   the other), a branch it does not return from an odd number; a dead end is turned round once.

Turns happen only at junctions and dead ends, and lanes meet only at junctions.
"""
import math

import cv2
import numpy as np

ROUTE_HOLES = 1.0       # m²: unseen specks inside the floor up to this are filled
SPECK = 0.05            # m²: seen specks up to this are ignored when finding corridors
ROUTE_SPUR = 1.0        # m: dead ends shorter than half-width + this are cut (alcoves, corners)
ROUTE_SIMPLIFY = 0.20   # m: lanes straightened within this (where it stays clear)
SAMPLE = 0.25           # m between lane points before straightening
SMOOTH = 1.0            # m: centre line direction averaged over this (for the side offsets)
JOIN = 1.0              # m: a lane starting this near is joined straight onto (no hop)
LANE_SPREAD = 1.5       # lanes pushed out from the middle (x their strip centre, capped)
LANE_SLACK = 0.25       # of a lane spacing: width over a whole number of lanes still left out


# ------------------------------------------------------------------------------- the space
def route_space(m, p, start=None, specks=False):
    """(mask where the robot centre may be, floor distance to anything in m, turning space
    mask) for the connected part holding `start` (or the largest). Unseen specks up to
    ROUTE_HOLES m² count as floor; specks=True also fills seen ones up to SPECK m² (LiDAR
    noise, a chair leg): for finding the corridors only, never for clearance."""
    from securebot_ros import coverage_logic as cl
    floor = cl.floor_mask(m, p.fill_holes)
    holes = _fill_small_holes(floor, ROUTE_HOLES / m.resolution ** 2, m.wall)
    if specks:
        holes = _fill_small_holes(holes, SPECK / m.resolution ** 2,
                                  np.zeros_like(holes) if m.wall is not None else None)
    padded = np.pad(holes.astype(np.uint8), 1)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1] * m.resolution
    ok = (dist >= p.pass_clearance - p.pass_slack).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ok, connectivity=8)
    if n <= 1:
        return np.zeros(ok.shape, bool), dist, np.zeros(ok.shape, bool)
    label = 0
    if start is not None:
        r, c = m.to_cell(*start)
        if 0 <= r < labels.shape[0] and 0 <= c < labels.shape[1]:
            label = labels[r, c]
            if label == 0:                          # robot just off the space: nearest part
                rr, cc = np.nonzero(labels)
                k = int(np.argmin((rr - r) ** 2 + (cc - c) ** 2))
                label = labels[rr[k], cc[k]]
    if label == 0:
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    space = labels == label
    return space, dist, space & (dist >= p.side_clearance)


def _fill_small_holes(mask, max_cells, wall=None):
    """`mask` with enclosed specks of unseen floor (up to max_cells) filled in; anything the
    LiDAR saw (`wall`; None = treat every non-floor cell as seen) is never filled."""
    if wall is None:
        return mask.copy()
    inv = (~mask).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = mask.copy()
    h, w = mask.shape
    seen = set(np.unique(labels[wall & ~mask])) - {0}
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area <= max_cells and i not in seen and x > 0 and y > 0 and x + bw < w \
                and y + bh < h:
            out[labels == i] = True
    return out


# ------------------------------------------------------------------------ skeleton graph
NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def skeleton_graph(space):
    """Graph of the centre lines of `space`: (nodes {id: [cells]}, edges {id: [n0, n1, cells]})
    where cells run from node n0 to node n1 (both ends included)."""
    skel = cv2.ximgproc.thinning(space.astype(np.uint8) * 255) > 0 \
        if hasattr(cv2, 'ximgproc') else _thin(space)
    k = np.ones((3, 3), np.uint8)
    nbrs = cv2.filter2D(skel.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT) - 1
    nbrs[~skel] = 0
    special = skel & (nbrs != 2)
    nl, lab = cv2.connectedComponents(special.astype(np.uint8), connectivity=8)
    node_of = {}
    nodes = {}
    for r, c in zip(*np.nonzero(special)):
        node_of[(r, c)] = int(lab[r, c])
        nodes.setdefault(int(lab[r, c]), []).append((int(r), int(c)))
    rows, cols = skel.shape

    def neighbours(cell):
        r, c = cell
        for dr, dc in NB:
            q = (r + dr, c + dc)
            if 0 <= q[0] < rows and 0 <= q[1] < cols and skel[q]:
                yield q

    edges, seen = {}, set()
    eid = 0

    def trace(n0, first, prev):
        cells, cur, back = [prev, first], first, prev
        while cur not in node_of:
            seen.add(cur)
            nxt = [q for q in neighbours(cur) if q != back and q not in cells[-3:-1]]
            # prefer a 4-neighbour step (thinning leaves corner pairs)
            if not nxt:
                return None, cells
            nxt.sort(key=lambda q: (q in node_of) * -1 + abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
            back, cur = cur, nxt[0]
            cells.append(cur)
        return node_of[cur], cells

    for n0, cells0 in nodes.items():
        for cell in cells0:
            for q in neighbours(cell):
                if q in node_of or q in seen:
                    if q in node_of and node_of[q] != n0 and cell < q:
                        # two node clusters touching directly: a zero-length edge
                        key = ('direct', min(n0, node_of[q]), max(n0, node_of[q]))
                        if key not in seen:
                            seen.add(key)
                            edges[eid] = [n0, node_of[q], [cell, q]]
                            eid += 1
                    continue
                n1, cells = trace(n0, q, cell)
                if n1 is None:
                    continue
                edges[eid] = [n0, n1, cells]
                eid += 1
    # rings with no junction at all (a corridor round an island): cut them open at one cell
    left = skel & (nbrs == 2)
    for r, c in zip(*np.nonzero(left)):
        if (r, c) in seen:
            continue
        nid = max(nodes, default=0) + 1
        nodes[nid] = [(int(r), int(c))]
        node_of[(int(r), int(c))] = nid
        seen.add((r, c))
        q = next(neighbours((r, c)), None)
        if q is None:
            continue
        n1, cells = trace(nid, q, (int(r), int(c)))
        if n1 is not None:
            edges[eid] = [nid, n1, cells]
            eid += 1
    # drop duplicate edges (traced from both ends)
    out, keys = {}, set()
    for i, (a, b, cells) in edges.items():
        key = (min(a, b), max(a, b), frozenset([cells[len(cells) // 2]]))
        if key in keys:
            continue
        keys.add(key)
        out[i] = [a, b, cells]
    return nodes, out


def _thin(mask):
    from securebot_ros import lane_logic as ll
    return ll._skeleton(mask)


def _cells_len(cells, res):
    a = np.array(cells, float)
    return float(np.hypot(*np.diff(a, axis=0).T).sum()) * res if len(a) > 1 else 0.0


def _degree(edges):
    deg = {}
    for a, b, _ in edges.values():
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    return deg


def prune(nodes, edges, dist, m, p, keep=None):
    """Cut short dead-end spurs (shorter than the corridor half-width + ROUTE_SPUR) and dead
    ends with nowhere to turn round, joining the pieces at nodes left with two edges first;
    then end every dead end front_clearance from the wall ahead. `keep`: a node never cut."""
    res = m.resolution
    while True:
        while _merge(edges, keep):
            pass
        deg = _degree(edges)
        cut = None
        for i, (a, b, cells) in edges.items():
            if a == b:
                continue
            for leaf, other, base in ((a, b, cells[-1]), (b, a, cells[0])):
                if deg.get(leaf, 0) != 1 or leaf == keep or deg.get(other, 0) < 2:
                    continue
                room = max(dist[c] for c in cells) >= p.side_clearance
                if _cells_len(cells, res) < dist[base] + ROUTE_SPUR or not room:
                    cut = i
                    break
            if cut is not None:
                break
        if cut is None:
            break
        del edges[cut]
    _trim_ends(edges, dist, p, keep)
    used = {n for a, b, _ in edges.values() for n in (a, b)}
    return {n: c for n, c in nodes.items() if n in used}, edges


def _trim_ends(edges, dist, p, keep):
    """Dead ends stop where the wall ahead is front_clearance away (the skeleton runs on up to
    the edge of the space), and where there is room to turn round."""
    deg = _degree(edges)
    need = max(p.front_clearance, p.side_clearance)
    for i, (a, b, cells) in list(edges.items()):
        cells = list(cells)
        if deg.get(b, 0) == 1 and b != keep and a != b:
            while len(cells) > 2 and dist[cells[-1]] < need:
                cells.pop()
        if deg.get(a, 0) == 1 and a != keep and a != b:
            while len(cells) > 2 and dist[cells[0]] < need:
                cells.pop(0)
        edges[i][2] = cells


def _merge(edges, keep):
    """Join the two pieces at a node with exactly two (not the robot's node). True if any."""
    deg = _degree(edges)
    for n, d in deg.items():
        if d != 2 or n == keep:
            continue
        two = [(i, e) for i, e in edges.items() if n in (e[0], e[1])]
        if len(two) != 2 or two[0][0] == two[1][0]:
            continue
        (i, (a1, b1, c1)), (j, (a2, b2, c2)) = two
        if a1 == b1 or a2 == b2:
            continue
        p1 = c1 if b1 == n else c1[::-1]           # ... -> n
        o1 = a1 if b1 == n else b1
        p2 = c2 if a2 == n else c2[::-1]           # n -> ...
        o2 = b2 if a2 == n else a2
        del edges[j]
        edges[i] = [o1, o2, p1 + p2[1:]]
        return True
    return False


def contract(nodes, edges, dist, m, keep=None):
    """Junctions a few cells apart (the skeleton splits one crossing into several) become
    one: pieces between two junctions shorter than the corridor half-width + 0.5 m are
    dropped and their ends joined. Small loops left over are dropped too."""
    res = m.resolution
    changed = True
    while changed:
        changed = False
        deg = _degree(edges)
        for i, (a, b, cells) in sorted(edges.items(), key=lambda e: len(e[1][2])):
            length = _cells_len(cells, res)
            if a == b:
                if length < 4.0:
                    del edges[i]
                    changed = True
                    break
                continue
            if deg.get(a, 0) < 2 or deg.get(b, 0) < 2:
                continue
            if length >= max(dist[cells[0]], dist[cells[-1]]) + 0.5:
                continue
            gone, stay = (a, b) if a != keep else (b, a)
            if gone == keep:
                continue
            del edges[i]
            for j, (x, y, c) in edges.items():
                edges[j] = [stay if x == gone else x, stay if y == gone else y, c]
            nodes[stay] = nodes.get(stay, []) + nodes.pop(gone, [])
            changed = True
            break
    return nodes, edges


def split_at(nodes, edges, cell):
    """Split the edge passing nearest `cell` there; returns the node id at that spot."""
    best = None
    for i, (a, b, cells) in edges.items():
        arr = np.array(cells)
        d = (arr[:, 0] - cell[0]) ** 2 + (arr[:, 1] - cell[1]) ** 2
        k = int(np.argmin(d))
        if best is None or d[k] < best[0]:
            best = (d[k], i, k)
    if best is None:
        return None
    _, i, k = best
    a, b, cells = edges[i]
    if k == 0:
        return a
    if k == len(cells) - 1:
        return b
    nid = max(nodes) + 1
    nodes[nid] = [cells[k]]
    del edges[i]
    j = max(edges, default=-1) + 1
    edges[i] = [a, nid, cells[:k + 1]]
    edges[j + 1] = [nid, b, cells[k:]]
    return nid


# ------------------------------------------------------------------------------- lanes
def _world(m, cells):
    return np.array([m.to_world(r, c) for r, c in cells], float)


def _resample(pts, step):
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return pts[:1].copy(), s[:1]
    n = max(2, int(math.ceil(s[-1] / step)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])]), t


def _normals(pts, window):
    """Left-hand unit normals of a polyline, from its direction averaged over `window` points."""
    d = np.gradient(pts, axis=0)
    if len(pts) > 2 and window > 1:
        k = np.ones(window) / window
        d = np.column_stack([np.convolve(np.pad(d[:, i], window // 2, mode='edge'), k, 'valid')
                             [:len(pts)] for i in range(2)])
    n = np.hypot(d[:, 0], d[:, 1])
    n[n < 1e-9] = 1.0
    return np.column_stack([-d[:, 1] / n, d[:, 0] / n])


def edge_lanes(m, p, cells, dist, space, count):
    """`count` lanes along the edge `cells` (all in its direction), spread across the turning
    space (lane i at (i + 0.5) / count of the width), each a list of (x, y)."""
    centre, _ = _resample(_world(m, cells), SAMPLE)
    rc = [m.to_cell(x, y) for x, y in centre]
    half = np.array([max(0.0, dist[min(max(r, 0), dist.shape[0] - 1),
                                   min(max(c, 0), dist.shape[1] - 1)] - p.side_clearance)
                     for r, c in rc])
    # one width for the whole piece where it can be (straight lanes), narrower where it must
    steady = float(np.percentile(half, 30)) if len(half) else 0.0
    half = np.minimum(half, max(steady, 0.0))
    normals = _normals(centre, max(1, int(round(SMOOTH / SAMPLE))))
    lanes = []
    for i in range(count):
        # -1..1: the centre of its strip, spread out by LANE_SPREAD (outer lanes nearer the
        # walls: wider loops that check more floor, with no more lanes)
        frac = max(-1.0, min(1.0, (-1.0 + (2 * i + 1) / count) * LANE_SPREAD))
        pts = centre + normals * (frac * half)[:, None]
        pts = [tuple(q) for q in pts]
        lanes.append(_clean(m, space, pts))
    return lanes


def _clean(m, space, pts):
    """Straighten within ROUTE_SIMPLIFY where the straight bits stay in `space`; points that
    fell outside `space` are pulled back by the way through it."""
    from securebot_ros import lane_logic as ll
    pts = [q for q in pts if _inside(m, space, q)] or pts[:1]
    out = _douglas(m, space, pts, ROUTE_SIMPLIFY)
    fixed = [out[0]]
    for a, b in zip(out, out[1:]):
        if ll._segment_clear(space, m, a, b):
            fixed.append(b)
            continue
        way = ll._shortest_way(space, m, a, b)
        fixed.extend(way[1:] if way else [b])
    return [tuple(map(float, q)) for q in fixed]


def _inside(m, space, q):
    r, c = m.to_cell(*q)
    return 0 <= r < space.shape[0] and 0 <= c < space.shape[1] and space[r, c]


def _douglas(m, space, pts, tol):
    from securebot_ros import lane_logic as ll
    if len(pts) <= 2:
        return list(pts)
    a, b = np.array(pts[0]), np.array(pts[-1])
    ab = b - a
    n = np.hypot(*ab)
    arr = np.array(pts[1:-1])
    if n < 1e-9:
        d = np.hypot(*(arr - a).T)
    else:
        d = np.abs(ab[0] * (arr[:, 1] - a[1]) - ab[1] * (arr[:, 0] - a[0])) / n
    k = int(np.argmax(d))
    if d[k] <= tol and ll._segment_clear(space, m, pts[0], pts[-1]):
        return [pts[0], pts[-1]]
    if d[k] <= tol:                       # close, but not clear straight: split at the middle
        k = len(arr) // 2
    left = _douglas(m, space, pts[:k + 2], tol)
    right = _douglas(m, space, pts[k + 1:], tol)
    return left[:-1] + right


# ------------------------------------------------------------------------------- order
def _lane_count(m, p, cells, dist):
    half = [max(0.0, dist[r, c] - p.side_clearance) for r, c in cells]
    width = 2.0 * float(np.percentile(half, 30)) if half else 0.0
    # a quarter of a lane spacing over is still one lane fewer (a 3 m hallway: one lane)
    return max(1, int(math.ceil(width / max(p.check_width, 0.1) - LANE_SLACK)))


def route(m, p, nodes, edges, dist, space, root):
    """Passes [(edge id, forward?, [(x, y), ...])] in drive order (depth-first from root)."""
    res = m.resolution
    adj = {}
    for i, (a, b, _) in edges.items():
        adj.setdefault(a, []).append(i)
        if b != a:
            adj.setdefault(b, []).append(i)
    # depth-first tree and the length of floor below every tree edge
    parent_edge, order, seen = {root: None}, [], {root}
    stack = [root]
    while stack:
        n = stack.pop()
        order.append(n)
        for i in adj.get(n, []):
            a, b, _ = edges[i]
            o = b if a == n else a
            if o not in seen:
                seen.add(o)
                parent_edge[o] = i
                stack.append(o)
    tree = {i for i in parent_edge.values() if i is not None}
    below = {}
    for n in reversed(order):
        tot = 0.0
        for i in adj.get(n, []):
            if i in tree and parent_edge.get(n) != i:
                a, b, cells = edges[i]
                child = b if a == n else a
                tot += _cells_len(cells, res) + below.get(child, 0.0)
        below[n] = tot

    lanes_of, done, passes = {}, set(), []

    def lanes(i, k):
        key = (i, k)
        if key not in lanes_of:
            lanes_of[key] = edge_lanes(m, p, edges[i][2], dist, space, k)
        return lanes_of[key]

    def drive(i, from_node, k, take):
        """Lanes `take` of edge i's k lanes, alternating direction, starting at from_node."""
        a, b, _ = edges[i]
        fwd = a == from_node
        all_lanes = lanes(i, k)
        for n, t in enumerate(take):
            lane = all_lanes[t]
            go_fwd = fwd if n % 2 == 0 else not fwd
            passes.append((i, go_fwd, lane if go_fwd else lane[::-1]))

    def visit(n, final):
        kids = [i for i in adj.get(n, []) if i not in done]

        def size(i):
            a, b, cells = edges[i]
            o = b if a == n else a
            return _cells_len(cells, res) + (below.get(o, 0.0) if i in tree else 0.0)
        kids.sort(key=size)                       # longest last: the robot ends there
        own = [i for i in kids if i in tree and parent_edge.get(n) != i]
        last = own[-1] if final and own else None
        if last is not None:                      # the one-way branch goes after the others
            kids.remove(last)
            kids.append(last)
        for i in kids:
            if i in done:
                continue
            done.add(i)
            a, b, _ = edges[i]
            o = b if a == n else a
            need = _lane_count(m, p, edges[i][2], dist)
            if i == last:                         # one way: an odd number of passes (one
                k = need if need % 2 else need - 1    # fewer: the LiDAR sees past them)
                drive(i, n, k, range(k))
                visit(o, True)
            else:                                 # there and back: even
                k = need if need % 2 == 0 else need + 1
                drive(i, n, k, [0])               # there along one side
                if i in tree and parent_edge.get(o) == i:
                    visit(o, False)
                drive(i, o, k, range(1, k))       # ... back on the others
    visit(root, True)
    return passes


# -------------------------------------------------------------------------------- plan
def corridor_plan(m, p, start=None):
    """Plan of corridor-following lanes (see the module notes)."""
    from securebot_ros import coverage_logic as cl
    from securebot_ros import lane_logic as ll
    space, dist, turn = route_space(m, p, start)            # clearance: everything seen
    graph_space, graph_dist, _ = route_space(m, p, start, specks=True)
    empty = cl.Plan(loops=[], levels=[], swept=np.zeros(m.free.shape, bool))
    if not space.any():
        return empty
    nodes, edges = skeleton_graph(graph_space)
    if not edges:
        return empty
    here = start if start is not None else m.to_world(*np.argwhere(space)[0])
    root = split_at(nodes, edges, m.to_cell(*here))
    for _ in range(2):
        nodes, edges = contract(nodes, edges, graph_dist, m, keep=root)
        nodes, edges = prune(nodes, edges, graph_dist, m, p, keep=root)
    passes = route(m, p, nodes, edges, graph_dist, space, root)
    paths, at = [], tuple(here)
    for _, _, lane in passes:
        if len(lane) < 2 or _length(lane) < 0.05:
            continue
        a = lane[0]
        if math.dist(at, a) < JOIN and len(lane) > 2 and ll._segment_clear(space, m, at, lane[1]):
            # the next lane starts next to here: straight onto it (one turn, not two)
            path = [at] + list(lane[1:])
        else:
            if ll._segment_clear(space, m, at, a):
                way = [at]
            else:
                way = ll._shortest_way(space, m, at, a) or [at, a]
                way = [at] + way[1:-1]
            path = way + list(lane)
        paths.append(_dedupe([tuple(map(float, q)) for q in path]))
        at = tuple(lane[-1])
    paths = [q for q in paths if len(q) >= 2]
    # waypoints at right angles only (90 deg corners, 180 deg turns back), an even number per
    # lane, on the building's grid (its main walls, +-6 deg: the fewest turns)
    paths, angle, _ = best_square(m, [turn, space], paths, ll.wall_angle(m))
    floor = cv2.dilate(space.astype(np.uint8), cl._disk(cl.WIDTH / 2 / m.resolution)) \
        .astype(bool) & cl.floor_mask(m, p.fill_holes)
    swept = cl.swept_floor(m, paths, width=max(cl.WIDTH, p.check_width), closed=False) & floor
    return cl.Plan(loops=paths, levels=[0] * len(paths), swept=swept,
                   area=float(floor.sum()) * m.resolution ** 2,
                   coverage=float(swept.sum()) / max(int(floor.sum()), 1),
                   length=sum(_length(q) for q in paths), angle=angle)


def _length(pts):
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


def _dedupe(pts, gap=0.15):
    """Drop points within `gap` m of the one before (they only make short, sharp hops); the
    last point stays."""
    out = [pts[0]]
    for q in pts[1:-1]:
        if math.dist(q, out[-1]) > gap:
            out.append(q)
    if len(pts) > 1:
        if len(out) > 1 and math.dist(pts[-1], out[-1]) <= gap:
            out[-1] = pts[-1]
        else:
            out.append(pts[-1])
    return out


# ------------------------------------------------------------------ right-angle waypoints
SQUARE_SIMPLIFY = 0.30  # m: shape of a path before it is squared (smaller wiggles ignored)
SQUARE_MIN_RUN = 0.40   # m: shorter sideways bits are folded into their neighbours
SQUARE_SHIFT = 1.50     # m: how far a straight run may move sideways to stay clear
SQUARE_STEP = 0.05      # m: ... in steps of this
SQUARE_SPLITS = 24      # most extra jogs added to one path to keep it clear
SQUARE_TINY = 0.10      # m: a path ending this near its last line ends on the line
SQUARE_SNAP = 0.50      # m: a path starting this near its first line starts on it


def _to_frame(pts, ang):
    c, s = math.cos(ang), math.sin(ang)
    return [(c * x + s * y, -s * x + c * y) for x, y in pts]


def _from_frame(pts, ang):
    c, s = math.cos(ang), math.sin(ang)
    return [(c * u - s * v, s * u + c * v) for u, v in pts]


def _plain_douglas(pts, tol):
    if len(pts) <= 2:
        return list(pts)
    a, b = np.array(pts[0]), np.array(pts[-1])
    ab = b - a
    n = np.hypot(*ab)
    arr = np.array(pts[1:-1])
    d = np.hypot(*(arr - a).T) if n < 1e-9 else \
        np.abs(ab[0] * (arr[:, 1] - a[1]) - ab[1] * (arr[:, 0] - a[0])) / n
    k = int(np.argmax(d))
    if d[k] <= tol:
        return [pts[0], pts[-1]]
    return _plain_douglas(pts[:k + 2], tol)[:-1] + _plain_douglas(pts[k + 1:], tol)


def _runs(fp, keep_back=False):
    """Straight runs of a path in frame coordinates: [{'h': along u?, 'c': coordinate across,
    'lo'/'hi': extent along}] from its (simplified) segments, alternating h / v."""
    runs = []
    for a, b in zip(fp, fp[1:]):
        du, dv = b[0] - a[0], b[1] - a[1]
        length = math.hypot(du, dv)
        if length < 1e-6:
            continue
        h = abs(du) >= abs(dv)
        across = (a[1] + b[1]) / 2 if h else (a[0] + b[0]) / 2
        along = (a[0], b[0]) if h else (a[1], b[1])
        sense = 1 if along[1] >= along[0] else -1
        if keep_back and runs and runs[-1]['h'] == h and runs[-1].get('sense', sense) != sense:
            # keep_back (the edge-case paths): a turn back along the same direction (180 deg)
            # keeps both legs, joined by a step across at the turning point
            runs.append({'h': not h, 'c': along[0], 'w': 0.0, 'lo': across, 'hi': across,
                         'back': True})
        if runs and runs[-1]['h'] == h:
            r = runs[-1]
            r['c'] = (r['c'] * r['w'] + across * length) / (r['w'] + length)
            r['w'] += length
            r['lo'], r['hi'] = min(r['lo'], *along), max(r['hi'], *along)
        else:
            runs.append({'h': h, 'c': across, 'w': length, 'lo': min(along), 'hi': max(along),
                         'sense': sense})
    # fold short runs into the neighbours on either side (which then join)
    changed = True
    while changed and len(runs) > 2:
        changed = False
        for j in range(1, len(runs) - 1):
            if runs[j].get('back') or runs[j - 1].get('back') or runs[j + 1].get('back'):
                continue                       # never fold away a turn back
            if runs[j]['hi'] - runs[j]['lo'] < SQUARE_MIN_RUN:
                a, b = runs[j - 1], runs[j + 1]
                w = a['w'] + b['w']
                runs[j - 1:j + 2] = [{'h': a['h'], 'c': (a['c'] * a['w'] + b['c'] * b['w']) / w,
                                      'w': w, 'lo': min(a['lo'], b['lo']),
                                      'hi': max(a['hi'], b['hi'])}]
                changed = True
                break
    return runs


def _corners(start, end, runs):
    """Waypoints (frame) for runs between start and end, and the run each segment lies on
    (-1: the first step from start onto the first run)."""
    def pt(h, c, along):
        return (along, c) if h else (c, along)
    wps, owner = [start], []
    r0 = runs[0]
    onto = pt(r0['h'], r0['c'], start[0] if r0['h'] else start[1])
    if math.dist(onto, start) > 1e-3:
        wps.append(onto)
        owner.append(-1)
    for j in range(len(runs) - 1):
        a, b = runs[j], runs[j + 1]
        wps.append((b['c'], a['c']) if a['h'] else (a['c'], b['c']))
        owner.append(j)
    rl = runs[-1]
    wps.append(pt(rl['h'], rl['c'], end[0] if rl['h'] else end[1]))
    owner.append(len(runs) - 1)
    return wps, owner


def square_path(m, space, pts, ang, start=None, keep_back=False, exact_end=False):
    """`pts` as right angles only (in the frame turned `ang`): straight runs along or across,
    joined by 90 deg corners (or a 180 deg turn back), clear of everything (`space`). Starts at
    `start` (default pts[0]). Returns (world points, True) or (pts, False) if it cannot."""
    start = tuple(start) if start is not None else tuple(pts[0])
    fp = _to_frame([start] + list(pts[1:]), ang)
    shape = _plain_douglas(fp, SQUARE_SIMPLIFY)
    if len(shape) < 2 or _length(shape) < 0.05:
        return [start], True
    runs = _runs(shape, keep_back)
    if not runs:
        return [start], True
    # a robot just off the first line: the line through it (no step of a few cm)
    across = fp[0][1] if runs[0]['h'] else fp[0][0]
    if abs(across - runs[0]['c']) < SQUARE_SNAP:
        runs[0]['c'] = across
    # exact_end (the way home): the last line through the end, so it ends exactly there
    end = shape[-1]
    across = end[1] if runs[-1]['h'] else end[0]
    if exact_end and len(runs) > 1 and abs(across - runs[-1]['c']) < SQUARE_SNAP:
        runs[-1]['c'] = across

    def clear(a, b):
        wa, wb = _from_frame([a, b], ang)
        return _clear(space, m, wa, wb)

    splits = 0
    for _ in range(200):
        wps, owner = _corners(fp[0], shape[-1], runs)
        if exact_end and math.dist(wps[-1], end) > SQUARE_TINY:  # a last step onto the end
            wps = wps + [end]
            owner = owner + [len(runs) - 1]
        bad = next((k for k in range(len(owner)) if not clear(wps[k], wps[k + 1])), None)
        if bad is None:
            return _straight_on(_from_frame(wps, ang)), True
        j = max(owner[bad], 0)
        base = runs[j]['c']
        best = None
        steps = int(SQUARE_SHIFT / SQUARE_STEP)
        for d in sorted(range(-steps, steps + 1), key=abs)[1:]:
            runs[j]['c'] = base + d * SQUARE_STEP
            w2, o2 = _corners(fp[0], shape[-1], runs)
            near = [k for k in range(len(o2)) if abs(o2[k] - j) <= 1 or o2[k] == -1]
            if all(clear(w2[k], w2[k + 1]) for k in near):
                best = runs[j]['c']
                break
        if best is not None:
            runs[j]['c'] = best
            continue
        runs[j]['c'] = base
        # no single straight line fits: split the run in two with a jog between
        r = runs[j]
        if splits >= SQUARE_SPLITS or r['hi'] - r['lo'] < 2 * SQUARE_MIN_RUN:
            return list(pts), False
        mid = (r['lo'] + r['hi']) / 2
        a = dict(r, hi=mid)
        b = dict(r, lo=mid)
        jog = {'h': not r['h'], 'c': mid, 'w': 0.0, 'lo': r['c'], 'hi': r['c']}
        runs[j:j + 1] = [a, jog, b]
        splits += 1
    return list(pts), False


def _clear(space, m, a, b):
    """The straight a-b stays in `space` (vectorised lane_logic._segment_clear)."""
    n = max(2, int(math.dist(a, b) / m.resolution) + 1)
    t = np.linspace(0.0, 1.0, n)
    x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
    cols = ((x - m.origin[0]) / m.resolution).astype(int)          # as Map.to_cell
    rows = (space.shape[0] - (y - m.origin[1]) / m.resolution).astype(int)
    if rows.min() < 0 or cols.min() < 0 or rows.max() >= space.shape[0] \
            or cols.max() >= space.shape[1]:
        return False
    return bool(space[rows, cols].all())


def square_path_in(m, spaces, pts, ang, start=None, keep_back=False, exact_end=False):
    """square_path in each of `spaces` (roomiest first), keeping the one with the fewest
    waypoints (the roomiest on a tie); if none works, a grid route (right angles by
    construction) along the path's shape in the last space."""
    best = None
    for space in spaces:
        sq, ok = square_path(m, space, pts, ang, start, keep_back, exact_end)
        if ok and (best is None or len(sq) < len(best)):
            best = sq
    if best is not None:
        return best, True
    start = tuple(start) if start is not None else tuple(pts[0])
    shape = _from_frame(_plain_douglas(_to_frame([start] + list(pts[1:]), ang),
                                       SQUARE_SIMPLIFY), ang)
    route = [start]
    for a, b in zip(shape, shape[1:]):
        leg = _grid_route(m, spaces[-1], route[-1], b, ang)
        if leg is None:
            return list(pts), False
        route.extend(leg[1:])
    return _straight_on(route), True


GRID_CELL = 0.10        # m: grid of the right-angle fallback route
GRID_TURN = 15.0        # cells: what a turn costs on that grid (fewer turns, a bit longer)


def _grid_route(m, space, a, b, ang):
    """Right-angle route a -> b through `space` on a GRID_CELL grid turned `ang` (A* with a
    cost per turn); the first and last stretch line up with a and b exactly. None if none."""
    import heapq
    h, w = space.shape
    k = max(1, int(round(GRID_CELL / m.resolution)))
    # frame -> canvas: rotate the map so the frame axes are the grid's
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), -math.degrees(ang), 1.0)
    diag = int(math.ceil(math.hypot(h, w))) + 2
    rot[0, 2] += (diag - w) / 2
    rot[1, 2] += (diag - h) / 2
    canvas = cv2.warpAffine(space.astype(np.uint8), rot, (diag, diag), flags=cv2.INTER_NEAREST)
    n = diag // k
    grid = canvas[:n * k, :n * k].reshape(n, k, n, k).min(axis=(1, 3)).astype(bool)
    inv = cv2.invertAffineTransform(rot)

    def to_grid(q):
        r, c = m.to_cell(*q)
        x, y = rot @ np.array([c + 0.5, r + 0.5, 1.0])
        return int(y // k), int(x // k)

    def to_world(cell):
        r, c = cell
        x, y = inv @ np.array([(c + 0.5) * k, (r + 0.5) * k, 1.0])
        return m.to_world(y - 0.5, x - 0.5)

    def snap(cell):
        r, c = cell
        for rad in range(0, 6):
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    q = (r + dr, c + dc)
                    if 0 <= q[0] < n and 0 <= q[1] < n and grid[q]:
                        return q
        return None
    s, g = snap(to_grid(a)), snap(to_grid(b))
    if s is None or g is None:
        return None
    steps = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    openq = [(0.0, 0.0, s, -1)]
    came, cost = {(s, -1): None}, {(s, -1): 0.0}
    end = None
    while openq:
        _, g0, cur, d0 = heapq.heappop(openq)
        if cur == g:
            end = (cur, d0)
            break
        if g0 > cost.get((cur, d0), math.inf):
            continue
        for d, (dr, dc) in enumerate(steps):
            nxt = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nxt[0] < n and 0 <= nxt[1] < n) or not grid[nxt]:
                continue
            g1 = g0 + 1.0 + (GRID_TURN if d0 not in (-1, d) else 0.0)
            if g1 < cost.get((nxt, d), math.inf):
                cost[(nxt, d)] = g1
                came[(nxt, d)] = (cur, d0)
                heur = abs(nxt[0] - g[0]) + abs(nxt[1] - g[1])
                heapq.heappush(openq, (g1 + heur, g1, nxt, d))
    if end is None:
        return None
    cells, node = [], end
    while node is not None:
        cells.append(node[0])
        node = came[node]
    cells = cells[::-1]
    corners = [cells[0]] + [cells[i] for i in range(1, len(cells) - 1)
                            if (cells[i][0] - cells[i - 1][0], cells[i][1] - cells[i - 1][1])
                            != (cells[i + 1][0] - cells[i][0], cells[i + 1][1] - cells[i][1])] \
        + [cells[-1]]
    fr = _to_frame([to_world(c) for c in corners], ang)
    fa, fb = _to_frame([a, b], ang)
    runs = _runs(fr) if len(fr) >= 2 else []
    if not runs:
        runs = [{'h': abs(fb[0] - fa[0]) >= abs(fb[1] - fa[1]), 'c': 0.0, 'w': 1.0,
                 'lo': 0.0, 'hi': 0.0}]
    # first and last stretches line up with a and b exactly (they move < one grid cell)
    runs[0]['c'] = fa[1] if runs[0]['h'] else fa[0]
    if len(runs) == 1:
        across_b = fb[1] if runs[0]['h'] else fb[0]
        if abs(across_b - runs[0]['c']) > 1e-3:          # a and b off one line: a jog
            r0 = runs[0]
            mid = ((fa[0] + fb[0]) / 2) if r0['h'] else ((fa[1] + fb[1]) / 2)
            runs = [r0, {'h': not r0['h'], 'c': mid, 'w': 0.0, 'lo': 0.0, 'hi': 0.0},
                    dict(r0, c=across_b)]
    else:
        runs[-1]['c'] = fb[1] if runs[-1]['h'] else fb[0]
    wps, _ = _corners(fa, fb, runs)
    if math.dist(wps[-1], fb) > 1e-3:
        wps.append(fb)
    fr = wps
    return _straight_on(_from_frame(fr, ang))


def _straight_on(pts, tol=1e-3):
    """Drop repeated points and middle points of straight stretches (a 180 deg waypoint)."""
    out = []
    for q in pts:
        if out and math.dist(q, out[-1]) < tol:
            continue
        out.append(q)
    k = 1
    while k < len(out) - 1:
        a, b, c = out[k - 1], out[k], out[k + 1]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        dot = (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1])
        if abs(cross) < 1e-6 * max(1.0, math.dist(a, b) * math.dist(b, c)) and dot > 0:
            del out[k]
        else:
            k += 1
    return out


def _even(pts, keep_last=False):
    """An even number of waypoints: if odd, the longest stretch gets a point in its middle
    (straight on: 180 deg). keep_last: never the last stretch (a straight lane's own two)."""
    if len(pts) % 2 == 0 or len(pts) < 2:
        return pts
    segs = range(len(pts) - 2 if keep_last and len(pts) > 2 else len(pts) - 1)
    k = max(segs, key=lambda i: math.dist(pts[i], pts[i + 1]))
    a, b = pts[k], pts[k + 1]
    return pts[:k + 1] + [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)] + pts[k + 1:]


def square_paths(m, spaces, paths, ang, keep_last=False, back_from=None):
    """Every path squared (square_path), each starting where the one before ended.
    Returns (paths, how many could not be squared)."""
    out, failed, at = [], 0, None
    for n, path in enumerate(paths):
        back = back_from is not None and n >= back_from
        lane_on_grid = len(path) >= 2 and _on_grid(path[-2], path[-1], ang)
        if keep_last and lane_on_grid:
            # a straight lane (already on the grid) stays as it is: square the way to it
            a, b = tuple(path[-2]), tuple(path[-1])
            way, ok = square_path_in(m, spaces, path[:-1], ang, at, back) if len(path) > 2 or \
                (at is not None and math.dist(at, a) > 1e-6) else ([a], True)
            if ok and math.dist(way[-1], a) > 1e-3:
                way = way + [a]                         # straight across onto the lane
            sq = _straight_on(list(way)) + [b] if ok else None
        else:
            sq, ok = square_path_in(m, spaces, path, ang, at, back)
        if not ok:
            failed += 1
            sq = ([at] if at is not None else []) + list(path[1:] if at is not None else path)
        if len(sq) < 2:
            continue
        sq = _even([tuple(map(float, q)) for q in sq], keep_last)
        out.append(sq)
        at = sq[-1]
    return out, failed


def _on_grid(a, b, ang, tol_deg=0.5):
    h = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]) - ang) % 90.0
    return min(h, 90.0 - h) < tol_deg


def square_turns(paths):
    """Waypoints where the route turns (90 or 180 deg), over all paths and between them."""
    route = []
    for p in paths:
        route.extend(p if not route else p[1:] if math.dist(p[0], route[-1]) < 1e-6 else p)
    return max(0, len(_straight_on(route)) - 2)


def best_square(m, spaces, paths, base_angle, spread=6, keep_last=False, back_from=None):
    """Squared paths in the frame (base_angle +- spread deg, 1 deg steps) with the fewest
    paths that could not be squared, then the fewest turns. Returns (paths, angle, failed)."""
    from securebot_ros.parallel import pmap
    angles = [base_angle + math.radians(d) for d in sorted(range(-spread, spread + 1), key=abs)]
    best = None
    for ang, (sq, failed) in zip(angles, pmap(_square_option, [
            (m, spaces, paths, ang, keep_last, back_from) for ang in angles])):
        key = (failed, square_turns(sq))
        if best is None or key < best[0]:
            best = (key, sq, ang, failed)
    return best[1], best[2], best[3]


def _square_option(job):
    """One best_square candidate: square_paths(...) in one frame."""
    m, spaces, paths, ang, keep_last, back_from = job
    return square_paths(m, spaces, paths, ang, keep_last, back_from)


# -------------------------------------------------------------------------- the way back
def route_back(paths, start, goal, merge=0.35, reach=3.0, clear=None, jump=3.0):
    """The shortest way from `start` to `goal` along the planned paths (they were checked
    clear on the map): the robot joins the nearest stretch, follows the lanes (points of
    different lanes within `merge` m count as one: junctions) and leaves at the stretch
    nearest `goal`. With `clear(a, b)` (the straight a-b is drivable), points up to `jump` m
    apart are joined too, so a detour into a side room and out again is skipped. None if
    start or goal is more than `reach` m from every stretch, or the two are not connected.
    Returns [start, ..., goal]."""
    import heapq
    pts, edges = [], []
    for path in paths:
        base = len(pts)
        pts.extend(tuple(map(float, q)) for q in path)
        edges.extend((base + k, base + k + 1) for k in range(len(path) - 1))
    if not edges:
        return None
    # junctions: points within `merge` m are one node
    parent = list(range(len(pts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    arr = np.array(pts)
    for i in range(len(pts)):
        near = np.nonzero(np.hypot(arr[i + 1:, 0] - arr[i, 0], arr[i + 1:, 1] - arr[i, 1])
                          <= merge)[0]
        for j in near + i + 1:
            parent[find(int(j))] = find(i)
    adj = {}

    def link(a, b, w=None):
        w = math.dist(pts[a], pts[b]) if w is None else w
        adj.setdefault(a, []).append((b, w))
        adj.setdefault(b, []).append((a, w))
    for a, b in edges:
        if find(a) != find(b):
            link(find(a), find(b))
    for i in range(len(pts)):
        if find(i) != i:                       # merged points: free hop to the junction
            link(i, find(i), math.dist(pts[i], pts[find(i)]))
    if clear is not None:                      # short clear hops between lanes: shortcuts
        roots = sorted({find(i) for i in range(len(pts))})
        for x, i in enumerate(roots):
            for j in roots[x + 1:]:
                d = math.dist(pts[i], pts[j])
                if merge < d <= jump and clear(pts[i], pts[j]):
                    link(i, j, d)

    def attach(q):
        """q as a node, joined to the nearest stretch it can reach straight and (with `clear`)
        to every route point within `reach` it can reach straight: the way can leave the
        lanes right where it heads for the goal (no hook past it). Index or None."""
        best = None
        for a, b in edges:
            (ax, ay), (bx, by) = pts[a], pts[b]
            dx, dy = bx - ax, by - ay
            l2 = dx * dx + dy * dy
            t = 0.0 if l2 < 1e-12 else max(0.0, min(1.0, ((q[0] - ax) * dx + (q[1] - ay) * dy)
                                                    / l2))
            p = (ax + t * dx, ay + t * dy)
            d = math.dist(q, p)
            if (best is None or d < best[0]) and d <= reach and \
                    (clear is None or d < 0.05 or clear(tuple(q), p)):
                best = (d, a, b, p)
        if best is None:
            return None
        _, a, b, p = best
        pts.append(tuple(map(float, q)))
        k = len(pts) - 1
        parent.append(k)
        pts.append(p)
        on = len(pts) - 1
        parent.append(on)
        link(on, find(a), math.dist(p, pts[find(a)]))
        link(on, find(b), math.dist(p, pts[find(b)]))
        link(k, on, math.dist(q, p))
        if clear is not None:
            for n in {find(i) for i in range(len(pts) - 2)}:
                d = math.dist(q, pts[n])
                if d <= reach and clear(tuple(q), pts[n]):
                    link(k, n, d)
        return k

    s, g = attach(start), attach(goal)
    if s is None or g is None:
        return None
    dist, came, queue = {s: 0.0}, {s: None}, [(0.0, s)]
    while queue:
        d, n = heapq.heappop(queue)
        if n == g:
            break
        if d > dist.get(n, math.inf):
            continue
        for o, w in adj.get(n, []):
            if d + w < dist.get(o, math.inf):
                dist[o], came[o] = d + w, n
                heapq.heappush(queue, (d + w, o))
    if g not in came:
        return None
    way, n = [], g
    while n is not None:
        way.append(pts[n])
        n = came[n]
    way = way[::-1]
    return _straight_on(_dedupe(way, gap=0.05))
