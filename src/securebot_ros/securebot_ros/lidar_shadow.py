"""Where the robot blinds its own LiDAR (camera tower, brackets), measured instead of assumed
(no ROS).

A sector without returns is either the robot's shadow or just open space (a doorway, a hallway
longer than the LiDAR's range). To tell them apart the robot turns once on the spot while scans
are collected: the shadow turns with the robot and stays blind at the same LiDAR angle the whole
time, while open space sweeps past every angle, so every angle sees walls somewhere during the
turn. What stays blind all the way round is the shadow. Parts of the robot itself (closer than
`self_range`, at the same angle in every scan) are masked too.

The result is written as the angular filters of config/lidar_filters.yaml (laser frame, ROS
angles: radians, counter-clockwise, -pi..pi), with `margin` degrees on each side.
"""
import math

import numpy as np

MIN_RANGE = 0.20        # m: closer returns are cut by the self_hits range filter anyway
MAX_RANGE = 12.0        # m
SELF_RANGE = 0.35       # m: a return this close at the same angle all the way round = the robot
BLIND_SHARE = 0.05      # an angle with returns in fewer scans than this is blind
SELF_SHARE = 0.8        # an angle with a near return in more scans than this is the robot
MIN_WIDTH = 3           # deg: narrower blind runs are noise
MARGIN = 3              # deg added on each side of a masked sector


def angle_stats(scans):
    """Per 1 deg of laser angle (index 0 = -180 deg ... 359 = +179 deg): share of scans with a
    return in [MIN_RANGE, MAX_RANGE], and share with a return closer than SELF_RANGE.
    `scans`: [(ranges, angle_min, angle_increment)]."""
    seen = np.zeros(360)
    near = np.zeros(360)
    for ranges, amin, ainc in scans:
        r = np.asarray(ranges, dtype=float)
        a = np.degrees(amin + np.arange(len(r)) * ainc)
        idx = (np.floor(a + 180.0).astype(int)) % 360
        ok = np.isfinite(r) & (r >= MIN_RANGE) & (r <= MAX_RANGE)
        close = ok & (r < SELF_RANGE)
        s = np.zeros(360, bool)
        n = np.zeros(360, bool)
        s[idx[ok]] = True
        n[idx[close]] = True
        seen += s
        near += n
    count = max(len(scans), 1)
    return seen / count, near / count


def masked_degrees(seen, near, blind_share=BLIND_SHARE, self_share=SELF_SHARE):
    """Indices (0..359, 0 = -180 deg) to mask: blind all the way round, or always the robot."""
    return {i for i in range(360) if seen[i] < blind_share or near[i] > self_share}


def sectors(degrees, min_width=MIN_WIDTH):
    """Contiguous runs of masked indices (wrapping past 359 -> 0) as (first, last) indices,
    dropping runs narrower than min_width."""
    deg = set(degrees)
    if not deg:
        return []
    if len(deg) == 360:
        return [(0, 359)]
    start = next(i for i in range(360) if i not in deg)       # begin outside a run
    out, run = [], None
    for k in range(1, 361):
        i = (start + k) % 360
        if i in deg:
            run = [i, i] if run is None else [run[0], i]
        elif run is not None:
            width = (run[1] - run[0]) % 360 + 1
            if width >= min_width:
                out.append((run[0], run[1]))
            run = None
    return out


def to_radian_bounds(sectors_idx, margin=MARGIN):
    """(first, last) index sectors -> [(lower, upper)] in radians, -pi..pi, `margin` deg wider
    on each side; a sector crossing +-180 deg is split in two (the filter cannot wrap)."""
    out = []
    for first, last in sectors_idx:
        lo = first - 180 - margin                 # index 0 = -180 deg; a bin spans 1 deg
        hi = last - 180 + 1 + margin
        width = (hi - lo) if hi >= lo else hi - lo + 360
        if width >= 360:
            return [(-math.pi, math.pi)]
        lo = (lo + 180) % 360 - 180
        hi = (hi + 180) % 360 - 180
        if lo < hi:
            out.append((math.radians(lo), math.radians(hi)))
        else:                                     # crosses +-180
            out.append((math.radians(lo), math.pi))
            out.append((-math.pi, math.radians(hi)))
    return out


def filters_yaml(bounds, note):
    """The whole lidar_filters.yaml: one angular filter per bound, then the self_hits filter."""
    lines = [
        '# Filter chain for the RPLidar A1M8: /scan_raw (driver) -> /scan (everything else).',
        '#',
        '# Angles: laser frame, ROS convention: radians, counter-clockwise, 0 = laser +x,',
        '# -pi..pi. The masked sectors are where the robot blinds its own LiDAR (camera',
        '# tower, brackets), measured, not guessed: securebot_find_shadow.launch.py turns the',
        '# robot once on the spot and masks what stayed blind (or always hit the robot) all',
        f'# the way round, with {MARGIN} deg of margin. Re-run it (save:=true) whenever the',
        '# tower or the LiDAR moves.',
        f'# {note}',
        'lidar_filters:',
        '  ros__parameters:',
    ]
    n = 0
    for lo, hi in bounds:
        n += 1
        lines += [
            f'    filter{n}:',
            f'      name: robot_shadow_{n}            # {math.degrees(lo):.0f} deg .. '
            f'{math.degrees(hi):.0f} deg',
            '      type: laser_filters/LaserScanAngularBoundsFilterInPlace',
            '      params:',
            f'        lower_angle: {lo:.4f}',
            f'        upper_angle: {hi:.4f}',
        ]
    n += 1
    lines += [
        f'    filter{n}:',
        '      name: self_hits                     # anything this close is the robot itself',
        '      type: laser_filters/LaserScanRangeFilter',
        '      params:',
        '        use_message_range_limits: false',
        f'        lower_threshold: {MIN_RANGE:.2f}',
        f'        upper_threshold: {MAX_RANGE:.1f}',
        '        lower_replacement_value: .inf     # "nothing seen", not an obstacle at 0 m',
        '        upper_replacement_value: .inf',
    ]
    return '\n'.join(lines) + '\n'


def describe(bounds):
    if not bounds:
        return 'nothing masked (the LiDAR sees all round)'
    return ', '.join(f'{math.degrees(lo):.0f}..{math.degrees(hi):.0f} deg' for lo, hi in bounds)
