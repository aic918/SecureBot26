"""Which way the LiDAR faces on the robot, measured instead of assumed (no ROS).

The robot drives a short, straight distance forward ("forward" = what the motors and the wheel
encoders call forward). The LiDAR moved with it, so between a scan before and a scan after, the
whole room shifted the other way in the LiDAR's own frame. Aligning the two scans gives the
LiDAR's movement (tx, ty) in its own frame; the robot's forward axis points that way, so

    laser_yaw = -atan2(ty, tx)

(laser_yaw: the turn from base_link to the laser frame, as in securebot_lidar.launch.py). This
needs nothing but walls in range: no blind spot, tower or marker to recognise.
"""
import math

import cv2
import numpy as np

CELL = 0.01                 # m: raster cell for the alignment
SEARCH = 0.8                # m: largest movement searched for, each way
COARSE = 0.02               # m: first grid step
MIN_RANGE = 0.25            # m: closer returns are the robot itself
MAX_RANGE = 8.0             # m: farther ones are noisy
SATURATE = 0.3              # m: distance to the nearest scan point is capped here (outliers)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def scan_points(ranges, angle_min, angle_increment, min_range=MIN_RANGE, max_range=MAX_RANGE):
    """LaserScan ranges -> Nx2 points in the laser frame (x along angle 0, y at +90 deg)."""
    r = np.asarray(ranges, dtype=float)
    a = angle_min + np.arange(len(r)) * angle_increment
    ok = np.isfinite(r) & (r >= min_range) & (r <= max_range)
    return np.column_stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])])


def nearest_clearance(points):
    """Distance (m) from the LiDAR to the nearest point, or inf."""
    return float(np.min(np.hypot(points[:, 0], points[:, 1]))) if len(points) else math.inf


def _distance_field(points):
    """Raster of 'distance to the nearest reference point' (m, capped), and its origin."""
    lo = points.min(axis=0) - SEARCH - SATURATE
    hi = points.max(axis=0) + SEARCH + SATURATE
    w, h = (np.ceil((hi - lo) / CELL).astype(int) + 1)
    occ = np.full((h, w), 255, np.uint8)
    idx = ((points - lo) / CELL).astype(int)
    occ[idx[:, 1], idx[:, 0]] = 0
    dist = cv2.distanceTransform(occ, cv2.DIST_L2, 5) * CELL
    return np.minimum(dist, SATURATE), lo


def _score(field, lo, pts, shifts):
    """Mean capped distance of `pts` moved by each shift (Kx2): lower = better aligned."""
    h, w = field.shape
    out = np.empty(len(shifts))
    for k, (dx, dy) in enumerate(shifts):
        idx = ((pts + (dx, dy) - lo) / CELL).astype(int)
        ix = np.clip(idx[:, 0], 0, w - 1)
        iy = np.clip(idx[:, 1], 0, h - 1)
        out[k] = field[iy, ix].mean()
    return out


def estimate_motion(before, after, rotation=0.0):
    """The LiDAR's movement (tx, ty) between two scans, in the frame of the first (m), and the
    alignment error (mean distance, m; a few cm when the scans really match). `rotation`: how
    much the LiDAR turned (rad, counter-clockwise; from the gyro), applied as known."""
    field, lo = _distance_field(before)
    c, s = math.cos(rotation), math.sin(rotation)
    rotated = after @ np.array([[c, s], [-s, c]])       # after's points, turned into before's axes
    steps = np.arange(-SEARCH, SEARCH + 1e-9, COARSE)
    grid = np.array([(x, y) for x in steps for y in steps])
    scores = _score(field, lo, rotated, grid)
    best = grid[int(np.argmin(scores))]
    fine = np.arange(-2 * COARSE, 2 * COARSE + 1e-9, CELL / 2)
    grid = np.array([(best[0] + x, best[1] + y) for x in fine for y in fine])
    scores = _score(field, lo, rotated, grid)
    k = int(np.argmin(scores))
    return float(grid[k][0]), float(grid[k][1]), float(scores[k])


def laser_yaw_from_motion(tx, ty):
    """laser_yaw (rad) from the LiDAR's movement in its own frame while the robot drove
    forward."""
    return wrap(-math.atan2(ty, tx))


def check(forward, backward, driven, max_angle=math.radians(15), max_error=0.05):
    """Judge a measurement: forward / backward = (tx, ty, error) of the drive out and the drive
    back, `driven` the distance the encoders measured (m). Returns (laser_yaw, problems)."""
    problems = []
    yaw_f = laser_yaw_from_motion(forward[0], forward[1])
    yaw_b = laser_yaw_from_motion(-backward[0], -backward[1])     # back: the opposite way
    for name, (tx, ty, err) in (('out', forward), ('back', backward)):
        moved = math.hypot(tx, ty)
        if err > max_error:
            problems.append(f'the scans before and after the drive {name} do not match well '
                            f'({err * 100:.1f} cm): too few walls in range, or something moved')
        if driven > 0 and not 0.6 < moved / driven < 1.5:
            problems.append(f'the LiDAR saw {moved:.2f} m on the drive {name} but the wheels '
                            f'{driven:.2f} m')
    if abs(wrap(yaw_f - yaw_b)) > max_angle:
        problems.append(f'the drive out and back disagree: {math.degrees(yaw_f):.0f} vs '
                        f'{math.degrees(yaw_b):.0f} deg')
    yaw = math.atan2(math.sin(yaw_f) + math.sin(yaw_b), math.cos(yaw_f) + math.cos(yaw_b))
    return yaw, problems
