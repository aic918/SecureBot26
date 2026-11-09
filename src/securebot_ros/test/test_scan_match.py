"""Scan-to-map pose correction on a synthetic room and on Kumar's Lab (no ROS)."""
import math
import os

import cv2
import numpy as np
import pytest

from securebot_ros.scan_match import ScanMatcher

RES = 0.05
LAB = os.path.expanduser('~/securebot_ws/maps/explore_1026_1922.yaml')


def room_image():
    """6 x 4 m room (walls 0, free 254) with a desk, origin (0, 0) at the bottom-left."""
    img = np.full((int(4 / RES), int(6 / RES)), 254, np.uint8)
    img[0, :] = img[-1, :] = 0
    img[:, 0] = img[:, -1] = 0
    img[30:40, 40:70] = 0                     # a desk, so the room is not symmetric
    return img


def fake_scan(img, pose, n=360, max_range=8.0):
    """Ray-cast scan points (robot frame) from `pose` in the image."""
    x, y, yaw = pose
    rows = img.shape[0]
    pts = []
    for k in range(n):
        a = 2 * math.pi * k / n
        for r in np.arange(0.1, max_range, RES / 2):
            wx, wy = x + r * math.cos(yaw + a), y + r * math.sin(yaw + a)
            row, col = int(rows - wy / RES), int(wx / RES)
            if not (0 <= row < rows and 0 <= col < img.shape[1]) or img[row, col] == 0:
                pts.append((r * math.cos(a), r * math.sin(a)))
                break
    return np.array(pts)


@pytest.fixture(scope='module')
def room():
    img = room_image()
    truth = (2.0, 1.2, math.radians(30))
    return ScanMatcher(img, RES, (0.0, 0.0)), fake_scan(img, truth), truth


def test_true_pose_matches_and_needs_no_correction(room):
    m, scan, truth = room
    score, better = m.correct(truth, scan)
    assert score > 0.9 and better is None


def test_drifted_pose_is_pulled_back(room):
    m, scan, truth = room
    drifted = (truth[0] + 0.25, truth[1] - 0.2, truth[2] + math.radians(8))
    score, better = m.correct(drifted, scan)
    assert score < 0.6 and better is not None
    assert math.hypot(better[0] - truth[0], better[1] - truth[1]) < 0.06
    assert abs(better[2] - truth[2]) < math.radians(3)


def test_no_correction_without_a_good_match(room):
    m, _, truth = room
    junk = np.random.default_rng(1).uniform(-3, 3, (200, 2))
    assert m.correct(truth, junk)[1] is None
    assert m.correct(truth, [])[1] is None


@pytest.mark.skipif(not os.path.exists(LAB), reason="Kumar's Lab map not on this machine")
def test_lab_map_loads():
    img = cv2.imread(LAB.replace('.yaml', '.pgm'), 0)
    m = ScanMatcher(img, 0.05, (-7.69, -9.55))
    assert m.dist.shape == img.shape and m.dist.max() > 1.0


def test_a_big_correction_needs_a_strong_match(room):
    """2026-11-07: a correction over 20 cm or 5 deg is taken only from a strong match."""
    m, scan, truth = room
    drifted = (truth[0] + 0.25, truth[1] - 0.2, truth[2] + math.radians(8))
    assert m.correct(drifted, scan)[1] is not None             # strong match: taken
    assert m.last_best > 0.85
    assert m.correct(drifted, scan, big_score=1.01)[1] is None  # not strong enough: left
    small = (truth[0] + 0.1, truth[1], truth[2])
    assert m.correct(small, scan, big_score=1.01, min_gain=0.0)[1] is not None


def test_lost_robot_is_found_metres_away():
    """2026-11-08 (hallway10): AMCL 3.4 m off; the map search finds the robot again. (The
    plain room above with one desk is nearly the same turned round: 93% there, too close
    to call; an L-shaped room makes it clear, as real rooms are.)"""
    img = room_image()
    img[0:30, 90:120] = 0                     # top-right 1.5 x 1.5 m corner walled off
    truth = (3.5, 3.2, math.radians(60))     # where the walled-off corner is in view
    m, scan = ScanMatcher(img, RES, (0.0, 0.0)), fake_scan(img, truth)
    lost = (truth[0] - 2.5, truth[1] - 1.5, truth[2] - math.radians(40))
    assert m.correct(lost, scan)[1] is None              # too far for the local check
    best, pose, second = m.relocalize(scan, around=lost[:2], radius=6.0)
    assert best > 0.9 and best - second >= 0.15
    assert math.hypot(pose[0] - truth[0], pose[1] - truth[1]) < 0.08
    assert abs(math.atan2(math.sin(pose[2] - truth[2]), math.cos(pose[2] - truth[2]))) \
        < math.radians(4)


def test_plain_corridor_is_ambiguous():
    """In a long corridor with nothing along it, every spot along it fits as well: the
    search must say so (no clear runner-up gap) instead of picking one."""
    img = np.full((int(2 / RES), int(16 / RES)), 254, np.uint8)
    img[0, :] = img[-1, :] = 0
    img[:, 0] = img[:, -1] = 0
    m = ScanMatcher(img, RES, (0.0, 0.0))
    scan = fake_scan(img, (8.0, 1.0, 0.0), max_range=3.0)     # sees only the side walls
    best, pose, second = m.relocalize(scan, around=(8.0, 1.0), radius=6.0)
    assert best - second < 0.15


def test_spots_stay_off_walls_and_inside_the_radius(room):
    m, _, _ = room
    xy = m.spots(0.1, around=(3.0, 2.0), radius=1.0)
    assert len(xy) and np.all(np.hypot(xy[:, 0] - 3.0, xy[:, 1] - 2.0) <= 1.0)
    rows = (m.rows - xy[:, 1] / RES).astype(int)
    cols = (xy[:, 0] / RES).astype(int)
    assert np.all(m.dist[rows, cols] >= 0.25 - RES)
