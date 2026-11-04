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
