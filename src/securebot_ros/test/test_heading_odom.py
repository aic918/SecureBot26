"""LiDAR + gyro odometry fusion (no ROS)."""
import math

import pytest

from securebot_ros.heading_odom import HeadingOdom


def test_heading_comes_from_the_gyro_not_rf2o():
    h = HeadingOdom()
    h.update_imu(0.0, 1.0, 0.0)                     # gyro zero is arbitrary
    h.update_laser(0.0, 0.0, 0.0, 0.0)
    h.update_imu(0.1, 1.0 + math.pi / 2, 0.0)       # gyro: turned 90 deg left
    x, y, yaw = h.update_laser(0.1, 0.0, 0.0, 0.3)  # rf2o thinks only ~17 deg
    assert yaw == pytest.approx(math.pi / 2)


def test_rf2o_movement_is_turned_with_the_gyro_heading():
    h = HeadingOdom()
    h.update_imu(0.0, 0.0, 0.0)
    h.update_laser(0.0, 0.0, 0.0, 0.0)
    h.update_imu(0.1, math.pi / 2, 0.0)             # facing +y (gyro)
    h.update_laser(0.1, 0.0, 0.0, 0.0)              # rf2o missed the turn ...
    x, y, _ = h.update_laser(0.2, 1.0, 0.0, 0.0)    # ... and says "1 m forward" along its x
    assert (x, y) == pytest.approx((0.0, 1.0))      # forward is +y now


def test_spinning_on_the_spot_ignores_rf2o_translation():
    h = HeadingOdom()
    h.update_imu(0.0, 0.0, 0.0)
    h.update_laser(0.0, 0.0, 0.0, 0.0)
    h.update_imu(0.1, 0.3, 2.5)                     # spinning at 2.5 rad/s
    x, y, _ = h.update_laser(0.1, 0.4, -0.2, 0.3)   # scan-matching garbage
    assert (x, y) == (0.0, 0.0)
    h.update_imu(0.2, 0.3, 0.1)                     # stopped spinning, driving
    x, y, _ = h.update_laser(0.2, 0.9, -0.2, 0.3)   # 0.5 m forward in rf2o's frame (along x)
    assert math.hypot(x, y) == pytest.approx(0.5)


def test_falls_back_to_rf2o_heading_when_gyro_is_silent_and_continues_after():
    h = HeadingOdom(imu_timeout=0.5)
    h.update_imu(0.0, 0.0, 0.0)
    h.update_laser(0.0, 0.0, 0.0, 0.0)
    h.update_laser(1.0, 0.0, 0.0, 0.2)              # gyro silent for 1 s: rf2o turns 0.2
    assert h.yaw == pytest.approx(0.2)
    h.update_imu(1.1, 5.0, 0.0)                     # gyro back with some other value
    assert h.yaw == pytest.approx(0.2)              # no jump
    h.update_imu(1.2, 5.1, 0.0)
    assert h.yaw == pytest.approx(0.3)
