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


# ------------------------------------------------------------------------------ wheel encoders
def test_wheels_give_the_distance_along_the_gyro_heading():
    h = HeadingOdom(ticks_per_meter=1000.0)
    h.update_imu(0.0, 0.0, 0.0)                      # the first gyro reading is heading 0
    h.update_imu(0.05, math.pi / 2, 0.0)             # turned to face +y
    h.update_wheels(0.05, 0, 0)
    x, y, yaw = h.update_wheels(0.1, 500, 300)       # wheels: 0.5 m and 0.3 m -> 0.4 m forward
    assert (x, y) == pytest.approx((0.0, 0.4))
    assert yaw == pytest.approx(math.pi / 2)         # heading from the gyro: the wheels
    #                                                  disagree (0.2 m difference = a turn)
    assert h.distance == pytest.approx(0.4)


def test_rf2o_distance_is_ignored_while_the_wheels_report():
    h = HeadingOdom(ticks_per_meter=1000.0)
    h.update_imu(0.0, 0.0, 0.0)
    h.update_wheels(0.0, 0, 0)
    h.update_laser(0.0, 0.0, 0.0, 0.0)
    for i in range(1, 11):                           # 1 m by the wheels, 0.1 m per update
        h.update_wheels(0.01 * i, 100 * i, 100 * i)
    h.update_imu(0.1, 0.0, 0.0)
    x, y, _ = h.update_laser(0.1, 0.3, 0.0, 0.0)     # rf2o, lost in a hallway: only 0.3 m
    assert (x, y) == pytest.approx((1.0, 0.0))


def test_rf2o_takes_over_when_the_wheels_go_silent():
    h = HeadingOdom(ticks_per_meter=1000.0, wheel_timeout=0.5)
    h.update_imu(0.0, 0.0, 0.0)
    h.update_wheels(0.0, 0, 0)
    h.update_laser(0.0, 0.0, 0.0, 0.0)
    h.update_imu(2.0, 0.0, 0.0)
    x, _, _ = h.update_laser(2.0, 0.5, 0.0, 0.0)     # no encoder update for 2 s
    assert x == pytest.approx(0.5)


def test_an_esp32_restart_only_rebases_the_counts():
    h = HeadingOdom(ticks_per_meter=1000.0)
    h.update_imu(0.0, 0.0, 0.0)
    h.update_wheels(0.0, 40000, 41000)
    h.update_wheels(0.05, 40100, 41100)              # 0.1 m
    h.update_wheels(0.10, 0, 0)                      # restarted: counts back to 0
    x, _, _ = h.update_wheels(0.15, 100, 100)        # 0.1 m more
    assert x == pytest.approx(0.2)


def test_without_ticks_per_meter_the_wheels_are_not_used():
    h = HeadingOdom()
    h.update_imu(0.0, 0.0, 0.0)
    h.update_wheels(0.0, 0, 0)
    h.update_laser(0.0, 0.0, 0.0, 0.0)
    h.update_wheels(0.1, 1000, 1000)
    h.update_imu(0.1, 0.0, 0.0)
    x, _, _ = h.update_laser(0.1, 0.3, 0.0, 0.0)
    assert x == pytest.approx(0.3)
