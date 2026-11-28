"""Pure logic of esp32_serial_bridge (no serial port, no ROS spin)."""
import math

from securebot_base.esp32_serial_bridge import (
    choose_command, parse_line, twist_to_pwm, twist_to_tps)

PWM_ARGS = (0.35, 1.5, 0.22, 150)       # max m/s, max rad/s, wheel base, max PWM
T = 0.5


def test_twist_to_pwm_straight_and_clamped():
    assert twist_to_pwm(0.35, 0.0, *PWM_ARGS) == (150, 150)
    assert twist_to_pwm(10.0, 0.0, *PWM_ARGS) == (150, 150)
    assert twist_to_pwm(-0.175, 0.0, *PWM_ARGS) == (-75, -75)


def test_twist_to_pwm_turn_in_place_left_wheel_backwards():
    left, right = twist_to_pwm(0.0, 1.0, *PWM_ARGS)
    assert left == -right and right > 0


def test_stop_wins_and_stale_mode_stops():
    assert choose_command('stop', 0.0, (0.3, 0.0), 0.0, T, PWM_ARGS) == 'stop'
    assert choose_command('patrol', 0.6, None, math.inf, T, PWM_ARGS) == 'stop'
    assert choose_command('garbage', 0.0, None, math.inf, T, PWM_ARGS) == 'stop'


def test_patrol_ignores_cmd_vel():
    assert choose_command('patrol', 0.1, (0.3, 0.0), 0.0, T, PWM_ARGS) == 'patrol'


def test_drive_uses_fresh_cmd_vel_and_zeroes_stale_one():
    assert choose_command('drive', 0.1, (0.35, 0.0), 0.1, T, PWM_ARGS) == 'M,150,150'
    assert choose_command('drive', 0.1, (0.35, 0.0), 0.6, T, PWM_ARGS) == 'M,0,0'
    assert choose_command('drive', 0.1, None, math.inf, T, PWM_ARGS) == 'M,0,0'


def test_parse_lines():
    assert parse_line('U,1.25,-1.00') == ('U', 1.25, -1.0)
    assert parse_line('W,655,-526') == ('W', 655, -526)
    assert parse_line('W,1.5,2') is None
    assert parse_line('S,drive,90,90,0') == ('S', 'drive,90,90,0')
    assert parse_line('E,watchdog: no command, stopped') == ('E', 'watchdog: no command, stopped')
    assert parse_line('U,abc,1') is None
    assert parse_line('G,-12.50,3.25') == ('G', -12.5, 3.25)
    assert parse_line('G,1') is None
    assert parse_line('LEFT: 9.28 ft') is None


def test_parse_battery_and_tamper_lines():
    assert parse_line('B,25200,1500') == ('B', 25.2, 1.5)
    assert parse_line('B,24000,-100') == ('B', 24.0, -0.1)
    assert parse_line('B,25.2,1') is None
    assert parse_line('B,1') is None
    assert parse_line('T,armed') == ('T', 'armed')
    assert parse_line('T,tripped,tilt') == ('T', 'tripped,tilt')


def test_min_pwm_lifts_small_commands_but_not_zero():
    args = PWM_ARGS + (70,)
    assert twist_to_pwm(0.0, 0.0, *args) == (0, 0)
    assert twist_to_pwm(0.05, 0.0, *args) == (70, 70)          # 21 -> 70
    assert twist_to_pwm(0.0, 0.8, *args) == (-70, 70)          # in-place turn, +/-38 -> +/-70
    assert twist_to_pwm(0.3, 0.0, *args) == (129, 129)         # already above the floor
    left, right = twist_to_pwm(0.06, 0.4, *args)               # gentle left curve, slow
    assert right == 70 and 0 < left < right                    # still curves left
    assert twist_to_pwm(-0.05, 0.0, *args) == (-70, -70)


def test_imu_message_heading_only():
    import math
    from securebot_base.esp32_serial_bridge import imu_message
    m = imu_message(math.pi / 2, 0.5)
    assert m.header.frame_id == 'base_link'
    assert abs(2 * math.atan2(m.orientation.z, m.orientation.w) - math.pi / 2) < 1e-9
    assert m.angular_velocity.z == 0.5
    assert m.linear_acceleration_covariance[0] == -1.0


def test_slow_requests_are_lifted_only_to_the_gentle_pwm():
    args = (0.35, 1.5, 0.44, 660, 660, 610, 250)
    left, right = twist_to_pwm(0.0, 0.45, *args)              # correction spin: ~190 raw
    assert (left, right) == (-610, 610)
    assert twist_to_pwm(0.0, 0.8, *args) == (-660, 660)       # normal turn: ~332 raw
    assert twist_to_pwm(0.2, 0.0, *args) == (660, 660)        # driving: ~377 raw
    assert twist_to_pwm(0.0, 0.45, *args[:5]) == (-660, 660)  # gentle band off


def test_slow_approach_with_steering_stays_gentle():
    args = (0.35, 1.5, 0.44, 660, 660, 610, 300)
    left, right = twist_to_pwm(0.1, 0.2, *args)               # approach + a correction
    assert max(abs(left), abs(right)) == 610
    assert twist_to_pwm(0.2, 0.0, *args) == (660, 660)        # normal driving
    assert twist_to_pwm(0.0, 0.8, *args) == (-660, 660)       # normal turn


# ------------------------------------------------------------------------------ speed control
SPEED_ARGS = (0.35, 1.5, 0.44, 1150.0, 40)     # max lin, max ang, wheel base, ticks/m, min t/s


def test_twist_to_wheel_speeds():
    assert twist_to_tps(0.2, 0.0, *SPEED_ARGS) == (230, 230)
    assert twist_to_tps(-0.2, 0.0, *SPEED_ARGS) == (-230, -230)
    left, right = twist_to_tps(0.0, 0.8, *SPEED_ARGS)          # spin left: 0.176 m/s per wheel
    assert (left, right) == (-202, 202)
    assert twist_to_tps(0.2, -0.31, *SPEED_ARGS) == (308, 152)  # a curve: both wheels turn


def test_slow_wheels_are_lifted_to_the_minimum_and_clamped():
    assert twist_to_tps(0.01, 0.0, *SPEED_ARGS) == (40, 40)
    assert twist_to_tps(0.0, 0.0, *SPEED_ARGS) == (0, 0)
    assert twist_to_tps(5.0, 0.0, *SPEED_ARGS) == (402, 402)    # max_linear 0.35 m/s


def test_speed_control_sends_v_commands():
    assert choose_command('drive', 0.1, (0.2, 0.0), 0.1, T, PWM_ARGS, SPEED_ARGS) == 'V,230,230'
    assert choose_command('drive', 0.1, (0.2, 0.0), 0.6, T, PWM_ARGS, SPEED_ARGS) == 'V,0,0'
    assert choose_command('stop', 0.1, (0.2, 0.0), 0.1, T, PWM_ARGS, SPEED_ARGS) == 'stop'
