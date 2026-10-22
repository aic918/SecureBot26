"""Pure logic of esp32_serial_bridge (no serial port, no ROS spin)."""
import math

from securebot_base.esp32_serial_bridge import choose_command, parse_line, twist_to_pwm

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
    assert parse_line('S,drive,90,90,0') == ('S', 'drive,90,90,0')
    assert parse_line('E,watchdog: no command, stopped') == ('E', 'watchdog: no command, stopped')
    assert parse_line('U,abc,1') is None
    assert parse_line('LEFT: 9.28 ft') is None
