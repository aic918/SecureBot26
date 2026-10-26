"""Key handling of teleop_keyboard (no terminal, no ROS)."""
from securebot_ros.teleop_keyboard import TeleopState


def state():
    return TeleopState(linear=0.15, angular=0.6, max_linear=0.25, max_angular=1.0, hold_time=0.7)


def test_nothing_pressed_sends_zero():
    assert state().command(0.0) == (0.0, 0.0)


def test_held_key_moves_and_release_stops_after_hold_time():
    s = state()
    s.key('w', 10.0)
    assert s.command(10.5) == (0.15, 0.0)
    assert s.command(10.8) == (0.0, 0.0)          # no repeat for > 0.7 s: key released


def test_turns_and_curves():
    s = state()
    s.key('a', 0.0)
    assert s.command(0.1) == (0.0, 0.6)           # left = counter-clockwise = +z
    s.key('e', 0.2)
    assert s.command(0.3) == (0.15, -0.6)


def test_space_stops_immediately_and_x_quits():
    s = state()
    s.key('w', 0.0)
    s.key(' ', 0.1)
    assert s.command(0.2) == (0.0, 0.0)
    assert s.key('x', 0.3) is False
    assert s.key('\x03', 0.3) is False


def test_speed_keys_stay_within_limits():
    s = state()
    for _ in range(20):
        s.key('+', 0.0)
    assert (s.linear, s.angular) == (0.25, 1.0)
    s.key('-', 0.0)
    assert abs(s.linear - 0.2) < 1e-9


def test_unknown_key_changes_nothing():
    s = state()
    s.key('w', 0.0)
    assert s.key('z', 0.1) is True
    assert s.command(0.2) == (0.15, 0.0)
