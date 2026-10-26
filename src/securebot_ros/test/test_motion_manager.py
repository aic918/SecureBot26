"""Decision rules of motion_manager_node."""
import math

from securebot_ros.motion_manager_node import decide

ST, NT = 1.0, 0.5


def test_security_stop_or_silence_stops():
    assert decide('stop', 0.0, 0.0, ST, NT) == 'stop'
    assert decide('patrol', 1.5, 0.0, ST, NT) == 'stop'        # vision node went quiet
    assert decide('patrol', math.inf, math.inf, ST, NT) == 'stop'


def test_patrol_without_planner_uses_firmware_patrol():
    assert decide('patrol', 0.05, math.inf, ST, NT) == 'patrol'
    assert decide('patrol', 0.05, 0.8, ST, NT) == 'patrol'     # planner stopped publishing


def test_fresh_planner_command_drives():
    assert decide('patrol', 0.05, 0.1, ST, NT) == 'drive'


def test_keyboard_mode_ignores_security_and_stops_when_idle():
    kb = dict(require_security=False, idle_action='stop')
    assert decide('stop', math.inf, 0.1, ST, NT, **kb) == 'drive'
    assert decide('stop', math.inf, 0.8, ST, NT, **kb) == 'stop'    # keys released
    assert decide('stop', math.inf, math.inf, ST, NT, **kb) == 'stop'


def test_nav2_mode_needs_security_and_never_patrols():
    nav = dict(require_security=True, idle_action='stop')
    assert decide('patrol', 0.05, 0.1, ST, NT, **nav) == 'drive'
    assert decide('patrol', 0.05, math.inf, ST, NT, **nav) == 'stop'
    assert decide('stop', 0.05, 0.1, ST, NT, **nav) == 'stop'
