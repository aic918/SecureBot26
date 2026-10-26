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
