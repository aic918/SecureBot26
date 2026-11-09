from builtin_interfaces.msg import Time
from securebot_ros.home_marker import home_path_marker
from visualization_msgs.msg import Marker


def test_home_path_is_a_magenta_line():
    m = home_path_marker([(0.0, 0.0), (1.0, 0.0), (1.0, 2.0)], Time())
    assert m.type == Marker.LINE_STRIP and m.action == Marker.ADD
    assert (m.color.r, m.color.g, m.color.b, m.color.a) == (1.0, 0.0, 1.0, 1.0)
    assert [(p.x, p.y) for p in m.points] == [(0.0, 0.0), (1.0, 0.0), (1.0, 2.0)]


def test_no_path_deletes_the_line():
    assert home_path_marker([], Time()).action == Marker.DELETE
    assert home_path_marker([(1.0, 1.0)], Time()).action == Marker.DELETE
