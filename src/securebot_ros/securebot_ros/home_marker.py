"""The HOME mark in RViz: where START was pressed (Return home drives back there)."""
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

DIAMETER = 0.4572                       # m (1.5 ft)
COLOR = (0x3f / 255, 0x48 / 255, 0xcc / 255)   # #3f48cc
PATH_COLOR = (1.0, 0.0, 1.0)            # magenta: the way home being driven
PATH_WIDTH = 0.06                       # m


def home_markers(x, y, stamp, frame='map', ns='home'):
    """[disc, label]: a #3f48cc circle 1.5 ft across with white "HOME" on it."""
    disc = Marker()
    disc.header.frame_id, disc.header.stamp = frame, stamp
    disc.ns, disc.id, disc.type, disc.action = ns, 0, Marker.CYLINDER, Marker.ADD
    disc.pose.position.x, disc.pose.position.y, disc.pose.position.z = float(x), float(y), 0.01
    disc.pose.orientation.w = 1.0
    disc.scale.x = disc.scale.y = DIAMETER
    disc.scale.z = 0.02
    disc.color.r, disc.color.g, disc.color.b = COLOR
    disc.color.a = 1.0
    label = Marker()
    label.header = disc.header
    label.ns, label.id, label.type, label.action = ns, 1, Marker.TEXT_VIEW_FACING, Marker.ADD
    label.pose.position.x, label.pose.position.y, label.pose.position.z = float(x), float(y), 0.1
    label.pose.orientation.w = 1.0
    label.scale.z = 0.12
    label.color.r = label.color.g = label.color.b = label.color.a = 1.0
    label.text = 'HOME'
    return [disc, label]


def home_path_marker(path, stamp, frame='map', ns='home_path'):
    """A magenta line along `path` [(x, y), ...] (the way home); DELETE if it has < 2 points."""
    line = Marker()
    line.header.frame_id, line.header.stamp = frame, stamp
    line.ns, line.id, line.type = ns, 0, Marker.LINE_STRIP
    if len(path) < 2:
        line.action = Marker.DELETE
        return line
    line.action = Marker.ADD
    line.pose.orientation.w = 1.0
    line.scale.x = PATH_WIDTH
    line.color.r, line.color.g, line.color.b = PATH_COLOR
    line.color.a = 1.0
    line.points = [Point(x=float(x), y=float(y), z=0.04) for x, y in path]
    return line
