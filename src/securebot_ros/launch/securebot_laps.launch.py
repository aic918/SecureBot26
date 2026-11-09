"""Run coverage laps round a saved room, with Start / Stop buttons.

  ros2 launch securebot_ros securebot_laps.launch.py room:="Kumar's Lab"
  ros2 launch securebot_ros securebot_laps.launch.py map:=$HOME/securebot_ws/maps/<name>.yaml

Starts: navigation on the room's saved map (LiDAR, laser odometry, AMCL, Nav2, RViz), the
robot side in keyboard mode (ESP32 bridge + motion manager: no camera needed, stops when
nothing is commanded), the laps node and the "SecureBot Laps" window. The room name is looked
up in the Rooms window's list (maps/rooms.json). Once AMCL is up, the robot is found on the
map from one LiDAR scan (auto_localize:=true, tools/global_localize.py), so it can start
anywhere in the room; if no clear match is found, use "2D Pose Estimate" in RViz.
auto_localize:=false: AMCL starts at the map's origin (where mapping started).
Nothing moves until you press START; STOP (or closing the window, or Ctrl+C here) stops it.

pattern:=lanes (default) drives straight parallel lanes along the room's main walls, keeping
side_clearance beside things and front_clearance ahead at lane ends; pattern:=contour drives
rings at wall_clearance from the walls, outside in.

rounds:=0 keeps lapping until Stop. Don't run it together with the live, teleop, explore or
other nav launches (only one program can own the ESP32 port).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, GroupAction, \
    IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from securebot_ros import room_store as rs


def resolve_map(room, map_yaml, maps_dir):
    """The .yaml to use: map:= as given, else the map named room:= in the Rooms list."""
    if map_yaml:
        return os.path.abspath(os.path.expanduser(map_yaml))
    if not room:
        raise RuntimeError('Give room:="<name>" (see the Rooms window) or map:=<file.yaml>')
    store = rs.RoomStore(os.path.expanduser(maps_dir))
    entry = store.find(room)
    if entry is None:
        raise RuntimeError(f'No room called "{room}" in {store.path}. Rooms: '
                           + (', '.join(store.names()) or 'none yet'))
    path = os.path.join(store.maps_dir, entry['map'] + '.yaml')
    if not store.map_exists(entry['map']):
        raise RuntimeError(f'The map for "{entry["name"]}" is missing: {path}')
    return path


def launch_setup(context):
    get = lambda n: context.launch_configurations[n]  # noqa: E731
    map_yaml = resolve_map(get('room'), get('map'), get('maps_dir'))
    share = get_package_share_directory('securebot_ros')
    launch = lambda name: os.path.join(share, 'launch', name)  # noqa: E731
    return [
        GroupAction(scoped=True, actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch('securebot_nav.launch.py')),
            launch_arguments={'map': map_yaml, 'rviz': get('rviz'),
                              'imu': get('imu')}.items())]),
        GroupAction(scoped=True, actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch('securebot_teleop.launch.py')),
            launch_arguments={'port': get('esp32_port')}.items())]),
        Node(package='securebot_ros', executable='laps_node', output='screen',
             parameters=[{'map': map_yaml, 'room': get('room') or os.path.basename(map_yaml),
                          'rounds': int(get('rounds')),
                          'lane_spacing': float(get('lane_spacing')),
                          'wall_clearance': float(get('wall_clearance')),
                          'pattern': get('pattern'),
                          'side_clearance': float(get('side_clearance')),
                          'front_clearance': float(get('front_clearance')),
                          'lane_gap': float(get('lane_gap')),
                          'min_pocket': float(get('min_pocket')),
                          'require_gyro': get('imu') == 'true',
                          'home': get('home')}]),
        Node(package='securebot_ros', executable='laps_panel', output='screen'),
    ] + auto_localize(get('auto_localize'), map_yaml)


GLOBAL_LOCALIZE = os.path.expanduser('~/securebot_ws/tools/global_localize.py')


def auto_localize(enabled, map_yaml):
    """Once AMCL is up, find the robot on the map from a LiDAR scan and send /initialpose."""
    if enabled.lower() != 'true':
        return []
    if not os.path.exists(GLOBAL_LOCALIZE):
        return [LogInfo(msg=f'auto_localize: {GLOBAL_LOCALIZE} not found; '
                            'set the pose with "2D Pose Estimate" in RViz')]
    return [ExecuteProcess(cmd=['python3', GLOBAL_LOCALIZE, '--map', map_yaml,
                                '--wait', '--tries', '5'],
                           name='global_localize', output='screen')]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('room', default_value='',
                              description='room name from the Rooms window'),
        DeclareLaunchArgument('map', default_value='',
                              description='or a saved map .yaml (overrides room)'),
        DeclareLaunchArgument('maps_dir', default_value=rs.DEFAULT_MAPS_DIR),
        DeclareLaunchArgument('rounds', default_value='0',
                              description='rounds of laps to drive; 0 = until Stop'),
        DeclareLaunchArgument('lane_spacing', default_value='0.45',
                              description='m between neighbouring laps (robot is 0.512 wide)'),
        DeclareLaunchArgument('pattern', default_value='auto',
                              description="'auto' (corridors unless lanes check much more), "
                                          "'corridors' (lanes along each corridor), 'lanes' "
                                          "(straight lanes) or 'contour' (rings)"),
        DeclareLaunchArgument('wall_clearance', default_value='0.65',
                              description='contour: m from the robot centre to walls'),
        DeclareLaunchArgument('side_clearance', default_value='0.50',
                              description='lanes: m from the robot centre to anything beside it'),
        DeclareLaunchArgument('front_clearance', default_value='0.60',
                              description='lanes: m from the robot centre to a wall ahead'),
        DeclareLaunchArgument('home', default_value='',
                              description="'x,y' on the map: HOME for Return home "
                                          "(default: where START is pressed)"),
        DeclareLaunchArgument('lane_gap', default_value='2.0',
                              description='lanes: m between lanes (fewer lanes and turns; '
                                          'was 1.1 before 2026-11-07)'),
        DeclareLaunchArgument('min_pocket', default_value='1.0',
                              description='lanes: m² of missed floor worth an extra lane '
                                          '(was 0.3)'),
        DeclareLaunchArgument('auto_localize', default_value='true',
                              description='find the robot on the map from a LiDAR scan at '
                                          'startup (tools/global_localize.py); false = AMCL '
                                          'starts at the map origin'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('imu', default_value='true',
                              description='use the gyro for heading; false = LiDAR only'),
        DeclareLaunchArgument(
            'esp32_port',
            default_value='/dev/serial/by-path/platform-3610000.usb-usb-0:2.4:1.0-port0',
            description='ESP32 serial port (by USB socket path)'),
        OpaqueFunction(function=launch_setup),
    ])
