"""RPLidar A1M8 -> /scan_raw -> tower/self-hit filter -> /scan, plus a temporary base_link -> laser TF.

  ros2 launch securebot_ros securebot_lidar.launch.py
  ros2 launch securebot_ros securebot_lidar.launch.py rviz:=true
  ros2 launch securebot_ros securebot_lidar.launch.py serial_port:=/dev/ttyUSB0

The default port is the USB *path* of the LiDAR's socket (2.2), which stays the same across
reboots as long as it is plugged into the same socket; /dev/ttyUSBn numbers can swap with
the ESP32 (both are CP2102 adapters with the same serial number).

The base_link -> laser transform is a placeholder until the URDF (Sprint 2, S2-3) has the
measured LiDAR position. robot_state_publisher publishes urdf/securebot.urdf (the case, to
scale: 495 x 360 mm) on /robot_description so RViz can draw the robot. laser_yaw turns the laser frame so its +x is the robot's front: the
LiDAR sits right behind the camera tower, and the tower's shadow is at 180 deg in the laser
frame, so the laser's +x points backwards and the default yaw is pi.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

LIDAR_PORT = '/dev/serial/by-path/platform-3610000.usb-usb-0:2.2:1.0-port0'


def generate_launch_description():
    share = get_package_share_directory('securebot_ros')
    arg = LaunchConfiguration

    args = [
        DeclareLaunchArgument('serial_port', default_value=LIDAR_PORT),
        DeclareLaunchArgument('scan_mode', default_value='Sensitivity',
                              description='Sensitivity (8 kHz, ~1080 points/turn) or Standard'),
        DeclareLaunchArgument('publish_tf', default_value='true',
                              description='static base_link -> laser TF (off once the URDF exists)'),
        DeclareLaunchArgument('laser_x', default_value='0.0'),
        DeclareLaunchArgument('laser_y', default_value='0.0'),
        DeclareLaunchArgument('laser_z', default_value='0.20'),
        DeclareLaunchArgument('laser_yaw', default_value='3.14159'),
        DeclareLaunchArgument('rviz', default_value='false'),
    ]

    with open(os.path.join(share, 'urdf', 'securebot.urdf')) as f:
        robot_description = f.read()
    model = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='robot_state_publisher', output='screen',
        parameters=[{'robot_description': robot_description}],
    )

    # Stop + soft-reset the A1 first: after an unclean shutdown it can still be scanning, and
    # rplidar_node then hangs on connect instead of exiting (so respawn would not help).
    reset = Node(package='securebot_ros', executable='lidar_reset', name='lidar_reset',
                 output='screen', arguments=[arg('serial_port')])

    lidar = Node(
        package='rplidar_ros', executable='rplidar_node', name='rplidar_node', output='screen',
        # The A1 sometimes misses the first request or refuses to scan right after the motor
        # starts; rplidar_node then exits, so start it again instead of giving up.
        respawn=True, respawn_delay=3.0,
        parameters=[{
            'channel_type': 'serial',
            'serial_port': arg('serial_port'),
            'serial_baudrate': 115200,
            'frame_id': 'laser',
            'inverted': False,
            'angle_compensate': True,
            'scan_mode': arg('scan_mode'),
        }],
        remappings=[('scan', 'scan_raw')],
    )

    filters = Node(
        package='laser_filters', executable='scan_to_scan_filter_chain', name='lidar_filters',
        output='screen',
        parameters=[os.path.join(share, 'config', 'lidar_filters.yaml')],
        remappings=[('scan', 'scan_raw'), ('scan_filtered', 'scan')],
    )

    tf = Node(
        package='tf2_ros', executable='static_transform_publisher', name='laser_tf',
        condition=IfCondition(arg('publish_tf')),
        arguments=['--x', arg('laser_x'), '--y', arg('laser_y'), '--z', arg('laser_z'),
                   '--yaw', arg('laser_yaw'), '--frame-id', 'base_link',
                   '--child-frame-id', 'laser'],
    )

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        condition=IfCondition(arg('rviz')),
        arguments=['-d', os.path.join(share, 'rviz', 'lidar.rviz')],
    )

    start_lidar = RegisterEventHandler(OnProcessExit(target_action=reset, on_exit=[lidar]))
    return LaunchDescription(args + [reset, filters, tf, model, rviz, start_lidar])
