"""Measure which way the LiDAR faces on the robot (laser_yaw), on a 0.3 m drive out and back.

  ros2 launch securebot_ros securebot_find_front.launch.py              # measure and print
  ros2 launch securebot_ros securebot_find_front.launch.py save:=true   # ... and save it

Put the robot where nothing is within 0.7 m of it, with walls in view of the LiDAR. It drives
0.3 m forward and back; everything stops when the measurement is done. Starts the LiDAR, the
ESP32 bridge + motion manager (teleop launch) and find_lidar_front (see that node). Don't run it
together with another launch that uses the ESP32 or the LiDAR.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, GroupAction,
                            IncludeLaunchDescription, RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

import yaml


def generate_launch_description():
    share = get_package_share_directory('securebot_ros')
    launch = lambda name: os.path.join(share, 'launch', name)  # noqa: E731
    try:
        with open(os.path.join(share, 'config', 'lidar_mount.yaml')) as f:
            current = float(yaml.safe_load(f)['laser_yaw'])
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
        current = float('nan')
    finder = Node(package='securebot_ros', executable='find_lidar_front', output='screen',
                  parameters=[{'save': ParameterValue(LaunchConfiguration('save'),
                                                      value_type=bool),
                               'distance': ParameterValue(LaunchConfiguration('distance'),
                                                          value_type=float),
                               'current_yaw': current}])
    return LaunchDescription([
        DeclareLaunchArgument('save', default_value='false',
                              description='write the measured laser_yaw to lidar_mount.yaml'),
        DeclareLaunchArgument('distance', default_value='0.3',
                              description='m to drive each way (at most ~0.7: the scan '
                                          'alignment searches 0.8 m)'),
        GroupAction(scoped=True, actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch('securebot_lidar.launch.py')),
            launch_arguments={'rviz': 'false'}.items())]),
        GroupAction(scoped=True, actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch('securebot_teleop.launch.py')))]),
        finder,
        RegisterEventHandler(OnProcessExit(target_action=finder,
                                           on_exit=[EmitEvent(event=Shutdown())])),
    ])
