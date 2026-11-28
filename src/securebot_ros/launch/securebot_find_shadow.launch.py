"""Measure where the robot blinds its own LiDAR (camera tower, brackets): one turn on the spot.

  ros2 launch securebot_ros securebot_find_shadow.launch.py              # measure and print
  ros2 launch securebot_ros securebot_find_shadow.launch.py save:=true   # ... and save it

Put the robot where nothing is within 0.5 m of it. It turns once on the spot; everything stops
when the measurement is done. Starts the LiDAR, the ESP32 bridge + motion manager (teleop
launch) and find_lidar_shadow (see that node). Don't run it together with another launch that
uses the ESP32 or the LiDAR.
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


def generate_launch_description():
    share = get_package_share_directory('securebot_ros')
    launch = lambda name: os.path.join(share, 'launch', name)  # noqa: E731
    finder = Node(package='securebot_ros', executable='find_lidar_shadow', output='screen',
                  parameters=[{'save': ParameterValue(LaunchConfiguration('save'),
                                                      value_type=bool)}])
    return LaunchDescription([
        DeclareLaunchArgument('save', default_value='false',
                              description='write the measured mask to lidar_filters.yaml'),
        GroupAction(scoped=True, actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch('securebot_lidar.launch.py')),
            launch_arguments={'rviz': 'false'}.items())]),
        GroupAction(scoped=True, actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch('securebot_teleop.launch.py')))]),
        finder,
        RegisterEventHandler(OnProcessExit(target_action=finder,
                                           on_exit=[EmitEvent(event=Shutdown())])),
    ])
