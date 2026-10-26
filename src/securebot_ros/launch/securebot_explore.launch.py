"""Let the robot explore and map the room on its own, with Start / Stop buttons.

  ros2 launch securebot_ros securebot_explore.launch.py

Starts: SLAM (LiDAR, laser odometry, slam_toolbox, RViz), the robot side in keyboard mode
(ESP32 bridge + motion manager: no camera needed, stops when nothing is commanded), the
explore node and the "SecureBot Explore" window. Nothing moves until you press START; STOP
(or closing the window, or Ctrl+C here) stops it. When the room is >= 90 % scanned the robot
stops by itself; save the map then:
  ros2 run nav2_map_server map_saver_cli -f maps/<name> --ros-args -p save_map_timeout:=10.0

Don't run it together with the live, teleop or nav launches (only one program can own the
ESP32 port, and only one SLAM may run).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('securebot_ros')
    launch = lambda name: os.path.join(share, 'launch', name)  # noqa: E731

    return LaunchDescription([
        DeclareLaunchArgument('coverage_goal', default_value='0.9',
                              description='stop when this fraction of the room is scanned'),
        GroupAction(scoped=True, actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch('securebot_slam.launch.py')))]),
        GroupAction(scoped=True, actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch('securebot_teleop.launch.py')))]),
        Node(package='securebot_ros', executable='explore_node', output='screen',
             parameters=[{'coverage_goal': ParameterValue(LaunchConfiguration('coverage_goal'),
                                                          value_type=float)}]),
        Node(package='securebot_ros', executable='explore_panel', output='screen'),
    ])
