"""See every sensor together in RViz: LiDAR (/scan) + both ultrasonics (/ultrasonic/left, /right).

  ros2 launch securebot_ros securebot_sensors.launch.py

Starts the LiDAR launch and the ESP32 bridge (for the ultrasonics). No motion manager runs, so
the bridge holds the robot in stop the whole time. Don't run it together with
securebot_live.launch.py: both start the bridge, and only one program can own the ESP32 port.

The ultrasonic positions are placeholders until the URDF has measured ones: both at the front
(us_x ahead of base_link), us_y to each side, pointing straight ahead plus/minus us_yaw.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def ultrasonic_tf(side, sign):
    arg = LaunchConfiguration
    return Node(
        package='tf2_ros', executable='static_transform_publisher', name=f'{side}_ultrasonic_tf',
        arguments=['--x', arg('us_x'), '--y', PythonExpression([sign, '*', arg('us_y')]),
                   '--z', arg('us_z'), '--yaw', PythonExpression([sign, '*', arg('us_yaw')]),
                   '--frame-id', 'base_link', '--child-frame-id', f'{side}_ultrasonic'],
    )


def generate_launch_description():
    share = get_package_share_directory('securebot_ros')

    args = [
        DeclareLaunchArgument('us_x', default_value='0.12'),
        DeclareLaunchArgument('us_y', default_value='0.06'),
        DeclareLaunchArgument('us_z', default_value='0.10'),
        DeclareLaunchArgument('us_yaw', default_value='0.0',
                              description='outward angle of each sensor in rad (left +, right -)'),
    ]

    # Scoped, so the included launch's rviz:=false stays inside it.
    lidar = GroupAction(scoped=True, actions=[IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'securebot_lidar.launch.py')),
        launch_arguments={'rviz': 'false'}.items(),
    )])

    bridge = Node(package='securebot_base', executable='esp32_serial_bridge', output='screen')

    rviz = Node(package='rviz2', executable='rviz2', name='rviz2',
                arguments=['-d', os.path.join(share, 'rviz', 'sensors.rviz')])

    return LaunchDescription(args + [lidar, bridge, ultrasonic_tf('left', '1'),
                                     ultrasonic_tf('right', '-1'), rviz])
