"""See every sensor together in RViz: LiDAR (/scan) + both ultrasonics (/ultrasonic/left, /right).

  ros2 launch securebot_ros securebot_sensors.launch.py

Starts the LiDAR launch and the ESP32 bridge (for the ultrasonics). No motion manager runs, so
the bridge holds the robot in stop the whole time. Don't run it together with
securebot_live.launch.py: both start the bridge, and only one program can own the ESP32 port.

The ultrasonic positions are placeholders until the URDF has measured ones; see
securebot_ultrasonic_tf.launch.py (us_x, us_y, us_z, us_yaw can be passed through here).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('securebot_ros')

    # Scoped, so the included launch's rviz:=false stays inside it.
    lidar = GroupAction(scoped=True, actions=[IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'securebot_lidar.launch.py')),
        launch_arguments={'rviz': 'false'}.items(),
    )])

    ultrasonic_tf = IncludeLaunchDescription(PythonLaunchDescriptionSource(
        os.path.join(share, 'launch', 'securebot_ultrasonic_tf.launch.py')))

    bridge = Node(package='securebot_base', executable='esp32_serial_bridge', output='screen')

    rviz = Node(package='rviz2', executable='rviz2', name='rviz2',
                arguments=['-d', os.path.join(share, 'rviz', 'sensors.rviz')])

    return LaunchDescription([lidar, ultrasonic_tf, bridge, rviz])
