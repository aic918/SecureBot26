"""Build a map: LiDAR (/scan) + odometry (odom -> base_link) + slam_toolbox (map -> odom).

  ros2 launch securebot_ros securebot_slam.launch.py              # with RViz showing the map
  ros2 launch securebot_ros securebot_slam.launch.py rviz:=false

Odometry (heading_odom_node): heading from the gyro, distance from the wheel encoders, both
from the ESP32 bridge (started by the explore / teleop launch). Whatever of that is missing
comes from rf2o (matching consecutive scans), e.g. with the robot pushed by hand and no bridge.
In long, plain hallways rf2o alone drifts (the LiDAR is blind straight ahead): that doubled a
hallway on 2026-11-22. imu:=false runs rf2o alone, as before. Move slowly and turn gently.

Save the map while this is running:
  ros2 run nav2_map_server map_saver_cli -f ~/securebot_ws/maps/<name> --ros-args -p save_map_timeout:=10.0
(the default 2 s timeout is too short: slam_toolbox only republishes the map every 2 s)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('securebot_ros')
    arg = LaunchConfiguration

    args = [
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('imu', default_value='true',
                              description='odometry from gyro + wheel encoders (false: rf2o only)'),
        DeclareLaunchArgument('slam_params',
                              default_value=os.path.join(share, 'config',
                                                         'slam_toolbox_mapping.yaml')),
    ]

    # Scoped, so the included launch's rviz:=false does not switch off this file's RViz too.
    lidar = GroupAction(scoped=True, actions=[IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'securebot_lidar.launch.py')),
        launch_arguments={'rviz': 'false'}.items(),
    )])

    # rf2o alone (imu:=false) publishes /odom and odom -> base_link itself; otherwise it feeds
    # heading_odom_node (gyro heading, encoder distance, rf2o for whatever is missing).
    def rf2o(condition, topic, tf):
        return Node(
            package='rf2o_laser_odometry', executable='rf2o_laser_odometry_node',
            name='rf2o_laser_odometry', output='screen', condition=condition,
            parameters=[{'laser_scan_topic': '/scan', 'odom_topic': topic, 'publish_tf': tf,
                         'base_frame_id': 'base_link', 'odom_frame_id': 'odom',
                         'init_pose_from_topic': '', 'freq': 10.0}],
        )
    odometry = [
        rf2o(UnlessCondition(arg('imu')), '/odom', True),
        rf2o(IfCondition(arg('imu')), '/odom_laser', False),
        Node(package='securebot_ros', executable='heading_odom_node', output='screen',
             condition=IfCondition(arg('imu'))),
    ]

    slam = Node(
        package='slam_toolbox', executable='async_slam_toolbox_node', name='slam_toolbox',
        output='screen', parameters=[arg('slam_params')],
        # the Rooms window's "Clear live map" stops it: it comes back with an empty map
        respawn=True, respawn_delay=1.0,
    )

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        condition=IfCondition(arg('rviz')),
        arguments=['-d', os.path.join(share, 'rviz', 'slam.rviz')],
    )

    return LaunchDescription(args + [lidar] + odometry + [slam, rviz])
