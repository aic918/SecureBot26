"""Build a map: LiDAR (/scan) + rf2o laser odometry (odom -> base_link) + slam_toolbox (map -> odom).

  ros2 launch securebot_ros securebot_slam.launch.py              # with RViz showing the map
  ros2 launch securebot_ros securebot_slam.launch.py rviz:=false

There are no wheel encoders in use yet, so odometry comes from matching consecutive scans
(rf2o). That works whether the robot drives itself or is pushed by hand. Move slowly
(walking pace or less, turn gently): rf2o loses track on fast turns.

Save the map while this is running:
  ros2 run nav2_map_server map_saver_cli -f ~/securebot_ws/maps/<name> --ros-args -p save_map_timeout:=10.0
(the default 2 s timeout is too short: slam_toolbox only republishes the map every 2 s)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('securebot_ros')
    arg = LaunchConfiguration

    args = [
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('slam_params',
                              default_value=os.path.join(share, 'config',
                                                         'slam_toolbox_mapping.yaml')),
    ]

    lidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'securebot_lidar.launch.py')),
        launch_arguments={'rviz': 'false'}.items(),
    )

    odometry = Node(
        package='rf2o_laser_odometry', executable='rf2o_laser_odometry_node',
        name='rf2o_laser_odometry', output='screen',
        parameters=[{
            'laser_scan_topic': '/scan',
            'odom_topic': '/odom',
            'publish_tf': True,
            'base_frame_id': 'base_link',
            'odom_frame_id': 'odom',
            'init_pose_from_topic': '',
            'freq': 10.0,
        }],
    )

    slam = Node(
        package='slam_toolbox', executable='async_slam_toolbox_node', name='slam_toolbox',
        output='screen', parameters=[arg('slam_params')],
    )

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        condition=IfCondition(arg('rviz')),
        arguments=['-d', os.path.join(share, 'rviz', 'slam.rviz')],
    )

    return LaunchDescription(args + [lidar, odometry, slam, rviz])
