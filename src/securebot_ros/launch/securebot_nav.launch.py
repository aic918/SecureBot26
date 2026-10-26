"""Navigate on a saved map: LiDAR + rf2o odometry + AMCL localization + Nav2 + collision monitor.

  terminal 1:  ros2 launch securebot_ros securebot_live.launch.py idle_action:=stop
  terminal 2:  ros2 launch securebot_ros securebot_nav.launch.py map:=$HOME/securebot_ws/maps/<name>.yaml

Terminal 1 is the robot side (camera security decision -> motion manager -> ESP32 bridge).
idle_action:=stop matters: when Nav2 has no goal the robot must stop, not start the firmware's
own patrol. Nav2 never talks to the wheels directly; its commands end on /nav_cmd_vel, and the
motion manager only passes them on while security allows motion.

In RViz: "2D Pose Estimate" to tell AMCL where the robot is, then "Nav2 Goal" to send a goal.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

LOCALIZATION = ['map_server', 'amcl']
NAVIGATION = ['controller_server', 'planner_server', 'behavior_server', 'bt_navigator',
              'waypoint_follower', 'velocity_smoother', 'collision_monitor']


def generate_launch_description():
    share = get_package_share_directory('securebot_ros')
    arg = LaunchConfiguration
    params = arg('params')

    args = [
        DeclareLaunchArgument('map', description='saved map .yaml (from map_saver_cli)'),
        DeclareLaunchArgument('params',
                              default_value=os.path.join(share, 'config', 'nav2_params.yaml')),
        DeclareLaunchArgument('rviz', default_value='true'),
    ]

    # Scoped, so the included launch's rviz:=false does not switch off this file's RViz too.
    lidar = GroupAction(scoped=True, actions=[IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'securebot_lidar.launch.py')),
        launch_arguments={'rviz': 'false'}.items(),
    )])

    ultrasonic_tf = IncludeLaunchDescription(PythonLaunchDescriptionSource(
        os.path.join(share, 'launch', 'securebot_ultrasonic_tf.launch.py')))

    odometry = Node(
        package='rf2o_laser_odometry', executable='rf2o_laser_odometry_node',
        name='rf2o_laser_odometry', output='screen',
        parameters=[{'laser_scan_topic': '/scan', 'odom_topic': '/odom', 'publish_tf': True,
                     'base_frame_id': 'base_link', 'odom_frame_id': 'odom',
                     'init_pose_from_topic': '', 'freq': 10.0}],
    )

    def nav2(package, executable, name, remappings=()):
        return Node(package=package, executable=executable, name=name, output='screen',
                    parameters=[params], remappings=list(remappings))

    to_smoother = [('cmd_vel', 'cmd_vel_nav')]
    nodes = [
        Node(package='nav2_map_server', executable='map_server', name='map_server',
             output='screen', parameters=[params, {'yaml_filename': arg('map')}]),
        nav2('nav2_amcl', 'amcl', 'amcl'),
        nav2('nav2_controller', 'controller_server', 'controller_server', to_smoother),
        nav2('nav2_planner', 'planner_server', 'planner_server'),
        nav2('nav2_behaviors', 'behavior_server', 'behavior_server', to_smoother),
        nav2('nav2_bt_navigator', 'bt_navigator', 'bt_navigator'),
        nav2('nav2_waypoint_follower', 'waypoint_follower', 'waypoint_follower'),
        # cmd_vel_nav -> cmd_vel_smoothed (its default output name)
        nav2('nav2_velocity_smoother', 'velocity_smoother', 'velocity_smoother', to_smoother),
        # cmd_vel_smoothed -> nav_cmd_vel (topics set in the params file)
        nav2('nav2_collision_monitor', 'collision_monitor', 'collision_monitor'),
    ]

    managers = [
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name=f'lifecycle_manager_{group}', output='screen',
             parameters=[{'autostart': True, 'node_names': names}])
        for group, names in (('localization', LOCALIZATION), ('navigation', NAVIGATION))
    ]

    rviz = Node(package='rviz2', executable='rviz2', name='rviz2',
                condition=IfCondition(arg('rviz')),
                arguments=['-d', os.path.join(share, 'rviz', 'nav.rviz')])

    return LaunchDescription(args + [lidar, ultrasonic_tf, odometry] + nodes + managers + [rviz])
