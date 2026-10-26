"""Static base_link -> left_ultrasonic / right_ultrasonic transforms (included by other launches).

PLACEHOLDER positions until the URDF has measured ones: both at the front (us_x ahead of
base_link), us_y to each side, pointing straight ahead plus/minus us_yaw (left +, right -).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
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
    return LaunchDescription([
        DeclareLaunchArgument('us_x', default_value='0.12'),
        DeclareLaunchArgument('us_y', default_value='0.06'),
        DeclareLaunchArgument('us_z', default_value='0.10'),
        DeclareLaunchArgument('us_yaw', default_value='0.0',
                              description='outward angle of each sensor in rad (left +, right -)'),
        ultrasonic_tf('left', '1'),
        ultrasonic_tf('right', '-1'),
    ])
