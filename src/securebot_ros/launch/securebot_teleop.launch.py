"""Keyboard driving (for mapping): ESP32 bridge + motion manager in keyboard mode.

  terminal 1:  ros2 launch securebot_ros securebot_slam.launch.py       (optional: build a map)
  terminal 2:  ros2 launch securebot_ros securebot_teleop.launch.py
  terminal 3:  ros2 run securebot_ros teleop_keyboard                  (hold keys to drive)

Keyboard mode: the camera's security decision is not required (require_security:=false), and
when no key is held the robot stops instead of patrolling (idle_action:=stop). The firmware
still refuses to drive forward into anything closer than 1 ft.

min_pwm lifts slow commands to a PWM that actually turns the wheels; raise it if a wheel only
hums at mapping speed, lower it if the robot jumps. Don't run this together with
securebot_live.launch.py (both start the bridge; only one program can own the ESP32 port).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    arg = LaunchConfiguration
    return LaunchDescription([
        DeclareLaunchArgument('min_pwm', default_value='70'),
        Node(package='securebot_ros', executable='motion_manager_node', output='screen',
             parameters=[{'require_security': False, 'idle_action': 'stop'}]),
        Node(package='securebot_base', executable='esp32_serial_bridge', output='screen',
             parameters=[{'min_pwm': ParameterValue(arg('min_pwm'), value_type=int)}]),
    ])
