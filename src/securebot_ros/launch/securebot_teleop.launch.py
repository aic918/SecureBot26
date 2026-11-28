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

It also starts guard_node: the battery percentage, return home when the battery is low and
stop when the robot is lifted or tipped (the ESP32 stops the motors itself first).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    arg = LaunchConfiguration
    return LaunchDescription([
        DeclareLaunchArgument('min_pwm', default_value='575'),     # of 1023 (10-bit duty)
        DeclareLaunchArgument('max_pwm', default_value='655'),
        DeclareLaunchArgument('gentle_pwm', default_value='565'),
        DeclareLaunchArgument('speed_control', default_value='true',
                              description='true: wheel speeds held by the encoders (V); false: '
                                          'fixed PWM between min_pwm and max_pwm (M)'),
        DeclareLaunchArgument(
            'port', default_value='/dev/serial/by-path/platform-3610000.usb-usb-0:2.4:1.0-port0',
            description='ESP32 serial port (by USB socket path)'),
        Node(package='securebot_ros', executable='motion_manager_node', output='screen',
             parameters=[{'require_security': False, 'idle_action': 'stop'}]),
        Node(package='securebot_base', executable='esp32_serial_bridge', output='screen',
             parameters=[{'min_pwm': ParameterValue(arg('min_pwm'), value_type=int),
                          'max_pwm': ParameterValue(arg('max_pwm'), value_type=int),
                          'gentle_pwm': ParameterValue(arg('gentle_pwm'), value_type=int),
                          'speed_control': ParameterValue(arg('speed_control'), value_type=bool),
                          'port': arg('port')}]),
        # battery %, return home when low, stop when tampered with (guard_node.py)
        Node(package='securebot_ros', executable='guard_node', output='screen'),
    ])
