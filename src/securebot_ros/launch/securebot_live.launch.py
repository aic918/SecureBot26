"""Run the live vision node with the project venv.

`ros2 run securebot_ros securebot_live_node` uses the system Python, which has
neither face_recognition nor torch/ultralytics. This launches the same node
with the venv's Python instead, and adds the venv's bundled NVIDIA libraries
(libcudss etc.) to LD_LIBRARY_PATH, which torch needs to import.

It also starts the motion chain, so only one program ever opens the ESP32 port:
  securebot_live_node --/security/motion--> motion_manager_node --/esp32/mode, /cmd_vel-->
  esp32_serial_bridge --serial--> ESP32

  ros2 launch securebot_ros securebot_live.launch.py
  ros2 launch securebot_ros securebot_live.launch.py enable_motors:=false show_camera:=false

enable_motors:=false runs vision only (no bridge, nothing opens the ESP32 port).
motor_speed sets the firmware's patrol PWM (10-bit duty, default 655); esp32_port overrides the ESP32's USB socket path.
idle_action:=stop (for Nav2) stops the robot when no /nav_cmd_vel arrives instead of patrolling.
Any parameter left empty keeps the node's own default.
"""
import glob
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

NODE_PARAMS = [
    'model_path', 'encodings_file', 'camera_index', 'yolo_confidence',
    'face_tolerance', 'face_scale', 'face_interval', 'show_camera',
]
# launch argument -> esp32_serial_bridge parameter
BRIDGE_PARAMS = {'esp32_port': 'port', 'motor_speed': 'patrol_speed'}


def _start_node(context):
    venv = os.path.expanduser(LaunchConfiguration('venv').perform(context))
    python = os.path.join(venv, 'bin', 'python')
    if not os.path.exists(python):
        raise RuntimeError(
            f'venv python not found: {python}. Create it (see README) or pass venv:=<path>.')

    nvidia_libs = glob.glob(
        os.path.join(venv, 'lib', 'python3*', 'site-packages', 'nvidia', '*', 'lib'))
    ld_path = os.pathsep.join(
        [p for p in nvidia_libs + [os.environ.get('LD_LIBRARY_PATH', '')] if p])

    cmd = [python, '-W', 'ignore::UserWarning', '-m', 'securebot_ros.securebot_live_node',
           '--ros-args']
    for name in NODE_PARAMS:
        value = LaunchConfiguration(name).perform(context)
        if value != '':
            cmd += ['-p', f'{name}:={value}']

    vision = ExecuteProcess(
        cmd=cmd,
        additional_env={'LD_LIBRARY_PATH': ld_path},
        output='screen',
    )
    if LaunchConfiguration('enable_motors').perform(context).lower() in ('false', '0', 'no'):
        return [vision]

    bridge_params = {}
    for arg, param in BRIDGE_PARAMS.items():
        value = LaunchConfiguration(arg).perform(context)
        if value != '':
            bridge_params[param] = int(value) if param == 'patrol_speed' else value
    return [
        vision,
        Node(package='securebot_ros', executable='motion_manager_node', output='screen',
             parameters=[{'idle_action': LaunchConfiguration('idle_action').perform(context)}]),
        Node(package='securebot_base', executable='esp32_serial_bridge', output='screen',
             parameters=[bridge_params]),
    ]


def generate_launch_description():
    args = [DeclareLaunchArgument('venv', default_value='~/yolo-env',
                                  description='Python venv with the vision dependencies')]
    args += [DeclareLaunchArgument(name, default_value='') for name in NODE_PARAMS]
    args += [DeclareLaunchArgument(name, default_value='') for name in BRIDGE_PARAMS]
    args += [DeclareLaunchArgument('enable_motors', default_value='true')]
    args += [DeclareLaunchArgument('idle_action', default_value='patrol',
                                   description='patrol | stop (use stop with Nav2)')]
    return LaunchDescription(args + [OpaqueFunction(function=_start_node)])
