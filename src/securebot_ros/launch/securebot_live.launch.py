"""Run the live vision node with the project venv.

`ros2 run securebot_ros securebot_live_node` uses the system Python, which has
neither face_recognition nor torch/ultralytics. This launches the same node
with the venv's Python instead, and adds the venv's bundled NVIDIA libraries
(libcudss etc.) to LD_LIBRARY_PATH, which torch needs to import.

  ros2 launch securebot_ros securebot_live.launch.py
  ros2 launch securebot_ros securebot_live.launch.py enable_motors:=false show_camera:=false

Any node parameter left empty keeps the node's own default.
"""
import glob
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration

NODE_PARAMS = [
    'model_path', 'encodings_file', 'camera_index', 'yolo_confidence',
    'face_tolerance', 'face_scale', 'face_interval', 'show_camera',
    'esp32_port', 'esp32_baud', 'enable_motors', 'motor_speed',
]


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

    return [ExecuteProcess(
        cmd=cmd,
        additional_env={'LD_LIBRARY_PATH': ld_path},
        output='screen',
    )]


def generate_launch_description():
    args = [DeclareLaunchArgument('venv', default_value='~/yolo-env',
                                  description='Python venv with the vision dependencies')]
    args += [DeclareLaunchArgument(name, default_value='') for name in NODE_PARAMS]
    return LaunchDescription(args + [OpaqueFunction(function=_start_node)])
