"""Pieces shared by the laps and explore launch files."""
import os

LIVE_PREVIEW = os.path.expanduser('~/securebot_ws/log/live_preview.jpg')


def faces_launch(faces):
    """Live face recognition next to the laps or exploring (faces:=true): vision only (the
    laps / explore stack owns the ESP32), no window, its newest frame saved to LIVE_PREVIEW
    for the web panel's camera view and tamper snapshots."""
    if str(faces).lower() != 'true':
        return []
    from ament_index_python.packages import get_package_share_directory
    from launch.actions import GroupAction, IncludeLaunchDescription
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    share = get_package_share_directory('securebot_ros')
    return [GroupAction(scoped=True, actions=[IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'securebot_live.launch.py')),
        launch_arguments={'enable_motors': 'false', 'show_camera': 'false',
                          'preview_file': LIVE_PREVIEW}.items())])]
