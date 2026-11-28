#!/usr/bin/env bash
# Start the SecureBot web panel: then open http://<robot-ip>:8080 in a browser on the same
# network (the panel prints the addresses to use; 0.0.0.0 is not one). Ctrl+C stops the panel and the robot software.
#   tools/securebot_panel.sh              # port 8080
#   tools/securebot_panel.sh --port 9000
set -e
source /opt/ros/humble/setup.bash
source "$HOME/securebot_ws/install/setup.bash"
exec ros2 run securebot_ros web_panel "$@"
