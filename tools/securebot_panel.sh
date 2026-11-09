#!/usr/bin/env bash
# Start the SecureBot web panel: then open http://<robot-ip>:8080 in a browser on the same
# network (this script prints the address). Ctrl+C stops the panel and the robot software.
#   tools/securebot_panel.sh              # port 8080
#   tools/securebot_panel.sh --port 9000
set -e
source /opt/ros/humble/setup.bash
source "$HOME/securebot_ws/install/setup.bash"
port=8080
for ((i = 1; i <= $#; i++)); do [[ ${!i} == --port ]] && j=$((i + 1)) && port=${!j}; done
for ip in $(hostname -I); do echo "SecureBot panel: http://$ip:$port"; done
exec ros2 run securebot_ros web_panel "$@"
