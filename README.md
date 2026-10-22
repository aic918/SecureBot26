# SecureBot

Autonomous security robot (Senior Design). Patrols, avoids obstacles, detects
people with YOLO11, and uses face recognition to decide whether to resume
patrol or hold a security stop.

## Architecture
- **Jetson Orin Nano (high level):** camera, YOLO11n person detection, face
  recognition, ROS 2 Humble nodes, security logic.
- **ESP32 (low level):** motor PWM, two HC-SR04 ultrasonics, obstacle avoidance;
  newline-terminated commands over USB serial, 115200 baud.

## Layout
```
src/securebot_base/   ESP32 serial bridge (/cmd_vel -> motors, ultrasonics -> Range)
src/securebot_ros/    live vision node, YOLO node, motion manager, patrol, enroll_faces
```

## Not in git (set up locally)
| What | Default location | Why |
|---|---|---|
| YOLO weights `yolo11n.pt` | `~/SecureBot/yolo11n.pt` | large binary; get from Ultralytics |
| Face photos | `~/yolo/securebot_face_id/authorized_faces/<name>/` | biometric data |
| Face encodings `.pkl` | `~/yolo/securebot_face_id/encodings/authorized_faces.pkl` | biometric data |
| Python venv | `~/yolo-env` | machine-specific |

Never commit these (they are in `.gitignore`).

## Build
```bash
source /opt/ros/humble/setup.bash
cd ~/securebot_ws
colcon build --symlink-install
source install/setup.bash
```

## Python environment
```bash
python3 -m venv ~/yolo-env --system-site-packages
source ~/yolo-env/bin/activate
pip install -r requirements.txt   # install Jetson torch/torchvision wheels separately
```

## Run
Build first (see above), then:
```bash
source /opt/ros/humble/setup.bash && source ~/securebot_ws/install/setup.bash
ros2 launch securebot_ros securebot_live.launch.py
```
The launch file runs the node with the `~/yolo-env` venv's Python and adds the venv's NVIDIA
libraries to `LD_LIBRARY_PATH` (torch fails with `libcudss.so.0` otherwise). Plain
`ros2 run securebot_ros securebot_live_node` uses the system Python and will fail on
`import face_recognition`; use the launch file.

Options (anything left out keeps the node's default):
```bash
ros2 launch securebot_ros securebot_live.launch.py \
    venv:=~/yolo-env model_path:=~/SecureBot/yolo11n.pt motor_speed:=88 \
    enable_motors:=false show_camera:=false
```
`enable_motors:=false` skips the ESP32 serial port, which is handy for testing vision alone.

Enroll authorized faces (one folder of photos per person) before the first run:
```bash
~/yolo-env/bin/python src/securebot_ros/securebot_ros/enroll_faces.py   # see --help for paths
```
In the camera window: **S** start, **X** stop, **Q** quit.

## LiDAR viewer (temporary, before ROS)
Live top-down view of the RPLidar A1M8 with the camera tower's blind spot measured and shaded.
Needs only pyserial and matplotlib (system Python); it never talks to the ESP32.
```bash
python3 tools/lidar_viewer.py               # finds the LiDAR (it answers a device-info request)
python3 tools/lidar_viewer.py /dev/ttyUSB0  # or name the port
python3 tools/lidar_viewer.py --demo        # simulated room + tower, no hardware
```
Orange points closer than `--tower-cm` (20) are the robot itself; gray wedges are angles with no
usable return in ~15 s. To measure the blind spot, press **c**, then walk once around the robot
holding a board ~50 cm away: what stays gray is blocked by the tower. Keys: **space** pause,
**s** save PNG + CSV, **+/-** zoom, **q** quit. Tests: `python3 -m unittest tools.test_lidar_viewer`.

## LiDAR in ROS (RPLidar A1M8)
One-time install: `sudo apt install ros-humble-rplidar-ros ros-humble-laser-filters`.
```bash
ros2 launch securebot_ros securebot_lidar.launch.py             # /scan_raw -> filter -> /scan
ros2 launch securebot_ros securebot_lidar.launch.py rviz:=true  # plus a top-down RViz view
```
`/scan` has the camera tower's shadow (laser frame 140-229 deg) and self-hits closer than
20 cm masked as "nothing seen" (`config/lidar_filters.yaml`). The LiDAR sits right behind the
tower, so the laser frame is turned 180 deg (`laser_yaw`) to make base_link +x the robot's
front. The port defaults to the LiDAR's USB socket path, so keep it in the same socket. The
`base_link -> laser` offsets (`laser_x/y/z`) are placeholders until the URDF exists.

## Mapping (SLAM)
rf2o laser odometry (`odom -> base_link`, from matching consecutive scans) + slam_toolbox
(`map -> odom`). rf2o is built from source: `vcs import src < deps.repos` (or
`git -C src clone -b ros2 https://github.com/MAPIRlab/rf2o_laser_odometry.git`), then
`colcon build`.
```bash
ros2 launch securebot_ros securebot_slam.launch.py        # opens RViz with the map
ros2 run nav2_map_server map_saver_cli -f maps/<name> --ros-args -p save_map_timeout:=10.0
```
Move the robot slowly (it can be pushed by hand; no wheel odometry is used): about walking pace,
gentle turns, and come back to the start so the map closes the loop. The tower blocks the view
straight ahead, so turn the robot now and then to let the LiDAR see what it is driving towards.

## Contributing
- Branch from `main` (`feat/<sprint>-<name>`) and open a PR; `main` needs owner review.
- Test motor changes with the wheels off the ground first.
- Never commit weights, face data, or secrets.
