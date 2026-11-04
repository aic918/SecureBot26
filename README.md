# SecureBot

Autonomous security robot (Senior Design). It maps and explores rooms with a LiDAR, navigates
saved maps, avoids obstacles, spots people with YOLO11 and uses face recognition to decide
whether to keep patrolling or hold a security stop.

## How it fits together
- **Jetson Orin Nano:** ROS 2 Humble, camera, person detection, face recognition, mapping,
  navigation and the security logic.
- **ESP32:** drives the motors and reads the two front ultrasonic sensors; it takes commands
  from the Jetson over USB.
- **RPLidar A1M8:** 360° scans for mapping, exploring and navigation.

```
src/securebot_base/   link between ROS and the ESP32 (motors, ultrasonics)
src/securebot_ros/    vision, security logic, mapping, exploring, navigation, Rooms window
firmware/esp32/       ESP32 firmware
tools/                stand-alone helpers (LiDAR viewer, motor tuning)
maps/                 saved maps and room names
```

## Setup (once)
Some files are not in git and live on the robot: YOLO weights (`~/SecureBot/yolo11n.pt`), the
authorized face photos and encodings (`~/yolo/securebot_face_id/`) and the Python venv
(`~/yolo-env`). Never commit them.

```bash
sudo apt install ros-humble-rplidar-ros ros-humble-laser-filters
cd ~/securebot_ws
vcs import src < deps.repos                  # LiDAR odometry package, built from source
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

Python environment for the vision node (install the Jetson torch wheels separately):
```bash
python3 -m venv ~/yolo-env --system-site-packages
~/yolo-env/bin/pip install -r requirements.txt
```

## Run
Start every terminal with:
```bash
source /opt/ros/humble/setup.bash && source ~/securebot_ws/install/setup.bash
```
Run one launch at a time: only one program can talk to the ESP32.

**Explore and map a room**
```bash
ros2 launch securebot_ros securebot_explore.launch.py
```
Opens RViz, the Explore window and the Rooms window. Press **START**; the robot drives around
the room by itself, avoiding obstacles, and stops when the room is about 90 % mapped
(or when you press **STOP**). Then save the map from the Rooms window.

**Manage rooms and view saved maps**
```bash
ros2 run securebot_ros room_manager
```
Lists every saved map. Click one to view it; name, rename or remove rooms; point a room at a
newer scan; delete old maps; save the map being built right now as a new room; or copy the
command that navigates on a map.

**Run laps round a saved room**
```bash
ros2 launch securebot_ros securebot_laps.launch.py room:="Kumar's Lab"
```
Plans laps that sweep the whole scanned floor: the first lap follows the walls and goes round
the furniture, and each next lap is one lane (0.45 m) further in, until the middle is reached.
The robot keeps 0.65 m from the walls: room to turn its 495 x 512 mm body (wheels included)
anywhere on a lap and to keep the front more than 1 ft from a wall it faces at a corner, where the
firmware would stop it (gaps narrower than ~1.3 m are skipped). Nav2 drives the laps and steers round anything new in the way. Put the
robot where mapping started (or set "2D Pose Estimate" in RViz), then press **START** in the
Laps window; it keeps lapping until **STOP** (`rounds:=N` stops after N rounds). The planned
laps show in RViz in blue.

**Navigate a saved map**
```bash
ros2 launch securebot_ros securebot_live.launch.py idle_action:=stop
ros2 launch securebot_ros securebot_nav.launch.py map:=$HOME/securebot_ws/maps/<name>.yaml
```
Set a goal with "Nav2 Goal" in RViz. The security logic can still stop the robot.

**Security patrol (camera + face recognition)**
```bash
ros2 launch securebot_ros securebot_live.launch.py
```
Enroll authorized faces first (`~/yolo-env/bin/python src/securebot_ros/securebot_ros/enroll_faces.py`).
In the camera window: **S** start, **X** stop, **Q** quit. `enable_motors:=false` tests vision
without driving.

**Map by hand (keyboard driving)**
```bash
ros2 launch securebot_ros securebot_slam.launch.py
ros2 launch securebot_ros securebot_teleop.launch.py
ros2 run securebot_ros teleop_keyboard       # w/a/s/d to drive, space to stop, x to quit
```

**Check the sensors**
```bash
ros2 launch securebot_ros securebot_sensors.launch.py   # LiDAR + ultrasonics, robot held still
python3 tools/lidar_viewer.py                           # LiDAR alone, no ROS
```

## Contributing
- Branch from `main` (`feat/<sprint>-<name>`) and open a PR; `main` needs owner review.
- Test motor changes with the wheels off the ground first.
- Never commit weights, face data or secrets.
