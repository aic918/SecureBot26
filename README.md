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


## Serial ownership and the motion chain
Only `esp32_serial_bridge` opens the ESP32 port. `securebot_live.launch.py` starts the chain:
```
securebot_live_node --/security/motion (stop|patrol)--> motion_manager_node
motion_manager_node --/esp32/mode (stop|patrol|drive), /cmd_vel--> esp32_serial_bridge --> ESP32
```
`motion_manager_node` stops when security says stop or goes quiet for 1 s, drives with
`/nav_cmd_vel` while one arrives (Nav2/teleop later), and otherwise uses the firmware's own
patrol (drive forward + ultrasonic avoidance). The bridge resends the command every 0.2 s,
treats inputs older than 0.5 s as stop, drops the serial backlog on startup, and publishes
`/ultrasonic/left|right` and `/esp32/status`.

See the LiDAR and both ultrasonics together (robot held in stop):
`ros2 launch securebot_ros securebot_sensors.launch.py`

## Contributing
- Branch from `main` (`feat/<sprint>-<name>`) and open a PR; `main` needs owner review.
- Test motor changes with the wheels off the ground first.
- Never commit weights, face data, or secrets.
