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
Enroll authorized faces (one folder of photos per person), then start the node:
```bash
source ~/yolo-env/bin/activate
python3 src/securebot_ros/securebot_ros/enroll_faces.py   # see --help for paths
python3 src/securebot_ros/securebot_ros/securebot_live_node.py
```
(`ros2 run` does not yet see the venv's `face_recognition`; tracked in Sprint 1.)

Override paths with ROS params, e.g. `--ros-args -p model_path:=/path/yolo11n.pt`.
In the camera window: **S** start, **X** stop, **Q** quit.

## Contributing
- Branch from `main` (`feat/<sprint>-<name>`) and open a PR; `main` needs owner review.
- Test motor changes with the wheels off the ground first.
- Never commit weights, face data, or secrets.
