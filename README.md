# SecureBot

Autonomous security robot (Senior Design). It maps rooms with a LiDAR, drives laps round saved
maps while avoiding obstacles, spots people with YOLO11 and uses face recognition to decide
whether to keep patrolling or hold a security stop.

## Hardware
- **Jetson Orin Nano:** ROS 2 Humble, camera, person detection, face recognition, mapping,
  navigation and the security logic.
- **ESP32:** motors, two front ultrasonics, MPU-6050 gyro and wheel encoders; takes commands
  from the Jetson over USB (protocol and pins: `firmware/esp32/README.md`).
- **RPLidar A1M8:** 360° scans for mapping and localization.

```
src/securebot_base/   ROS <-> ESP32 bridge
src/securebot_ros/    vision, security logic, mapping, exploring, laps, navigation, web panel
firmware/esp32/       ESP32 firmware and its tests
tools/                web panel launcher, re-localization, run recorder, LiDAR viewer, motor tuning
maps/                 saved maps, room names and no-pass lines
```

## Setup (once)
```bash
sudo apt install ros-humble-rplidar-ros ros-humble-laser-filters ros-humble-slam-toolbox \
  ros-humble-navigation2
cd ~/securebot_ws
vcs import src < deps.repos                  # LiDAR odometry, built from source
source /opt/ros/humble/setup.bash
colcon build --symlink-install

python3 -m venv ~/yolo-env --system-site-packages      # vision venv
~/yolo-env/bin/pip install -r requirements.txt         # plus NVIDIA's Jetson torch wheels
```
Kept on the robot only, never committed: YOLO weights (`~/SecureBot/yolo11n.pt`), face photos
and encodings (`~/yolo/securebot_face_id/`) and the venv.

## Using the robot (web panel)
The web panel runs the robot from a phone or laptop: laps round a saved room, and enrolling
the faces the robot recognizes. A room needs a saved map first (one-time, from a terminal:
`ros2 launch securebot_ros securebot_explore.launch.py room:="<name>"`, then press **START**).

### Launching
Plug in the ESP32 and the LiDAR, then on the robot:
```bash
~/securebot_ws/tools/securebot_panel.sh        # prints the addresses to open
```
Keep that terminal open and open one of the printed addresses, `:8080` included:
`http://127.0.0.1:8080` on the robot's own screen, `http://<robot-ip>:8080` from a phone or
laptop on the same network. `0.0.0.0` is not an address to browse to. Ctrl+C in that
terminal stops the panel and the robot software. `--port <n>` picks another port;
`--host 127.0.0.1` allows only the robot's own screen (use it on an untrusted network: anyone
who can open the panel can drive the robot and enroll faces). Don't run robot launches from a terminal at the same time.

### Using it
1. Pick the **Room** and press **Power on**. Wait until the pill at the top reads **Ready**.
2. Press **Start**. The robot drives its lanes; watch it on the **Map** and the **Status** card.
3. Press **Go home** when done (or **Stop** to halt where it is), then **Power off**.

### Buttons
**1 · Robot software**

| Control | What it does |
|---|---|
| **Room** | The saved room to run in. |
| **Options → Home point** | `x,y` on the map to use as home. Blank: wherever **Start** is pressed. |
| **Options → Rounds** | How many rounds of laps to drive. 0 keeps lapping until **Stop**. |
| **Power on** | Starts the robot software for that room (no RViz or windows): finds the ESP32's USB port, finds the robot on the map and records the run in `log/laps_runs/`. Refused if the ESP32/LiDAR isn't plugged in or robot software is already running from a terminal. |
| **Power off** | Stops the robot and shuts the robot software down (asks first). |

**2 · Drive** (needs **Power on**)

| Control | What it does |
|---|---|
| **Start** | Starts the laps from where the robot stands. Clears last run's obstacle boxes. |
| **Stop** | Stops the robot where it is. |
| **Go home** | Stops the laps and drives back to home (the shortest clear way), then stops. |
| **Re-localize** | Stops the robot and finds where it is on the map again. Use it after moving the robot by hand; keep it still until it finishes. |
| **■ STOP** (corner) | Shown while the robot is driving: same as **Stop**, always in reach. |

**Map:** the robot, the planned lanes, the path being driven now, home, and orange boxes where
something blocked the robot (it plans round them). Tap a box to remove it.
**Status:** laps state, what it's doing, round and lane, coverage, position and ESP32 link.
**Log and past runs:** the robot software's log and the recorded runs.

**Faces** (who counts as authorized; stop the live face node first, capture needs the camera)

| Control | What it does |
|---|---|
| People list | Everyone enrolled and how many photos each has ("N in database" means a rebuild is due). |
| **Add or update a person → Name** | New name adds a person; an existing name adds photos to them. |
| **Photos to take** | 3–40 (default 15). |
| **Start capture** | Takes photos from the camera. Only that person in view, 1–1.5 m away: look at the camera, then turn slowly left, right, up and down until the bar fills. The live view gives hints. |
| **Stop** | Ends the capture early (keeps what was taken). |
| **Add ticked photos** | Adds the ticked photos to that person. Flagged ones (look like someone else, unlike the rest) come unticked with the reason; also untick blurry ones. |
| **Discard all** | Throws away this capture (kept aside in `captures/`). |
| **Rebuild face database** | Re-encodes everyone's photos so new ones count. Backs up the old database and warns about people who look alike. Restart the live face node afterwards. |

### Troubleshooting
- **"Cannot find the ESP32 USB port":** plug in both the ESP32 and the LiDAR, then retry.
- **"Robot software started from a terminal is running":** Ctrl+C it there first.
- **Robot is in the wrong place on the map:** **Re-localize** with the robot standing still.
- **Gyro not found** (`imu not found ... SDA=0` in the log): power-cycle the robot; if it
  persists, check the MPU-6050 wiring, then **Power off** / **Power on**.

## Contributing
- Branch from `main` (`feat/<sprint>-<name>`) and open a PR; `main` needs owner review.
- Test motor changes with the wheels off the ground first.
- Never commit weights, face data or secrets.
