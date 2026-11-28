# SecureBot

Autonomous security robot (Senior Design). It maps rooms with a LiDAR, drives laps round saved
maps while avoiding obstacles, spots people with YOLO11 and uses face recognition to decide
whether to keep patrolling or hold a security stop.

## Hardware
- **Jetson Orin Nano:** ROS 2 Humble, camera, person detection, face recognition, mapping,
  navigation and the security logic.
- **ESP32:** motors, two front ultrasonics, MPU-6050 gyro and accelerometer (heading, tamper
  detection), INA226 battery monitor and wheel encoders; takes commands from the Jetson over
  USB (protocol and pins: `firmware/esp32/README.md`).
- **Battery:** 24 V 3.2 Ah lithium-ion pack (6S, 25.2 V full; 7S packs read 29.4 V full),
  measured by the INA226 (36 V max; the INA219's 26 V is too low for this pack).
- **RPLidar A1M8:** 360° scans for mapping and localization.

```
src/securebot_base/   ROS <-> ESP32 bridge
src/securebot_ros/    vision, security logic, mapping, exploring, laps, navigation, web panel
firmware/esp32/       ESP32 firmware and its tests
tools/                web panel launcher, re-localization, run recorder, face + SLAM check,
                      LiDAR viewer, motor tuning
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
The web panel runs the robot from a phone or laptop: laps round a saved room, mapping new
rooms, no-pass lines, and enrolling the faces the robot recognizes. The navbar at the top
switches between the three pages: **Robot**, **Facial recognition manager** and
**Map manager**. On a phone (screens narrower than 900 px) the navbar folds into the **≡** menu
at the top left.

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
who can open the panel can drive the robot and enroll faces). Don't run robot launches
from a terminal at the same time.

### Using it
1. Pick the room in the box at the top (or **Show on main screen** in the Map manager).
2. Press **Start**. If the robot software is off, Start powers it on, waits until the robot has
   found itself on the map, then starts the laps; the pill at the top shows what it's doing.
3. Press **Return home** when done (or **Stop** to halt where it is), then **Power off**.

### Robot page
| Control | What it does |
|---|---|
| **Start** | Starts the laps (powering the robot software on for the room first if needed: finds the ESP32's USB port, finds the robot on the map, records the run in `log/laps_runs/`). While mapping a new room: starts exploring. |
| **Stop** | Stops the robot where it is (also cancels a Start that is still waiting to power on). While mapping: stops and saves the map. |
| **Swap to camera / map** | Switches the big view between the map and the robot's webcam (not while face photos are being taken or the live face node has the camera). |
| **Manual re-localize** | Stops the robot and finds where it is on the map again. Use it after moving the robot by hand; keep it still until it finishes. |
| **Return home** | Stops the laps and drives back to home (the shortest clear way), then stops. |
| **Face recognition** (switch) | Runs live face recognition alongside the laps or mapping, from the next Power on. The **Faces** line shows what it sees (authorized / unauthorized / scanning, with names). The camera view keeps working: it shows the face node's frames. |
| **Power off** | Shuts the robot software down (asks first). |
| **■ STOP** (corner) | Shown while the robot moves: same as **Stop**, always in reach. |
| **Battery** (map corner) | The pack's charge while the robot software is on (`—%` with no reading); green, amber at 50 % and under, red when low. Click it to set the **return-home level** (default 20 %) and the pack type (6S / 7S). |

**Map view:** the room, the robot, the planned lanes, the path being driven now, home, the
no-pass lines (black, outlined) and orange boxes where something blocked the robot. Tap a box
to remove it; scroll or pinch to zoom, drag to pan, **Fit** to see it all.
**Logs** (under the view): this run's log and the last 10 runs' files.
**Alerts** (under the logs): every tamper alert with its time and snapshot.

**Low battery:** when the charge stays at or under the return-home level for 10 s, the robot
stops its laps or mapping and drives home (the pill says *battery low*), and Start is refused
until it is charged.

**Tamper alert:** with the robot software on, lifting or tipping the robot stops the motors at
once (in the ESP32 itself) and a red banner appears on every page with the time and a webcam
snapshot (click it to enlarge), with a beep. The motors stay off and Start is refused until
you press **Acknowledge**; then press Start to carry on. Alerts are kept in `alerts/`
(newest 100). A pager can be hooked in later (`web_alerts.AlertStore.hooks`).

### Map manager
Every room as a map preview. Click to select, double-click to see it full screen (**Esc** or
Back to return).

| Control | What it does |
|---|---|
| **Show on main screen** | Makes the selected room the one Start runs in. |
| **Rename** | Renames the room. |
| **Delete** | Removes the room; its map files and lines move to `maps/deleted/`. |
| **Add** | Maps a new room: name it, put the robot in the room, press **Start** on the Robot page and it explores on its own while the map grows on screen. **Stop** (or a fully scanned room) saves it as that room. |
| **Edit** | Highlights the previews; pick one to draw **no-pass lines** on it full screen: drag to draw a line (the robot treats it as a wall), **Move** (or two fingers / right-drag) to pan, **Undo**, **Delete all lines**, **Save**. **Esc** leaves, asking first if lines are unsaved. Saving while the robot runs in that room re-plans with the new lines. |

### Facial recognition manager
Everyone the robot treats as authorized, with a photo each. Click to select, double-click to
open their photos. Stop the live face node first: taking photos needs the camera.

| Control | What it does |
|---|---|
| **Add person** | Name, number of photos (3–40), **Start capture**: only that person in view, 1–1.5 m away; look at the camera, then turn slowly left, right, up and down until the bar fills. Then tick the photos to keep (flagged ones come unticked with the reason; untick blurry ones too) and **Add ticked photos**, or **Discard all**. |
| **View captures** | The person's photos: **✕** deletes one, **Retake photos** takes a new set whose kept photos replace the old ones. |
| **Delete person** | Removes the person (photos kept aside in `captures/`). |
| **Update face database** | Shown when photos changed: re-encodes everyone so the changes count, backs up the old database and warns about people who look alike. Restart the live face node afterwards. |

### Troubleshooting
- **"Cannot find the ESP32 USB port":** plug in both the ESP32 and the LiDAR, then retry.
- **"Robot software started from a terminal is running":** Ctrl+C it there first.
- **Robot is in the wrong place on the map:** **Manual re-localize** with the robot standing still.
- **Gyro not found** (`imu not found ... SDA=0` in the log): power-cycle the robot; if it
  persists, check the MPU-6050 wiring, then **Power off** and **Start** again.
- **Battery shows `—%` while on:** `power monitor not found` in the log: check the INA226
  wiring (`firmware/esp32/README.md`). A percentage that looks wrong: check the pack type (6S
  or 7S) in the battery dialog.
- **Tamper detection "Unavailable":** the accelerometer did not answer at boot (same wiring as
  the gyro). **Tamper alerts while just driving:** raise `TAMPER_LIFT_G` / `TAMPER_TILT_DEG`
  in the firmware.

### Testing face recognition and SLAM together
Turn the **Face recognition** switch on and start laps (or map a room), then beside it:
```bash
python3 ~/securebot_ws/tools/faces_slam_check.py --minutes 10
```
It records the face frame rate, LiDAR rate, position updates and jumps, errors, CPU / GPU load
and temperatures every second, and ends with PASS or what failed (saved in `log/faces_slam/`).
From a terminal instead of the panel: `ros2 launch securebot_ros securebot_laps.launch.py
room:="<room>" faces:=true`.

## Contributing
- Branch from `main` (`feat/<sprint>-<name>`) and open a PR; `main` needs owner review.
- Test motor changes with the wheels off the ground first.
- Never commit weights, face data or secrets.
