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
ros2 launch securebot_ros securebot_explore.launch.py room:="ENGR Hallway"
```
Opens RViz, the Explore window and the Rooms window. Press **START**; the robot drives around
the room by itself, avoiding obstacles, and stops when the room is about 90 % mapped
(or when you press **STOP**). It keeps going while an opening the robot fits through (end of
a hallway, a doorway) is still unexplored. With `room:="<name>"` the map is saved and filed
under that name when it finishes or when you press **STOP** (a room that already exists gets
the new scan), ready for laps. Without it, save the map from the Rooms window.
When driving stops showing new floor (less than 1 m² in the last 4 m), it plans a way to the
nearest unexplored opening, drives there and explores on from it, instead of going over
scanned parts again. **RETURN HOME** drives the robot back to where **START** was first pressed
(a path planned on the map so far, shown as a magenta line) and stops it there. No-pass lines can be drawn here too
(same buttons as in the Laps window): exploring treats them as walls, and they are saved with
the room's map, so laps keep them; RViz marks that spot with a blue
1.5 ft **HOME** circle (laps show it too).

**Manage rooms and view saved maps**
```bash
ros2 run securebot_ros room_manager
```
Lists every saved map. Click one to view it; name, rename or remove rooms; point a room at a
newer scan; delete old maps; save the map being built right now as a new room; or copy the
command that navigates on a map. SLAM's live map holds everything scanned since the launch:
**Clear live map** (also offered after deleting a map) restarts SLAM with an empty map, so a
deleted scan does not come back with the next save.

**Run laps round a saved room**
```bash
ros2 launch securebot_ros securebot_laps.launch.py room:="Kumar's Lab" imu:=true rounds:=1
```
Opens RViz and the Laps window; nothing moves until **START**. Lane 1 starts wherever the robot
is. If the robot was moved by hand, re-localize it first (it must be standing still):
```bash
python3 tools/global_localize.py --map maps/<map>.yaml     # --dry-run only prints the match
```
or set "2D Pose Estimate" in RViz.

How the laps drive (`laps_node`, `lap_follower`, `coverage_logic`, `lane_logic`):
- **Plan:** `pattern:=auto` (default) picks between two planners, keeping corridor routes
  unless they check much less floor than lanes (under 80 %, counting what the LiDAR sees
  2 m either side of the robot):
  - `pattern:=corridors` (`route_logic`): lanes follow the corridors. The planner finds
    each corridor's centre line, cuts door alcoves and corners, and gives each corridor one
    lane per 2 m of width. A hallway up to ~3.5 m wide is one lane down the middle, end to
    end, even where the map bends slightly. Side branches are driven first and the longest
    last, so the robot never drives back along it; dead ends are turned round once. Turns
    happen only at junctions and dead ends. Picked for hallways.
  - `pattern:=lanes`: straight parallel lanes, each part of the room along its own long
    side. Picked for rooms with desks.
  Lanes are `lane_gap:=2.0` m apart; with straight lanes, missed bits of floor get an extra
  lane only from `min_pocket:=1.0` m². `pattern:=contour` drives rings in from the walls.
  Waypoints are at right angles only, on a grid lined up with the building's main walls
  (the angle within 6 deg that needs the fewest turns). At each waypoint the route either
  goes straight on (180 deg) or turns 90 deg (or turns back, 180 deg, at a dead end). Every
  lane has an even number of waypoints; a straight line that would not stay clear of the
  walls gets a square jog instead.
  **START** drives the plan RViz shows, if it was made from where the robot stands (no
  wait for re-planning).
  RViz shows the plan (blue), the path being driven now (green) and the numbered lane points
  (black numbers, white dots). The robot slows only before real corners and lane ends, not
  at the slight bends of a lane that follows a corridor.
- **Driving:** the robot holds each lane with the gyro and LiDAR odometry (AMCL is blended in
  slowly, so it does not chase AMCL's jumps in open space). Off the line, it angles back to a
  point ahead at up to 30 deg instead of going back. Corners of up to 100 deg are driven as
  rounded curves (0.5 m radius) that ease off as the exit heading nears; sharper ones are
  turned on the spot (small turns at the gentle 610 PWM, big ones at full power). A small
  correction (under 30 deg) turns only half the error and the lane hold steers off the rest;
  the follow-up spins of a big turn finish it. While standing still, the scan is matched
  against the map to correct the position; a correction over 20 cm or 5 deg needs an 85 %
  match. It slows to
  the gentle PWM for the last 8 in before every lane point and before every stop.
- **Things on the floor:** when something stops the robot for good (too low for the LiDAR and
  ultrasonics, or it stays in the way), an orange 8 x 8 in box is put on the map for this run
  only and Nav2 plans round it; the boxes are cleared at the next **START**. A lane is only
  skipped after every clear way to it has been tried. Back-ups are short (0.5 s).
- **Remove an orange box:** in RViz select **Publish Point** in the toolbar and click the box.
- **No-pass lines:** in the Laps window press **Draw no-pass line**, select **Publish Point**
  in RViz and click the corners of the line; press **Finish line** when done. Lines show as thick
  black lines and act as walls: the plan stays off them, Nav2 routes round them and the robot
  stops before them. They belong to the room (saved as `maps/<map>.nopass.json`) and are kept
  for later runs. **Undo line** / **Clear lines** remove them, or click a line with Publish
  Point while not drawing. Draw them before **START**: lanes are re-planned at the next start.
- **RETURN HOME** (Laps window) stops the laps, drives back to where **START** was pressed
  and stops there. It goes back along the planned lanes, which start at HOME and are known
  clear on the map. It takes the shortest way, skips side rooms with short clear hops, leaves
  the lanes straight for HOME, and keeps to right angles on the plan's grid. It uses Nav2's
  route (round boxes and no-pass lines) only if the robot is off the lanes or on a last try.
  The way home shows as a magenta line.
  `home:="x,y"` fixes HOME at a map point instead (e.g. `home:="0,0"`, where the room's
  exploring started); it shows from launch.
- **Safety stops:** the run stops with a message if the gyro, LiDAR or ESP32 goes quiet, if
  Nav2 finds no way to three lanes in a row, or if the robot is boxed in for good.

**Record a run in detail** (beside the laps launch; Ctrl+C to finish):
```bash
python3 tools/run_recorder.py --name 20261107_2000
```
Writes `log/laps_runs/`:
- `track_<name>.csv`: map pose, odometry heading and laps state, 20 times a second.
- `imu_cmd_<name>.csv`: gyro and every motor command.
- `paths_<name>.jsonl`: each path driven.
- `events_<name>.log`: every status change.

**Gyro (MPU-6050) not found:** the ESP32 checks the I2C wires at boot and reports e.g.
`imu not found ... wire check (1 = connected): SDA=0 SCL=1`. A 0 means that wire does not reach
the MPU-6050 (loose jumper or joint), or that the MPU-6050 is holding SDA low because a
reset cut it off mid-transfer. Since firmware of 2026-11-07 the ESP32 frees a held bus at
every boot (`imu bus freed` in the log); if it still says SDA=0, power-cycle, and if that
does not help either, check the wiring. The gyro is only looked for at boot: restart the
launch after a fix.

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
