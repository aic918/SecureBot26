"""SecureBot web panel: everything except ROS (no rclpy here, so it is tested without ROS).

The panel replaces the terminal commands and the "SecureBot Laps" / RViz windows for running
laps: a browser on the same network opens http://<robot>:8080 and gets

  Power on   (StackManager)  starts securebot_laps.launch.py for the chosen room without
                             RViz or the Tk window (panel:=web), on the ESP32's USB
                             socket (found here),
                             logging to log/laps_runs/run_<stamp>.log, plus run_recorder.py
  Start / Stop / Go home     "start" / "stop" / "home" on /laps/command (the bridge)
  Re-localize                tools/global_localize.py on the room's map
  the live map               the room's map image, robot pose, lanes, boxes and home
  Power off                  Ctrl+C (SIGINT) to the launch, then SIGTERM / SIGKILL
  Faces                      enroll people for face recognition (web_faces.FaceEnroller):
                             capture from the camera, check, keep, rebuild the database,
                             delete or retake a person's photos
  Map manager                rename / delete rooms, draw no-pass lines on a room's map, and
                             Add: map a new room (the explore launch, its live SLAM map)
  Camera view                the webcam while someone watches it (web_camera.CameraFeed)
  Battery                    the guard node's percentage (/power/status), the return-home
                             level (power_logic settings) and a refused Start when it is low
  Tamper alerts              lifted / tipped (/tamper): an alert with its time and a webcam
                             snapshot (web_alerts.AlertStore); acknowledging it re-enables the
                             motors ("clear" on /tamper/command)
  Face recognition           optional live face recognition during laps / mapping (faces:=true)

The page's Start works from off: it powers the laps launch on for the active room, waits until
the robot has found itself on the map, then starts the laps.

PanelApp holds that state and answers the HTTP API (make_server); the ROS side is passed in
as a `bridge` (web_panel.RosBridge on the robot, a fake one in the tests) with
send_command(cmd, target='laps'|'explore'), click(x, y) and snapshot(kind) -> dict, and
optionally power() -> dict | None, tamper() -> (state, reason) | None, faces() -> dict | None
and tamper_command('clear').
"""
import glob
import json
import math
import os
import shutil
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import yaml

from securebot_ros import nopass as npz
from securebot_ros import power_logic as pl
from securebot_ros import room_store as rs
from securebot_ros.web_alerts import AlertStore
from securebot_ros.web_camera import CameraFeed
from securebot_ros.web_faces import FaceEnroller, camera_in_use, read_json

WS = os.path.expanduser('~/securebot_ws')
MAPS_DIR = os.path.join(WS, 'maps')
LOG_DIR = os.path.join(WS, 'log')
RUNS_DIR = os.path.join(LOG_DIR, 'laps_runs')
KEEP_LOGS = 10                # newest runs / log files / build logs kept in log/ (prune_logs)
TOOLS_DIR = os.path.join(WS, 'tools')
SERIAL_DIR = '/dev/serial/by-path'
LIDAR_PORT = '/dev/serial/by-path/platform-3610000.usb-usb-0:2.3:1.0-port0'

COMMANDS = ('start', 'stop', 'home')
MAX_NOPASS = 500              # lines per room (a sanity limit for what the page sends)
DELETED_DIR = 'deleted'       # maps/deleted/: where Delete moves a room's map files
STATUS_FRESH = 3.0            # s: /laps/status older than this = laps node not answering
STOP_GRACE = (20.0, 5.0)      # s after SIGINT, then after SIGTERM, before SIGKILL
PANEL_SETTINGS = os.path.join(WS, 'config', 'panel.json')   # {"faces": bool}


# ------------------------------------------------------------------------------ rooms, maps
def load_rooms(maps_dir=MAPS_DIR):
    """[{'name', 'map', 'yaml'}] of rooms.json whose map file exists."""
    try:
        with open(os.path.join(maps_dir, 'rooms.json')) as f:
            rooms = json.load(f).get('rooms', [])
    except (OSError, ValueError):
        return []
    out = []
    for r in rooms:
        path = os.path.join(maps_dir, f"{r.get('map', '')}.yaml")
        if r.get('name') and os.path.exists(path):
            out.append({'name': r['name'], 'map': r['map'], 'yaml': path})
    return out


def find_room(rooms, name):
    return next((r for r in rooms if r['name'] == name), None)


def load_map(yaml_path):
    """(png bytes, meta {resolution, origin [x, y], width, height}) of a saved map."""
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    image = meta['image']
    if not os.path.isabs(image):
        image = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), image)
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'cannot read map image {image}')
    ok, png = cv2.imencode('.png', img)
    if not ok:
        raise ValueError(f'cannot encode {image}')
    return png.tobytes(), {'resolution': float(meta['resolution']),
                           'origin': [float(v) for v in meta['origin'][:2]],
                           'width': int(img.shape[1]), 'height': int(img.shape[0])}


# ------------------------------------------------------------------------------ hardware
def esp32_port(ports, lidar_port=LIDAR_PORT):
    """The ESP32's port: the one USB serial port that is not the LiDAR's (both are CP2102s,
    so only the socket tells them apart; the ESP32 has moved between sockets 2.3 and 2.4).
    None if there is not exactly one other port."""
    others = sorted(p for p in ports if p != lidar_port)
    return others[0] if len(others) == 1 else None


def serial_ports(serial_dir=SERIAL_DIR):
    return sorted(glob.glob(os.path.join(serial_dir, '*')))


def other_stack_running(ps_output):
    """True if `ps -eo args` output shows a SecureBot launch the panel did not start."""
    for line in ps_output.splitlines():
        if 'ros2 launch' in line and 'securebot_' in line and 'panel:=web' not in line:
            return True
    return False


# ------------------------------------------------------------------------------ the stack
def launch_command(yaml_path, room, port, home='', rounds=0, near=None, faces=False):
    """The laps launch, as the panel runs it: no RViz, no Tk window, auto-localize (looking
    near `near` (x, y) first: where the last run in this room ended), face recognition
    alongside when `faces`."""
    cmd = ['ros2', 'launch', 'securebot_ros', 'securebot_laps.launch.py',
           f'map:={yaml_path}', f'room:={room}', f'esp32_port:={port}',
           'rviz:=false', 'panel:=web', 'auto_localize:=true', f'rounds:={int(rounds)}',
           f'faces:={"true" if faces else "false"}']
    if home:
        cmd.append(f'home:={home}')
    if near is not None:
        cmd.append(f'localize_near:={near[0]:.2f},{near[1]:.2f}')
    return cmd


def explore_command(room, port, faces=False):
    """The explore launch, as the panel runs it: no RViz, no Tk windows; it saves the map as
    `room` when the room is scanned or Stop is pressed."""
    return ['ros2', 'launch', 'securebot_ros', 'securebot_explore.launch.py',
            f'room:={room}', f'esp32_port:={port}', 'rviz:=false', 'panel:=web',
            'rooms:=false', f'faces:={"true" if faces else "false"}']


def last_pose(runs_dir, room):
    """(x, y) where the newest recorded run in `room` ended (meta_<run>.json names the room,
    track_<run>.csv the poses), or None."""
    for r in list_runs(runs_dir, limit=None):
        meta = os.path.join(runs_dir, f'meta_{r["name"]}.json')
        track = os.path.join(runs_dir, f'track_{r["name"]}.csv')
        try:
            with open(meta) as f:
                if json.load(f).get('room') != room:
                    continue
            with open(track) as f:
                lines = f.read().strip().splitlines()
        except (OSError, ValueError):
            continue
        for line in reversed(lines[1:]):
            parts = line.split(',')
            try:
                x, y = float(parts[1]), float(parts[2])
            except (IndexError, ValueError):
                continue
            if (x, y) != (0.0, 0.0):              # 0,0 = the track before localizing
                return x, y
    return None


def parse_home(text):
    """'x,y' -> 'x,y' (normalised) or '' for empty; ValueError if malformed."""
    text = (text or '').strip()
    if not text:
        return ''
    x, y = (float(v) for v in text.split(','))
    return f'{x:.2f},{y:.2f}'


def localization_from_log(lines):
    """What auto-localize said last: (state, text); state None, 'searching', 'ok', 'failed'."""
    state, text = None, ''
    for line in lines:
        if 'global_localize: sent /initialpose' in line:
            state, text = 'ok', line.split('global_localize: ', 1)[1].strip()
        elif 'global_localize: not sending a pose' in line or (
                'No clear match' in line and 'trying again' not in line):
            state, text = 'failed', 'No clear match: the robot could not find itself on the map'
        elif 'global_localize: waiting for AMCL' in line or 'global_localize: attempt' in line:
            state, text = 'searching', 'Finding the robot on the map'
    return state, text


class StackManager:
    """Starts and stops the laps launch (and the run recorder) as process groups."""

    def __init__(self, runs_dir=RUNS_DIR, tools_dir=TOOLS_DIR, popen=subprocess.Popen,
                 killpg=os.killpg, clock=time.monotonic, sleep=time.sleep, record=True):
        self.runs_dir, self.tools_dir = runs_dir, tools_dir
        self.popen, self.killpg, self.clock, self.sleep = popen, killpg, clock, sleep
        self.record = record
        self.proc = self.recorder = None
        self.room = self.run_name = self.log_path = None
        self.kind = 'laps'                       # what is running: 'laps' or 'explore'
        self.state, self.message = 'off', ''
        self.lock = threading.Lock()
        self.log_file = None
        self._loc_path, self._loc_offset, self._loc = None, 0, (None, '')

    def start(self, room, yaml_path, port, home='', rounds=0, name=None, near=None,
              kind='laps', faces=False):
        """Start the laps launch for `room` (kind 'laps') or map it (kind 'explore':
        yaml_path unused, the map is saved as `room`)."""
        with self.lock:
            if self.state in ('starting', 'running', 'stopping'):
                raise RuntimeError(f'The robot is already {self.state}')
            os.makedirs(self.runs_dir, exist_ok=True)
            self.run_name = name or time.strftime('%Y%m%d_%H%M%S')
            self.log_path = os.path.join(self.runs_dir, f'run_{self.run_name}.log')
            self.log_file = open(self.log_path, 'w')
            with open(os.path.join(self.runs_dir, f'meta_{self.run_name}.json'), 'w') as f:
                json.dump({'room': room, 'map': yaml_path, 'kind': kind}, f)
            prune_runs(self.runs_dir)             # the new run is the newest: it is kept
            cmd = explore_command(room, port, faces) if kind == 'explore' else \
                launch_command(yaml_path, room, port, home, rounds, near, faces)
            self.proc = self.popen(cmd, stdout=self.log_file, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
            self.recorder = None
            self.kind = kind
            rec = os.path.join(self.tools_dir, 'run_recorder.py')
            if self.record and kind == 'laps' and os.path.exists(rec):
                self.recorder = self.popen(
                    ['python3', rec, '--name', self.run_name, '--dir', self.runs_dir],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL, start_new_session=True)
            self.room = room
            self.state, self.message = 'starting', (
                f'Starting up to map {room}' if kind == 'explore' else f'Starting up in {room}')

    def mark_running(self):
        """The laps node answers: starting -> running."""
        with self.lock:
            if self.state == 'starting':
                self.state, self.message = 'running', (
                    f'Ready to map {self.room}: press Start' if self.kind == 'explore'
                    else f'Ready in {self.room}')

    def poll(self):
        """Notice a launch that ended by itself (crash, Ctrl+C elsewhere)."""
        with self.lock:
            if self.proc is not None and self.state in ('starting', 'running') \
                    and self.proc.poll() is not None:
                code = self.proc.returncode
                self._stop_group(self.recorder)
                self._closed()
                self.state = 'crashed'
                self.message = (f'The robot software stopped by itself (exit code {code}); '
                                'see the log')
        return self.state

    def stop(self):
        """Stop everything (blocking: up to sum(STOP_GRACE) s); call it from a thread."""
        with self.lock:
            if self.proc is None or self.state == 'stopping':
                return
            self.state, self.message = 'stopping', 'Shutting down'
            proc, rec = self.proc, self.recorder
        self._stop_group(rec)
        self._stop_group(proc)
        with self.lock:
            self._closed()
            self.state, self.message = 'off', 'Robot software is off'

    def _stop_group(self, proc):
        if proc is None or proc.poll() is not None:
            return
        for sig, grace in ((signal.SIGINT, STOP_GRACE[0]), (signal.SIGTERM, STOP_GRACE[1]),
                           (signal.SIGKILL, 2.0)):
            try:
                self.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            end = self.clock() + grace
            while self.clock() < end:
                if proc.poll() is not None:
                    return
                self.sleep(0.2)

    def _closed(self):
        self.proc = self.recorder = None
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def log_tail(self, n=40, max_bytes=16384):
        """The log's last n lines (reads only its last max_bytes)."""
        if not self.log_path or not os.path.exists(self.log_path):
            return []
        with open(self.log_path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - max_bytes))
            text = f.read().decode(errors='replace')
        return text.splitlines()[-n:]

    def localization(self):
        """localization_from_log over the run's log, reading only what is new since last time
        (the page asks twice a second and a run's log grows to megabytes)."""
        if self.log_path != self._loc_path:
            self._loc_path, self._loc_offset, self._loc = self.log_path, 0, (None, '')
        if not self.log_path or not os.path.exists(self.log_path):
            return self._loc
        with open(self.log_path, 'rb') as f:
            f.seek(self._loc_offset)
            data = f.read()
        end = data.rfind(b'\n') + 1                # whole lines only
        self._loc_offset += end
        state, text = localization_from_log(data[:end].decode(errors='replace').splitlines())
        if state:
            self._loc = (state, text)
        return self._loc


RUN_PREFIXES = ('run_', 'track_', 'imu_cmd_', 'paths_', 'events_', 'meta_')


def prune_runs(runs_dir=RUNS_DIR, keep=KEEP_LOGS):
    """Delete all but the `keep` newest runs (each run's files together). Returns the names
    of the runs deleted."""
    old = list_runs(runs_dir, limit=None)[keep:]
    for r in old:
        for f in r['files']:
            try:
                os.remove(os.path.join(runs_dir, f))
            except OSError:
                pass
    return [r['name'] for r in old]


def prune_logs(log_dir=LOG_DIR, keep=KEEP_LOGS):
    """Keep only the `keep` newest of each kind of log in log/: laps runs (laps_runs/, a run's
    files together), files in explore_runs/, and colcon's build_* and test_* folders (never one
    that latest / latest_build / latest_test points to). Returns how many were deleted."""
    if not os.path.isdir(log_dir):
        return 0
    deleted = len(prune_runs(os.path.join(log_dir, 'laps_runs'), keep))
    explore = os.path.join(log_dir, 'explore_runs')
    files = sorted((p for p in glob.glob(os.path.join(explore, '*')) if os.path.isfile(p)),
                   key=os.path.getmtime, reverse=True)
    for path in files[keep:]:
        try:
            os.remove(path)
            deleted += 1
        except OSError:
            pass
    linked = {os.path.realpath(os.path.join(log_dir, n))
              for n in ('latest', 'latest_build', 'latest_test')
              if os.path.islink(os.path.join(log_dir, n))}
    for kind in ('build_', 'test_'):
        dirs = sorted((p for p in glob.glob(os.path.join(log_dir, kind + '*'))
                       if os.path.isdir(p) and not os.path.islink(p)),
                      key=os.path.getmtime, reverse=True)
        for path in dirs[keep:]:
            if os.path.realpath(path) in linked:
                continue
            shutil.rmtree(path, ignore_errors=True)
            deleted += 1
    return deleted


def list_runs(runs_dir=RUNS_DIR, limit=20):
    """Newest first: [{'name', 'files': [...], 'time'}] grouped by run name (limit None: all)."""
    runs = {}
    for path in glob.glob(os.path.join(runs_dir, '*')):
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]
        for prefix in RUN_PREFIXES:
            if stem.startswith(prefix):
                name = stem[len(prefix):]
                r = runs.setdefault(name, {'name': name, 'files': [], 'time': 0.0})
                r['files'].append(base)
                r['time'] = max(r['time'], os.path.getmtime(path))
                break
    out = sorted(runs.values(), key=lambda r: -r['time'])[:limit]
    for r in out:
        r['files'].sort()
    return out


# ------------------------------------------------------------------------------ ROS messages
# (duck-typed: they only read attributes, so the tests pass simple stand-ins)
CUBE, CYLINDER, ADD = 1, 3, 0          # visualization_msgs/Marker type / action values


def boxes_from_markers(markers):
    """[[x, y, size]] of the orange boxes in a /laps/debris MarkerArray (the full set)."""
    return [[round(m.pose.position.x, 3), round(m.pose.position.y, 3), round(m.scale.x, 3)]
            for m in markers if m.type == CUBE and m.action == ADD]


def home_from_markers(markers):
    """[x, y] of the HOME disc in a /laps/home MarkerArray, or None."""
    for m in markers:
        if m.type == CYLINDER and m.action == ADD and m.ns == 'home':
            return [round(m.pose.position.x, 3), round(m.pose.position.y, 3)]
    return None


def thin_path(points, max_points=1500):
    """[[x, y]] rounded to cm, at most max_points (every n-th, always keeping the last)."""
    pts = [[round(x, 2), round(y, 2)] for x, y in points]
    if len(pts) <= max_points:
        return pts
    every = -(-len(pts) // max_points)
    out = pts[::every]
    if out[-1] != pts[-1]:
        out.append(pts[-1])
    return out


def occupancy_png(data, width, height):
    """A nav_msgs/OccupancyGrid's cells (row 0 = bottom, -1 unknown, 0..100) as a PNG the way
    map_saver writes maps (free 254, wall 0, unknown 205; row 0 = top)."""
    grid = np.asarray(data, dtype=np.int16).reshape(height, width)
    img = np.full((height, width), 205, np.uint8)
    img[(grid >= 0) & (grid <= 25)] = 254
    img[grid >= 65] = 0
    ok, png = cv2.imencode('.png', np.flipud(img))
    if not ok:
        raise ValueError('cannot encode the live map')
    return png.tobytes()


def parse_lines(lines):
    """No-pass lines from the page, [[[x, y], [x, y]], ...] -> [((x, y), (x, y))]; ValueError
    if malformed."""
    if not isinstance(lines, list) or len(lines) > MAX_NOPASS:
        raise ValueError(f'lines must be a list of up to {MAX_NOPASS} lines')
    out = []
    for line in lines:
        try:
            (ax, ay), (bx, by) = line
            a, b = (float(ax), float(ay)), (float(bx), float(by))
        except (TypeError, ValueError):
            raise ValueError('Each line is [[x1, y1], [x2, y2]]') from None
        if not all(math.isfinite(v) and abs(v) < 1e4 for v in a + b):
            raise ValueError('Line ends must be finite map coordinates')
        out.append((a, b))
    return out


# ------------------------------------------------------------------------------ the app
class PanelApp:
    """The panel's state and actions, used by the HTTP handler."""

    def __init__(self, bridge, stack=None, maps_dir=MAPS_DIR, runs_dir=RUNS_DIR,
                 tools_dir=TOOLS_DIR, ports=serial_ports, ps=None, clock=time.monotonic,
                 spawn=None, faces=None, camera=None, alerts=None,
                 power_file=pl.SETTINGS_FILE, settings_file=PANEL_SETTINGS):
        self.bridge, self.maps_dir, self.runs_dir = bridge, maps_dir, runs_dir
        self.alerts = alerts or AlertStore()
        self.power_file, self.settings_file = power_file, settings_file
        self.tamper_lock = threading.Lock()
        self.tamper_seen = 'unavailable'     # last /tamper state (for the tripped edge)
        self.faces = faces or FaceEnroller()
        self.tools_dir = tools_dir
        self.stack = stack or StackManager(runs_dir, tools_dir)
        self.ports, self.clock = ports, clock
        self.ps = ps or (lambda: subprocess.run(['ps', '-eo', 'args'], capture_output=True,
                                                text=True).stdout)
        self.spawn = spawn or (lambda target: threading.Thread(target=target,
                                                               daemon=True).start())
        self.camera = camera or CameraFeed(busy=self.camera_busy)
        self.relocalizing = False
        self.reloc_text = ''
        self.map_cache = {}
        self.active = None            # the room on the main screen (set_active)
        self.pending_start = False    # Start pressed while off: start laps once localized
        self.note = ''                # what the last Start-from-off is waiting for

    def camera_busy(self):
        """Why the camera view cannot open the camera now, or None."""
        f = self.faces.state()
        if f['busy'] == 'capture':
            return 'The camera is taking face photos; the view comes back after that'
        if camera_in_use(self.ps()):
            return 'The live face node has the camera'
        return None

    # actions -------------------------------------------------------------------------------
    def _ready_to_power_on(self):
        if self.stack.state in ('starting', 'running', 'stopping'):
            raise RuntimeError(f'The robot is already {self.stack.state}')
        if other_stack_running(self.ps()):
            raise RuntimeError('Robot software started from a terminal is running; stop it '
                               'there first (Ctrl+C)')
        port = esp32_port(self.ports())
        if port is None:
            raise RuntimeError('Cannot find the ESP32 USB port: check that the ESP32 and the '
                               'LiDAR are both plugged in')
        reset = getattr(self.bridge, 'reset', None)
        if reset:
            reset()                       # forget the last run's lanes, boxes and home
        return port

    def power_on(self, room_name, home='', rounds=0):
        room = find_room(load_rooms(self.maps_dir), room_name)
        if room is None:
            raise ValueError(f'Unknown room "{room_name}"')
        home = parse_home(home)
        if int(rounds) < 0:
            raise ValueError('Rounds must be 0 (until Stop) or more')
        port = self._ready_to_power_on()
        near = last_pose(self.stack.runs_dir, room['name'])
        self.stack.start(room['name'], room['yaml'], port, home, rounds, near=near,
                         faces=self.settings()['faces'])
        self.active = room['name']
        self.pending_start, self.note = False, ''
        return f'Starting up in {room["name"]}'

    def start(self, room_name=''):
        """The Start button. Off: power on for the room (default: the active one) and start
        the laps once the robot has found itself. On: start (laps or exploring)."""
        self.check_can_drive()
        st = self.stack.state
        if st == 'stopping':
            raise RuntimeError('The robot software is shutting down; wait a moment')
        if st in ('starting', 'running'):
            room_name = room_name or self.stack.room
            if self.stack.kind == 'laps' and room_name != self.stack.room:
                raise RuntimeError(f'The robot is running in {self.stack.room}: Power off '
                                   f'first to switch to {room_name}')
            if st == 'starting':
                if self.stack.kind == 'explore':
                    raise RuntimeError('Still starting up: press Start when it says Ready')
                self.pending_start = True
                self.note = 'Starts the laps once the robot has found itself on the map'
                return self.note
            return self.command('start')
        room_name = room_name or self.active_room()
        if not room_name:
            raise RuntimeError('No room yet: add one in the Map Manager')
        self.power_on(room_name)
        self.pending_start = True
        self.note = 'Starts the laps once the robot has found itself on the map'
        return f'Powering on in {room_name}; the laps start once it has found itself'

    def explore(self, name):
        """Map Manager's Add: power the explore launch on to map a new room called `name`."""
        name = rs.clean_name(name)
        store = rs.RoomStore(self.maps_dir)
        if store.find(name):
            raise RuntimeError(f'A room called "{name}" already exists: pick another name '
                               'or delete it first')
        port = self._ready_to_power_on()
        self.stack.start(name, '', port, kind='explore', faces=self.settings()['faces'])
        self.active = name
        self.pending_start, self.note = False, ''
        return (f'Starting up to map {name}. Put the robot in the room, then press Start; '
                'Stop saves the map')

    def active_room(self):
        """The room on the main screen: what runs, else the one picked, else the first."""
        rooms = [r['name'] for r in load_rooms(self.maps_dir)]
        if self.stack.state in ('starting', 'running', 'stopping') and self.stack.room:
            return self.stack.room
        if self.active in rooms:
            return self.active
        return rooms[0] if rooms else None

    def set_active(self, room_name):
        if find_room(load_rooms(self.maps_dir), room_name) is None:
            raise ValueError(f'Unknown room "{room_name}"')
        if self.stack.state in ('starting', 'running', 'stopping') and \
                room_name != self.stack.room:
            raise RuntimeError(f'The robot is running in {self.stack.room}: Power off first')
        self.active = room_name
        return f'{room_name} is on the main screen'

    # map manager ---------------------------------------------------------------------------
    def _room_in_use(self, name):
        if self.stack.state in ('starting', 'running', 'stopping') and \
                self.stack.room == name:
            raise RuntimeError(f'The robot is using {name}: Power off first')

    def maps(self):
        """The rooms for the Map Manager: name, map stem, size, saved time, no-pass lines."""
        out = []
        for name, info in rs.RoomStore(self.maps_dir).entries():
            if name is None or info is None:
                continue
            w, h = info.size_m
            lines = npz.NoPass(npz.path_for(info.yaml_path)).lines
            out.append({'name': name, 'map': info.stem, 'width_m': round(w, 1),
                        'height_m': round(h, 1), 'modified': info.modified,
                        'nopass': len(lines)})
        return out

    def rename_room(self, old, new):
        self._room_in_use(old)
        new = rs.RoomStore(self.maps_dir).rename(old, new)
        if self.active == old:
            self.active = new
        return f'Renamed to {new}'

    def delete_room(self, name):
        """Forget the room and move its map files (and no-pass lines) to maps/deleted/."""
        self._room_in_use(name)
        store = rs.RoomStore(self.maps_dir)
        room = store.find(name)
        if room is None:
            raise ValueError(f'There is no room called "{name}"')
        stem = room['map']
        info = rs.read_map(os.path.join(self.maps_dir, stem + '.yaml'))
        files = [os.path.join(self.maps_dir, stem + '.yaml'),
                 npz.path_for(os.path.join(self.maps_dir, stem + '.yaml'))]
        if info and os.path.dirname(os.path.abspath(info.image_path)) == \
                os.path.abspath(self.maps_dir):
            files.append(info.image_path)
        if not any(r['map'] == stem for r in store.rooms if r is not room):
            trash = os.path.join(self.maps_dir, DELETED_DIR)
            os.makedirs(trash, exist_ok=True)
            stamp = time.strftime('%Y%m%d_%H%M%S')
            for f in files:
                if os.path.isfile(f):
                    shutil.move(f, os.path.join(trash, f'{stamp}_{os.path.basename(f)}'))
        store.remove(room['name'])
        if self.active == room['name']:
            self.active = None
        return f'Deleted {room["name"]} (its map files are in maps/{DELETED_DIR}/)'

    def nopass_lines(self, room_name):
        room = find_room(load_rooms(self.maps_dir), room_name)
        if room is None:
            raise ValueError(f'Unknown room "{room_name}"')
        np_ = npz.NoPass(npz.path_for(room['yaml']))
        return {'width': np_.width, 'lines': [[list(a), list(b)] for a, b in np_.lines]}

    def save_nopass(self, room_name, lines):
        """Replace the room's no-pass lines; a laps run in that room re-plans with them."""
        room = find_room(load_rooms(self.maps_dir), room_name)
        if room is None:
            raise ValueError(f'Unknown room "{room_name}"')
        lines = parse_lines(lines)
        np_ = npz.NoPass(npz.path_for(room['yaml']))
        np_.lines = lines
        np_.save()
        live = self.stack.state in ('starting', 'running') and \
            self.stack.kind == 'laps' and self.stack.room == room['name']
        if live:
            self.bridge.send_command('nopass reload')
        return f'Saved {len(lines)} no-pass line{"s" if len(lines) != 1 else ""}' + (
            ': the robot re-plans with them' if live else '')

    def power_off(self):
        self.pending_start, self.note = False, ''
        if self.stack.state in ('off', 'crashed'):
            self.stack.state, self.stack.message = 'off', 'Robot software is off'
            return 'Already off'
        if self.stack.kind == 'explore':
            self.bridge.send_command('stop', 'explore')
        else:
            self.bridge.send_command('stop')
        self.spawn(self.stack.stop)
        return 'Shutting down'

    def command(self, cmd):
        if cmd not in COMMANDS:
            raise ValueError(f'Unknown command "{cmd}"')
        if cmd == 'stop':
            self.pending_start, self.note = False, ''
        else:
            self.check_can_drive(home=cmd == 'home')
        if self.stack.state not in ('running', 'starting'):
            if cmd == 'stop':
                return 'The robot software is off: nothing is moving'
            raise RuntimeError('Press Start first (the robot software is off)')
        explore = self.stack.kind == 'explore'
        if explore:
            self.bridge.send_command(cmd, 'explore')
        else:
            self.bridge.send_command(cmd)
        return {'start': 'Exploring' if explore else 'Starting laps',
                'stop': 'Stopping; the map is saved' if explore else 'Stopping',
                'home': 'Going home'}[cmd]

    def click(self, x, y):
        if self.stack.state != 'running' or self.stack.kind != 'laps':
            raise RuntimeError('Power on the robot first')
        self.bridge.click(float(x), float(y))
        return 'Removing the box there (if any)'

    def relocalize(self):
        if self.stack.state != 'running':
            raise RuntimeError('Press Start first (the robot software is off)')
        if self.stack.kind == 'explore':
            raise RuntimeError('While mapping, the robot is always on its own map')
        if self.relocalizing:
            raise RuntimeError('Already re-localizing')
        room = find_room(load_rooms(self.maps_dir), self.stack.room)
        if room is None:
            raise RuntimeError('Room map not found')
        self.bridge.send_command('stop')
        self.pending_start = False
        self.relocalizing, self.reloc_text = True, 'Finding the robot on the map'
        cmd = ['python3', os.path.join(self.tools_dir, 'global_localize.py'),
               '--map', room['yaml'], '--tries', '3', '--turn']

        def run():
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                state, text = localization_from_log(out.stdout.splitlines())
                self.reloc_text = text if state else 'Re-localize gave no answer'
            except (OSError, subprocess.TimeoutExpired) as e:
                self.reloc_text = f'Re-localize failed: {e}'
            finally:
                self.relocalizing = False
        self.spawn(run)
        return 'Re-localizing (the robot stays still)'

    # battery, tamper, settings -------------------------------------------------------------
    def settings(self):
        """Panel settings: {'faces': run face recognition during laps / mapping}."""
        data = read_json(self.settings_file)
        return {'faces': bool(data.get('faces', False)) if isinstance(data, dict) else False}

    def set_faces(self, on):
        os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
        with open(self.settings_file + '.tmp', 'w') as f:
            json.dump({**self.settings(), 'faces': bool(on)}, f)
        os.replace(self.settings_file + '.tmp', self.settings_file)
        running = self.stack.state in ('starting', 'running')
        return ('Face recognition ' + ('on' if on else 'off')
                + (' from the next Power on' if running else ' during laps and mapping'))

    def power(self):
        """The guard node's battery status (pct None: no reading), or None when it is not
        running."""
        get = getattr(self.bridge, 'power', None)
        return get() if get else None

    def tamper(self):
        """(state, reason) from the ESP32, or ('unavailable', '') when nothing reports."""
        get = getattr(self.bridge, 'tamper', None)
        return (get() if get else None) or ('unavailable', '')

    def check_can_drive(self, home=False):
        """RuntimeError when the robot must not start driving: tampered with (until the
        alert is acknowledged) or, for Start, a low battery."""
        state, reason = self.tamper()
        if state == 'tripped':
            raise RuntimeError(f'The robot was {pl.tamper_text(reason)}: acknowledge the alert '
                               'first (the motors stay off until then)')
        power = self.power()
        if not home and power and power.get('low'):
            raise RuntimeError(f'Battery low ({power["pct"]:.0f} %, returns home at '
                               f'{power["return_home_pct"]} %): charge it first')

    def power_settings(self):
        return pl.load_settings(self.power_file)

    def set_power_settings(self, changes):
        if not isinstance(changes, dict):
            raise ValueError('Expected settings')
        new = {**self.power_settings(),
               **{k: changes[k] for k in pl.DEFAULTS if k in changes}}
        saved = pl.save_settings(new, self.power_file)
        return f'Returns home at {saved["return_home_pct"]} % ({saved["cells"]}S pack)'

    def tamper_event(self, state, reason):
        """Every /tamper message (ROS thread). A new 'tripped' becomes an alert with a
        snapshot (taken in the background: the camera may need a moment to start)."""
        with self.tamper_lock:
            new_trip = state == 'tripped' and self.tamper_seen != 'tripped'
            self.tamper_seen = state
        if not new_trip:
            return
        self.pending_start, self.note = False, ''

        def record():
            jpeg = None
            try:
                jpeg = self.camera.snapshot()
            except Exception:
                pass
            self.alerts.add('tamper', reason,
                            f'Robot {pl.tamper_text(reason)}: motors stopped', jpeg)
        self.spawn(record)

    def ack_alert(self, alert_id=''):
        """Acknowledge one alert (or all); if the robot is still in its tamper alarm, let the
        motors run again."""
        n = self.alerts.ack(alert_id or None)
        state, _ = self.tamper()
        if state == 'tripped':
            send = getattr(self.bridge, 'tamper_command', None)
            if send:
                send('clear')
            return 'Acknowledged: the motors are enabled again (press Start to continue)'
        return 'Acknowledged' if n else 'Nothing to acknowledge'

    # state ---------------------------------------------------------------------------------
    def state(self):
        stack_state = self.stack.poll()
        kind = self.stack.kind
        robot = self.bridge.snapshot(kind) if stack_state in ('starting', 'running') else {}
        status = robot.get('status')
        age = robot.get('status_age')
        fresh = status is not None and age is not None and age < STATUS_FRESH
        if fresh:
            self.stack.mark_running()
            stack_state = self.stack.state
        loc_state, loc_text = self.stack.localization()
        if self.relocalizing or self.reloc_text:
            loc_state = 'searching' if self.relocalizing else loc_state
            loc_text = self.reloc_text or loc_text
        if stack_state not in ('starting', 'running'):
            self.pending_start, self.note = False, ''
        elif self.pending_start and fresh and kind == 'laps' and not self.relocalizing:
            if loc_state == 'ok':
                self.pending_start, self.note = False, ''
                self.bridge.send_command('start')
            elif loc_state == 'failed':
                self.pending_start = False
                self.note = ('Not started: the robot could not find itself. Use Manual '
                             're-localize, then Start')
        return {
            'stack': {'state': stack_state, 'message': self.stack.message,
                      'room': self.stack.room, 'run': self.stack.run_name,
                      'kind': kind, 'log': self.stack.log_tail(30)},
            'active': self.active_room(),
            'pending_start': self.pending_start,
            'note': self.note,
            'battery': self.power(),
            'tamper': dict(zip(('state', 'reason'), self.tamper())),
            'alerts': self.alerts.unacked()[:5],
            'settings': self.settings(),
            'face_live': (getattr(self.bridge, 'faces', None) or (lambda: None))(),
            'camera': self.camera.state(),
            'live_map': robot.get('live_map'),
            'laps': status if fresh and kind == 'laps' else None,
            'explore': status if fresh and kind == 'explore' else None,
            'connected': fresh,
            'localization': {'state': loc_state, 'text': loc_text,
                             'busy': self.relocalizing},
            'pose': robot.get('pose'),
            'path': robot.get('path', []),
            'current': robot.get('current', []),
            'boxes': robot.get('boxes', []),
            'home': robot.get('home'),
            'esp32': robot.get('esp32'),
            'rooms': [r['name'] for r in load_rooms(self.maps_dir)],
            'port': esp32_port(self.ports()),
            'faces': self.faces.state(),
        }

    def live_map_png(self):
        """The SLAM map being built (exploring), as PNG bytes, or None."""
        get = getattr(self.bridge, 'live_png', None)
        exploring = self.stack.kind == 'explore' and self.stack.state in ('starting', 'running')
        return get() if get and exploring else None

    def map_for(self, room_name):
        room = find_room(load_rooms(self.maps_dir), room_name)
        if room is None:
            raise ValueError(f'Unknown room "{room_name}"')
        key = (room['yaml'], os.path.getmtime(room['yaml']))
        if key not in self.map_cache:
            self.map_cache[key] = load_map(room['yaml'])
        return self.map_cache[key]


# ------------------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    app = None            # set by make_server
    static_dir = None
    server_version = 'SecureBotPanel/1.0'

    def log_message(self, fmt, *args):  # quiet: the page polls twice a second
        pass

    def _send(self, code, body, ctype='application/json'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, text):
        self._send(code, {'ok': False, 'error': text})

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            if url.path in ('/', '/index.html'):
                with open(os.path.join(self.static_dir, 'index.html'), 'rb') as f:
                    return self._send(200, f.read(), 'text/html; charset=utf-8')
            if url.path == '/api/state':
                return self._send(200, self.app.state())
            if url.path == '/api/map.png':
                png, _ = self.app.map_for(q.get('room', [''])[0])
                return self._send(200, png, 'image/png')
            if url.path == '/api/map':
                _, meta = self.app.map_for(q.get('room', [''])[0])
                return self._send(200, meta)
            if url.path == '/api/runs':
                return self._send(200, list_runs(self.app.runs_dir, limit=KEEP_LOGS))
            if url.path == '/api/maps':
                return self._send(200, self.app.maps())
            if url.path == '/api/alerts':
                return self._send(200, self.app.alerts.list(20))
            if url.path == '/api/alerts/snapshot.jpg':
                return self._send(200, self.app.alerts.snapshot(q.get('id', [''])[0]),
                                  'image/jpeg')
            if url.path == '/api/power/settings':
                return self._send(200, self.app.power_settings())
            if url.path == '/api/nopass':
                return self._send(200, self.app.nopass_lines(q.get('room', [''])[0]))
            if url.path == '/api/live_map.png':
                png = self.app.live_map_png()
                return self._send(200, png, 'image/png') if png else \
                    self._error(404, 'No live map yet')
            if url.path == '/api/camera.jpg':
                try:
                    jpg = self.app.camera.frame()
                except RuntimeError as e:
                    return self._error(409, str(e))
                return self._send(200, jpg, 'image/jpeg') if jpg else \
                    self._error(503, 'The camera is starting')
            if url.path == '/api/faces/photos':
                return self._send(200, self.app.faces.photos(q.get('name', [''])[0]))
            if url.path == '/api/faces/preview.jpg':
                jpg = self.app.faces.preview()
                return self._send(200, jpg, 'image/jpeg') if jpg else \
                    self._error(404, 'No camera view yet')
            if url.path == '/api/faces/photo':
                jpg = self.app.faces.photo(q.get('kind', [''])[0], q.get('name', [''])[0],
                                           q.get('file', [''])[0])
                return self._send(200, jpg, 'image/jpeg')
            if url.path == '/api/log':
                name = os.path.basename(q.get('file', [''])[0])
                path = os.path.join(self.app.runs_dir, name)
                if not name or not os.path.isfile(path):
                    return self._error(404, 'No such file')
                with open(path, 'rb') as f:
                    return self._send(200, f.read(), 'text/plain; charset=utf-8')
            return self._error(404, 'Not found')
        except ValueError as e:
            return self._error(404, str(e))
        except Exception as e:                  # never drop the connection without an answer
            return self._error(500, f'{type(e).__name__}: {e}')

    def do_POST(self):  # noqa: N802
        try:
            n = int(self.headers.get('Content-Length') or 0)
            body = json.loads(self.rfile.read(n) or b'{}') if n else {}
            if not isinstance(body, dict):
                raise ValueError('Expected a JSON object')
        except ValueError:
            return self._error(400, 'Bad JSON')
        path = urlparse(self.path).path
        try:
            if path == '/api/start':
                text = self.app.start(str(body.get('room', '')))
            elif path == '/api/active':
                text = self.app.set_active(str(body.get('room', '')))
            elif path == '/api/maps/rename':
                text = self.app.rename_room(str(body.get('old', '')), str(body.get('new', '')))
            elif path == '/api/maps/delete':
                text = self.app.delete_room(str(body.get('name', '')))
            elif path == '/api/alerts/ack':
                text = self.app.ack_alert(str(body.get('id', '') or ''))
            elif path == '/api/power/settings':
                text = self.app.set_power_settings(body)
            elif path == '/api/settings/faces':
                text = self.app.set_faces(bool(body.get('on')))
            elif path == '/api/maps/add':
                text = self.app.explore(str(body.get('name', '')))
            elif path == '/api/nopass':
                text = self.app.save_nopass(str(body.get('room', '')), body.get('lines'))
            elif path == '/api/faces/delete_photo':
                text = self.app.faces.delete_photo(str(body.get('name', '')),
                                                   str(body.get('file', '')))
            elif path == '/api/faces/delete_person':
                text = self.app.faces.delete_person(str(body.get('name', '')))
            elif path == '/api/power/on':
                text = self.app.power_on(body.get('room', ''), body.get('home', ''),
                                         body.get('rounds', 0))
            elif path == '/api/power/off':
                text = self.app.power_off()
            elif path == '/api/command':
                text = self.app.command(str(body.get('cmd', '')).lower())
            elif path == '/api/click':
                text = self.app.click(body['x'], body['y'])
            elif path == '/api/relocalize':
                text = self.app.relocalize()
            elif path == '/api/faces/capture':
                text = self.app.faces.start_capture(body.get('name', ''),
                                                    body.get('count', 15),
                                                    bool(body.get('replace', False)))
            elif path == '/api/faces/stop':
                text = self.app.faces.stop_capture()
            elif path == '/api/faces/keep':
                files = body.get('files', [])
                if not isinstance(files, list):
                    raise ValueError('files must be a list')
                text = self.app.faces.keep(str(body.get('name', '')), files)
            elif path == '/api/faces/discard':
                text = self.app.faces.discard(str(body.get('name', '')))
            elif path == '/api/faces/rebuild':
                text = self.app.faces.rebuild()
            else:
                return self._error(404, 'Not found')
        except (ValueError, KeyError, TypeError) as e:
            return self._error(400, str(e) or 'Bad request')
        except RuntimeError as e:
            return self._error(409, str(e))
        except Exception as e:
            return self._error(500, f'{type(e).__name__}: {e}')
        return self._send(200, {'ok': True, 'message': text})


# interfaces nobody browses from: Docker's bridges and container links
HIDDEN_IFACES = ('docker', 'br-', 'veth')


def panel_urls(host, port, ip_output):
    """The addresses to open the panel at, as 'http://<ip>:<port>  (<where>)' lines.

    `ip_output` is `ip -o -4 addr show`. A host of 0.0.0.0 (every network) is not an address
    a browser can open, so it is expanded into this machine's own (127.0.0.1) and one line
    per network interface, leaving out Docker's."""
    if host not in ('0.0.0.0', ''):
        return [f'http://{host}:{port}']
    urls = []
    for line in ip_output.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[2] != 'inet':
            continue
        iface, ip = parts[1], parts[3].split('/')[0]
        if iface.startswith(HIDDEN_IFACES):
            continue
        if iface == 'lo':
            where = 'on the robot itself'
        elif iface.startswith('tailscale'):
            where = 'over Tailscale'
        else:
            where = f'same network, {iface}'
        urls.append((iface != 'lo', f'http://{ip}:{port}  ({where})'))
    if not any(not remote for remote, _ in urls):
        urls.append((False, f'http://127.0.0.1:{port}  (on the robot itself)'))
    return [u for _, u in sorted(urls, key=lambda r: r[0])]


def local_panel_urls(host, port):
    """panel_urls() for this machine's interfaces."""
    try:
        out = subprocess.run(['ip', '-o', '-4', 'addr', 'show'], capture_output=True,
                             text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        out = ''
    return panel_urls(host, port, out)


def make_server(app, host='0.0.0.0', port=8080, static_dir=None):
    handler = type('PanelHandler', (Handler,), {'app': app, 'static_dir': static_dir})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server
