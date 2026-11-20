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
                             capture from the camera, check, keep, rebuild the database

PanelApp holds that state and answers the HTTP API (make_server); the ROS side is passed in
as a `bridge` (web_panel.RosBridge on the robot, a fake one in the tests) with
send_command(cmd), click(x, y) and snapshot() -> dict.
"""
import glob
import json
import os
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import yaml

from securebot_ros.web_faces import FaceEnroller

WS = os.path.expanduser('~/securebot_ws')
MAPS_DIR = os.path.join(WS, 'maps')
RUNS_DIR = os.path.join(WS, 'log', 'laps_runs')
TOOLS_DIR = os.path.join(WS, 'tools')
SERIAL_DIR = '/dev/serial/by-path'
LIDAR_PORT = '/dev/serial/by-path/platform-3610000.usb-usb-0:2.2:1.0-port0'

COMMANDS = ('start', 'stop', 'home')
STATUS_FRESH = 3.0            # s: /laps/status older than this = laps node not answering
STOP_GRACE = (20.0, 5.0)      # s after SIGINT, then after SIGTERM, before SIGKILL


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
def launch_command(yaml_path, room, port, home='', rounds=0):
    """The laps launch, as the panel runs it: no RViz, no Tk window, auto-localize."""
    cmd = ['ros2', 'launch', 'securebot_ros', 'securebot_laps.launch.py',
           f'map:={yaml_path}', f'room:={room}', f'esp32_port:={port}',
           'rviz:=false', 'panel:=web', 'auto_localize:=true', f'rounds:={int(rounds)}']
    if home:
        cmd.append(f'home:={home}')
    return cmd


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
        self.state, self.message = 'off', ''
        self.lock = threading.Lock()
        self.log_file = None
        self._loc_path, self._loc_offset, self._loc = None, 0, (None, '')

    def start(self, room, yaml_path, port, home='', rounds=0, name=None):
        with self.lock:
            if self.state in ('starting', 'running', 'stopping'):
                raise RuntimeError(f'The robot is already {self.state}')
            os.makedirs(self.runs_dir, exist_ok=True)
            self.run_name = name or time.strftime('%Y%m%d_%H%M%S')
            self.log_path = os.path.join(self.runs_dir, f'run_{self.run_name}.log')
            self.log_file = open(self.log_path, 'w')
            self.proc = self.popen(launch_command(yaml_path, room, port, home, rounds),
                                   stdout=self.log_file, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
            self.recorder = None
            rec = os.path.join(self.tools_dir, 'run_recorder.py')
            if self.record and os.path.exists(rec):
                self.recorder = self.popen(
                    ['python3', rec, '--name', self.run_name, '--dir', self.runs_dir],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL, start_new_session=True)
            self.room = room
            self.state, self.message = 'starting', f'Starting up in {room}'

    def mark_running(self):
        """The laps node answers: starting -> running."""
        with self.lock:
            if self.state == 'starting':
                self.state, self.message = 'running', f'Ready in {self.room}'

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


def list_runs(runs_dir=RUNS_DIR, limit=20):
    """Newest first: [{'name', 'files': [...], 'time'}] grouped by run name."""
    runs = {}
    for path in glob.glob(os.path.join(runs_dir, '*')):
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]
        for prefix in ('run_', 'track_', 'imu_cmd_', 'paths_', 'events_'):
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


# ------------------------------------------------------------------------------ the app
class PanelApp:
    """The panel's state and actions, used by the HTTP handler."""

    def __init__(self, bridge, stack=None, maps_dir=MAPS_DIR, runs_dir=RUNS_DIR,
                 tools_dir=TOOLS_DIR, ports=serial_ports, ps=None, clock=time.monotonic,
                 spawn=None, faces=None):
        self.bridge, self.maps_dir, self.runs_dir = bridge, maps_dir, runs_dir
        self.faces = faces or FaceEnroller()
        self.tools_dir = tools_dir
        self.stack = stack or StackManager(runs_dir, tools_dir)
        self.ports, self.clock = ports, clock
        self.ps = ps or (lambda: subprocess.run(['ps', '-eo', 'args'], capture_output=True,
                                                text=True).stdout)
        self.spawn = spawn or (lambda target: threading.Thread(target=target,
                                                               daemon=True).start())
        self.relocalizing = False
        self.reloc_text = ''
        self.map_cache = {}

    # actions -------------------------------------------------------------------------------
    def power_on(self, room_name, home='', rounds=0):
        room = find_room(load_rooms(self.maps_dir), room_name)
        if room is None:
            raise ValueError(f'Unknown room "{room_name}"')
        home = parse_home(home)
        if int(rounds) < 0:
            raise ValueError('Rounds must be 0 (until Stop) or more')
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
        self.stack.start(room['name'], room['yaml'], port, home, rounds)
        return f'Starting up in {room["name"]}'

    def power_off(self):
        if self.stack.state in ('off', 'crashed'):
            self.stack.state, self.stack.message = 'off', 'Robot software is off'
            return 'Already off'
        self.bridge.send_command('stop')
        self.spawn(self.stack.stop)
        return 'Shutting down'

    def command(self, cmd):
        if cmd not in COMMANDS:
            raise ValueError(f'Unknown command "{cmd}"')
        if self.stack.state not in ('running', 'starting'):
            raise RuntimeError('Power on the robot first')
        self.bridge.send_command(cmd)
        return {'start': 'Starting laps', 'stop': 'Stopping',
                'home': 'Going home'}[cmd]

    def click(self, x, y):
        if self.stack.state != 'running':
            raise RuntimeError('Power on the robot first')
        self.bridge.click(float(x), float(y))
        return 'Removing the box there (if any)'

    def relocalize(self):
        if self.stack.state != 'running':
            raise RuntimeError('Power on the robot first')
        if self.relocalizing:
            raise RuntimeError('Already re-localizing')
        room = find_room(load_rooms(self.maps_dir), self.stack.room)
        if room is None:
            raise RuntimeError('Room map not found')
        self.bridge.send_command('stop')
        self.relocalizing, self.reloc_text = True, 'Finding the robot on the map'
        cmd = ['python3', os.path.join(self.tools_dir, 'global_localize.py'),
               '--map', room['yaml'], '--tries', '3']

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

    # state ---------------------------------------------------------------------------------
    def state(self):
        stack_state = self.stack.poll()
        robot = self.bridge.snapshot() if stack_state in ('starting', 'running') else {}
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
        return {
            'stack': {'state': stack_state, 'message': self.stack.message,
                      'room': self.stack.room, 'run': self.stack.run_name,
                      'log': self.stack.log_tail(30)},
            'laps': status if fresh else None,
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
                return self._send(200, list_runs(self.app.runs_dir))
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
            if path == '/api/power/on':
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
                                                    body.get('count', 15))
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


def make_server(app, host='0.0.0.0', port=8080, static_dir=None):
    handler = type('PanelHandler', (Handler,), {'app': app, 'static_dir': static_dir})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server
