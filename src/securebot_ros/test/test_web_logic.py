"""The web panel without ROS or hardware: rooms and maps, the ESP32 port, starting and
stopping the laps launch, the HTTP API (on a real local socket, with a fake ROS bridge)."""
import importlib.util
import json
import os
import signal
import threading
import types
import urllib.error
import urllib.request

import numpy as np
import cv2
import pytest

from securebot_ros import web_logic as wl
from securebot_ros.web_alerts import AlertStore
from securebot_ros.web_faces import FaceEnroller

LIDAR = wl.LIDAR_PORT
ESP = '/dev/serial/by-path/platform-3610000.usb-usb-0:2.4:1.0-port0'
HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------------------ fixtures
@pytest.fixture
def maps(tmp_path):
    """A maps dir with two rooms in rooms.json, one of them with a missing map."""
    d = tmp_path / 'maps'
    d.mkdir()
    img = np.full((40, 60), 254, np.uint8)
    img[0, :] = 0
    cv2.imwrite(str(d / 'lab.pgm'), img)
    (d / 'lab.yaml').write_text('image: lab.pgm\nresolution: 0.05\norigin: [-1.0, -2.0, 0.0]\n'
                                'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
    (d / 'rooms.json').write_text(json.dumps({'rooms': [
        {'name': "Kumar's Lab", 'map': 'lab'}, {'name': 'Gone', 'map': 'missing'}]}))
    return str(d)


class FakeProc:
    """A process that ends on the signals in `dies_on` (or by calling end())."""
    count = 0

    def __init__(self, cmd, dies_on=(signal.SIGINT,), **kw):
        FakeProc.count += 1
        self.pid, self.cmd, self.kw = 1000 + FakeProc.count, cmd, kw
        self.dies_on, self.returncode, self.signals = dies_on, None, []

    def poll(self):
        return self.returncode

    def end(self, code):
        self.returncode = code


class FakeOS:
    """popen / killpg / clock / sleep for StackManager: no real processes, no waiting."""

    def __init__(self, dies_on=(signal.SIGINT,)):
        self.procs, self.t, self.dies_on = [], 0.0, dies_on

    def popen(self, cmd, **kw):
        p = FakeProc(cmd, self.dies_on, **kw)
        self.procs.append(p)
        return p

    def killpg(self, pid, sig):
        p = next(p for p in self.procs if p.pid == pid)
        p.signals.append(sig)
        if sig in p.dies_on or sig == signal.SIGKILL:
            p.returncode = -sig

    def clock(self):
        return self.t

    def sleep(self, s):
        self.t += s


def stack_for(tmp_path, fos, tools=True):
    tools_dir = tmp_path / 'tools'
    tools_dir.mkdir(exist_ok=True)
    if tools:
        (tools_dir / 'run_recorder.py').write_text('')
    return wl.StackManager(str(tmp_path / 'runs'), str(tools_dir), popen=fos.popen,
                           killpg=fos.killpg, clock=fos.clock, sleep=fos.sleep)


class FakeBridge:
    def __init__(self):
        self.sent, self.clicks, self.resets = [], [], 0
        self.snap = {}

    def send_command(self, cmd, target='laps'):
        self.sent.append(cmd if target == 'laps' else f'{target}:{cmd}')

    def click(self, x, y):
        self.clicks.append((x, y))

    def reset(self):
        self.resets += 1

    def snapshot(self, kind='laps'):
        self.kind = kind
        return dict(self.snap)

    # guard node / ESP32 tamper / live face node (None: not running)
    power_status = tamper_status = face_status = None

    def power(self):
        return self.power_status

    def tamper(self):
        return self.tamper_status

    def faces(self):
        return self.face_status

    def tamper_command(self, cmd):
        self.sent.append(f'tamper:{cmd}')


class FakeCamera:
    def __init__(self):
        self.jpg, self.why = b'\xff\xd8jpg', None

    def frame(self):
        if self.why:
            raise RuntimeError(self.why)
        return self.jpg

    def state(self):
        return {'on': bool(self.jpg), 'error': ''}

    def snapshot(self):
        return self.jpg


def ticking_clock(start=1764280800.0):
    """A wall clock that moves 1 s per reading (alerts get distinct ids)."""
    t = [start - 1.0]

    def clock():
        t[0] += 1.0
        return t[0]
    return clock


def app_for(tmp_path, maps, fos, ports=(LIDAR, ESP), ps=''):
    bridge = FakeBridge()
    faces = FaceEnroller(base=str(tmp_path / 'faces'), venv=str(tmp_path / 'venv'),
                         popen=fos.popen, killpg=fos.killpg, ps=lambda: ps)
    app = wl.PanelApp(bridge, stack=stack_for(tmp_path, fos), maps_dir=maps,
                      runs_dir=str(tmp_path / 'runs'), tools_dir=str(tmp_path / 'tools'),
                      ports=lambda: list(ports), ps=lambda: ps, spawn=lambda f: f(),
                      faces=faces, camera=FakeCamera(),
                      alerts=AlertStore(str(tmp_path / 'alerts'), clock=ticking_clock()),
                      power_file=str(tmp_path / 'config' / 'power.json'),
                      settings_file=str(tmp_path / 'config' / 'panel.json'))
    return app, bridge


# ------------------------------------------------------------------------------ rooms, maps
def test_rooms_with_a_missing_map_are_left_out(maps):
    rooms = wl.load_rooms(maps)
    assert [r['name'] for r in rooms] == ["Kumar's Lab"]
    assert wl.find_room(rooms, "Kumar's Lab")['yaml'].endswith('lab.yaml')
    assert wl.find_room(rooms, 'Gone') is None
    assert wl.load_rooms('/no/such/dir') == []


def test_map_is_sent_as_png_with_its_placement(maps):
    png, meta = wl.load_map(os.path.join(maps, 'lab.yaml'))
    assert png[:8] == b'\x89PNG\r\n\x1a\n'
    assert meta == {'resolution': 0.05, 'origin': [-1.0, -2.0], 'width': 60, 'height': 40}
    back = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert back[0, 0] == 0 and back[5, 5] == 254          # walls stay walls, no flip


# ------------------------------------------------------------------------------ hardware
def test_esp32_is_the_port_that_is_not_the_lidar():
    assert wl.esp32_port([LIDAR, ESP]) == ESP
    assert wl.esp32_port([ESP.replace('2.4', '2.5'), LIDAR]) == ESP.replace('2.4', '2.5')
    assert wl.esp32_port([LIDAR]) is None                 # ESP32 unplugged
    assert wl.esp32_port([LIDAR, ESP, ESP + 'x']) is None  # can't tell which


def test_a_stack_started_from_a_terminal_is_noticed():
    term = '/usr/bin/python3 /opt/ros/humble/bin/ros2 launch securebot_ros ' \
           'securebot_laps.launch.py room:=Lab\n'
    ours = term.replace('room:=Lab', 'room:=Lab panel:=web')
    assert wl.other_stack_running('bash\n' + term)
    assert not wl.other_stack_running(ours)
    assert not wl.other_stack_running('bash\nros2 run securebot_ros web_panel\n')


# ------------------------------------------------------------------------------ the stack
def test_launch_command_has_no_windows_and_keeps_home():
    cmd = wl.launch_command('/m/lab.yaml', "Kumar's Lab", ESP, '0.18,0.04', 2)
    assert cmd[:4] == ['ros2', 'launch', 'securebot_ros', 'securebot_laps.launch.py']
    for arg in ('map:=/m/lab.yaml', "room:=Kumar's Lab", f'esp32_port:={ESP}', 'rviz:=false',
                'panel:=web', 'auto_localize:=true', 'rounds:=2', 'home:=0.18,0.04'):
        assert arg in cmd
    assert not any(a.startswith('home:=') for a in wl.launch_command('/m', 'L', ESP))


def test_home_text_is_checked():
    assert wl.parse_home(' 0.18, 0.04 ') == '0.18,0.04'
    assert wl.parse_home('') == ''
    for bad in ('1', '1,2,3', 'a,b'):
        with pytest.raises(ValueError):
            wl.parse_home(bad)


def test_localization_messages_from_the_launch_log():
    assert wl.localization_from_log([]) == (None, '')
    lines = ['[global_localize-9] global_localize: waiting for AMCL (/initialpose subscriber)']
    assert wl.localization_from_log(lines)[0] == 'searching'
    lines.append('[global_localize-9] No clear match: trying again with a new scan')
    assert wl.localization_from_log(lines)[0] == 'searching'
    ok = lines + ['[global_localize-9] global_localize: sent /initialpose x 0.18 y 0.04 '
                  'yaw 2.0 deg (score 91%)']
    state, text = wl.localization_from_log(ok)
    assert state == 'ok' and text.startswith('sent /initialpose x 0.18')
    bad = lines + ['[global_localize-9] No clear match',
                   '[global_localize-9] global_localize: not sending a pose; set it ...']
    assert wl.localization_from_log(bad)[0] == 'failed'


def test_power_on_starts_launch_and_recorder_in_their_own_groups(tmp_path):
    fos = FakeOS()
    st = stack_for(tmp_path, fos)
    st.start("Kumar's Lab", '/m/lab.yaml', ESP, name='t1')
    launch, rec = fos.procs
    assert launch.cmd[1] == 'launch' and launch.kw['start_new_session']
    assert rec.cmd[-4:] == ['--name', 't1', '--dir', str(tmp_path / 'runs')]
    assert st.state == 'starting' and st.log_path.endswith('run_t1.log')
    assert os.path.exists(st.log_path)
    with pytest.raises(RuntimeError):
        st.start("Kumar's Lab", '/m/lab.yaml', ESP)       # one robot, one launch
    st.mark_running()
    assert st.state == 'running'


def test_power_off_uses_ctrl_c_first_and_escalates(tmp_path):
    fos = FakeOS()
    st = stack_for(tmp_path, fos)
    st.start('L', '/m', ESP, name='a')
    st.stop()
    assert [p.signals for p in fos.procs] == [[signal.SIGINT], [signal.SIGINT]]
    assert st.state == 'off' and st.proc is None
    fos = FakeOS(dies_on=())                               # ignores Ctrl+C and SIGTERM
    st = stack_for(tmp_path, fos, tools=False)
    st.start('L', '/m', ESP, name='b')
    st.stop()
    assert fos.procs[0].signals == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
    assert fos.t >= sum(wl.STOP_GRACE)                     # waited before escalating
    assert st.state == 'off'


def test_a_launch_that_dies_shows_as_crashed(tmp_path):
    fos = FakeOS()
    st = stack_for(tmp_path, fos)
    st.start('L', '/m', ESP, name='c')
    assert st.poll() == 'starting'
    fos.procs[0].end(1)
    assert st.poll() == 'crashed' and 'exit code 1' in st.message
    assert fos.procs[1].signals == [signal.SIGINT]         # recorder stopped too
    st.start('L', '/m', ESP, name='d')                     # can power on again
    assert st.state == 'starting'


def test_log_is_read_incrementally(tmp_path):
    fos = FakeOS()
    st = stack_for(tmp_path, fos)
    st.start('L', '/m', ESP, name='e')
    with open(st.log_path, 'a') as f:
        f.write('[global_localize-9] global_localize: waiting for AMCL\n')
        f.write('half a line')
    st.log_file.flush()
    assert st.localization()[0] == 'searching'
    with open(st.log_path, 'a') as f:
        f.write(' ... \n[g] global_localize: sent /initialpose x 1.00 y 2.00 yaw 0 deg\n'
                'other output\n')
    assert st.localization() == ('ok', 'sent /initialpose x 1.00 y 2.00 yaw 0 deg')
    assert st.localization()[0] == 'ok'                    # remembered, nothing new
    assert st.log_tail(1) == ['other output']


def test_runs_are_grouped_newest_first(tmp_path):
    d = tmp_path / 'runs'
    d.mkdir()
    for k, name in enumerate(('run_a.log', 'track_a.csv', 'events_a.log', 'run_b.log',
                              'notes.txt')):
        (d / name).write_text('x')
        os.utime(d / name, (k, k))
    runs = wl.list_runs(str(d))
    assert [r['name'] for r in runs] == ['b', 'a']
    assert runs[1]['files'] == ['events_a.log', 'run_a.log', 'track_a.csv']


# ------------------------------------------------------------------------------ ROS messages
def _marker(type_, x, y, size=0.2, ns='debris', action=0):
    pos = types.SimpleNamespace(x=x, y=y)
    return types.SimpleNamespace(type=type_, action=action, ns=ns,
                                 pose=types.SimpleNamespace(position=pos),
                                 scale=types.SimpleNamespace(x=size))


def test_boxes_and_home_from_markers():
    wipe = _marker(1, 0, 0, action=3)                     # DELETEALL first, as laps_node does
    boxes = [wipe, _marker(wl.CUBE, 1.0, 2.0, 0.2032), _marker(wl.CUBE, -1, 0.5)]
    assert wl.boxes_from_markers(boxes) == [[1.0, 2.0, 0.203], [-1, 0.5, 0.2]]
    assert wl.boxes_from_markers([wipe]) == []
    home = [_marker(wl.CYLINDER, 0.18, 0.04, ns='home'), _marker(9, 0.18, 0.04, ns='home')]
    assert wl.home_from_markers(home) == [0.18, 0.04]
    assert wl.home_from_markers([_marker(wl.CYLINDER, 1, 1, ns='other')]) is None


def test_long_paths_are_thinned_but_keep_their_end():
    pts = [(k * 0.01, 0.0) for k in range(5001)]
    thin = wl.thin_path(pts, 1000)
    assert len(thin) <= 1001 and thin[0] == [0.0, 0.0] and thin[-1] == [50.0, 0.0]
    assert wl.thin_path([(0.123, 4.567)]) == [[0.12, 4.57]]


# ------------------------------------------------------------------------------ the app
def test_power_on_checks_room_port_and_other_launches(tmp_path, maps):
    app, bridge = app_for(tmp_path, maps, FakeOS())
    with pytest.raises(ValueError):
        app.power_on('Nowhere')
    with pytest.raises(ValueError):
        app.power_on("Kumar's Lab", home='x')
    app_np, _ = app_for(tmp_path, maps, FakeOS(), ports=[LIDAR])
    with pytest.raises(RuntimeError, match='ESP32'):
        app_np.power_on("Kumar's Lab")
    app_ps, _ = app_for(tmp_path, maps, FakeOS(),
                        ps='ros2 launch securebot_ros securebot_laps.launch.py room:=L')
    with pytest.raises(RuntimeError, match='terminal'):
        app_ps.power_on("Kumar's Lab")
    app.power_on("Kumar's Lab", home='0.18,0.04')
    cmd = app.stack.proc.cmd
    assert f'esp32_port:={ESP}' in cmd and 'home:=0.18,0.04' in cmd
    assert bridge.resets == 1
    with pytest.raises(RuntimeError, match='already'):
        app.power_on("Kumar's Lab")


def test_buttons_need_the_robot_on(tmp_path, maps):
    app, bridge = app_for(tmp_path, maps, FakeOS())
    with pytest.raises(RuntimeError):
        app.command('start')
    with pytest.raises(RuntimeError):
        app.click(1, 2)
    app.power_on("Kumar's Lab")
    with pytest.raises(ValueError):
        app.command('selfdestruct')
    for cmd in ('start', 'stop', 'home'):
        app.command(cmd)
    assert bridge.sent == ['start', 'stop', 'home']


def test_state_turns_ready_when_the_laps_node_answers(tmp_path, maps):
    app, bridge = app_for(tmp_path, maps, FakeOS())
    s = app.state()
    assert s['stack']['state'] == 'off' and s['rooms'] == ["Kumar's Lab"] and s['port'] == ESP
    app.power_on("Kumar's Lab")
    bridge.snap = {'status': {'state': 'idle'}, 'status_age': 10.0}   # stale: old run
    s = app.state()
    assert s['stack']['state'] == 'starting' and not s['connected'] and s['laps'] is None
    bridge.snap = {'status': {'state': 'idle', 'message': 'Ready'}, 'status_age': 0.3,
                   'pose': [0.1, 0.2, 0.0], 'boxes': [[1, 1, 0.2]], 'home': [0.18, 0.04]}
    s = app.state()
    assert s['stack']['state'] == 'running' and s['connected']
    assert s['laps']['message'] == 'Ready' and s['pose'] == [0.1, 0.2, 0.0]
    assert s['boxes'] == [[1, 1, 0.2]] and s['home'] == [0.18, 0.04]
    app.click(1.0, 1.0)
    assert bridge.clicks == [(1.0, 1.0)]


def test_power_off_stops_laps_then_the_software(tmp_path, maps):
    fos = FakeOS()
    app, bridge = app_for(tmp_path, maps, fos)
    assert app.power_off() == 'Already off'
    app.power_on("Kumar's Lab")
    app.power_off()
    assert bridge.sent == ['stop']
    assert app.stack.state == 'off' and fos.procs[0].signals == [signal.SIGINT]


def test_relocalize_runs_global_localize_on_the_room_map(tmp_path, maps, monkeypatch):
    app, bridge = app_for(tmp_path, maps, FakeOS())
    with pytest.raises(RuntimeError):
        app.relocalize()
    app.power_on("Kumar's Lab")
    app.stack.mark_running()
    ran = []

    def fake_run(cmd, **kw):
        ran.append(cmd)
        out = 'global_localize: sent /initialpose x 1.00 y 2.00 yaw 90.0 deg (score 88%)\n'
        return types.SimpleNamespace(stdout=out, returncode=0)
    monkeypatch.setattr(wl.subprocess, 'run', fake_run)
    app.relocalize()
    assert ran[0][1].endswith('global_localize.py') and ran[0][2:4] == [
        '--map', os.path.join(maps, 'lab.yaml')]
    assert bridge.sent == ['stop']                         # robot stands still for it
    s = app.state()
    assert s['localization']['text'].startswith('sent /initialpose x 1.00')
    assert not s['localization']['busy']


# ------------------------------------------------------------------------------ HTTP
@pytest.fixture
def server(tmp_path, maps):
    fos = FakeOS()
    app, bridge = app_for(tmp_path, maps, fos)
    srv = wl.make_server(app, '127.0.0.1', 0, os.path.join(HERE, '..', 'web'))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{srv.server_address[1]}'
    yield base, app, bridge
    srv.shutdown()
    srv.server_close()


def call(url, body=None, raw=None):
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.headers.get('Content-Type'), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get('Content-Type'), e.read()


def test_http_page_state_and_map(server):
    base, _, _ = server
    code, ctype, body = call(base + '/')
    assert code == 200 and ctype.startswith('text/html') and b'SecureBot' in body
    code, _, body = call(base + '/api/state')
    assert code == 200 and json.loads(body)['stack']['state'] == 'off'
    code, ctype, body = call(base + "/api/map.png?room=Kumar's%20Lab")
    assert code == 200 and ctype == 'image/png' and body.startswith(b'\x89PNG')
    code, _, body = call(base + "/api/map?room=Kumar%27s%20Lab")
    assert json.loads(body)['width'] == 60
    assert call(base + '/api/map?room=Nowhere')[0] == 404
    assert call(base + '/nope')[0] == 404


def test_http_buttons(server):
    base, app, bridge = server
    code, _, body = call(base + '/api/command', {'cmd': 'start'})
    assert code == 409 and 'Press Start' in json.loads(body)['error']
    code, _, body = call(base + '/api/power/on', {'room': "Kumar's Lab"})
    assert code == 200 and json.loads(body)['ok']
    assert call(base + '/api/command', {'cmd': 'START'})[0] == 200
    assert call(base + '/api/command', {'cmd': 'dance'})[0] == 400
    assert call(base + '/api/click', {'x': 1})[0] == 400       # y missing
    assert call(base + '/api/power/on', raw=b'{not json')[0] == 400
    assert call(base + '/api/power/on', raw=b'[1, 2]')[0] == 400
    assert call(base + '/api/power/off', {})[0] == 200
    assert bridge.sent == ['start', 'stop'] and app.stack.state == 'off'


def test_http_logs_only_from_the_runs_folder(server, tmp_path):
    base, app, _ = server
    os.makedirs(app.runs_dir, exist_ok=True)
    with open(os.path.join(app.runs_dir, 'run_x.log'), 'w') as f:
        f.write('hello')
    (tmp_path / 'secret.txt').write_text('no')
    assert call(base + '/api/log?file=run_x.log')[2] == b'hello'
    assert call(base + '/api/log?file=../secret.txt')[0] == 404
    code, _, body = call(base + '/api/runs')
    assert code == 200 and json.loads(body)[0]['name'] == 'x'


def test_http_faces_capture_review_keep(server, tmp_path):
    base, app, _ = server
    faces = app.faces
    code, _, body = call(base + '/api/faces/capture', {'name': '../evil'})
    assert code == 400
    assert call(base + '/api/faces/capture', {'name': 'Dan', 'count': 5})[0] == 200
    assert call(base + '/api/faces/capture', {'name': 'Dan'})[0] == 409       # busy
    staged = tmp_path / 'faces' / 'staging' / 'Dan'
    for i in range(2):
        cv2.imwrite(str(staged / f'cam_{i}.jpg'), np.zeros((8, 8, 3), np.uint8))
    faces.job['proc'].end(0)                       # capture finished -> review starts
    assert json.loads(call(base + '/api/state')[2])['faces']['busy'] == 'review'
    (staged / '.review.json').write_text(json.dumps({'name': 'Dan', 'photos': [
        {'file': 'cam_0.jpg', 'keep': True, 'reasons': [], 'notes': []},
        {'file': 'cam_1.jpg', 'keep': False, 'reasons': ['looks like X (0.40)'], 'notes': []}]}))
    faces.job['proc'].end(0)
    s = json.loads(call(base + '/api/state')[2])['faces']
    assert s['busy'] is None and s['review']['name'] == 'Dan'
    code, ctype, _ = call(base + '/api/faces/photo?kind=staged&name=Dan&file=cam_0.jpg')
    assert code == 200 and ctype == 'image/jpeg'
    assert call(base + '/api/faces/photo?kind=staged&name=Dan&file=../../x.jpg')[0] == 404
    assert call(base + '/api/faces/keep', {'name': 'Dan', 'files': 'cam_0.jpg'})[0] == 400
    assert call(base + '/api/faces/keep', {'name': 'Dan', 'files': ['cam_0.jpg']})[0] == 200
    assert os.listdir(tmp_path / 'faces' / 'authorized_faces' / 'Dan') == ['cam_0.jpg']
    s = json.loads(call(base + '/api/state')[2])['faces']
    assert s['review'] is None
    assert s['people'] == [{'name': 'Dan', 'photos': 1, 'enrolled': 0, 'cover': 'cam_0.jpg'}]
    assert s['database']['outdated']
    assert call(base + '/api/faces/rebuild', {})[0] == 200


PAGE_ROUTES = ('/api/state', '/api/map.png', '/api/map?', '/api/runs', '/api/log', '/api/start',
               '/api/power/off', '/api/command', '/api/click', '/api/relocalize', '/api/active',
               '/api/maps', '/api/maps/rename', '/api/maps/delete', '/api/maps/add',
               '/api/nopass', '/api/live_map.png', '/api/camera.jpg', '/api/faces/capture',
               '/api/faces/stop', '/api/faces/keep', '/api/faces/discard', '/api/faces/rebuild',
               '/api/faces/preview.jpg', '/api/faces/photo', '/api/faces/photos',
               '/api/faces/delete_photo', '/api/faces/delete_person')


def test_page_only_calls_api_routes_that_exist():
    with open(os.path.join(HERE, '..', 'web', 'index.html')) as f:
        page = f.read()
    for route in PAGE_ROUTES:
        assert route in page
    src = open(wl.__file__).read()
    for route in PAGE_ROUTES:
        assert ("'/api/map'" if route == '/api/map?' else route) in src
    assert '/api/power/on' in src                   # still there for scripts / curl


# ------------------------------------------------------------------------------ launch file
def test_laps_launch_leaves_the_window_out_for_the_web_panel():
    pytest.importorskip('launch_ros')
    path = os.path.join(HERE, '..', 'launch', 'securebot_laps.launch.py')
    spec = importlib.util.spec_from_file_location('laps_launch', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.laps_window('web') == []
    assert len(mod.laps_window('true')) == 1
    names = [getattr(a, 'name', None) for a in mod.generate_launch_description().entities]
    assert 'panel' in names


# ------------------------------------------------------------------------------ panel_urls
IP_OUTPUT = """1: lo    inet 127.0.0.1/8 scope host lo\\       valid_lft forever
2: wlP1p1s0    inet 10.115.14.57/24 brd 10.115.14.255 scope global wlP1p1s0\\  valid_lft 2303sec
8: tailscale0    inet 100.100.137.104/32 scope global tailscale0\\       valid_lft forever
9: docker0    inet 172.17.0.1/16 brd 172.17.255.255 scope global docker0\\       valid_lft forever
"""


def test_panel_urls_expand_every_network_into_real_addresses():
    urls = wl.panel_urls('0.0.0.0', 8080, IP_OUTPUT)
    assert urls == ['http://127.0.0.1:8080  (on the robot itself)',
                    'http://10.115.14.57:8080  (same network, wlP1p1s0)',
                    'http://100.100.137.104:8080  (over Tailscale)']
    assert not any('0.0.0.0' in u or '172.17' in u for u in urls)


def test_panel_urls_keep_a_specific_host_and_port():
    assert wl.panel_urls('127.0.0.1', 9000, IP_OUTPUT) == ['http://127.0.0.1:9000']


def test_panel_urls_without_ip_output_still_offer_the_robot_itself():
    assert wl.panel_urls('0.0.0.0', 8080, '') == ['http://127.0.0.1:8080  (on the robot itself)']


# ------------------------------------------------------------------------------ log retention
def _touch(path, mtime):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w').close()
    os.utime(path, (mtime, mtime))


def test_prune_runs_keeps_the_newest_runs_with_all_their_files(tmp_path):
    runs = tmp_path / 'laps_runs'
    for i in range(14):
        for prefix, ext in (('run_', 'log'), ('track_', 'csv'), ('events_', 'log')):
            _touch(str(runs / f'{prefix}2026112{i:02d}.{ext}'), 1000 + i)
    deleted = wl.prune_runs(str(runs), keep=10)
    assert sorted(deleted) == [f'2026112{i:02d}' for i in range(4)]
    left = [r['name'] for r in wl.list_runs(str(runs), limit=None)]
    assert left == [f'2026112{i:02d}' for i in range(13, 3, -1)]
    assert len(os.listdir(runs)) == 30                       # 10 runs x 3 files


def test_prune_logs_trims_explore_logs_and_colcon_folders_but_not_latest(tmp_path):
    log = tmp_path / 'log'
    for i in range(13):
        _touch(str(log / 'explore_runs' / f'explore_{i:02d}.log'), 1000 + i)
        d = log / f'build_2026-11-{i + 1:02d}_10-00-00'
        _touch(str(d / 'events.log'), 1000 + i)
        os.utime(str(d), (1000 + i, 1000 + i))
    oldest = log / 'build_2026-11-01_10-00-00'
    os.symlink(str(oldest), str(log / 'latest_build'))       # odd, but never delete it
    _touch(str(log / 'laps_runs' / 'run_x.log'), 2000)
    deleted = wl.prune_logs(str(log), keep=10)
    assert len(os.listdir(log / 'explore_runs')) == 10
    builds = sorted(p for p in os.listdir(log) if p.startswith('build_'))
    assert len(builds) == 11 and oldest.name in builds        # 10 newest + the linked one
    assert os.path.exists(log / 'laps_runs' / 'run_x.log')
    assert deleted == 3 + 2


def test_starting_a_run_prunes_old_runs_but_keeps_the_new_one(tmp_path):
    runs = tmp_path / 'laps_runs'
    for i in range(12):
        _touch(str(runs / f'run_old{i:02d}.log'), 1000 + i)

    class P:
        def __init__(self, *a, **k):
            pass

        def poll(self):
            return None
    sm = wl.StackManager(runs_dir=str(runs), tools_dir=str(tmp_path), popen=P, record=False)
    sm.start('Lab', '/x.yaml', '/dev/null', name='new')
    names = [r['name'] for r in wl.list_runs(str(runs), limit=None)]
    assert len(names) == 10 and names[0] == 'new'
    sm.log_file.close()


# ------------------------------------------------------------------------------ last position
def _run(runs, name, room, poses, mtime):
    import json as _json
    with open(runs / f'meta_{name}.json', 'w') as f:
        _json.dump({'room': room}, f)
    with open(runs / f'track_{name}.csv', 'w') as f:
        f.write('t,x,y,yaw,odom_yaw,path_id,state,loop,message\n')
        for i, (x, y) in enumerate(poses):
            f.write(f'{i},{x},{y},0,0,1,running,1,"x"\n')
    for p in runs.glob(f'*_{name}.*'):
        os.utime(p, (mtime, mtime))


def test_last_pose_is_where_the_newest_run_in_that_room_ended(tmp_path):
    runs = tmp_path / 'laps_runs'
    runs.mkdir()
    _run(runs, 'a', 'Lab', [(0, 0), (1.0, 2.0)], 1000)
    _run(runs, 'b', 'Hall', [(5.0, 5.0)], 2000)                   # newer, other room
    _run(runs, 'c', 'Lab', [(0, 0), (-3.1, 2.4), (0.0, 0.0)], 1500)
    assert wl.last_pose(str(runs), 'Lab') == (-3.1, 2.4)          # ignores the 0,0 rows
    assert wl.last_pose(str(runs), 'Hall') == (5.0, 5.0)
    assert wl.last_pose(str(runs), 'Nowhere') is None


def test_launch_command_passes_the_last_position():
    cmd = wl.launch_command('/m/a.yaml', 'Lab', ESP, near=(-3.1, 2.4))
    assert 'localize_near:=-3.10,2.40' in cmd
    assert not any(c.startswith('localize_near') for c in wl.launch_command('/m', 'L', ESP))


def test_starting_a_run_records_its_room(tmp_path):
    runs = tmp_path / 'laps_runs'

    class P:
        def __init__(self, *a, **k):
            pass

        def poll(self):
            return None
    sm = wl.StackManager(runs_dir=str(runs), tools_dir=str(tmp_path), popen=P, record=False)
    sm.start('Lab', '/x.yaml', '/dev/null', name='r1')
    import json as _json
    assert _json.load(open(runs / 'meta_r1.json'))['room'] == 'Lab'
    sm.log_file.close()


# ------------------------------------------------------------------------------ redesign
def localized(app, ok=True):
    """Write what auto-localize says into the run's log."""
    with open(app.stack.log_path, 'a') as f:
        f.write('[global_localize-9] global_localize: waiting for AMCL\n')
        f.write('[global_localize-9] global_localize: sent /initialpose x 1.00 y 2.00\n' if ok
                else '[global_localize-9] global_localize: not sending a pose\n')
        f.flush()


def test_start_from_off_powers_on_and_starts_once_localized(tmp_path, maps):
    app, bridge = app_for(tmp_path, maps, FakeOS())
    assert app.state()['active'] == "Kumar's Lab"            # the only room
    msg = app.start()
    assert 'Powering on' in msg and app.stack.state == 'starting'
    assert app.state()['pending_start']
    bridge.snap = {'status': {'state': 'idle'}, 'status_age': 0.2}
    app.state()
    assert bridge.sent == []                                  # not found itself yet
    localized(app)
    s = app.state()
    assert bridge.sent == ['start'] and not s['pending_start']
    app.state()
    assert bridge.sent == ['start']                           # only once
    assert app.start() == 'Starting laps' and bridge.sent == ['start', 'start']


def test_start_from_off_gives_up_when_it_cannot_localize(tmp_path, maps):
    app, bridge = app_for(tmp_path, maps, FakeOS())
    app.start("Kumar's Lab")
    bridge.snap = {'status': {'state': 'idle'}, 'status_age': 0.2}
    localized(app, ok=False)
    s = app.state()
    assert bridge.sent == [] and not s['pending_start'] and 'Manual re-localize' in s['note']


def test_stop_cancels_a_pending_start(tmp_path, maps):
    app, bridge = app_for(tmp_path, maps, FakeOS())
    app.start()
    app.command('stop')
    bridge.snap = {'status': {'state': 'idle'}, 'status_age': 0.2}
    localized(app)
    app.state()
    assert bridge.sent == ['stop']                            # never started
    app.power_off()
    assert 'nothing is moving' in app.command('stop')         # Stop is always safe to press


def test_start_refuses_another_room_while_running(tmp_path, maps):
    (tmp_path / 'maps' / 'rooms.json').write_text(json.dumps({'rooms': [
        {'name': "Kumar's Lab", 'map': 'lab'}, {'name': 'Hall', 'map': 'lab'}]}))
    app, _ = app_for(tmp_path, maps, FakeOS())
    app.start("Kumar's Lab")
    with pytest.raises(RuntimeError, match='Power off'):
        app.start('Hall')
    with pytest.raises(RuntimeError, match='Power off'):
        app.set_active('Hall')
    app.power_off()
    assert app.set_active('Hall') and app.state()['active'] == 'Hall'
    with pytest.raises(ValueError):
        app.set_active('Nowhere')


def test_add_maps_a_new_room_with_the_explore_launch(tmp_path, maps):
    app, bridge = app_for(tmp_path, maps, FakeOS())
    with pytest.raises(RuntimeError, match='already exists'):
        app.explore("kumar's lab")
    with pytest.raises(ValueError):
        app.explore('   ')
    app.explore('Library')
    cmd = app.stack.proc.cmd
    assert cmd[3] == 'securebot_explore.launch.py'
    for arg in ('room:=Library', f'esp32_port:={ESP}', 'rviz:=false', 'panel:=web',
                'rooms:=false'):
        assert arg in cmd
    assert app.stack.recorder is None                         # laps-only recorder
    assert not wl.other_stack_running(' '.join(cmd))
    s = app.state()
    assert s['active'] == 'Library' and s['stack']['kind'] == 'explore'
    bridge.snap = {'status': {'state': 'idle', 'coverage': 0.1}, 'status_age': 0.2}
    s = app.state()
    assert bridge.kind == 'explore' and s['explore']['coverage'] == 0.1 and s['laps'] is None
    assert 'press Start' in s['stack']['message']
    app.start()
    app.command('home')
    assert 'map is saved' in app.command('stop')
    assert bridge.sent == ['explore:start', 'explore:home', 'explore:stop']
    with pytest.raises(RuntimeError, match='mapping'):
        app.relocalize()
    with pytest.raises(RuntimeError):
        app.click(1, 2)
    app.power_off()
    assert bridge.sent[-1] == 'explore:stop'


def test_maps_list_rename_and_delete(tmp_path, maps):
    app, _ = app_for(tmp_path, maps, FakeOS())
    (tmp_path / 'maps' / 'lab.nopass.json').write_text(json.dumps(
        {'width': 0.1, 'lines': [[[0, 0], [1, 1]]]}))
    [m] = app.maps()
    assert m['name'] == "Kumar's Lab" and m['map'] == 'lab' and m['nopass'] == 1
    assert m['width_m'] == 3.0 and m['height_m'] == 2.0
    app.set_active("Kumar's Lab")
    app.rename_room("Kumar's Lab", 'Lab 2')
    assert [r['name'] for r in wl.load_rooms(maps)] == ['Lab 2']
    assert app.state()['active'] == 'Lab 2'
    with pytest.raises(ValueError):
        app.rename_room('Nope', 'X')
    app.power_on('Lab 2')
    with pytest.raises(RuntimeError, match='Power off'):
        app.delete_room('Lab 2')
    with pytest.raises(RuntimeError, match='Power off'):
        app.rename_room('Lab 2', 'Lab 3')
    app.power_off()
    msg = app.delete_room('Lab 2')
    assert 'deleted' in msg.lower() and wl.load_rooms(maps) == []
    trash = sorted(os.listdir(os.path.join(maps, 'deleted')))
    assert [t.split('_', 2)[2] for t in trash] == ['lab.nopass.json', 'lab.pgm', 'lab.yaml']
    assert not os.path.exists(os.path.join(maps, 'lab.yaml'))
    assert app.state()['active'] is None
    with pytest.raises(ValueError):
        app.delete_room('Lab 2')


def test_delete_keeps_a_map_another_room_still_uses(tmp_path, maps):
    (tmp_path / 'maps' / 'rooms.json').write_text(json.dumps({'rooms': [
        {'name': 'A', 'map': 'lab'}, {'name': 'B', 'map': 'lab'}]}))
    app, _ = app_for(tmp_path, maps, FakeOS())
    app.delete_room('A')
    assert os.path.exists(os.path.join(maps, 'lab.yaml'))
    assert [r['name'] for r in wl.load_rooms(maps)] == ['B']


def test_nopass_lines_are_saved_and_a_running_room_reloads(tmp_path, maps):
    app, bridge = app_for(tmp_path, maps, FakeOS())
    room = "Kumar's Lab"
    assert app.nopass_lines(room) == {'width': 0.1, 'lines': []}
    app.save_nopass(room, [[[0, 0], [1, 0]], [[0, 1], [0, 2.5]]])
    data = json.loads((tmp_path / 'maps' / 'lab.nopass.json').read_text())
    assert data['lines'] == [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 2.5]]]
    assert app.nopass_lines(room)['lines'][1] == [[0.0, 1.0], [0.0, 2.5]]
    assert bridge.sent == []                                  # nothing running
    app.power_on(room)
    assert 're-plans' in app.save_nopass(room, [])
    assert bridge.sent == ['nopass reload'] and app.nopass_lines(room)['lines'] == []
    for bad in ('x', [[1, 2]], [[[0, 0], [1, 'a']]], [[[0, 0], [float('nan'), 1]]],
                [[[0, 0], [1, 1]]] * (wl.MAX_NOPASS + 1)):
        with pytest.raises(ValueError):
            app.save_nopass(room, bad)
    with pytest.raises(ValueError):
        app.save_nopass('Nowhere', [])


def test_occupancy_grid_becomes_a_map_image():
    # 3 x 2 grid, row 0 = bottom: free, wall, unknown / free, free, wall
    png = wl.occupancy_png([0, 100, -1, 0, 0, 90], 3, 2)
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert img.tolist() == [[254, 254, 0], [254, 0, 205]]    # flipped: top row first


def test_live_map_only_while_exploring(tmp_path, maps):
    app, bridge = app_for(tmp_path, maps, FakeOS())
    bridge.live_png = lambda: b'png'
    assert app.live_map_png() is None
    app.explore('Library')
    assert app.live_map_png() == b'png'
    app.power_off()
    assert app.live_map_png() is None                         # mapping over


def test_http_new_endpoints(server):
    base, app, bridge = server
    code, _, body = call(base + '/api/maps')
    assert code == 200 and json.loads(body)[0]['name'] == "Kumar's Lab"
    assert call(base + '/api/nopass', {'room': "Kumar's Lab",
                                       'lines': [[[0, 0], [1, 1]]]})[0] == 200
    code, _, body = call(base + "/api/nopass?room=Kumar%27s%20Lab")
    assert json.loads(body)['lines'] == [[[0.0, 0.0], [1.0, 1.0]]]
    assert call(base + '/api/nopass', {'room': "Kumar's Lab", 'lines': 'x'})[0] == 400
    code, ctype, body = call(base + '/api/camera.jpg')
    assert code == 200 and ctype == 'image/jpeg'
    app.camera.why = 'The live face node has the camera'
    code, _, body = call(base + '/api/camera.jpg')
    assert code == 409 and 'live face' in json.loads(body)['error']
    assert call(base + '/api/live_map.png')[0] == 404
    assert call(base + '/api/active', {'room': "Kumar's Lab"})[0] == 200
    assert call(base + '/api/maps/rename', {'old': "Kumar's Lab", 'new': ''})[0] == 400
    assert call(base + '/api/maps/add', {'name': 'Library'})[0] == 200
    assert app.stack.kind == 'explore'
    assert call(base + '/api/maps/add', {'name': 'Other'})[0] == 409      # already on
    assert call(base + '/api/power/off', {})[0] == 200
    assert call(base + '/api/start', {})[0] == 200 and app.stack.kind == 'laps'
    assert call(base + '/api/power/off', {})[0] == 200
    assert call(base + '/api/maps/rename', {'old': "Kumar's Lab", 'new': 'Lab'})[0] == 200
    assert call(base + '/api/maps/delete', {'name': 'Lab'})[0] == 200
    assert json.loads(call(base + '/api/maps')[2]) == []
    s = json.loads(call(base + '/api/state')[2])
    assert s['battery'] is None and 'camera' in s


def test_http_faces_manage(server, tmp_path):
    base, app, _ = server
    d = tmp_path / 'faces' / 'authorized_faces' / 'Ana'
    d.mkdir(parents=True)
    for f in ('1.jpg', '2.jpg'):
        (d / f).write_bytes(b'jpg')
    code, _, body = call(base + '/api/faces/photos?name=Ana')
    assert code == 200 and json.loads(body) == ['1.jpg', '2.jpg']
    assert call(base + '/api/faces/photos?name=..')[0] == 404
    code, ctype, _ = call(base + '/api/faces/photo?kind=enrolled&name=Ana&file=1.jpg')
    assert code == 200 and ctype == 'image/jpeg'
    assert call(base + '/api/faces/delete_photo', {'name': 'Ana', 'file': '1.jpg'})[0] == 200
    assert os.listdir(d) == ['2.jpg']
    assert call(base + '/api/faces/capture', {'name': 'Ana', 'count': 5,
                                              'replace': True})[0] == 200
    assert app.faces.replace == 'Ana'
    assert call(base + '/api/faces/delete_person', {'name': 'Ana'})[0] == 409   # capturing
    app.faces.job['proc'].end(0)                     # no photos taken -> capture over
    assert call(base + '/api/faces/delete_person', {'name': 'Ana'})[0] == 200
    assert not d.exists()


# ------------------------------------------------------------------------------ guard
def test_tamper_trip_records_one_alert_with_snapshot(tmp_path, maps):
    app, bridge = app_for(tmp_path, maps, FakeOS())
    app.tamper_event('armed', '')
    assert app.alerts.list() == []
    bridge.tamper_status = ('tripped', 'lift')
    app.tamper_event('tripped', 'lift')
    app.tamper_event('tripped', 'lift')            # the ESP32 repeats it twice a second
    alerts = app.alerts.list()
    assert len(alerts) == 1
    a = alerts[0]
    assert a['kind'] == 'tamper' and a['reason'] == 'lift' and a['snapshot']
    assert a['text'] == 'Robot lifted: motors stopped' and not a['acked']
    assert a['time_text'].startswith('20')
    assert app.alerts.snapshot(a['id']) == app.camera.jpg
    s = app.state()
    assert s['tamper'] == {'state': 'tripped', 'reason': 'lift'}
    assert [x['id'] for x in s['alerts']] == [a['id']]


def test_tamper_trip_without_a_camera_still_alerts(tmp_path, maps):
    app, bridge = app_for(tmp_path, maps, FakeOS())
    app.camera.jpg = None
    app.tamper_event('tripped', 'tilt')
    a = app.alerts.list()[0]
    assert not a['snapshot'] and a['text'] == 'Robot tipped over: motors stopped'
    with pytest.raises(ValueError):
        app.alerts.snapshot(a['id'])


def test_a_new_trip_after_clearing_is_a_new_alert(tmp_path, maps):
    app, _ = app_for(tmp_path, maps, FakeOS())
    app.tamper_event('tripped', 'tilt')
    app.tamper_event('armed', '')
    app.tamper_event('tripped', 'lift')
    assert [a['reason'] for a in app.alerts.list()] == ['lift', 'tilt']


def test_tampered_robot_refuses_to_drive_until_acknowledged(tmp_path, maps):
    fos = FakeOS()
    app, bridge = app_for(tmp_path, maps, fos)
    app.power_on("Kumar's Lab")
    bridge.tamper_status = ('tripped', 'tilt')
    app.tamper_event('tripped', 'tilt')
    with pytest.raises(RuntimeError, match='tipped over: acknowledge'):
        app.start()
    with pytest.raises(RuntimeError, match='acknowledge'):
        app.command('home')
    assert app.command('stop')                     # stopping is always allowed
    assert 'enabled again' in app.ack_alert()
    assert 'tamper:clear' in bridge.sent
    assert app.alerts.unacked() == []
    bridge.tamper_status = ('armed', '')
    app.check_can_drive()                          # no error now
    assert app.ack_alert() == 'Nothing to acknowledge'


def test_ack_one_alert_by_id(tmp_path, maps):
    app, bridge = app_for(tmp_path, maps, FakeOS())
    app.tamper_event('tripped', 'tilt')
    app.tamper_event('off', '')
    app.tamper_event('tripped', 'lift')
    first = app.alerts.list()[-1]['id']
    assert app.ack_alert(first) == 'Acknowledged'
    assert [a['reason'] for a in app.alerts.unacked()] == ['lift']
    assert 'tamper:clear' not in bridge.sent       # not tripped any more: nothing to clear
    with pytest.raises(ValueError):
        app.ack_alert('19990101_000000_000')


def test_low_battery_refuses_start_but_not_home(tmp_path, maps):
    fos = FakeOS()
    app, bridge = app_for(tmp_path, maps, fos)
    bridge.power_status = {'pct': 17.6, 'volts': 22.1, 'amps': 0.8, 'low': True,
                           'return_home_pct': 20, 'cells': 6}
    with pytest.raises(RuntimeError, match=r'Battery low \(18 %, returns home at 20 %\)'):
        app.start()
    app.power_on("Kumar's Lab")
    app.stack.mark_running()
    assert app.command('home') == 'Going home'
    assert app.state()['battery']['pct'] == 17.6
    bridge.power_status['low'] = False
    assert app.start()


def test_power_settings_saved_and_checked(tmp_path, maps):
    app, _ = app_for(tmp_path, maps, FakeOS())
    assert app.power_settings()['return_home_pct'] == 20
    assert app.set_power_settings({'return_home_pct': 30}) == 'Returns home at 30 % (7S pack)'
    assert json.load(open(tmp_path / 'config' / 'power.json'))['return_home_pct'] == 30
    with pytest.raises(ValueError):
        app.set_power_settings({'return_home_pct': 95})
    with pytest.raises(ValueError):
        app.set_power_settings({'cells': 5})
    assert app.power_settings()['return_home_pct'] == 30


def test_faces_setting_goes_into_the_launch(tmp_path, maps):
    fos = FakeOS()
    app, _ = app_for(tmp_path, maps, fos)
    assert app.settings() == {'faces': False}
    app.power_on("Kumar's Lab")
    assert 'faces:=false' in fos.procs[0].cmd
    app.power_off()
    assert 'next Power on' not in app.set_faces(True)
    app.power_on("Kumar's Lab")
    assert 'faces:=true' in fos.procs[-2].cmd
    app.stack.mark_running()
    assert 'next Power on' in app.set_faces(False)
    assert wl.explore_command('Lib', '/dev/x', True)[-1] == 'faces:=true'


def test_face_status_in_state(tmp_path, maps):
    app, bridge = app_for(tmp_path, maps, FakeOS())
    assert app.state()['face_live'] is None
    bridge.face_status = {'status': 'AUTHORIZED', 'names': 'Ana'}
    assert app.state()['face_live']['names'] == 'Ana'


def test_http_alerts_and_power(server):
    base, app, bridge = server
    bridge.tamper_status = ('tripped', 'lift')
    app.tamper_event('tripped', 'lift')
    code, _, body = call(base + '/api/alerts')
    alerts = json.loads(body)
    assert code == 200 and len(alerts) == 1
    code, ctype, body = call(base + '/api/alerts/snapshot.jpg?id=' + alerts[0]['id'])
    assert code == 200 and ctype == 'image/jpeg' and body == app.camera.jpg
    assert call(base + '/api/alerts/snapshot.jpg?id=../../etc/passwd')[0] == 404
    code, _, body = call(base + '/api/start', {})
    assert code == 409 and 'acknowledge' in json.loads(body)['error']
    assert call(base + '/api/alerts/ack', {'id': alerts[0]['id']})[0] == 200
    assert 'tamper:clear' in bridge.sent
    code, _, body = call(base + '/api/power/settings')
    assert json.loads(body)['return_home_pct'] == 20
    assert call(base + '/api/power/settings', {'return_home_pct': 25})[0] == 200
    assert call(base + '/api/power/settings', {'return_home_pct': 'x'})[0] == 400
    assert call(base + '/api/settings/faces', {'on': True})[0] == 200
    assert json.loads(call(base + '/api/state')[2])['settings'] == {'faces': True}
