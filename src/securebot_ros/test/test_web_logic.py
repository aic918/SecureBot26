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

LIDAR = wl.LIDAR_PORT
ESP = '/dev/serial/by-path/platform-3610000.usb-usb-0:2.3:1.0-port0'
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

    def send_command(self, cmd):
        self.sent.append(cmd)

    def click(self, x, y):
        self.clicks.append((x, y))

    def reset(self):
        self.resets += 1

    def snapshot(self):
        return dict(self.snap)


def app_for(tmp_path, maps, fos, ports=(LIDAR, ESP), ps=''):
    bridge = FakeBridge()
    app = wl.PanelApp(bridge, stack=stack_for(tmp_path, fos), maps_dir=maps,
                      runs_dir=str(tmp_path / 'runs'), tools_dir=str(tmp_path / 'tools'),
                      ports=lambda: list(ports), ps=lambda: ps, spawn=lambda f: f())
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
    assert wl.esp32_port([ESP.replace('2.3', '2.4'), LIDAR]) == ESP.replace('2.3', '2.4')
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
    assert code == 409 and 'Power on' in json.loads(body)['error']
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


def test_page_only_calls_api_routes_that_exist():
    with open(os.path.join(HERE, '..', 'web', 'index.html')) as f:
        page = f.read()
    for route in ('/api/state', '/api/map.png', '/api/map?', '/api/runs', '/api/log',
                  '/api/power/on', '/api/power/off', '/api/command', '/api/click',
                  '/api/relocalize'):
        assert route in page
    src = open(wl.__file__).read()
    for route in ('/api/state', '/api/map.png', "'/api/map'", '/api/runs', '/api/log',
                  '/api/power/on', '/api/power/off', '/api/command', '/api/click',
                  '/api/relocalize'):
        assert route in src


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
