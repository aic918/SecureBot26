#!/usr/bin/env python3
"""Check that face recognition and SLAM / localization keep up when they run together.

Start laps (or mapping) with face recognition on, e.g. from the web panel with the Face
recognition switch on, or:
  ros2 launch securebot_ros securebot_laps.launch.py room:="Kumar's Lab" faces:=true
then, beside it:
  python3 tools/faces_slam_check.py --minutes 10          # Ctrl+C ends it early

Every second it records the live face node's frame rate (/face_status arrives once per
processed frame), the LiDAR rate (/scan), how fresh the robot's position on the map is
(TF map -> base_link) and whether it jumped, the laps / explore state, the battery (guard
node) and the Jetson's load and temperatures (tegrastats). At the end it writes
log/faces_slam/<stamp>.csv (one row per second) and <stamp>.txt (the verdict) and prints the
verdict: PASS when every check held, else what did not.

The checks (thresholds are options): face frame rate (on average, and under the minimum for
at most a tenth of the time), LiDAR rate, position updates (no gap
over --max-tf-gap s), no position jumps (more than --jump-m in one update; a localization
correction while driving), no laps / explore error, CPU / GPU temperature under --max-temp.
"""
import argparse
import csv
import json
import math
import os
import re
import subprocess
import threading
import time

SLOW_SHARE = 0.1        # face recognition may be under --min-face-fps for this share of seconds
FIELDS = ['t', 'face_fps', 'face_status', 'scan_hz', 'tf_age', 'jump_m', 'job', 'job_state',
          'battery_pct', 'cpu_pct', 'gpu_pct', 'ram_mb', 'cpu_temp', 'gpu_temp']

TEGRA_RE = {
    'ram_mb': re.compile(r'RAM (\d+)/\d+MB'),
    'cpu': re.compile(r'CPU \[([^\]]*)\]'),
    'gpu_pct': re.compile(r'GR3D_FREQ (\d+)%'),
    'cpu_temp': re.compile(r'\bcpu@([\d.]+)C', re.IGNORECASE),
    'gpu_temp': re.compile(r'\bgpu@([\d.]+)C', re.IGNORECASE),
}


def parse_tegrastats(line):
    """One tegrastats line -> {cpu_pct (mean of the cores), gpu_pct, ram_mb, cpu_temp,
    gpu_temp}; missing values are None."""
    out = {k: None for k in ('cpu_pct', 'gpu_pct', 'ram_mb', 'cpu_temp', 'gpu_temp')}
    m = TEGRA_RE['cpu'].search(line)
    if m:
        cores = [int(c.split('%')[0]) for c in m.group(1).split(',') if '%' in c]
        if cores:
            out['cpu_pct'] = round(sum(cores) / len(cores), 1)
    for key in ('gpu_pct', 'ram_mb'):
        m = TEGRA_RE[key].search(line)
        if m:
            out[key] = int(m.group(1))
    for key in ('cpu_temp', 'gpu_temp'):
        m = TEGRA_RE[key].search(line)
        if m:
            out[key] = float(m.group(1))
    return out


def summarize(rows, opts):
    """rows: the per-second dicts. Returns (passed, lines of text)."""
    if not rows:
        return False, ['No data recorded.']

    def values(key):
        return [r[key] for r in rows if r.get(key) is not None]

    def mean(v):
        return sum(v) / len(v) if v else None

    face = values('face_fps')
    scan = values('scan_hz')
    tf_age = values('tf_age')
    jumps = [j for j in values('jump_m') if j > opts['jump_m']]
    errors = sorted({r['job_state'] for r in rows if r.get('job_state') == 'error'})
    temps = values('cpu_temp') + values('gpu_temp')
    face_mean, scan_mean = mean(face), mean(scan)
    face_low = sum(1 for f in face if f < opts['min_face_fps'])
    jobs = sorted({r['job'] for r in rows if r.get('job')})
    checks = [
        ('Face recognition runs', bool(face) and face_mean >= opts['min_face_fps']
         and face_low <= SLOW_SHARE * len(face),
         f'{face_mean:.1f} fps average, under {opts["min_face_fps"]} fps for {face_low} s'
         if face else 'no /face_status at all: is faces:=true on?'),
        ('LiDAR keeps its rate', bool(scan) and scan_mean >= opts['min_scan_hz'],
         f'{scan_mean:.1f} Hz average, lowest {min(scan):.1f} Hz' if scan else 'no /scan'),
        ('Position keeps updating', bool(tf_age) and max(tf_age) <= opts['max_tf_gap'],
         f'longest gap {max(tf_age):.2f} s' if tf_age else 'no map -> base_link (not localized)'),
        ('No position jumps', not jumps,
         f'{len(jumps)} jumps over {opts["jump_m"]} m (largest {max(jumps):.2f} m)' if jumps
         else 'none'),
        ('Laps / mapping without errors', not errors and bool(jobs),
         ('ran: ' + ', '.join(jobs)) if jobs and not errors else
         ('error state seen' if errors else 'no laps or explore status: is it running?')),
        ('Temperatures OK', not temps or max(temps) < opts['max_temp'],
         f'highest {max(temps):.0f} C' if temps else 'no tegrastats readings'),
    ]
    cpu, gpu = values('cpu_pct'), values('gpu_pct')
    lines = [f'{len(rows)} s recorded.']
    for name, ok, detail in checks:
        lines.append(f'{"PASS" if ok else "FAIL"}  {name}: {detail}')
    if cpu or gpu:
        lines.append('Load: CPU ' + (f'{mean(cpu):.0f} % average, {max(cpu):.0f} % peak'
                                     if cpu else '?') +
                     '; GPU ' + (f'{mean(gpu):.0f} % average, {max(gpu):.0f} % peak'
                                 if gpu else '?'))
    passed = all(ok for _, ok, _ in checks)
    lines.insert(0, 'PASS: face recognition and SLAM ran well together' if passed else
                 'FAIL: see the checks below')
    return passed, lines


class Counter:
    """Messages per second for one topic."""

    def __init__(self):
        self.n, self.lock = 0, threading.Lock()

    def hit(self, *_):
        with self.lock:
            self.n += 1

    def take(self):
        with self.lock:
            n, self.n = self.n, 0
        return n


def run(opts):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformListener

    rclpy.init()
    node = Node('faces_slam_check')
    faces, scans = Counter(), Counter()
    latest = {'face': None, 'job': None, 'battery': None}

    def on_status(kind, msg):
        try:
            latest['job'] = (kind, json.loads(msg.data).get('state'), time.monotonic())
        except ValueError:
            pass

    def on_power(msg):
        try:
            latest['battery'] = json.loads(msg.data).get('pct')
        except ValueError:
            pass

    def on_face(msg):
        faces.hit()
        latest['face'] = msg.data

    node.create_subscription(String, '/face_status', on_face, 10)
    node.create_subscription(LaserScan, '/scan', scans.hit, qos_profile_sensor_data)
    node.create_subscription(String, '/laps/status', lambda m: on_status('laps', m), 10)
    node.create_subscription(String, '/explore/status', lambda m: on_status('explore', m), 10)
    node.create_subscription(String, '/power/status', on_power, 10)
    tf = Buffer()
    TransformListener(tf, node)
    tegra = {}

    def read_tegrastats():
        try:
            p = subprocess.Popen(['tegrastats', '--interval', '1000'], stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, text=True)
        except OSError:
            return
        for line in p.stdout:
            tegra.update(parse_tegrastats(line))

    threading.Thread(target=read_tegrastats, daemon=True).start()
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()

    stamp = time.strftime('%Y%m%d_%H%M%S')
    os.makedirs(opts['dir'], exist_ok=True)
    csv_path = os.path.join(opts['dir'], f'{stamp}.csv')
    rows, last_pose, last_tf_time = [], None, None
    t0 = time.monotonic()
    print(f'Recording for {opts["minutes"]:.0f} min to {csv_path} (Ctrl+C to finish early)')
    try:
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, FIELDS)
            w.writeheader()
            while time.monotonic() - t0 < opts['minutes'] * 60:
                time.sleep(1.0)
                now = time.monotonic()
                row = {'t': round(now - t0, 1), 'face_fps': faces.take(),
                       'face_status': latest['face'], 'scan_hz': scans.take(),
                       'battery_pct': latest['battery'], 'tf_age': None, 'jump_m': None}
                try:
                    tr = tf.lookup_transform('map', 'base_link', rclpy.time.Time())
                    stamp_s = tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9
                    pose = (tr.transform.translation.x, tr.transform.translation.y)
                    clock_now = node.get_clock().now().nanoseconds * 1e-9
                    row['tf_age'] = round(max(0.0, clock_now - stamp_s), 2)
                    if last_pose is not None and stamp_s != last_tf_time:
                        row['jump_m'] = round(math.dist(pose, last_pose), 3)
                    last_pose, last_tf_time = pose, stamp_s
                except Exception:
                    pass
                job = latest['job']
                if job and now - job[2] < 3.0:
                    row['job'], row['job_state'] = job[0], job[1]
                row.update(tegra)
                rows.append(row)
                w.writerow({k: row.get(k) for k in FIELDS})
                f.flush()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
    passed, lines = summarize(rows, opts)
    txt = os.path.join(opts['dir'], f'{stamp}.txt')
    with open(txt, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'Saved {csv_path} and {txt}')
    return 0 if passed else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--minutes', type=float, default=10.0)
    ap.add_argument('--dir', default=os.path.expanduser('~/securebot_ws/log/faces_slam'))
    ap.add_argument('--min-face-fps', type=float, default=2.0)
    ap.add_argument('--min-scan-hz', type=float, default=5.0)
    ap.add_argument('--max-tf-gap', type=float, default=1.0,
                    help='s: longest acceptable time without a new map -> base_link')
    ap.add_argument('--jump-m', type=float, default=0.5,
                    help='m: a position change bigger than this in one second is a jump')
    ap.add_argument('--max-temp', type=float, default=85.0)
    a = ap.parse_args()
    raise SystemExit(run({'minutes': a.minutes, 'dir': a.dir, 'min_face_fps': a.min_face_fps,
                          'min_scan_hz': a.min_scan_hz, 'max_tf_gap': a.max_tf_gap,
                          'jump_m': a.jump_m, 'max_temp': a.max_temp}))


if __name__ == '__main__':
    main()
