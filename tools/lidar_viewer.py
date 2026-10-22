#!/usr/bin/env python3
"""Live view of what the RPLidar A1M8 sees, with the tower's blind spot measured and shaded.

Temporary bring-up tool (Sprint 2, before ROS / sllidar_ros2). Talks to the LiDAR directly over
its USB serial adapter with pyserial; nothing else is needed. Nothing is sent to the ESP32.

  python3 tools/lidar_viewer.py                    # finds the LiDAR port itself
  python3 tools/lidar_viewer.py /dev/ttyUSB0       # or name the port
  python3 tools/lidar_viewer.py --demo             # simulated room + tower, no hardware
  python3 tools/lidar_viewer.py --snapshot out.png # no window: collect, save a picture, exit

The plot is a top-down view with the robot in the middle and the robot's front at the top.
Points are colored by distance; orange points closer than --tower-cm are the robot itself
(the camera tower). Gray wedges are blind sectors: angles where the LiDAR got no usable
return in nearly all of the last ~15 s. On exit the measured blind sectors are printed, ready
to be used for the ROS laser filter later.

Measuring the tower's blind spot: an empty or dark part of the room also gives no returns, so
press c, then walk once around the robot holding a flat board (or book) about 50 cm away.
Everything the LiDAR can see turns white; what stays gray is the real blind spot.

Keys: space pause, s save PNG + CSV, c reset blind-spot stats, +/- zoom, q quit.
"""
import argparse
import csv
import glob
import math
import random
import signal
import sys
import textwrap
import threading
import time
from collections import deque

# ---------------------------------------------------------------- RPLidar A1 protocol
SYNC = 0xA5
CMD_STOP, CMD_RESET, CMD_SCAN = 0x25, 0x40, 0x20
CMD_GET_INFO, CMD_GET_HEALTH = 0x50, 0x52
SCAN_DESCRIPTOR = bytes([0xA5, 0x5A, 0x05, 0x00, 0x00, 0x40, 0x81])
HEALTH_NAMES = {0: "good", 1: "warning", 2: "ERROR"}
BAUD = 115200


def parse_node(b):
    """One 5-byte scan node -> (start, quality, angle_deg, dist_mm), or None if it fails the
    protocol's check bits (used to re-synchronize). dist_mm 0 means "no measurement"."""
    start = b[0] & 0x01
    not_start = (b[0] >> 1) & 0x01
    if start == not_start or not (b[1] & 0x01):
        return None
    quality = b[0] >> 2
    angle = ((b[2] << 7) | (b[1] >> 1)) / 64.0
    dist = (b[3] | (b[4] << 8)) / 4.0
    if angle >= 360.0:
        return None
    return bool(start), quality, angle, dist


class NodeStream:
    """Turns raw serial bytes into nodes, dropping bytes until the check bits line up."""

    def __init__(self):
        self.buf = bytearray()
        self.dropped = 0

    def feed(self, data):
        self.buf += data
        nodes = []
        while len(self.buf) >= 5:
            node = parse_node(self.buf[:5])
            if node is None:
                del self.buf[0]
                self.dropped += 1
                continue
            nodes.append(node)
            del self.buf[:5]
        return nodes


class RevolutionAssembler:
    """Groups nodes into full 360° revolutions, split on the start flag."""

    def __init__(self):
        self.points = []

    def add(self, node):
        """Returns a finished revolution [(angle, dist_mm, quality), ...] or None."""
        start, quality, angle, dist = node
        done = None
        if start and self.points:
            done, self.points = self.points, []
        self.points.append((angle, dist, quality))
        return done


class RPLidar:
    def __init__(self, port):
        import serial
        self.port = port
        self.ser = serial.Serial(port, BAUD, timeout=1)
        self.ser.dtr = True            # A1 USB adapter: DTR asserted = motor stopped
        self.stream = NodeStream()
        self.assembler = RevolutionAssembler()

    def _cmd(self, cmd):
        self.ser.write(bytes([SYNC, cmd]))

    def _response(self, length):
        head = self.ser.read(7)
        if len(head) != 7 or head[:2] != b"\xa5\x5a":
            return None
        data = self.ser.read(length)
        return data if len(data) == length else None

    def _info_once(self):
        self._cmd(CMD_STOP)
        time.sleep(0.05)
        self.ser.reset_input_buffer()
        self._cmd(CMD_GET_INFO)
        d = self._response(20)
        if d is None:
            return None
        return {"model": d[0], "firmware": f"{d[2]}.{d[1]:02d}", "hardware": d[3],
                "serial": d[4:].hex().upper()}

    def info(self, tries=4):
        """The A1 often ignores the first request after the port opens, so retry, then
        soft-reset it (it reboots in ~2 s) and try again."""
        for attempt in range(2):
            for _ in range(tries):
                info = self._info_once()
                if info:
                    return info
                time.sleep(0.3)
            if attempt == 0:
                self._cmd(CMD_RESET)
                time.sleep(2.0)
                self.ser.reset_input_buffer()
        return None

    def health(self):
        self._cmd(CMD_GET_HEALTH)
        d = self._response(3)
        return None if d is None else (d[0], d[1] | (d[2] << 8))

    def start(self):
        self.ser.dtr = False           # motor on
        time.sleep(1.0)                # let it reach speed
        for attempt in range(6):
            if attempt == 3:           # still refusing: soft-reset (reboots in ~2 s) and retry
                self._cmd(CMD_RESET)
                time.sleep(2.0)
            self._cmd(CMD_STOP)
            time.sleep(0.05)
            self.ser.reset_input_buffer()
            self._cmd(CMD_SCAN)
            if self.ser.read(7) == SCAN_DESCRIPTOR:
                return
        raise RuntimeError("the LiDAR answers but will not start scanning. Is the head spinning? "
                           "If not, check nothing touches it and that it gets enough USB power "
                           "(unplug and replug it, or use a powered hub).")

    def revolutions(self, stop_event):
        while not stop_event.is_set():
            for node in self.stream.feed(self.ser.read(max(5, self.ser.in_waiting))):
                rev = self.assembler.add(node)
                if rev:
                    yield rev

    def close(self):
        try:
            self._cmd(CMD_STOP)
            time.sleep(0.05)
            self.ser.dtr = True        # motor off
        finally:
            self.ser.close()


def find_lidar(ports=None):
    """Probe serial ports with GET_INFO; the ESP32 never answers, the LiDAR does."""
    for port in ports or sorted(glob.glob("/dev/ttyUSB*")):
        try:
            lidar = RPLidar(port)
        except Exception:
            continue
        info = lidar.info(tries=2)
        if info:
            return lidar, info
        lidar.ser.close()
    return None, None


# ---------------------------------------------------------------- demo source
class DemoLidar:
    """Simulated 4 m x 3 m room with a box, and the camera tower behind the LiDAR:
    a post 9 cm away at 165-195° that returns orange self-hits at its edges and nothing
    (too close for the A1) in the middle."""

    port = "demo"

    def __init__(self, seed=1):
        self.rng = random.Random(seed)

    def info(self):
        return {"model": "demo", "firmware": "-", "hardware": "-", "serial": "-"}

    def health(self):
        return (0, 0)

    def start(self):
        pass

    def close(self):
        pass

    @staticmethod
    def _wall_mm(angle):
        a = math.radians(angle)
        dx, dy = math.sin(a), math.cos(a)          # 0° = front (up), clockwise
        hits = []
        for wall, d in ((1500, dx), (-2500, dx), (1800, dy), (-1200, dy)):
            if d * wall > 0:
                hits.append(wall / d)
        # 40 cm box, front-right
        bx, by, half = 900, 900, 200
        for t in range(400, 1600, 5):
            x, y = dx * t, dy * t
            if abs(x - bx) < half and abs(y - by) < half:
                hits.append(t)
                break
        return min(hits)

    def revolutions(self, stop_event):
        while not stop_event.is_set():
            time.sleep(0.18)                       # ~5.5 Hz like the A1M8
            rev = []
            angle = self.rng.uniform(0, 0.9)
            while angle < 360:
                if 165 <= angle <= 195:
                    dist = 90.0 if (angle < 170 or angle > 190) and self.rng.random() < 0.6 else 0.0
                elif self.rng.random() < 0.03:
                    dist = 0.0                     # occasional dropout
                else:
                    dist = self._wall_mm(angle) + self.rng.gauss(0, 8)
                rev.append((angle, dist, 47 if dist else 0))
                angle += 0.9 + self.rng.uniform(-0.05, 0.05)
            yield rev


# ---------------------------------------------------------------- blind spot measurement
class BlindSpotTracker:
    """Per-degree record of which angles produced a usable return (not zero, not the robot)
    over the last `window` revolutions (~15 s at 7 Hz). A degree is blind if it was usable in
    fewer than `min_fraction` of them."""

    def __init__(self, tower_mm, window=100, min_fraction=0.03):
        self.tower_mm = tower_mm
        self.history = deque(maxlen=window)
        self.min_fraction = min_fraction

    def reset(self):
        self.history.clear()

    def add(self, rev):
        seen = [False] * 360
        for angle, dist, _ in rev:
            if dist > self.tower_mm:
                seen[int(angle) % 360] = True
        self.history.append(seen)

    def blind_degrees(self):
        if not self.history:
            return []
        n = len(self.history)
        return [d for d in range(360)
                if sum(h[d] for h in self.history) < self.min_fraction * n]

    def sectors(self, min_width=8):
        """Contiguous blind runs as (start_deg, end_deg) inclusive, wrapping past 360.
        Runs narrower than min_width degrees (scattered dropouts) are ignored."""
        return blind_sectors(self.blind_degrees(), min_width)


def blind_sectors(degrees, min_width=8):
    blind = set(degrees)
    if not blind:
        return []
    if len(blind) == 360:
        return [(0, 359)]
    start = next(d for d in range(360) if d not in blind)   # begin outside a run
    runs, run = [], []
    for i in range(1, 361):
        d = (start + i) % 360
        if d in blind:
            run.append(d)
        elif run:
            runs.append(run)
            run = []
    return [(r[0], r[-1]) for r in runs if len(r) >= min_width]


def sector_width(sector):
    s, e = sector
    return (e - s) % 360 + 1


def describe_sectors(sectors):
    if not sectors:
        return "none"
    return ", ".join(f"{s}°–{e}° ({sector_width((s, e))}°)" for s, e in sectors)


def parse_blind_arg(text):
    """'160:200' -> (160, 200)."""
    try:
        s, e = (int(v) % 360 for v in text.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError("use START:END in degrees, e.g. 160:200")
    return s, e


# ---------------------------------------------------------------- scan summary
def scan_stats(rev, tower_mm):
    valid = [(a, d) for a, d, _ in rev if d > tower_mm]
    tower = sum(1 for _, d, _ in rev if 0 < d <= tower_mm)
    nearest = min(valid, key=lambda p: p[1]) if valid else None
    return {"points": len(rev), "valid": len(valid), "tower": tower, "nearest": nearest}


def save_csv(path, rev, front_offset):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lidar_angle_deg", "robot_angle_deg", "distance_mm", "quality"])
        for angle, dist, q in rev:
            w.writerow([f"{angle:.2f}", f"{(angle - front_offset) % 360:.2f}", f"{dist:.1f}", q])


# ---------------------------------------------------------------- reader thread
class Reader(threading.Thread):
    def __init__(self, lidar, tracker):
        super().__init__(daemon=True)
        self.lidar, self.tracker = lidar, tracker
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.recent = deque(maxlen=3)
        self.count = 0
        self.times = deque(maxlen=20)
        self.error = None

    def run(self):
        try:
            for rev in self.lidar.revolutions(self.stop_event):
                with self.lock:
                    self.latest = rev
                    self.recent.append(rev)
                    self.count += 1
                    self.times.append(time.time())
                    self.tracker.add(rev)
        except Exception as e:      # serial unplugged etc.: show it, don't hang the GUI
            self.error = str(e)

    def snapshot(self):
        with self.lock:
            hz = ((len(self.times) - 1) / (self.times[-1] - self.times[0])
                  if len(self.times) > 1 and self.times[-1] > self.times[0] else 0.0)
            points = [p for rev in self.recent for p in rev]
            return self.latest, points, self.count, hz, self.tracker.sectors(), len(self.tracker.history)


# ---------------------------------------------------------------- plotting
class Viewer:
    def __init__(self, plt, reader, args, info):
        import numpy as np
        self.np, self.plt, self.reader, self.args, self.info = np, plt, reader, args, info
        self.paused = False
        self.range_m = args.range_m
        self.last_rev = None
        self.last_points = []
        self.last_sectors = None

        self.fig = plt.figure("SecureBot LiDAR viewer", figsize=(9, 8))
        self.ax = self.fig.add_axes([0.05, 0.17, 0.9, 0.75], projection="polar")
        self.ax.set_theta_zero_location("N")          # robot front at the top
        self.ax.set_theta_direction(-1)               # RPLidar angles run clockwise
        self.ax.set_rlim(0, self.range_m)
        self.ax.set_rlabel_position(112)
        self.ax.grid(alpha=0.35)
        self.scatter = self.ax.scatter([], [], s=6, c=[], cmap="viridis_r",
                                       vmin=0, vmax=self.range_m)
        self.tower = self.ax.scatter([], [], s=14, c="tab:orange", label="robot / tower")
        self.ax.plot([0], [0], marker="o", color="black", markersize=8)
        self.ax.annotate("", xy=(0, self.range_m * 0.12), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="->", color="tab:red", lw=2))
        self.wedges = []
        self.manual = [parse_blind_arg(b) for b in args.blind]
        self.title = self.fig.text(0.5, 0.965, "", ha="center", fontsize=12, weight="bold")
        self.status = self.fig.text(0.02, 0.115, "", family="monospace", fontsize=9, va="top")
        self.fig.text(0.98, 0.015, "space pause · s save · c reset blind · +/- zoom · q quit",
                      ha="right", fontsize=8, color="gray")
        self.fig.colorbar(self.scatter, ax=self.ax, shrink=0.6, pad=0.08, label="distance (m)")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def robot_theta(self, angles):
        return self.np.radians((self.np.asarray(angles) - self.args.front_offset) % 360)

    def draw_wedges(self, sectors):
        for w in self.wedges:
            w.remove()
        self.wedges = []
        for (s, e), style in [(x, "auto") for x in sectors] + [(x, "manual") for x in self.manual]:
            width = sector_width((s, e))
            center = (s + width / 2.0 - self.args.front_offset) % 360
            bars = self.ax.bar([math.radians(center)], [self.range_m], width=math.radians(width),
                               bottom=0, color="gray" if style == "auto" else "tab:purple",
                               alpha=0.22, edgecolor="none", zorder=0)
            self.wedges.extend(bars)

    def update(self, _frame=None):
        rev, points, count, hz, sectors, measured = self.reader.snapshot()
        if self.reader.error:
            self.title.set_text(f"LiDAR read error: {self.reader.error}")
            self.title.set_color("tab:red")
        if rev is None:
            if not self.reader.error:
                self.title.set_text("waiting for the first revolution…")
            return
        if self.paused:
            rev, points = self.last_rev, self.last_points
        self.last_rev, self.last_points = rev, points
        np = self.np
        tower_mm = self.args.tower_cm * 10
        far = [(a, d / 1000.0) for a, d, _ in points if d > tower_mm]       # last 3 revolutions
        near = [(a, d / 1000.0) for a, d, _ in points if 0 < d <= tower_mm]
        if far:
            a, r = zip(*far)
            self.scatter.set_offsets(np.column_stack([self.robot_theta(a), r]))
            self.scatter.set_array(np.asarray(r))
        else:
            self.scatter.set_offsets(np.empty((0, 2)))
        if near:
            a, r = zip(*near)
            self.tower.set_offsets(np.column_stack([self.robot_theta(a), r]))
        else:
            self.tower.set_offsets(np.empty((0, 2)))
        if sectors != self.last_sectors:
            self.draw_wedges(sectors)
            self.last_sectors = sectors

        st = scan_stats(rev, tower_mm)
        coverage = 360 - sum(sector_width(s) for s in sectors)
        nearest = (f"{st['nearest'][1] / 10:.0f} cm at {(st['nearest'][0] - self.args.front_offset) % 360:.0f}°"
                   if st["nearest"] else "-")
        if not self.reader.error:
            self.title.set_text(f"{'PAUSED  ' if self.paused else ''}RPLidar {self.reader.lidar.port}"
                                f"  ·  {hz:.1f} Hz  ·  revolution {count}")
            self.title.set_color("black")
        self.status.set_text(
            f"points {st['points']:4d}   usable {st['valid']:4d}   robot/tower hits {st['tower']:3d}"
            f"   nearest {nearest}\n"
            + textwrap.fill(f"coverage {coverage}° of 360°   blind (gray, last {measured} revs): "
                            f"{describe_sectors(sectors)}", 115)
            + (f"\nmanual blind (purple): {describe_sectors(self.manual)}" if self.manual else ""))

    def on_key(self, event):
        if event.key == " ":
            self.paused = not self.paused
        elif event.key == "c":
            with self.reader.lock:
                self.reader.tracker.reset()
        elif event.key in ("+", "="):
            self.set_range(max(0.5, self.range_m / 1.5))
        elif event.key in ("-", "_"):
            self.set_range(min(12.0, self.range_m * 1.5))
        elif event.key == "s" and self.last_rev:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.fig.savefig(f"lidar_{stamp}.png", dpi=110)
            save_csv(f"lidar_{stamp}.csv", self.last_rev, self.args.front_offset)
            print(f"saved lidar_{stamp}.png and lidar_{stamp}.csv")

    def set_range(self, r):
        self.range_m = r
        self.ax.set_rlim(0, r)
        self.scatter.set_clim(0, r)
        self.last_sectors = None        # redraw wedges at the new length


# ---------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("port", nargs="?", help="LiDAR serial port (default: probe /dev/ttyUSB*)")
    p.add_argument("--demo", action="store_true", help="simulated scans, no hardware")
    p.add_argument("--range", dest="range_m", type=float, default=4.0, help="plot radius in m (4)")
    p.add_argument("--tower-cm", type=float, default=20.0,
                   help="returns closer than this are the robot itself (20)")
    p.add_argument("--front-offset", type=float, default=0.0,
                   help="LiDAR angle that points at the robot's front, in degrees (0)")
    p.add_argument("--blind", action="append", default=[], metavar="START:END",
                   help="also shade a known blind sector (LiDAR degrees), e.g. 160:200")
    p.add_argument("--snapshot", metavar="PNG", help="no window: collect, save a picture, exit")
    p.add_argument("--revs", type=int, default=30, help="revolutions to collect for --snapshot (30)")
    args = p.parse_args(argv)

    if args.demo:
        lidar, info = DemoLidar(), DemoLidar().info()
    elif args.port:
        lidar = RPLidar(args.port)
        info = lidar.info()
        if not info:
            lidar.ser.close()
            sys.exit(f"No RPLidar answer on {args.port}. Check the USB cable and that nothing "
                     "else has the port open.")
    else:
        lidar, info = find_lidar()
        if not lidar:
            sys.exit("No RPLidar found on /dev/ttyUSB*. Is it plugged into the Jetson? "
                     "(Try --demo to see the viewer without hardware.)")

    print(f"LiDAR on {lidar.port}: model {info['model']}, firmware {info['firmware']}, "
          f"hardware {info['hardware']}, serial {info['serial']}")
    health = lidar.health()
    if not health:
        print("health: no answer")
    else:
        print(f"health: {HEALTH_NAMES.get(health[0], health[0])} (code {health[1]})")
        if health[0] == 2:
            lidar.close()
            sys.exit("LiDAR reports an error; power-cycle it (unplug USB) and try again.")

    tracker = BlindSpotTracker(args.tower_cm * 10)
    reader = Reader(lidar, tracker)
    try:
        try:
            lidar.start()
        except RuntimeError as e:
            sys.exit(f"Cannot scan: {e}")
        reader.start()
        import matplotlib
        if args.snapshot:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        viewer = Viewer(plt, reader, args, info)
        if args.snapshot:
            deadline = time.time() + args.revs / 3.0 + 10
            while reader.count < args.revs and time.time() < deadline and not reader.error:
                time.sleep(0.1)
            viewer.update()
            viewer.fig.savefig(args.snapshot, dpi=110)
            print(f"saved {args.snapshot} after {reader.count} revolutions")
        else:
            from matplotlib.animation import FuncAnimation
            # Ctrl+C (or a kill) closes the window, so the finally block stops the motor.
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, lambda *_: plt.close("all"))
            anim = FuncAnimation(viewer.fig, viewer.update, interval=150, cache_frame_data=False)  # noqa: F841
            plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop_event.set()
        if reader.is_alive():
            reader.join(timeout=2)
        lidar.close()
        if not tracker.history:
            return
        sectors = tracker.sectors()
        print(f"\nMeasured blind sectors over the last {len(tracker.history)} revolutions "
              f"(LiDAR degrees, clockwise from the LiDAR's 0°): {describe_sectors(sectors)}")
        if sectors:
            print("Keep these: the ROS laser filter in Sprint 2 will mask exactly these angles.")


if __name__ == "__main__":
    main()
