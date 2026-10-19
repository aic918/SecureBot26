#!/usr/bin/env python3
"""Live view of the two ultrasonic sensors, read from the ESP32's serial output.

Works with the robot firmware (securebot_esp32), which prints `U,<left_ft>,<right_ft>`
about 8 times a second; -1 means "no echo". Read-only: it never sends anything, so the
motors cannot be affected. Opening the port resets the ESP32 (normal).

  python3 tools/ultrasonic_monitor.py                   # finds the ESP32 port itself
  python3 tools/ultrasonic_monitor.py /dev/ttyUSB1      # or name the port
  python3 tools/ultrasonic_monitor.py --seconds 20      # stop after 20 s and print a summary

Close anything else using the serial port first (PlatformIO/Arduino serial monitor, ROS bridge).
Press Ctrl+C to quit; a summary of how often each sensor got an echo is printed.
"""
import argparse
import glob
import sys
import time

FT_TO_CM = 30.48
OBSTACLE_CM = 30.48          # firmware STOP_DISTANCE_FEET = 1.0 ft
NEAR_CM = 60.0
UNITS = {"cm": (FT_TO_CM, "cm"), "in": (12.0, "in"), "ft": (1.0, "ft")}

RED, YELLOW, GREEN, GRAY, RESET = "\x1b[31m", "\x1b[33m", "\x1b[32m", "\x1b[90m", "\x1b[0m"


def parse_u_line(line):
    """`U,1.25,-1.00` -> (1.25, -1.0); anything else -> None."""
    if not line.startswith("U,"):
        return None
    parts = line[2:].split(",")
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


class Sensor:
    def __init__(self, name):
        self.name = name
        self.total = 0
        self.echoes = 0
        self.last_ft = -1.0
        self.min_cm = None
        self.max_cm = None

    def update(self, ft):
        self.total += 1
        self.last_ft = ft
        if ft > 0:
            self.echoes += 1
            cm = ft * FT_TO_CM
            self.min_cm = cm if self.min_cm is None else min(self.min_cm, cm)
            self.max_cm = cm if self.max_cm is None else max(self.max_cm, cm)

    @property
    def has_echo(self):
        return self.last_ft > 0

    @property
    def percent(self):
        return 100.0 * self.echoes / self.total if self.total else 0.0


def render_sensor(sensor, unit="cm", width=36, scale_cm=200.0, color=False):
    """One display line for one sensor."""
    per_ft, label = UNITS[unit]
    c = (lambda code, text: f"{code}{text}{RESET}") if color else (lambda code, text: text)

    if sensor.has_echo:
        cm = sensor.last_ft * FT_TO_CM
        filled = max(1, int(min(cm, scale_cm) / scale_cm * width))
        bar = "█" * filled + "·" * (width - filled)
        tone = RED if cm < OBSTACLE_CM else YELLOW if cm < NEAR_CM else GREEN
        reading = f"{sensor.last_ft * per_ft:7.1f} {label}"
        zone = "OBSTACLE" if cm < OBSTACLE_CM else "near" if cm < NEAR_CM else "clear"
        bar, reading, zone = c(tone, bar), c(tone, reading), c(tone, f"{zone:<8}")
    else:
        bar = c(GRAY, "·" * width)
        reading = c(GRAY, f"{'NO ECHO':>10}")
        zone = c(GRAY, f"{'':<8}")

    span = ""
    if sensor.min_cm is not None:
        span = f"  min {sensor.min_cm / FT_TO_CM * per_ft:.1f} max {sensor.max_cm / FT_TO_CM * per_ft:.1f}"
    return (f"{sensor.name:<5} [{bar}] {reading} {zone}"
            f" echoes {sensor.echoes}/{sensor.total} ({sensor.percent:.0f}%){span}")


def summary(sensors):
    lines = ["", "Summary:"]
    for s in sensors:
        verdict = "WORKING" if s.echoes else "NO ECHO AT ALL"
        lines.append(f"  {s.name:<5} {s.echoes}/{s.total} readings had an echo -> {verdict}")
    return "\n".join(lines)


def monitor(ser, out, seconds=None, unit="cm", scale_cm=200.0, color=False, redraw=True,
            clock=time.monotonic):
    """Read `ser` (anything with readline() -> bytes) and draw live. Returns the sensors."""
    sensors = [Sensor("LEFT"), Sensor("RIGHT")]
    status = ""
    start = last_draw = clock()
    drawn = 0
    lines_seen = 0

    def draw(force=False):
        nonlocal drawn, last_draw
        now = clock()
        if not force and now - last_draw < 0.1:
            return
        last_draw = now
        rate = lines_seen / max(now - start, 1e-6)
        body = [render_sensor(s, unit, 36, scale_cm, color) for s in sensors]
        body.append(f"robot status: {status or '(waiting for an S line)'}   |   {rate:.0f} lines/s"
                    "   |   Ctrl+C to quit")
        if redraw and drawn:
            out.write(f"\x1b[{drawn}A")
        for line in body:
            out.write(("\x1b[2K" if redraw else "") + line + "\n")
        drawn = len(body) if redraw else 0
        out.flush()

    try:
        while seconds is None or clock() - start < seconds:
            raw = ser.readline()
            if not raw:
                draw()
                continue
            line = raw.decode("latin-1").strip()
            lines_seen += 1
            parsed = parse_u_line(line)
            if parsed:
                sensors[0].update(parsed[0])
                sensors[1].update(parsed[1])
            elif line.startswith("S,"):
                status = line[2:]
            if parsed or redraw:
                draw()
    except KeyboardInterrupt:
        pass
    draw(force=True)
    return sensors


def drain(ser, sleep=time.sleep, max_s=3.0, quiet_bytes=300):
    """Throw away the backlog the OS keeps while no program has the port open.

    On opening, the driver hands over everything the ESP32 sent in the meantime (thousands of
    old lines, delivered in the first ~0.2 s). Keep flushing until the input is quiet, i.e. only
    the normal live rate (about 15 bytes per 100 ms) is arriving.
    """
    waited = 0.0
    while waited < max_s:
        ser.reset_input_buffer()
        sleep(0.15)
        waited += 0.15
        if ser.in_waiting < quiet_bytes:
            return waited
    return waited


def find_port(serial_module, baud=115200, listen_s=2.5):
    """Return the first /dev/ttyUSB*/ttyACM* that prints `U,` lines (the ESP32 firmware)."""
    for port in sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")):
        try:
            ser = serial_module.Serial(port, baud, timeout=0.2)
        except (OSError, serial_module.SerialException):
            continue
        try:
            end = time.monotonic() + listen_s
            while time.monotonic() < end:
                line = ser.readline().decode("latin-1").strip()
                if parse_u_line(line):
                    return port
        finally:
            ser.close()
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("port", nargs="?", help="serial port (default: auto-detect)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--unit", choices=UNITS, default="cm")
    ap.add_argument("--scale", type=float, default=200.0, help="bar full-scale in cm (default 200)")
    ap.add_argument("--seconds", type=float, help="stop after this many seconds")
    ap.add_argument("--plain", action="store_true", help="no colors / in-place redraw; print lines")
    args = ap.parse_args(argv)

    try:
        import serial
    except ImportError:
        sys.exit("pyserial is missing: sudo apt install python3-serial")

    port = args.port or find_port(serial, args.baud)
    if not port:
        sys.exit("No port printing `U,` lines was found. Is the ESP32 plugged in, running the "
                 "robot firmware, and not in use by another program?")
    try:
        ser = serial.Serial(port, args.baud, timeout=0.2)
    except (OSError, serial.SerialException) as exc:
        sys.exit(f"Could not open {port}: {exc}\nClose any serial monitor or ROS node using it.")

    drain(ser)
    live = sys.stdout.isatty() and not args.plain
    print(f"SecureBot ultrasonic monitor on {port}  (read-only, nothing is sent to the robot)")
    try:
        sensors = monitor(ser, sys.stdout, args.seconds, args.unit, args.scale,
                          color=live, redraw=live)
    except (OSError, serial.SerialException) as exc:
        sys.exit(f"\nLost the serial port: {exc}")
    finally:
        ser.close()
    print(summary(sensors))


if __name__ == "__main__":
    main()
