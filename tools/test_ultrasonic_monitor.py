"""Tests for ultrasonic_monitor.py. Run: python3 -m unittest tools.test_ultrasonic_monitor -v"""
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import ultrasonic_monitor as um  # noqa: E402


class FakeSerial:
    def __init__(self, lines):
        self.lines = [l.encode() + b"\n" for l in lines]

    def readline(self):
        return self.lines.pop(0) if self.lines else b""


class Clock:
    """Fake clock: every call advances 50 ms, so --seconds runs end deterministically."""
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.05
        return self.t


class ParseTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(um.parse_u_line("U,1.25,-1.00"), (1.25, -1.0))

    def test_rejects_other_lines(self):
        for bad in ("S,stop,0,0,0", "U,1.0", "U,a,b", "", "U,1,2,3", "I,hello"):
            self.assertIsNone(um.parse_u_line(bad), bad)


class SensorTests(unittest.TestCase):
    def test_stats(self):
        s = um.Sensor("LEFT")
        for ft in (-1, 1.0, 2.0, -1):
            s.update(ft)
        self.assertEqual((s.total, s.echoes), (4, 2))
        self.assertAlmostEqual(s.min_cm, 30.48)
        self.assertAlmostEqual(s.max_cm, 60.96)
        self.assertEqual(s.percent, 50.0)
        self.assertFalse(s.has_echo)

    def test_render_no_echo(self):
        s = um.Sensor("LEFT"); s.update(-1)
        line = um.render_sensor(s)
        self.assertIn("NO ECHO", line); self.assertIn("0/1 (0%)", line); self.assertNotIn("█", line)

    def test_render_zones(self):
        for ft, zone in ((0.5, "OBSTACLE"), (1.5, "near"), (4.0, "clear")):
            s = um.Sensor("RIGHT"); s.update(ft)
            line = um.render_sensor(s)
            self.assertIn(zone, line, ft); self.assertIn("█", line)

    def test_render_units_and_bar_length(self):
        s = um.Sensor("LEFT"); s.update(1.0)
        self.assertIn("30.5 cm", um.render_sensor(s, "cm"))
        self.assertIn("12.0 in", um.render_sensor(s, "in"))
        self.assertIn("1.0 ft", um.render_sensor(s, "ft"))
        self.assertEqual(len(um.render_sensor(s, width=20).split("[")[1].split("]")[0]), 20)

    def test_color_codes_only_when_asked(self):
        s = um.Sensor("LEFT"); s.update(0.5)
        self.assertNotIn("\x1b[", um.render_sensor(s, color=False))
        self.assertIn(um.RED, um.render_sensor(s, color=True))


class DrainTests(unittest.TestCase):
    class Port:
        """in_waiting stays high for `busy` flushes (stale backlog), then drops to the live rate."""
        def __init__(self, busy):
            self.busy, self.flushes = busy, 0

        def reset_input_buffer(self):
            self.flushes += 1

        @property
        def in_waiting(self):
            return 5000 if self.flushes <= self.busy else 15

    def test_flushes_until_quiet(self):
        port = self.Port(busy=3)
        waited = um.drain(port, sleep=lambda s: None)
        self.assertEqual(port.flushes, 4)           # three busy rounds, then one quiet check
        self.assertAlmostEqual(waited, 0.6)

    def test_gives_up_after_max(self):
        port = self.Port(busy=10 ** 6)
        waited = um.drain(port, sleep=lambda s: None, max_s=1.0)
        self.assertLessEqual(waited, 1.2)
        self.assertGreaterEqual(port.flushes, 5)


class MonitorTests(unittest.TestCase):
    def run_monitor(self, lines, **kw):
        out = io.StringIO()
        sensors = um.monitor(FakeSerial(lines), out, seconds=5, clock=Clock(), **kw)
        return sensors, out.getvalue()

    def test_only_left_working(self):
        lines = ["S,stop,0,0,0"] + ["U,1.00,-1.00"] * 10
        (left, right), out = self.run_monitor(lines, redraw=False)
        self.assertEqual((left.echoes, left.total), (10, 10))
        self.assertEqual((right.echoes, right.total), (0, 10))
        self.assertIn("robot status: stop,0,0,0", out)
        self.assertIn("NO ECHO", out)

    def test_ignores_garbage_and_survives_silence(self):
        lines = ["\xff\xff", "U,bad", "U,0.5,0.6", "I,hi"]
        (left, right), _ = self.run_monitor(lines, redraw=False)
        self.assertEqual((left.total, right.total), (1, 1))

    def test_redraw_moves_cursor_up(self):
        _, out = self.run_monitor(["U,1,1"] * 5, redraw=True)
        self.assertIn("\x1b[3A", out)         # 2 sensor lines + 1 status line

    def test_summary_verdicts(self):
        (left, right), _ = self.run_monitor(["U,1.00,-1.00"] * 3, redraw=False)
        text = um.summary([left, right])
        self.assertIn("LEFT  3/3", text); self.assertIn("WORKING", text)
        self.assertIn("RIGHT 0/3", text); self.assertIn("NO ECHO AT ALL", text)


if __name__ == "__main__":
    unittest.main()
