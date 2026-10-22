"""Tests for lidar_viewer.py. Run: python3 -m unittest tools.test_lidar_viewer -v"""
import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import lidar_viewer as lv  # noqa: E402


def node_bytes(start, quality, angle, dist_mm):
    """Encode one scan node the way the A1 sends it."""
    a = int(round(angle * 64))
    d = int(round(dist_mm * 4))
    return bytes([(quality << 2) | ((not start) << 1) | start,
                  ((a & 0x7F) << 1) | 1, a >> 7, d & 0xFF, d >> 8])


class ParseNode(unittest.TestCase):
    def test_round_trip(self):
        start, q, angle, dist = lv.parse_node(node_bytes(1, 47, 123.5, 1532.25))
        self.assertEqual((start, q), (True, 47))
        self.assertAlmostEqual(angle, 123.5, places=1)
        self.assertAlmostEqual(dist, 1532.25)

    def test_bad_start_bits_rejected(self):
        b = bytearray(node_bytes(1, 47, 10, 500))
        b[0] |= 0x02                       # S and !S both set
        self.assertIsNone(lv.parse_node(b))

    def test_bad_check_bit_rejected(self):
        b = bytearray(node_bytes(0, 47, 10, 500))
        b[1] &= 0xFE
        self.assertIsNone(lv.parse_node(b))

    def test_zero_distance_kept_as_no_measurement(self):
        self.assertEqual(lv.parse_node(node_bytes(0, 0, 90, 0))[3], 0.0)


class Stream(unittest.TestCase):
    def test_resyncs_after_garbage_and_split_reads(self):
        data = b"\x00\x13" + node_bytes(1, 40, 0.5, 300) + node_bytes(0, 40, 1.5, 310)
        s = lv.NodeStream()
        nodes = s.feed(data[:4]) + s.feed(data[4:9]) + s.feed(data[9:])
        self.assertEqual([round(n[2], 1) for n in nodes], [0.5, 1.5])
        self.assertGreaterEqual(s.dropped, 2)

    def test_revolutions_split_on_start_flag(self):
        asm = lv.RevolutionAssembler()
        out = []
        for start, angle in [(1, 0), (0, 120), (0, 240), (1, 1), (0, 180)]:
            rev = asm.add((bool(start), 40, angle, 1000))
            if rev:
                out.append(rev)
        self.assertEqual(len(out), 1)
        self.assertEqual([p[0] for p in out[0]], [0, 120, 240])


class Blind(unittest.TestCase):
    def rev(self, blind=range(0), tower=range(0)):
        return [(a + 0.5, 90.0 if a in tower else 0.0 if a in blind else 1500.0, 40)
                for a in range(360)]

    def test_tower_and_no_return_both_count_as_blind(self):
        t = lv.BlindSpotTracker(tower_mm=200)
        for _ in range(10):
            t.add(self.rev(blind=range(170, 191), tower=range(160, 170)))
        self.assertEqual(t.sectors(), [(160, 190)])

    def test_sector_wrapping_through_zero(self):
        self.assertEqual(lv.blind_sectors(list(range(350, 360)) + list(range(0, 6))), [(350, 5)])
        self.assertEqual(lv.sector_width((350, 5)), 16)

    def test_scattered_dropouts_ignored(self):
        self.assertEqual(lv.blind_sectors([10, 50, 51, 200]), [])

    def test_occasional_return_still_blind(self):
        t = lv.BlindSpotTracker(tower_mm=200, min_fraction=0.2)
        for i in range(10):
            t.add(self.rev(blind=range(100, 110) if i else range(0)))   # seen 1 in 10
        self.assertEqual(t.sectors(), [(100, 109)])

    def test_all_blind_and_none(self):
        self.assertEqual(lv.blind_sectors(range(360)), [(0, 359)])
        self.assertEqual(lv.BlindSpotTracker(200).sectors(), [])
        self.assertEqual(lv.describe_sectors([]), "none")

    def test_parse_blind_arg(self):
        self.assertEqual(lv.parse_blind_arg("160:200"), (160, 200))
        self.assertEqual(lv.parse_blind_arg("350:370"), (350, 10))


class Demo(unittest.TestCase):
    def test_demo_has_tower_blind_spot_near_180(self):
        demo, stop, t = lv.DemoLidar(), threading.Event(), lv.BlindSpotTracker(200)
        for i, rev in enumerate(demo.revolutions(stop)):
            t.add(rev)
            if i == 9:
                break
        sectors = t.sectors()
        self.assertEqual(len(sectors), 1)
        s, e = sectors[0]
        self.assertTrue(160 <= s <= 166 and 194 <= e <= 200, sectors)

    def test_scan_stats_and_csv(self):
        rev = [(0.0, 1000.0, 40), (90.0, 80.0, 40), (180.0, 0.0, 0), (270.0, 450.0, 40)]
        st = lv.scan_stats(rev, 200)
        self.assertEqual((st["points"], st["valid"], st["tower"]), (4, 2, 1))
        self.assertEqual(st["nearest"], (270.0, 450.0))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.csv")
            lv.save_csv(path, rev, front_offset=90)
            with open(path) as f:
                lines = f.read().splitlines()
        self.assertEqual(lines[0], "lidar_angle_deg,robot_angle_deg,distance_mm,quality")
        self.assertEqual(lines[1], "0.00,270.00,1000.0,40")


class Snapshot(unittest.TestCase):
    def test_demo_snapshot_renders(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "demo.png")
            lv.main(["--demo", "--snapshot", out, "--revs", "8"])
            self.assertGreater(os.path.getsize(out), 10000)


if __name__ == "__main__":
    unittest.main()
