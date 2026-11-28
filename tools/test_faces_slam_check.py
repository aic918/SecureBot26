"""Tests for faces_slam_check.py. Run: python3 -m unittest tools.test_faces_slam_check -v"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import faces_slam_check as fs  # noqa: E402

OPTS = {'min_face_fps': 2.0, 'min_scan_hz': 5.0, 'max_tf_gap': 1.0, 'jump_m': 0.5,
        'max_temp': 85.0}
LINE = ('11-27-2026 20:26:17 RAM 5059/7607MB (lfb 3x4MB) SWAP 451/3804MB (cached 0MB) '
        'CPU [12%@729,4%@729,6%@729,6%@729,6%@729,3%@729] GR3D_FREQ 37% cpu@45.781C '
        'soc2@45.281C soc0@45.062C gpu@45.5C tj@45.906C soc1@45.906C VDD_IN 5680mW/5680mW')


def good_row(t):
    return {'t': t, 'face_fps': 6, 'face_status': 'NO_PERSON', 'scan_hz': 10, 'tf_age': 0.1,
            'jump_m': 0.05, 'job': 'laps', 'job_state': 'running', 'cpu_pct': 40.0,
            'gpu_pct': 60, 'cpu_temp': 55.0, 'gpu_temp': 54.0}


class CheckTest(unittest.TestCase):
    def test_parse_tegrastats(self):
        p = fs.parse_tegrastats(LINE)
        self.assertEqual(p['ram_mb'], 5059)
        self.assertAlmostEqual(p['cpu_pct'], 6.2, places=1)
        self.assertEqual(p['gpu_pct'], 37)
        self.assertAlmostEqual(p['cpu_temp'], 45.781)
        self.assertAlmostEqual(p['gpu_temp'], 45.5)
        self.assertEqual(fs.parse_tegrastats('junk')['cpu_pct'], None)

    def test_all_good_passes(self):
        ok, lines = fs.summarize([good_row(t) for t in range(60)], OPTS)
        self.assertTrue(ok, lines)
        self.assertIn('PASS: face recognition and SLAM ran well together', lines[0])
        self.assertTrue(any(line.startswith('Load: CPU 40 %') for line in lines))

    def test_each_failure_is_named(self):
        cases = {
            'Face recognition runs': {'face_fps': 1},
            'LiDAR keeps its rate': {'scan_hz': 2},
            'Position keeps updating': {'tf_age': 3.0},
            'No position jumps': {'jump_m': 1.2},
            'Laps / mapping without errors': {'job_state': 'error'},
            'Temperatures OK': {'gpu_temp': 91.0},
        }
        for name, change in cases.items():
            rows = [good_row(t) for t in range(60)]
            for r in rows[20:]:
                r.update(change)
            ok, lines = fs.summarize(rows, OPTS)
            self.assertFalse(ok, name)
            self.assertTrue(any(line.startswith(f'FAIL  {name}') for line in lines), lines)

    def test_missing_streams(self):
        rows = [{'t': t, 'face_fps': None, 'scan_hz': None} for t in range(5)]
        ok, lines = fs.summarize(rows, OPTS)
        self.assertFalse(ok)
        self.assertIn('is faces:=true on?', '\n'.join(lines))
        self.assertEqual(fs.summarize([], OPTS), (False, ['No data recorded.']))


if __name__ == '__main__':
    unittest.main()
