"""Tests for motor_tune.py key handling. Run: python3 -m unittest tools.test_motor_tune -v"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import motor_tune as mt  # noqa: E402


class TuneStateTest(unittest.TestCase):
    def test_idle_sends_stop(self):
        self.assertEqual(mt.TuneState().command(0.0), 'stop')

    def test_held_key_drives_and_release_stops(self):
        s = mt.TuneState(power=500)
        s.key('w', 10.0)
        self.assertEqual(s.command(10.3), 'M,500,500')
        self.assertEqual(s.command(10.7), 'stop')          # no key repeat for > 0.6 s

    def test_spins_and_backward(self):
        s = mt.TuneState(power=400)
        s.key('a', 0.0)
        self.assertEqual(s.command(0.1), 'M,-400,400')     # spin left: left wheel back
        s.key('d', 0.2)
        self.assertEqual(s.command(0.3), 'M,400,-400')
        s.key('s', 0.4)
        self.assertEqual(s.command(0.5), 'M,-400,-400')

    def test_space_stops_and_q_quits(self):
        s = mt.TuneState()
        s.key('w', 0.0)
        s.key(' ', 0.1)
        self.assertEqual(s.command(0.2), 'stop')
        self.assertFalse(s.key('q', 0.3))

    def test_frequency_keys_queue_a_freq_command(self):
        s = mt.TuneState()
        s.key('4', 0.0)
        self.assertEqual((s.freq, s.pending), (1000, ['freq 1000']))
        s.key('1', 0.0)
        self.assertEqual(s.pending[-1], 'freq 100')

    def test_power_limits(self):
        s = mt.TuneState(power=500)
        for _ in range(40):
            s.key('+', 0.0)
        self.assertEqual(s.power, mt.POWER_MAX)
        for _ in range(40):
            s.key('-', 0.0)
        self.assertEqual(s.power, mt.POWER_MIN)
        self.assertLessEqual(mt.POWER_MAX, 850)            # firmware cap


if __name__ == '__main__':
    unittest.main()
