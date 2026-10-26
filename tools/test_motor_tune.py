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

    def test_frequency_step_keys(self):
        s = mt.TuneState(freq=10000)
        steps = {'1': 9950, '2': 10000, '3': 9900, '4': 10000, '5': 9500, '6': 10000,
                 '7': 9000, '8': 10000}
        for key, expected in steps.items():
            s.key(key, 0.0)
            self.assertEqual(s.freq, expected, key)
            self.assertEqual(s.pending[-1], f'freq {expected}')

    def test_frequency_stays_in_firmware_range(self):
        s = mt.TuneState(freq=100)
        s.key('7', 0.0)                                    # -1000 from 100
        self.assertEqual(s.freq, mt.FREQ_MIN)
        n = len(s.pending)
        s.key('1', 0.0)                                    # already at the minimum
        self.assertEqual((s.freq, len(s.pending)), (mt.FREQ_MIN, n))   # nothing new sent
        s = mt.TuneState(freq=24800)
        s.key('6', 0.0)
        self.assertEqual(s.freq, mt.FREQ_MAX)

    def test_power_step_keys(self):
        s = mt.TuneState(power=500)
        steps = {'!': 490, '@': 500, '#': 475, '$': 500, '%': 450, '^': 500, '&': 400, '*': 500,
                 '-': 475, '+': 500}
        for key, expected in steps.items():
            s.key(key, 0.0)
            self.assertEqual(s.power, expected, key)
        self.assertEqual(s.pending, [])                    # power is sent with the next M,l,r

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
