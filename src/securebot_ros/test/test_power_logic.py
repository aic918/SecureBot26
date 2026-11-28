"""Battery percentage, the low-battery decision and the guard's commands (power_logic.py)."""
import json

import pytest

from securebot_ros import power_logic as pl


def test_cell_curve_ends_and_middle():
    assert pl.cell_percent(4.25) == 100.0
    assert pl.cell_percent(4.20) == 100.0
    assert pl.cell_percent(3.0) == 0.0
    assert pl.cell_percent(3.84) == pytest.approx(50.0)
    values = [pl.cell_percent(v / 100) for v in range(320, 425)]
    assert values == sorted(values)                       # never goes down as voltage rises


def test_pack_percent_is_of_the_full_voltage():
    s = pl.DEFAULTS
    assert pl.pack_percent(27.8, s) == 100.0              # the robot's pack, fully charged
    assert pl.pack_percent(28.3, s) == 100.0
    assert 10 < pl.pack_percent(s['return_home_volts'], s) < 25
    assert pl.pack_percent(20.0, s) == 0.0
    values = [pl.pack_percent(v / 10, s) for v in range(220, 280)]
    assert values == sorted(values)


def test_settings_defaults_validation_and_round_trip(tmp_path):
    path = str(tmp_path / 'config' / 'power.json')
    assert pl.load_settings(path) == pl.DEFAULTS
    saved = pl.save_settings({**pl.DEFAULTS, 'return_home_volts': '26.2'}, path)
    assert saved['return_home_volts'] == 26.2
    assert pl.load_settings(path) == saved
    for bad in ({'return_home_volts': 20.0},              # under 3.3 V a cell
                {'return_home_volts': 27.6},              # less than 0.5 V under full
                {'full_volts': 31.0},
                {'cells': 8},
                {'pack_ohms': -1},
                {'max_amps': 0},
                {'return_home_volts': 'x'}):
        with pytest.raises(ValueError):
            pl.save_settings({**pl.DEFAULTS, **bad}, path)
    with open(path, 'w') as f:
        f.write('{broken')
    assert pl.load_settings(path) == pl.DEFAULTS
    with open(path, 'w') as f:                            # the old %-based file: defaults
        json.dump({'return_home_pct': 20, 'cells': 7, 'pack_ohms': 0.25}, f)
    assert pl.load_settings(path) == pl.DEFAULTS


def test_load_compensation_below_the_current_limit():
    m = pl.BatteryMonitor()
    m.update(0.0, 26.5, 0.0)
    m.update(0.5, 26.5 - 0.6 * 0.25, 0.6)                  # 0.6 A: sag added back
    assert m.rest == pytest.approx(26.5)
    assert not m.held


def test_low_only_after_holding_under_the_level():
    level = pl.DEFAULTS['return_home_volts']
    m = pl.BatteryMonitor()
    assert m.update(0.0, level + 1.0, 0.0) is None
    m2 = pl.BatteryMonitor({**pl.DEFAULTS, 'pack_ohms': 0.0})   # a moment's sag: not low
    m2.update(0.0, level + 0.2, 0.0)
    m2.update(0.5, level - 2.0, 0.5)
    assert not m2.low
    m = pl.BatteryMonitor()
    m.update(0.0, level - 0.1, 0.0)
    events = []
    t = 0.0
    for i in range(1, 30):
        t = i * 0.5
        events.append(m.update(t, level - 0.1, 0.0))
    assert events.count('low') == 1
    assert events.index('low') * 0.5 + 0.5 >= pl.LOW_HOLD_S
    s = m.status(t)
    assert s['low'] and s['return_home_volts'] == level


def test_low_clears_only_when_charged_above_the_margin():
    level = pl.DEFAULTS['return_home_volts']
    m = pl.BatteryMonitor()
    t = 0.0
    while not m.low:
        t += 0.5
        m.update(t, level - 0.3, 0.0)
    for _ in range(200):                     # just over the level: still low (no flapping)
        t += 0.5
        m.update(t, level + 0.1, 0.0)
    assert m.low
    for _ in range(200):                     # charged
        t += 0.5
        m.update(t, level + 1.0, 0.0)
    assert not m.low


def test_current_at_the_limit_holds_the_last_good_voltage():
    s = {**pl.DEFAULTS, 'pack_ohms': 0.25}
    m = pl.BatteryMonitor(s)
    t = 0.0
    for _ in range(10):                      # standing: 26.5 V
        t += 0.5
        m.update(t, 26.5, 0.3)
    good = m.rest
    for _ in range(40):                      # driving at 3 A: 0.75 V sag, reads 0.82 A
        t += 0.5
        assert m.update(t, 26.5 + 0.3 * 0.25 - 3 * 0.25, 0.82) is None
    assert m.rest == pytest.approx(good) and m.held
    assert m.status(t)['held']
    m.update(t + 0.5, 26.5, 0.3)             # stopped again: updates
    assert not m.held


def test_a_long_drive_at_the_limit_uses_the_low_reading():
    """Held for longer than HOLD_MAX_S, the clipped (too low) voltage counts: the robot then
    goes home early instead of running flat."""
    level = pl.DEFAULTS['return_home_volts']
    m = pl.BatteryMonitor()
    t = 0.0
    m.update(t, level + 0.3, 0.3)
    events = []
    while t < pl.HOLD_MAX_S + pl.SMOOTH_S * 3 + pl.LOW_HOLD_S:
        t += 0.5
        events.append(m.update(t, level - 0.5, 0.82))
    assert 'low' in events and not m.held


def test_no_reading_means_unknown():
    m = pl.BatteryMonitor()
    assert m.status(0.0)['pct'] is None
    m.update(0.0, 26.5, 0.0)
    assert m.status(1.0)['pct'] is not None
    s = m.status(1.0 + pl.STALE_S + 1)
    assert s['pct'] is None and s['rest_volts'] is None and not s['low']
    assert s['return_home_volts'] == pl.DEFAULTS['return_home_volts']


def test_impossible_voltage_is_no_reading():
    m = pl.BatteryMonitor()
    for i in range(60):                      # 30 s of 0 V (VBS not wired)
        assert m.update(i * 0.5, 0.0, 6.0) is None
    assert m.status(30.0)['pct'] is None and not m.low


def test_parse_tamper():
    assert pl.parse_tamper('armed') == ('armed', '')
    assert pl.parse_tamper('tripped,lift') == ('tripped', 'lift')
    assert pl.parse_tamper('bogus') is None
    assert pl.tamper_text('tilt') == 'tipped over'
    assert pl.tamper_text('') == 'tampered with'


def test_guard_sends_home_once_per_period_while_low_and_moving():
    g = pl.Guard()
    states = {'laps': 'running', 'explore': None}
    assert g.commands(0.0, False, 'armed', states) == []
    assert g.commands(1.0, True, 'armed', states) == [('laps', 'home')]
    assert g.commands(2.0, True, 'armed', states) == []                 # not again at once
    assert g.commands(2.0, True, 'armed', {'laps': 'home'}) == []       # already going home
    assert g.commands(40.0, True, 'armed', states) == [('laps', 'home')]   # still out: again
    assert g.commands(80.0, True, 'armed', {'laps': 'idle'}) == []      # stopped: leave it
    assert g.commands(0.0, True, 'armed', {'explore': 'seek'}) == [('explore', 'home')]


def test_guard_stops_everything_on_tamper():
    g = pl.Guard()
    out = g.commands(0.0, False, 'tripped', {'laps': 'running', 'explore': None})
    assert out == [('laps', 'stop'), ('explore', 'stop')]
    assert g.commands(1.0, False, 'tripped', {'laps': 'idle', 'explore': None}) == []
    # something still moving 30 s later: stop again
    assert g.commands(31.0, False, 'tripped', {'laps': 'running'}) == [('laps', 'stop')]
    # tamper beats a low battery: stop, not home
    assert g.commands(70.0, True, 'tripped', {'laps': 'running'}) == [('laps', 'stop')]
    # cleared, then tripped again: stops at once
    assert g.commands(71.0, False, 'armed', {'laps': 'idle'}) == []
    assert g.commands(72.0, False, 'tripped', {'laps': 'running'}) == [('laps', 'stop')]
