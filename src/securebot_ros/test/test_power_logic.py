"""Battery percentage, the low-battery decision and the guard's commands (power_logic.py)."""
import json

import pytest

from securebot_ros import power_logic as pl


def test_cell_curve_ends_and_middle():
    assert pl.cell_percent(4.25) == 100.0
    assert pl.cell_percent(4.20) == 100.0
    assert pl.cell_percent(3.0) == 0.0
    assert pl.cell_percent(3.84) == pytest.approx(50.0)
    assert 20.0 < pl.cell_percent(3.74) < 25.0
    values = [pl.cell_percent(v / 100) for v in range(320, 425)]
    assert values == sorted(values)                       # never goes down as voltage rises


def test_pack_percent_6s_and_7s_and_load():
    assert pl.pack_percent(25.2, 0.0, 6, 0.25) == 100.0   # 6S full
    assert pl.pack_percent(25.2, 0.0, 7, 0.25) < 10.0     # the same voltage is nearly empty 7S
    assert pl.pack_percent(29.4, 0.0, 7, 0.25) == 100.0
    rest = pl.pack_percent(23.04, 0.0, 6, 0.25)           # 3.84 V per cell: 50 %
    sagging = pl.pack_percent(22.54, 2.0, 6, 0.25)        # 0.5 V lower while 2 A flow
    assert sagging == pytest.approx(rest)
    assert pl.pack_percent(23.04, -1.0, 6, 0.25) == pytest.approx(rest)   # charging: no lift


def test_settings_defaults_validation_and_round_trip(tmp_path):
    path = str(tmp_path / 'config' / 'power.json')
    assert pl.load_settings(path) == pl.DEFAULTS
    saved = pl.save_settings({'return_home_pct': '25', 'cells': 7, 'pack_ohms': 0.3}, path)
    assert saved == {'return_home_pct': 25, 'cells': 7, 'pack_ohms': 0.3}
    assert pl.load_settings(path) == saved
    for bad in ({'return_home_pct': 2, 'cells': 6, 'pack_ohms': 0.2},
                {'return_home_pct': 20, 'cells': 8, 'pack_ohms': 0.2},
                {'return_home_pct': 20, 'cells': 6, 'pack_ohms': -1},
                {'return_home_pct': 'x', 'cells': 6, 'pack_ohms': 0.2}):
        with pytest.raises(ValueError):
            pl.save_settings(bad, path)
    with open(path, 'w') as f:
        f.write('{broken')
    assert pl.load_settings(path) == pl.DEFAULTS
    with open(path, 'w') as f:
        json.dump({'return_home_pct': 999}, f)
    assert pl.load_settings(path) == pl.DEFAULTS


def volts_for(pct, cells=pl.DEFAULTS["cells"]):
    """Resting pack voltage at about `pct` (inverse of the curve, by search)."""
    lo, hi = 3.0 * cells, 4.25 * cells
    for _ in range(60):
        mid = (lo + hi) / 2
        if pl.pack_percent(mid, 0, cells, 0) < pct:
            lo = mid
        else:
            hi = mid
    return hi


def test_low_only_after_holding_under_the_level():
    m = pl.BatteryMonitor()
    assert m.update(0.0, volts_for(60), 0.0) is None
    assert m.status(0.0)['pct'] == pytest.approx(60, abs=0.5)
    t = 0.0
    # a motor start sags the voltage for a moment: not low
    m2 = pl.BatteryMonitor({**pl.DEFAULTS, 'pack_ohms': 0.0})
    m2.update(0.0, volts_for(25), 0.0)
    m2.update(0.5, volts_for(5), 6.0)
    assert not m2.low
    # really under the level: low after LOW_HOLD_S, once
    m = pl.BatteryMonitor()
    m.update(0.0, volts_for(19), 0.0)
    events = []
    for i in range(1, 30):
        t = i * 0.5
        events.append(m.update(t, volts_for(19), 0.0))
    assert events.count('low') == 1
    assert events.index('low') * 0.5 + 0.5 >= pl.LOW_HOLD_S
    assert m.status(t)['low']


def test_low_clears_only_when_charged_above_the_margin():
    m = pl.BatteryMonitor()
    t = 0.0
    while not m.low:
        t += 0.5
        m.update(t, volts_for(15), 0.0)
    for _ in range(200):                     # back to 22 %: still low (no flapping)
        t += 0.5
        m.update(t, volts_for(22), 0.0)
    assert m.low
    for _ in range(200):                     # charged to 40 %
        t += 0.5
        m.update(t, volts_for(40), 0.0)
    assert not m.low


def test_no_reading_means_unknown():
    m = pl.BatteryMonitor()
    assert m.status(0.0)['pct'] is None
    m.update(0.0, volts_for(50), 0.0)
    assert m.status(1.0)['pct'] is not None
    s = m.status(1.0 + pl.STALE_S + 1)
    assert s['pct'] is None and not s['low'] and s['return_home_pct'] == 20


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
