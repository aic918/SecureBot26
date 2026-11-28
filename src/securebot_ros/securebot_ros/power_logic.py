"""SecureBot power monitor and tamper logic (no ROS here, so it can be tested on its own).

Battery: a 24 V 3.2 Ah lithium-ion pack (the robot's is 7S, 29.4 V full, the default; 6S
packs, 25.2 V full, are also sold as "24 V") measured by an INA226 on the ESP32's I2C bus.
The firmware sends the pack voltage and current (`B,<mV>,<mA>`); BatteryMonitor turns them
into a percentage:

  * load compensation: under load the voltage sags by current x pack resistance, so
    V_rest = V + I * R before looking the voltage up (otherwise driving reads ~10 % lower
    than standing still);
  * the per-cell resting-voltage curve of a lithium-ion (NMC) cell;
  * smoothing over SMOOTH_S, so a motor start does not swing the number.

Low battery: the percentage at or under the return-home level for LOW_HOLD_S makes it "low"
(once); it is "low" until the percentage is back REARM_MARGIN above the level (charged).

Settings (return-home level, cells, pack resistance) live in a small JSON file that the web
panel writes and the guard node re-reads when it changes.
"""
import json
import os

SETTINGS_FILE = os.path.expanduser('~/securebot_ws/config/power.json')
DEFAULTS = {'return_home_pct': 20, 'cells': 7, 'pack_ohms': 0.25}   # robot's pack: 7S
MIN_RETURN_PCT, MAX_RETURN_PCT = 5, 80
CELL_CHOICES = (6, 7)

SMOOTH_S = 20.0          # s: time constant of the percentage smoothing
LOW_HOLD_S = 10.0        # s at or under the level before it counts as low
REARM_MARGIN = 5.0       # % above the level before "low" clears
STALE_S = 5.0            # s without a reading: battery unknown
MIN_CELL_V = 2.0         # V per cell: anything lower is not a battery reading (sensor/wiring)

# resting voltage of one lithium-ion (NMC) cell -> % charge
CELL_CURVE = [(3.27, 0), (3.61, 5), (3.69, 10), (3.71, 15), (3.73, 20), (3.75, 25),
              (3.77, 30), (3.79, 35), (3.80, 40), (3.82, 45), (3.84, 50), (3.85, 55),
              (3.87, 60), (3.91, 65), (3.95, 70), (3.98, 75), (4.02, 80), (4.08, 85),
              (4.11, 90), (4.15, 95), (4.20, 100)]


def cell_percent(volts):
    """% charge (0..100) of one lithium-ion cell at rest at `volts`."""
    if volts <= CELL_CURVE[0][0]:
        return 0.0
    for (v0, p0), (v1, p1) in zip(CELL_CURVE, CELL_CURVE[1:]):
        if volts <= v1:
            return p0 + (p1 - p0) * (volts - v0) / (v1 - v0)
    return 100.0


def pack_percent(volts, amps, cells, pack_ohms):
    """% charge of a `cells`-series pack reading `volts` while `amps` flow out of it."""
    return cell_percent((volts + max(0.0, amps) * pack_ohms) / cells)


def load_settings(path=SETTINGS_FILE):
    """The power settings, defaults for anything missing or broken."""
    settings = dict(DEFAULTS)
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            settings.update({k: data[k] for k in DEFAULTS if k in data})
    except (OSError, ValueError):
        pass
    try:
        return check_settings(settings)
    except ValueError:
        return dict(DEFAULTS)


def check_settings(settings):
    """Validated copy of settings, or ValueError saying what is wrong."""
    try:
        pct = int(settings['return_home_pct'])
        cells = int(settings['cells'])
        ohms = float(settings['pack_ohms'])
    except (KeyError, TypeError, ValueError):
        raise ValueError('Settings need return_home_pct, cells and pack_ohms as numbers')
    if not MIN_RETURN_PCT <= pct <= MAX_RETURN_PCT:
        raise ValueError(f'Return home at {MIN_RETURN_PCT}-{MAX_RETURN_PCT} %')
    if cells not in CELL_CHOICES:
        raise ValueError('The pack has 6 or 7 cells in series (25.2 V or 29.4 V full)')
    if not 0.0 <= ohms <= 2.0:
        raise ValueError('Pack resistance must be 0-2 ohm')
    return {'return_home_pct': pct, 'cells': cells, 'pack_ohms': ohms}


def save_settings(settings, path=SETTINGS_FILE):
    settings = check_settings(settings)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(settings, f, indent=2)
        f.write('\n')
    os.replace(tmp, path)
    return settings


class BatteryMonitor:
    """Readings in, smoothed percentage and the low-battery decision out."""

    def __init__(self, settings=None):
        self.settings = dict(settings or DEFAULTS)
        self.pct = None
        self.volts = self.amps = None
        self.last_t = None
        self.under_since = None
        self.low = False

    def update(self, t, volts, amps):
        """A reading at time t (s). Returns 'low' the moment the battery becomes low, else
        None. A voltage no pack can have (VBS not wired, no battery) is ignored, so the battery
        shows as unknown instead of 0 % and nobody is sent home on a wiring fault."""
        s = self.settings
        if volts < MIN_CELL_V * s['cells']:
            return None
        raw = pack_percent(volts, amps, s['cells'], s['pack_ohms'])
        if self.pct is None or self.last_t is None or t - self.last_t > STALE_S:
            self.pct = raw
        else:
            a = min(1.0, max(0.0, t - self.last_t) / SMOOTH_S)
            self.pct += a * (raw - self.pct)
        self.volts, self.amps, self.last_t = volts, amps, t
        level = s['return_home_pct']
        if self.low:
            if self.pct >= level + REARM_MARGIN:
                self.low, self.under_since = False, None
            return None
        if self.pct > level:
            self.under_since = None
            return None
        if self.under_since is None:
            self.under_since = t
        if t - self.under_since >= LOW_HOLD_S:
            self.low = True
            return 'low'
        return None

    def status(self, t):
        """For /power/status: None fields when no recent reading (no sensor)."""
        fresh = self.last_t is not None and t - self.last_t <= STALE_S
        return {'pct': round(self.pct, 1) if fresh else None,
                'volts': round(self.volts, 2) if fresh else None,
                'amps': round(self.amps, 2) if fresh else None,
                'low': self.low and fresh,
                'return_home_pct': self.settings['return_home_pct'],
                'cells': self.settings['cells']}


# ------------------------------------------------------------------------------ tamper
TAMPER_STATES = ('off', 'armed', 'tripped', 'unavailable')
TAMPER_REASONS = {'tilt': 'tipped over', 'lift': 'lifted'}


def parse_tamper(text):
    """The firmware's T line without "T," ('armed', 'tripped,tilt', ...) as
    (state, reason) or None."""
    parts = text.strip().split(',')
    if parts[0] not in TAMPER_STATES:
        return None
    return parts[0], (parts[1] if len(parts) > 1 else '')


def tamper_text(reason):
    return TAMPER_REASONS.get(reason, reason or 'tampered with')


# ------------------------------------------------------------------------------ guard
RESEND_S = 30.0          # s between repeated home / stop commands while the cause lasts
LAPS_MOVING = ('running',)                 # laps 'home' = already heading home
EXPLORE_IDLE = ('idle', 'done', 'error', 'home')


def moving(kind, state):
    """Whether a laps / explore status state means the robot is out driving its job (not
    already going home, not stopped)."""
    if state is None:
        return False
    if kind == 'laps':
        return state in LAPS_MOVING
    return state not in EXPLORE_IDLE


class Guard:
    """What the guard node sends: 'home' while the battery is low and the robot is out on
    its job, 'stop' when tampered with (and again while it is tripped and something moves)."""

    def __init__(self):
        self.last_sent = {}           # (kind, cmd) -> time

    def commands(self, t, low, tamper_state, states):
        """states: {'laps': state or None, 'explore': state or None}. Returns [(kind, cmd)]."""
        out = []
        for kind, state in states.items():
            cmd = None
            if tamper_state == 'tripped':
                if moving(kind, state) or state == 'home' or (kind, 'stop') not in self.last_sent:
                    cmd = 'stop'
            elif low and moving(kind, state):
                cmd = 'home'
            if cmd is None:
                continue
            last = self.last_sent.get((kind, cmd))
            if last is None or t - last >= RESEND_S:
                self.last_sent[(kind, cmd)] = t
                out.append((kind, cmd))
        if tamper_state != 'tripped':
            for k in [k for k in self.last_sent if k[1] == 'stop']:
                del self.last_sent[k]
        return out
