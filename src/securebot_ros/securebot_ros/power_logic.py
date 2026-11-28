"""SecureBot power monitor and tamper logic (no ROS here, so it can be tested on its own).

Battery: the robot's 24 V 3.2 Ah lithium-ion pack (PB240A1, 7 cells in series), 27.8 V when
fully charged, measured by an INA226 on the ESP32's I2C bus. The firmware sends the pack
voltage and current (`B,<mV>,<mA>`); BatteryMonitor turns them into a resting voltage:

  * load compensation: under load the voltage sags by current x pack resistance, so
    V_rest = V + I * R;
  * current limit: the module's R100 shunt reads at most ~0.82 A, while driving takes more.
    Then V + I * R comes out too low, so while the current is at the limit the last good
    resting voltage is held. Held for longer than HOLD_MAX_S (a long drive), the clipped,
    too-low value is used after all: the robot then goes home early rather than running flat;
  * smoothing over SMOOTH_S, so a motor start does not swing the number.

Low battery is decided on that voltage: at or under the return-home voltage for LOW_HOLD_S
makes it "low" (once); it is "low" until the voltage is back REARM_V above the level
(charged). The percentage shown is for people: the lithium-ion cell curve, scaled so the
pack's full voltage is 100 %.

Settings (return-home voltage, full voltage, cells, pack resistance, current limit) live in
a small JSON file that the web panel writes and the guard node re-reads when it changes.
"""
import json
import os

SETTINGS_FILE = os.path.expanduser('~/securebot_ws/config/power.json')
DEFAULTS = {'return_home_volts': 25.9,   # 3.7 V a cell: about 18 % of the pack's charge left
            'full_volts': 27.8,          # the robot's pack after a full charge
            'cells': 7,
            'pack_ohms': 0.25,
            'max_amps': 0.8}             # current the INA226 can read (R100 shunt: 0.82 A)
CELL_CHOICES = (6, 7)

SMOOTH_S = 20.0          # s: time constant of the voltage smoothing
LOW_HOLD_S = 10.0        # s at or under the level before it counts as low
REARM_V = 0.3            # V above the level before "low" clears
HOLD_MAX_S = 60.0        # s at the current limit before the clipped reading is used anyway
STALE_S = 5.0            # s without a reading: battery unknown
MIN_CELL_V = 2.0         # V per cell: anything lower is not a battery reading (sensor/wiring)
MIN_CELL_LEVEL, MAX_CELL_LEVEL = 3.3, 4.0   # V per cell the return-home level may be set to

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


def pack_percent(rest_volts, settings):
    """% (0..100) of the pack's usable charge at this resting voltage: its full voltage is
    100 %."""
    cells = settings['cells']
    full = cell_percent(settings['full_volts'] / cells)
    if full <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * cell_percent(rest_volts / cells) / full))


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
        level = round(float(settings['return_home_volts']), 2)
        full = round(float(settings['full_volts']), 2)
        cells = int(settings['cells'])
        ohms = float(settings['pack_ohms'])
        max_amps = float(settings['max_amps'])
    except (KeyError, TypeError, ValueError):
        raise ValueError('Settings need return_home_volts, full_volts, cells, pack_ohms and '
                         'max_amps as numbers')
    if cells not in CELL_CHOICES:
        raise ValueError('The pack has 6 or 7 cells in series')
    if not 3.6 * cells <= full <= 4.25 * cells:
        raise ValueError(f'A full {cells}-cell pack reads {3.6 * cells:.1f}-{4.25 * cells:.1f} V')
    lo, hi = MIN_CELL_LEVEL * cells, min(MAX_CELL_LEVEL * cells, full - 0.5)
    if not lo <= level <= hi:
        raise ValueError(f'Return home at {lo:.1f}-{hi:.1f} V')
    if not 0.0 <= ohms <= 2.0:
        raise ValueError('Pack resistance must be 0-2 ohm')
    if not 0.1 <= max_amps <= 20.0:
        raise ValueError('Current limit must be 0.1-20 A')
    return {'return_home_volts': level, 'full_volts': full, 'cells': cells,
            'pack_ohms': ohms, 'max_amps': max_amps}


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
    """Readings in, smoothed resting voltage, percentage and the low-battery decision out."""

    def __init__(self, settings=None):
        self.settings = dict(settings or DEFAULTS)
        self.rest = None                  # smoothed resting voltage (V)
        self.volts = self.amps = None     # the last reading as measured
        self.last_t = None                # time of the last reading
        self.clip_since = None            # current at the limit since (None: below it)
        self.held = False                 # the last reading was held (current at the limit)
        self.under_since = None
        self.low = False

    def update(self, t, volts, amps):
        """A reading at time t (s). Returns 'low' the moment the battery becomes low, else
        None. A voltage no pack can have (VBS not wired, no battery) is ignored, so the battery
        shows as unknown instead of 0 V and nobody is sent home on a wiring fault."""
        s = self.settings
        if volts < MIN_CELL_V * s['cells']:
            return None
        fresh = self.rest is not None and self.last_t is not None and t - self.last_t <= STALE_S
        self.volts, self.amps = volts, amps
        estimate = volts + max(0.0, amps) * s['pack_ohms']
        if amps >= s['max_amps']:
            if self.clip_since is None:
                self.clip_since = t
            self.held = fresh and t - self.clip_since < HOLD_MAX_S
        else:
            self.clip_since, self.held = None, False
        if not fresh:
            self.rest = estimate
        elif not self.held:
            a = min(1.0, max(0.0, t - self.last_t) / SMOOTH_S)
            self.rest += a * (estimate - self.rest)
        self.last_t = t
        return self._judge(t)

    def _judge(self, t):
        level = self.settings['return_home_volts']
        if self.low:
            if self.rest >= level + REARM_V:
                self.low, self.under_since = False, None
            return None
        if self.rest > level:
            self.under_since = None
            return None
        if self.under_since is None:
            self.under_since = t
        if t - self.under_since >= LOW_HOLD_S:
            self.low = True
            return 'low'
        return None

    @property
    def pct(self):
        return None if self.rest is None else pack_percent(self.rest, self.settings)

    def status(self, t):
        """For /power/status: None fields when no recent reading (no sensor)."""
        s = self.settings
        fresh = self.last_t is not None and t - self.last_t <= STALE_S

        def r(v, n=2):
            return round(v, n) if fresh else None
        return {'pct': r(self.pct, 1) if fresh else None,
                'volts': r(self.volts), 'amps': r(self.amps),
                'rest_volts': r(self.rest) if fresh else None,
                'held': self.held and fresh,
                'low': self.low and fresh,
                'return_home_volts': s['return_home_volts'],
                'return_home_pct': round(pack_percent(s['return_home_volts'], s)),
                'full_volts': s['full_volts'],
                'cells': s['cells']}


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
