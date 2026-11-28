"""SecureBot web panel, alerts: tampering, unauthorized faces and a low battery, each with its time
and (when the camera gave one) a webcam snapshot.

Each alert is two files in ALERTS_DIR: <id>.json (time, kind, reason, text, acknowledged) and,
when the camera gave a picture, <id>.jpg. The newest KEEP alerts are kept.

Other receivers (the pager, later) subscribe with AlertStore.hooks.append(fn): fn(alert) is
called once per new alert, after its files are written, and must not block for long (run
slow work in a thread). A failing hook is reported in the alert's 'hook_errors' and does not
stop the others.
"""
import json
import os
import re
import threading
import time

ALERTS_DIR = os.path.expanduser('~/securebot_ws/alerts')
KEEP = 100
ID_RE = re.compile(r'^\d{8}_\d{6}_\d{3}$')


class AlertStore:
    def __init__(self, directory=ALERTS_DIR, clock=time.time, keep=KEEP):
        self.dir, self.clock, self.keep = directory, clock, keep
        self.lock = threading.Lock()
        self.hooks = []

    def _path(self, alert_id, ext):
        if not ID_RE.match(alert_id or ''):
            raise ValueError('No such alert')
        return os.path.join(self.dir, f'{alert_id}.{ext}')

    def add(self, kind, reason, text, jpeg=None):
        """Store a new alert (and its snapshot); returns it."""
        t = self.clock()
        stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(t)) + f'_{int(t * 1000) % 1000:03d}'
        alert = {'id': stamp, 'time': t,
                 'time_text': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t)),
                 'kind': kind, 'reason': reason, 'text': text,
                 'snapshot': bool(jpeg), 'acked': False}
        with self.lock:
            os.makedirs(self.dir, exist_ok=True)
            if jpeg:
                with open(self._path(stamp, 'jpg'), 'wb') as f:
                    f.write(jpeg)
            self._write(alert)
            self._prune()
        errors = []
        for hook in list(self.hooks):
            try:
                hook(dict(alert))
            except Exception as e:          # one broken receiver must not lose the alert
                errors.append(f'{getattr(hook, "__name__", "hook")}: {e}')
        if errors:
            alert['hook_errors'] = errors
            with self.lock:
                self._write(alert)
        return alert

    def _write(self, alert):
        path = self._path(alert['id'], 'json')
        with open(path + '.tmp', 'w') as f:
            json.dump(alert, f)
        os.replace(path + '.tmp', path)

    def _ids(self):
        try:
            names = os.listdir(self.dir)
        except OSError:
            return []
        return sorted((n[:-5] for n in names if n.endswith('.json') and ID_RE.match(n[:-5])),
                      reverse=True)

    def _prune(self):
        for old in self._ids()[self.keep:]:
            for ext in ('json', 'jpg'):
                try:
                    os.remove(self._path(old, ext))
                except OSError:
                    pass

    def list(self, limit=20):
        out = []
        with self.lock:
            for alert_id in self._ids()[:limit]:
                try:
                    with open(self._path(alert_id, 'json')) as f:
                        out.append(json.load(f))
                except (OSError, ValueError):
                    continue
        return out

    def unacked(self):
        return [a for a in self.list(self.keep) if not a.get('acked')]

    def ack(self, alert_id=None):
        """Acknowledge one alert, or all of them (alert_id None). Returns how many."""
        n = 0
        with self.lock:
            ids = [alert_id] if alert_id else self._ids()
            for i in ids:
                path = self._path(i, 'json')         # ValueError for a malformed id
                try:
                    with open(path) as f:
                        alert = json.load(f)
                except OSError:
                    if alert_id:
                        raise ValueError('No such alert')
                    continue
                except ValueError:
                    continue
                if not alert.get('acked'):
                    alert['acked'] = True
                    self._write(alert)
                    n += 1
        return n

    def snapshot(self, alert_id):
        try:
            with open(self._path(alert_id, 'jpg'), 'rb') as f:
                return f.read()
        except OSError:
            raise ValueError('No snapshot for this alert')
