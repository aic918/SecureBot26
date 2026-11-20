"""
SecureBot web panel, Faces card: enroll people for face recognition from the browser.

The same steps as by hand, as buttons:

  Capture   capture_faces in the vision venv saves sharp, new-angle face photos of one
            person from the robot's camera into a staging folder (not yet enrolled),
            with a live preview and progress for the page
  Review    review_faces checks them (face_logic.review_photos): photos that could be
            mistaken for someone else, that are unlike the rest (side views, heavy blur,
            someone else) or that don't match the person's enrollment come back unticked
  Keep      the ticked photos move to authorized_faces/<name>/, the rest to captures/
  Rebuild   enroll_faces --backup rebuilds the database (the old one is kept) and writes
            a report: photos per person and who could be confused with whom

The face libraries live in the vision venv (~/yolo-env), not the system Python the panel
runs in, so each step is a child process of that venv; nothing here imports them. Only
one step runs at a time, and capture refuses to start while the live face node has the
camera. The live node reads the database when it starts: restart it to use a new one.

Anyone who can open the panel can enroll themselves as authorized: keep the panel on a
trusted network or start it with --host 127.0.0.1.
"""
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time

FACES_BASE = os.path.expanduser('~/yolo/securebot_face_id')
VENV = os.path.expanduser('~/yolo-env')
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png')
NAME_RE = re.compile(r'^[A-Za-z][A-Za-z0-9_-]{0,31}$')
DEFAULT_COUNT = 15


def venv_command(venv, module, *args):
    """Run securebot_ros.<module> with the venv's Python."""
    return [os.path.join(venv, 'bin', 'python'), '-W', 'ignore', '-m',
            f'securebot_ros.{module}', *[str(a) for a in args]]


def venv_env(venv, env=None):
    """
    Return the environment for the venv, with its libcudss folder first on LD_LIBRARY_PATH.

    Only that folder: the venv's pip cuBLAS breaks torch on the Jetson (see
    securebot_live.launch.py).
    """
    env = dict(os.environ if env is None else env)
    dirs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(
        venv, 'lib', 'python3*', 'site-packages', 'nvidia', '*', 'lib', 'libcudss.so*'))})
    env['LD_LIBRARY_PATH'] = os.pathsep.join(
        [d for d in dirs + [env.get('LD_LIBRARY_PATH', '')] if d])
    return env


def photos_in(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder)
                  if f.lower().endswith(IMAGE_SUFFIXES) and not f.startswith('.'))


def person_folders(faces_dir):
    if not os.path.isdir(faces_dir):
        return []
    return sorted((d for d in os.listdir(faces_dir)
                   if os.path.isdir(os.path.join(faces_dir, d)) and not d.startswith('.')),
                  key=str.lower)


def canonical_name(name, faces_dir):
    """Return a safe folder name; an existing person's spelling wins ('balde' -> 'Balde')."""
    name = (name or '').strip()
    if not NAME_RE.match(name):
        raise ValueError('Use a name of letters, digits, - or _ (up to 32), starting with '
                         'a letter')
    return next((d for d in person_folders(faces_dir) if d.lower() == name.lower()), name)


def camera_in_use(ps_output):
    """Return whether `ps -eo args` shows the live face node or a capture using the camera."""
    return any('securebot_live_node' in line or 'securebot_ros.capture_faces' in line
               for line in ps_output.splitlines())


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class FaceEnroller:
    """The Faces card's state and actions; one child process (step) at a time."""

    def __init__(self, base=FACES_BASE, venv=VENV, popen=subprocess.Popen, killpg=os.killpg,
                 ps=None, clock=time.time):
        self.base, self.venv = base, venv
        self.faces_dir = os.path.join(base, 'authorized_faces')
        self.staging = os.path.join(base, 'staging')
        self.captures = os.path.join(base, 'captures')
        self.db = os.path.join(base, 'encodings', 'authorized_faces.pkl')
        self.report_file = self.db + '.report.json'
        self.preview_file = os.path.join(self.staging, '.preview.jpg')
        self.status_file = os.path.join(self.staging, '.capture.json')
        self.popen, self.killpg, self.clock = popen, killpg, clock
        self.ps = ps or (lambda: subprocess.run(['ps', '-eo', 'args'], capture_output=True,
                                                text=True).stdout)
        self.lock = threading.Lock()
        self.job = None            # {'kind', 'name', 'proc', 'log'}
        self.message, self.error = '', False
        self.review = self._waiting_review()

    # helpers -------------------------------------------------------------------------------
    def _staged(self, name):
        return os.path.join(self.staging, name)

    def _review_file(self, name):
        return os.path.join(self._staged(name), '.review.json')

    def _waiting_review(self):
        """Return a review left from before a panel restart (photos still in staging)."""
        for name in person_folders(self.staging):
            review = read_json(self._review_file(name))
            if review and photos_in(self._staged(name)):
                return review
        return None

    def _say(self, text, error=False):
        self.message, self.error = text, error

    def _start(self, kind, name, cmd):
        os.makedirs(self.staging, exist_ok=True)
        log = open(os.path.join(self.staging, f'.{kind}.log'), 'w')
        try:
            proc = self.popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL, env=venv_env(self.venv),
                              start_new_session=True)
        except OSError as e:
            log.close()
            raise RuntimeError(f'Cannot start {kind}: {e} (is the vision venv at {self.venv}?)')
        self.job = {'kind': kind, 'name': name, 'proc': proc, 'log': log}

    def _busy(self):
        if self.job is not None:
            raise RuntimeError(f'Busy: {self.job["kind"]} for {self.job["name"]} is running')

    def _log_tail(self, kind, n=6):
        try:
            with open(os.path.join(self.staging, f'.{kind}.log'), errors='replace') as f:
                return [ln.rstrip() for ln in f.readlines()[-n:] if 'NvMap' not in ln]
        except OSError:
            return []

    # actions -------------------------------------------------------------------------------
    def start_capture(self, name, count=DEFAULT_COUNT):
        with self.lock:
            self._poll()
            self._busy()
            name = canonical_name(name, self.faces_dir)
            count = int(count)
            if not 3 <= count <= 40:
                raise ValueError('Take between 3 and 40 photos')
            if self.review is not None:
                raise RuntimeError(f'First keep or discard the photos of {self.review["name"]} '
                                   'waiting below')
            if camera_in_use(self.ps()):
                raise RuntimeError('The camera is in use (the live face node or another '
                                   'capture): stop it first')
            staged = self._staged(name)
            if photos_in(staged):
                raise RuntimeError(f'Photos of {name} are already waiting in {staged}')
            os.makedirs(staged, exist_ok=True)
            for f in (self.preview_file, self.status_file):
                if os.path.exists(f):
                    os.remove(f)
            self._start('capture', name, venv_command(
                self.venv, 'capture_faces', '--name', name, '--out-dir', staged,
                '--count', count, '--no-window', '--preview', self.preview_file,
                '--status', self.status_file, '--encodings', self.db))
            self._say(f'Capturing {name}: look at the camera, then slowly turn your head')
            return self.message

    def stop_capture(self):
        with self.lock:
            if self.job is None or self.job['kind'] != 'capture':
                raise RuntimeError('No capture is running')
            try:
                self.killpg(self.job['proc'].pid, signal.SIGINT)   # releases the camera
            except ProcessLookupError:
                pass
            return 'Stopping the capture'

    def keep(self, name, files):
        """Move the ticked photos into the person's folder and the rest to captures/."""
        with self.lock:
            self._poll()
            if self.review is None or self.review['name'] != name:
                raise RuntimeError('No photos of that person are waiting')
            staged = self._staged(name)
            waiting = photos_in(staged)
            keep = {os.path.basename(f) for f in files}
            unknown = keep - set(waiting)
            if unknown:
                raise ValueError(f'Not a waiting photo: {sorted(unknown)[0]}')
            dest = os.path.join(self.faces_dir, name)
            os.makedirs(dest, exist_ok=True)
            put_aside = self._put_aside(name, 'removed')
            for f in waiting:
                target = os.path.join(dest if f in keep else put_aside, f)
                shutil.move(os.path.join(staged, f), target)
                if f in keep:
                    os.utime(target)      # newer than the database: it needs rebuilding
            self._clear_staging(name)
            self._say(f'Added {len(keep)} photos to {name}; {len(waiting) - len(keep)} put '
                      f'aside. Rebuild the database to use them')
            return self.message

    def discard(self, name):
        with self.lock:
            self._poll()
            if self.review is None or self.review['name'] != name:
                raise RuntimeError('No photos of that person are waiting')
            put_aside = self._put_aside(name, 'discarded')
            for f in photos_in(self._staged(name)):
                shutil.move(os.path.join(self._staged(name), f), os.path.join(put_aside, f))
            self._clear_staging(name)
            self._say(f'Discarded the photos of {name} (kept in {put_aside})')
            return self.message

    def rebuild(self):
        with self.lock:
            self._poll()
            self._busy()
            if not any(photos_in(os.path.join(self.faces_dir, p))
                       for p in person_folders(self.faces_dir)):
                raise RuntimeError('Nobody has photos yet')
            self._start('rebuild', 'everyone', venv_command(
                self.venv, 'enroll_faces', '--backup', '--faces-dir', self.faces_dir,
                '--output', self.db))
            self._say('Rebuilding the face database (about 1-2 s per photo)')
            return self.message

    def shutdown(self):
        """Stop a running step (the panel is closing)."""
        with self.lock:
            if self.job is not None:
                try:
                    self.killpg(self.job['proc'].pid, signal.SIGINT)
                except ProcessLookupError:
                    pass

    def _put_aside(self, name, why):
        d = os.path.join(self.captures, f'{time.strftime("%Y-%m-%d_%H%M%S")}_{name}_{why}')
        os.makedirs(d, exist_ok=True)
        return d

    def _clear_staging(self, name):
        shutil.rmtree(self._staged(name), ignore_errors=True)
        self.review = None

    # progress ------------------------------------------------------------------------------
    def _poll(self):
        """Notice a step that finished and move on (capture -> review -> waiting for you)."""
        job = self.job
        if job is None or job['proc'].poll() is None:
            return
        code = job['proc'].returncode
        job['log'].close()
        self.job = None
        name = job['name']
        if job['kind'] == 'capture':
            status = read_json(self.status_file) or {}
            n = len(photos_in(self._staged(name)))
            if n == 0:
                shutil.rmtree(self._staged(name), ignore_errors=True)
                self._say(status.get('error') or f'No photos of {name} were taken', True)
                return
            self._start('review', name, venv_command(
                self.venv, 'review_faces', '--name', name, '--dir', self._staged(name),
                '--out', self._review_file(name), '--encodings', self.db))
            self._say(f'Took {n} photos of {name}; checking them')
        elif job['kind'] == 'review':
            review = read_json(self._review_file(name))
            if code != 0 or review is None:
                n = len(photos_in(self._staged(name)))
                review = {'name': name, 'photos': [
                    {'file': f, 'keep': True, 'reasons': [], 'notes': ['not checked'],
                     'closest': None, 'rest': None, 'own': None}
                    for f in photos_in(self._staged(name))]}
                self._say(f'The automatic check failed (see the log); look at the {n} photos '
                          'yourself', True)
            else:
                kept = sum(p['keep'] for p in review['photos'])
                self._say(f'{kept} of {len(review["photos"])} photos of {name} look good: '
                          'check them below')
            self.review = review
        elif job['kind'] == 'rebuild':
            if code == 0:
                self._say('Face database rebuilt. Restart the live face node to use it')
            else:
                self._say(f'Rebuilding failed (exit code {code}); see the log', True)

    def state(self):
        with self.lock:
            self._poll()
            job = self.job
            capture = read_json(self.status_file) if job and job['kind'] == 'capture' else None
            report = read_json(self.report_file)
            built = report.get('time') if report else None
            enrolled = {p['name']: p['photos'] for p in (report or {}).get('people', [])}
            people, newest = [], 0.0
            for name in person_folders(self.faces_dir):
                folder = os.path.join(self.faces_dir, name)
                files = photos_in(folder)
                for f in files:
                    newest = max(newest, os.path.getmtime(os.path.join(folder, f)))
                people.append({'name': name, 'photos': len(files),
                               'enrolled': enrolled.get(name, 0)})
            outdated = built is None or newest > built or \
                set(enrolled) != {p['name'] for p in people if p['photos']}
            return {
                'busy': job['kind'] if job else None,
                'name': job['name'] if job else None,
                'message': self.message, 'error': self.error,
                'capture': capture,
                'review': self.review,
                'people': people,
                'database': {'built': built, 'outdated': bool(people) and outdated,
                             'total': (report or {}).get('total'),
                             'pairs': (report or {}).get('pairs', []),
                             'backup': (report or {}).get('backup')},
                'log': self._log_tail(job['kind']) if job else [],
            }

    # files ---------------------------------------------------------------------------------
    def preview(self):
        try:
            with open(self.preview_file, 'rb') as f:
                return f.read()
        except OSError:
            return None

    def photo(self, kind, name, file):
        """Return a waiting (kind 'staged') or enrolled photo's bytes; ValueError if none."""
        folder = {'staged': self.staging, 'enrolled': self.faces_dir}.get(kind)
        if folder is None or name not in person_folders(folder):    # no '..', no paths
            raise ValueError('No such photo')
        file = os.path.basename(file or '')
        if file not in photos_in(os.path.join(folder, name)):
            raise ValueError('No such photo')
        with open(os.path.join(folder, name, file), 'rb') as f:
            return f.read()
