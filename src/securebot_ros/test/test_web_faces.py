"""The web panel's Faces card without a camera, GPU or face libraries: capture, review,
keep / discard and rebuild as fake child processes over a temporary faces folder."""
import json
import os
import signal

import pytest

from securebot_ros import web_faces as wf


class FakeProc:
    count = 0

    def __init__(self, cmd, **kw):
        FakeProc.count += 1
        self.pid, self.cmd, self.kw, self.returncode = 5000 + FakeProc.count, cmd, kw, None

    def poll(self):
        return self.returncode


class Fake:
    """popen / killpg / ps for FaceEnroller."""

    def __init__(self, ps=''):
        self.procs, self.signals, self.ps_out = [], [], ps

    def popen(self, cmd, **kw):
        self.procs.append(FakeProc(cmd, **kw))
        return self.procs[-1]

    def killpg(self, pid, sig):
        self.signals.append((pid, sig))

    def ps(self):
        return self.ps_out


@pytest.fixture
def setup(tmp_path):
    fake = Fake()
    base = tmp_path / 'faces'
    (base / 'authorized_faces' / 'Balde').mkdir(parents=True)
    (base / 'authorized_faces' / 'Balde' / 'a.jpg').write_bytes(b'jpg')
    fe = wf.FaceEnroller(base=str(base), venv=str(tmp_path / 'venv'), popen=fake.popen,
                         killpg=fake.killpg, ps=fake.ps)
    return fe, fake, base


def stage(base, name, files):
    d = base / 'staging' / name
    d.mkdir(parents=True, exist_ok=True)
    for f in files:
        (d / f).write_bytes(b'jpg')
    return d


def finish(fe, code=0):
    fe.job['proc'].returncode = code
    return fe.state()


def review_json(d, name, verdicts):
    (d / '.review.json').write_text(json.dumps({'name': name, 'photos': [
        {'file': f, 'keep': k, 'reasons': [] if k else ['why'], 'notes': []}
        for f, k in verdicts]}))


def test_names_are_checked_and_match_existing_people(setup):
    fe, _, _ = setup
    assert wf.canonical_name('balde', fe.faces_dir) == 'Balde'
    assert wf.canonical_name(' New_Person-2 ', fe.faces_dir) == 'New_Person-2'
    for bad in ('', '../x', 'a/b', '1abc', 'x' * 40, 'a b'):
        with pytest.raises(ValueError):
            wf.canonical_name(bad, fe.faces_dir)


def test_capture_runs_in_the_venv_into_staging(setup, tmp_path):
    fe, fake, base = setup
    fe.start_capture('balde', 12)
    cmd = fake.procs[0].cmd
    assert cmd[0] == str(tmp_path / 'venv' / 'bin' / 'python')
    assert cmd[cmd.index('-m') + 1] == 'securebot_ros.capture_faces'
    assert cmd[cmd.index('--out-dir') + 1] == str(base / 'staging' / 'Balde')
    assert cmd[cmd.index('--count') + 1] == '12' and '--no-window' in cmd
    assert fake.procs[0].kw['start_new_session']
    assert fe.state()['busy'] == 'capture'


def test_capture_refuses_while_the_camera_is_in_use(setup):
    fe, fake, _ = setup
    fake.ps_out = 'python -m securebot_ros.securebot_live_node --ros-args\n'
    with pytest.raises(RuntimeError, match='camera is in use'):
        fe.start_capture('Dan')
    with pytest.raises(ValueError):
        fe.start_capture('Dan', 2)


def test_stop_sends_ctrl_c_so_the_camera_is_released(setup):
    fe, fake, _ = setup
    with pytest.raises(RuntimeError):
        fe.stop_capture()
    fe.start_capture('Dan')
    fe.stop_capture()
    assert fake.signals == [(fake.procs[0].pid, signal.SIGINT)]


def test_capture_then_review_then_keep(setup):
    fe, fake, base = setup
    fe.start_capture('Dan')
    d = stage(base, 'Dan', ['1.jpg', '2.jpg', '3.jpg'])
    s = finish(fe)
    assert s['busy'] == 'review' and 'securebot_ros.review_faces' in fake.procs[1].cmd
    with pytest.raises(RuntimeError, match='Busy'):
        fe.rebuild()
    review_json(d, 'Dan', [('1.jpg', True), ('2.jpg', True), ('3.jpg', False)])
    s = finish(fe)
    assert s['busy'] is None and [p['keep'] for p in s['review']['photos']] == \
        [True, True, False]
    with pytest.raises(RuntimeError, match='keep or discard'):
        fe.start_capture('Other')
    with pytest.raises(ValueError):
        fe.keep('Dan', ['nope.jpg'])
    fe.keep('Dan', ['1.jpg', '2.jpg'])
    assert sorted(os.listdir(base / 'authorized_faces' / 'Dan')) == ['1.jpg', '2.jpg']
    aside = [d for d in os.listdir(base / 'captures') if d.endswith('_Dan_removed')]
    assert os.listdir(base / 'captures' / aside[0]) == ['3.jpg']
    assert not (base / 'staging' / 'Dan').exists()
    s = fe.state()
    assert s['review'] is None and s['database']['outdated']
    assert {p['name']: p['photos'] for p in s['people']} == {'Balde': 1, 'Dan': 2}


def test_discard_puts_every_photo_aside(setup):
    fe, _, base = setup
    fe.start_capture('Dan')
    d = stage(base, 'Dan', ['1.jpg'])
    finish(fe)
    review_json(d, 'Dan', [('1.jpg', True)])
    finish(fe)
    fe.discard('Dan')
    assert not (base / 'authorized_faces' / 'Dan').exists()
    assert any(x.endswith('_Dan_discarded') for x in os.listdir(base / 'captures'))


def test_a_capture_without_photos_reports_why(setup):
    fe, _, base = setup
    fe.start_capture('Dan')
    (base / 'staging' / '.capture.json').write_text(json.dumps(
        {'saved': 0, 'count': 15, 'done': True, 'error': 'Could not open camera 0'}))
    s = finish(fe, 1)
    assert s['busy'] is None and s['error'] and 'Could not open camera' in s['message']


def test_a_failed_check_still_lets_you_choose(setup):
    fe, _, base = setup
    fe.start_capture('Dan')
    stage(base, 'Dan', ['1.jpg', '2.jpg'])
    finish(fe)
    s = finish(fe, 1)                                 # review_faces crashed
    assert s['error'] and [p['keep'] for p in s['review']['photos']] == [True, True]


def test_rebuild_backs_up_and_reads_the_report(setup):
    fe, fake, base = setup
    fe.rebuild()
    cmd = fake.procs[0].cmd
    assert 'securebot_ros.enroll_faces' in cmd and '--backup' in cmd
    assert cmd[cmd.index('--output') + 1] == str(base / 'encodings' / 'authorized_faces.pkl')
    (base / 'encodings').mkdir()
    (base / 'encodings' / 'authorized_faces.pkl.report.json').write_text(json.dumps({
        'time': 9e9, 'total': 1, 'people': [{'name': 'Balde', 'photos': 1}],
        'pairs': [], 'backup': None}))
    s = finish(fe)
    assert 'rebuilt' in s['message'] and not s['database']['outdated']
    assert s['people'] == [{'name': 'Balde', 'photos': 1, 'enrolled': 1, 'cover': 'a.jpg'}]


def test_a_review_survives_a_panel_restart(setup, tmp_path):
    fe, fake, base = setup
    d = stage(base, 'Dan', ['1.jpg'])
    review_json(d, 'Dan', [('1.jpg', True)])
    again = wf.FaceEnroller(base=str(base), venv=str(tmp_path / 'venv'), popen=fake.popen,
                            killpg=fake.killpg, ps=fake.ps)
    assert again.state()['review']['name'] == 'Dan'


def test_photos_are_only_served_from_the_faces_folders(setup):
    fe, _, base = setup
    assert fe.photo('enrolled', 'Balde', 'a.jpg') == b'jpg'
    (base / 'secret.jpg').write_bytes(b'no')
    for kind, name, f in (('enrolled', '..', 'secret.jpg'), ('enrolled', '.', 'secret.jpg'),
                          ('enrolled', 'Balde/..', 'secret.jpg'), ('other', 'Balde', 'a.jpg'),
                          ('enrolled', 'Balde', 'b.jpg'), ('staged', '..', 'secret.jpg')):
        with pytest.raises(ValueError):
            fe.photo(kind, name, f)
    assert fe.photo('enrolled', 'Balde', '../a.jpg') == b'jpg'      # reduced to a.jpg


def test_venv_environment_puts_only_libcudss_first(tmp_path):
    lib = tmp_path / 'venv' / 'lib' / 'python3.10' / 'site-packages' / 'nvidia'
    for pkg, so in (('cu12', 'libcudss.so.0'), ('cublas', 'libcublas.so.12')):
        (lib / pkg / 'lib').mkdir(parents=True)
        (lib / pkg / 'lib' / so).write_bytes(b'')
    env = wf.venv_env(str(tmp_path / 'venv'), {'LD_LIBRARY_PATH': '/usr/lib'})
    assert env['LD_LIBRARY_PATH'] == f"{lib / 'cu12' / 'lib'}:/usr/lib"


# ------------------------------------------------------------------------------ manage
def reviewed(base, name, **keep):
    """Write the review result for staged photos: file stem -> ticked."""
    (base / 'staging' / name / '.review.json').write_text(json.dumps({'name': name, 'photos': [
        {'file': f'{stem}.jpg', 'keep': k, 'reasons': [] if k else ['x'], 'notes': []}
        for stem, k in keep.items()]}))


def built(base, people):
    (base / 'encodings').mkdir(exist_ok=True)
    (base / 'encodings' / 'authorized_faces.pkl.report.json').write_text(json.dumps({
        'time': 9e9, 'total': sum(people.values()),
        'people': [{'name': n, 'photos': c} for n, c in people.items()], 'pairs': []}))


def test_delete_one_photo_moves_it_aside_and_outdates_the_database(setup):
    fe, _, base = setup
    (base / 'authorized_faces' / 'Balde' / 'b.jpg').write_bytes(b'jpg')
    built(base, {'Balde': 2})
    assert not fe.state()['database']['outdated']
    assert fe.photos('Balde') == ['a.jpg', 'b.jpg']
    assert 'deleted' in fe.delete_photo('Balde', 'a.jpg').lower()
    assert fe.photos('Balde') == ['b.jpg']
    aside = [p for p in (base / 'captures').iterdir() if p.name.endswith('_Balde_deleted')]
    assert len(aside) == 1 and (aside[0] / 'a.jpg').exists()          # not really gone
    assert fe.state()['database']['outdated']           # 1 photo now, 2 in the database
    with pytest.raises(ValueError):
        fe.delete_photo('Balde', '../../x.jpg')
    with pytest.raises(ValueError):
        fe.delete_photo('../Balde', 'b.jpg')
    with pytest.raises(ValueError):
        fe.photos('Nobody')


def test_delete_person_moves_the_folder_aside(setup):
    fe, _, base = setup
    built(base, {'Balde': 1})
    fe.delete_person('Balde')
    assert not (base / 'authorized_faces' / 'Balde').exists()
    s = fe.state()
    assert s['people'] == [] and s['database']['outdated']   # still in the database
    with pytest.raises(ValueError):
        fe.delete_person('Balde')


def test_cannot_delete_someone_being_captured(setup):
    fe, _, _ = setup
    fe.start_capture('Balde', 5)
    with pytest.raises(RuntimeError):
        fe.delete_person('Balde')
    with pytest.raises(RuntimeError):
        fe.delete_photo('Balde', 'a.jpg')


def test_retake_replaces_the_old_photos_at_keep(setup):
    fe, _, base = setup
    fe.start_capture('Balde', 5, replace=True)
    assert fe.state()['replace'] == 'Balde'
    stage(base, 'Balde', ['n1.jpg', 'n2.jpg'])
    finish(fe)                                        # capture done -> review
    reviewed(base, 'Balde', n1=True, n2=True)
    finish(fe)
    msg = fe.keep('Balde', ['n1.jpg'])
    assert 'replacing the 1 old' in msg
    assert fe.photos('Balde') == ['n1.jpg']
    replaced = [p for p in (base / 'captures').iterdir() if p.name.endswith('_replaced')]
    assert (replaced[0] / 'a.jpg').exists()
    assert fe.state()['replace'] is None


def test_plain_add_keeps_the_old_photos(setup):
    fe, _, base = setup
    fe.start_capture('Balde', 5)
    stage(base, 'Balde', ['n1.jpg'])
    finish(fe)
    reviewed(base, 'Balde', n1=True)
    finish(fe)
    fe.keep('Balde', ['n1.jpg'])
    assert fe.photos('Balde') == ['a.jpg', 'n1.jpg']


def test_retake_keeping_nothing_leaves_the_old_photos(setup):
    fe, _, base = setup
    fe.start_capture('Balde', 5, replace=True)
    stage(base, 'Balde', ['n1.jpg'])
    finish(fe)
    reviewed(base, 'Balde', n1=False)
    finish(fe)
    fe.keep('Balde', [])
    assert fe.photos('Balde') == ['a.jpg']
