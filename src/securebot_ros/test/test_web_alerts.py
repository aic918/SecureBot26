"""Alerts (web_alerts.AlertStore) and the camera's snapshot / live-node preview."""
import os
import time

import pytest

from securebot_ros import web_camera as wc
from securebot_ros.web_alerts import AlertStore


def clock_from(start=1764280800.0):
    t = [start - 1.0]

    def clock():
        t[0] += 1.0
        return t[0]
    return clock


def test_add_list_ack_and_snapshot(tmp_path):
    store = AlertStore(str(tmp_path / 'alerts'), clock=clock_from())
    a = store.add('tamper', 'lift', 'Robot lifted: motors stopped', b'JPEG')
    b = store.add('tamper', 'tilt', 'Robot tipped over: motors stopped')
    assert [x['id'] for x in store.list()] == [b['id'], a['id']]       # newest first
    assert store.snapshot(a['id']) == b'JPEG'
    with pytest.raises(ValueError):
        store.snapshot(b['id'])
    assert len(store.unacked()) == 2
    assert store.ack(a['id']) == 1
    assert [x['id'] for x in store.unacked()] == [b['id']]
    assert store.ack() == 1 and store.unacked() == []


def test_ids_cannot_reach_other_files(tmp_path):
    store = AlertStore(str(tmp_path / 'alerts'))
    for bad in ('../x', '', 'abc', '20261127_120000_000/../../x'):
        with pytest.raises(ValueError):
            store.snapshot(bad)
        if bad:                              # '' means "all" for ack
            with pytest.raises(ValueError):
                store.ack(bad)


def test_keeps_the_newest(tmp_path):
    store = AlertStore(str(tmp_path / 'alerts'), clock=clock_from(), keep=3)
    ids = [store.add('tamper', 'lift', 't', b'J')['id'] for _ in range(5)]
    assert [a['id'] for a in store.list()] == ids[:1:-1]
    assert len(os.listdir(tmp_path / 'alerts')) == 6                   # 3 x (json + jpg)


def test_hooks_get_each_alert_and_a_broken_one_is_recorded(tmp_path):
    store = AlertStore(str(tmp_path / 'alerts'), clock=clock_from())
    got = []

    def pager(alert):
        got.append(alert['text'])

    def broken(alert):
        raise OSError('no signal')
    store.hooks += [broken, pager]
    a = store.add('tamper', 'tilt', 'Robot tipped over: motors stopped')
    assert got == ['Robot tipped over: motors stopped']
    assert store.list()[0]['hook_errors'] == ['broken: no signal']
    assert a['id'] == store.list()[0]['id']


# ------------------------------------------------------------------------------ camera
class Cap:
    def __init__(self):
        self.released = False

    def read(self):
        import numpy as np
        return True, np.zeros((48, 64, 3), np.uint8)

    def release(self):
        self.released = True


def test_snapshot_starts_the_camera_and_waits_for_a_frame():
    threads = []
    cam = wc.CameraFeed(opener=lambda i: Cap(), start_thread=threads.append,
                        preview=lambda: None)
    waits = []

    def sleep(s):
        waits.append(s)
        if len(waits) == 2:                  # the capture thread delivers a frame meanwhile
            cam.jpeg = b'FRAME'
    assert cam.snapshot(sleep=sleep) == b'FRAME'
    assert len(threads) == 1


def test_snapshot_gives_up_without_a_camera():
    t = [0.0]
    cam = wc.CameraFeed(busy=lambda: 'The live face node has the camera', preview=lambda: None,
                        clock=lambda: t[0])

    def sleep(s):
        t[0] += s
    assert cam.snapshot(timeout=1.0, sleep=sleep) is None


def test_busy_camera_shows_the_live_nodes_frame():
    cam = wc.CameraFeed(busy=lambda: 'The live face node has the camera',
                        preview=lambda: b'LIVE')
    assert cam.frame() == b'LIVE'
    assert cam.snapshot() == b'LIVE'


def test_live_preview_only_when_recent(tmp_path):
    path = tmp_path / 'live.jpg'
    assert wc.live_preview(str(path)) is None
    path.write_bytes(b'JPG')
    assert wc.live_preview(str(path)) == b'JPG'
    old = time.time() - 60
    os.utime(path, (old, old))
    assert wc.live_preview(str(path)) is None
