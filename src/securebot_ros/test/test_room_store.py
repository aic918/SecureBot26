"""Room names and saved-map listing on a temporary maps folder (no ROS, no Tk)."""
import json
import os

import pytest

from securebot_ros import room_store as rs


def write_map(d, stem, w=40, h=30, mtime=None):
    with open(os.path.join(d, stem + '.pgm'), 'wb') as f:
        f.write(b'P5\n# CREATOR: map_saver\n%d %d\n255\n' % (w, h) + bytes(w * h))
    with open(os.path.join(d, stem + '.yaml'), 'w') as f:
        f.write(f'image: {stem}.pgm\nmode: trinary\nresolution: 0.05\n'
                'origin: [-1.5, -2.0, 0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')
    if mtime:
        os.utime(os.path.join(d, stem + '.pgm'), (mtime, mtime))


@pytest.fixture
def maps(tmp_path):
    write_map(tmp_path, 'explore_a', mtime=1000)
    write_map(tmp_path, 'explore_b', w=80, h=60, mtime=2000)
    return tmp_path


def test_lists_maps_newest_first_with_size(maps):
    (maps / 'broken.yaml').write_text('image: nothere.pgm\n')        # image missing: skipped
    (maps / 'notes.txt').write_text('x')
    found = rs.list_maps(str(maps))
    assert [m.stem for m in found] == ['explore_b', 'explore_a']
    b = found[0]
    assert (b.width, b.height, b.resolution) == (80, 60, 0.05)
    assert b.size_m == pytest.approx((4.0, 3.0))
    assert b.origin == [-1.5, -2.0, 0.0]


def test_no_folder_is_no_maps(tmp_path):
    assert rs.list_maps(str(tmp_path / 'none')) == []
    assert rs.RoomStore(str(tmp_path / 'none')).entries() == []


def test_add_rename_change_remove_persist(maps):
    s = rs.RoomStore(str(maps))
    assert s.entries() == [(None, s.entries()[0][1]), (None, s.entries()[1][1])]
    s.add('  Living   room ', 'explore_a')
    assert s.names() == ['Living room']
    assert rs.RoomStore(str(maps)).name_for('explore_a') == 'Living room'   # saved to disk

    s.rename('living ROOM', 'Lounge')                                     # case-insensitive find
    s.set_map('Lounge', 'explore_b')
    again = rs.RoomStore(str(maps))
    assert again.rooms == [{'name': 'Lounge', 'map': 'explore_b'}]
    assert [(n, m.stem) for n, m in again.entries()] == [('Lounge', 'explore_b'),
                                                         (None, 'explore_a')]
    again.remove('Lounge')
    assert rs.RoomStore(str(maps)).rooms == []
    assert len(rs.list_maps(str(maps))) == 2                              # files kept


def test_refuses_bad_requests(maps):
    s = rs.RoomStore(str(maps))
    s.add('Kitchen', 'explore_a')
    for call in (lambda: s.add('kitchen', 'explore_b'),     # duplicate name
                 lambda: s.add('Hall', 'explore_a'),        # map already named
                 lambda: s.add('Hall', 'nope'),             # no such map
                 lambda: s.add('   ', 'explore_b'),         # empty name
                 lambda: s.add('x' * 41, 'explore_b'),      # too long
                 lambda: s.rename('Garage', 'Shed'),        # no such room
                 lambda: s.set_map('Kitchen', 'nope'),
                 lambda: s.remove('Garage')):
        with pytest.raises(rs.RoomError):
            call()
    s.add('Hall', 'explore_b')
    with pytest.raises(rs.RoomError):
        s.rename('Hall', 'KITCHEN')
    with pytest.raises(rs.RoomError):
        s.set_map('Hall', 'explore_a')
    s.rename('Hall', 'hall')                                 # changing only the case is fine
    assert s.names() == ['Kitchen', 'hall']


def test_room_with_missing_map_still_listed(maps):
    s = rs.RoomStore(str(maps))
    s.add('Kitchen', 'explore_a')
    os.remove(maps / 'explore_a.pgm')
    assert rs.RoomStore(str(maps)).entries()[0] == ('Kitchen', None)


def test_delete_map_removes_files_and_name(maps):
    s = rs.RoomStore(str(maps))
    s.add('Kitchen', 'explore_a')
    s.delete_map('explore_a')
    assert not (maps / 'explore_a.yaml').exists() and not (maps / 'explore_a.pgm').exists()
    assert (maps / 'explore_b.pgm').exists()
    assert rs.RoomStore(str(maps)).rooms == []


def test_delete_map_keeps_images_outside_the_folder(maps, tmp_path_factory):
    other = tmp_path_factory.mktemp('elsewhere')
    write_map(other, 'keep')
    (maps / 'link.yaml').write_text(f'image: {other / "keep.pgm"}\nresolution: 0.05\n')
    rs.RoomStore(str(maps)).delete_map('link')
    assert not (maps / 'link.yaml').exists() and (other / 'keep.pgm').exists()


def test_bad_rooms_file_is_reported(maps):
    (maps / rs.ROOMS_FILE).write_text('{not json')
    with pytest.raises(rs.RoomError):
        rs.RoomStore(str(maps))


def test_rooms_file_ignores_junk_entries(maps):
    (maps / rs.ROOMS_FILE).write_text(json.dumps(
        {'rooms': [{'name': 'Ok', 'map': 'explore_a'}, {'name': ''}, 'x', {'map': 'b'}]}))
    assert rs.RoomStore(str(maps)).rooms == [{'name': 'Ok', 'map': 'explore_a'}]


def test_stem_for():
    assert rs.stem_for('Living Room #2!', now=0).startswith('living_room_2_')
    assert rs.stem_for('!!!', now=0).startswith('room_')
