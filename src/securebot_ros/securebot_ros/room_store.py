"""Named rooms and saved LiDAR maps (no ROS, no Tk): what the room manager window works on.

A saved map is the pair map_saver_cli writes: <stem>.yaml (resolution, origin, image name) and
its image (<stem>.pgm). A room is a name the user gives one of those maps; the names live in
<maps_dir>/rooms.json as {"rooms": [{"name": "Kitchen", "map": "explore_1026_1922"}, ...]}.
Removing a room only forgets the name; the map files stay until delete_map() is called.
"""
import json
import os
import re
import time
from dataclasses import dataclass, field

import yaml

DEFAULT_MAPS_DIR = os.path.expanduser('~/securebot_ws/maps')
ROOMS_FILE = 'rooms.json'
MAX_NAME = 40


class RoomError(ValueError):
    """A request the store refuses (bad or duplicate name, unknown room or map)."""


@dataclass
class MapInfo:
    stem: str                    # file name without .yaml, e.g. explore_1026_1922
    yaml_path: str
    image_path: str
    resolution: float = 0.0      # m per pixel
    origin: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    width: int = 0               # pixels
    height: int = 0
    modified: float = 0.0        # mtime of the image, seconds since the epoch
    size_bytes: int = 0

    @property
    def size_m(self):
        return self.width * self.resolution, self.height * self.resolution


def pgm_size(path):
    """(width, height) from a PGM header (P2 or P5), or (0, 0) if it is not one."""
    try:
        with open(path, 'rb') as f:
            head = f.read(512)
    except OSError:
        return 0, 0
    # Header tokens: magic, width, height, maxval; '#' starts a comment up to the line end.
    tokens = re.sub(rb'#[^\n]*', b' ', head).split()
    if len(tokens) < 3 or tokens[0] not in (b'P2', b'P5'):
        return 0, 0
    try:
        return int(tokens[1]), int(tokens[2])
    except ValueError:
        return 0, 0


def read_map(yaml_path):
    """MapInfo for one map's .yaml, or None if the yaml or its image is missing or unreadable."""
    try:
        with open(yaml_path) as f:
            meta = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return None
    image = meta.get('image') if isinstance(meta, dict) else None
    if not image:
        return None
    image_path = image if os.path.isabs(image) else os.path.join(os.path.dirname(yaml_path), image)
    if not os.path.isfile(image_path):
        return None
    st = os.stat(image_path)
    width, height = pgm_size(image_path)
    origin = meta.get('origin') or [0.0, 0.0, 0.0]
    return MapInfo(stem=os.path.splitext(os.path.basename(yaml_path))[0], yaml_path=yaml_path,
                   image_path=image_path, resolution=float(meta.get('resolution', 0.0)),
                   origin=[float(v) for v in origin], width=width, height=height,
                   modified=st.st_mtime, size_bytes=st.st_size)


def list_maps(maps_dir):
    """Every complete saved map in maps_dir, newest first."""
    if not os.path.isdir(maps_dir):
        return []
    maps = [read_map(os.path.join(maps_dir, n)) for n in os.listdir(maps_dir)
            if n.endswith('.yaml')]
    return sorted((m for m in maps if m), key=lambda m: m.modified, reverse=True)


def clean_name(name):
    """The name trimmed, inner whitespace collapsed; RoomError if empty, too long or odd."""
    name = ' '.join(str(name).split())
    if not name:
        raise RoomError('Give the room a name.')
    if len(name) > MAX_NAME:
        raise RoomError(f'Keep the name to {MAX_NAME} characters or fewer.')
    if any(ord(c) < 32 for c in name):
        raise RoomError('The name has control characters in it.')
    return name


def stem_for(name, now=None):
    """A file stem for saving a new map for room `name`: kitchen_1103_1742."""
    slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_') or 'room'
    return f'{slug[:24]}_{time.strftime("%m%d_%H%M", time.localtime(now))}'


class RoomStore:
    """Room names <-> map stems, saved to <maps_dir>/rooms.json after every change."""

    def __init__(self, maps_dir=DEFAULT_MAPS_DIR):
        self.maps_dir = maps_dir
        self.path = os.path.join(maps_dir, ROOMS_FILE)
        self.rooms = []                  # [{'name': str, 'map': stem}], in the user's order
        self.load()

    # --- file ---------------------------------------------------------------------------
    def load(self):
        try:
            with open(self.path) as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}
        except (OSError, ValueError) as e:
            raise RoomError(f'Could not read {self.path}: {e}') from e
        rooms = data.get('rooms', []) if isinstance(data, dict) else []
        self.rooms = [{'name': str(r['name']), 'map': str(r['map'])} for r in rooms
                      if isinstance(r, dict) and r.get('name') and r.get('map')]

    def save(self):
        os.makedirs(self.maps_dir, exist_ok=True)
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'rooms': self.rooms}, f, indent=2)
            f.write('\n')
        os.replace(tmp, self.path)       # never leaves a half-written rooms.json behind

    # --- queries ------------------------------------------------------------------------
    def names(self):
        return [r['name'] for r in self.rooms]

    def find(self, name):
        key = clean_name(name).casefold()
        for r in self.rooms:
            if r['name'].casefold() == key:
                return r
        return None

    def name_for(self, stem):
        """The room name given to map `stem`, or None."""
        for r in self.rooms:
            if r['map'] == stem:
                return r['name']
        return None

    def map_exists(self, stem):
        return read_map(os.path.join(self.maps_dir, stem + '.yaml')) is not None

    def entries(self):
        """[(room name or None, MapInfo or None)]: named rooms first (missing maps included, so
        they can be fixed or removed), then the saved maps nobody has named yet."""
        maps = {m.stem: m for m in list_maps(self.maps_dir)}
        out = [(r['name'], maps.get(r['map'])) for r in self.rooms]
        named = {r['map'] for r in self.rooms}
        out += [(None, m) for s, m in maps.items() if s not in named]
        return out

    # --- changes ------------------------------------------------------------------------
    def _require(self, name):
        r = self.find(name)
        if r is None:
            raise RoomError(f'There is no room called "{name}".')
        return r

    def _check_free(self, name, ignore=None):
        other = self.find(name)
        if other is not None and other is not ignore:
            raise RoomError(f'A room called "{other["name"]}" already exists.')

    def add(self, name, stem):
        """Give map `stem` the room name `name` (one name per map)."""
        name = clean_name(name)
        self._check_free(name)
        if not self.map_exists(stem):
            raise RoomError(f'No saved map called "{stem}" in {self.maps_dir}.')
        taken = self.name_for(stem)
        if taken:
            raise RoomError(f'That map is already the room "{taken}"; rename it instead.')
        self.rooms.append({'name': name, 'map': stem})
        self.save()
        return name

    def rename(self, old, new):
        r = self._require(old)
        new = clean_name(new)
        self._check_free(new, ignore=r)
        r['name'] = new
        self.save()
        return new

    def set_map(self, name, stem):
        """Point room `name` at a different saved map (e.g. a newer scan of the same room)."""
        r = self._require(name)
        if not self.map_exists(stem):
            raise RoomError(f'No saved map called "{stem}" in {self.maps_dir}.')
        taken = self.name_for(stem)
        if taken and taken != r['name']:
            raise RoomError(f'That map is already the room "{taken}".')
        r['map'] = stem
        self.save()

    def remove(self, name):
        """Forget the room name; the map files are kept."""
        r = self._require(name)
        self.rooms.remove(r)
        self.save()

    def delete_map(self, stem):
        """Delete map `stem`'s .yaml and image from disk, and any room name pointing at it."""
        info = read_map(os.path.join(self.maps_dir, stem + '.yaml'))
        paths = [os.path.join(self.maps_dir, stem + '.yaml')]
        # Only the image next to the yaml: a yaml naming a file elsewhere must not delete it.
        if info and os.path.dirname(os.path.abspath(info.image_path)) == \
                os.path.abspath(self.maps_dir):
            paths.append(info.image_path)
        for p in paths:
            if os.path.isfile(p):
                os.remove(p)
        before = len(self.rooms)
        self.rooms = [r for r in self.rooms if r['map'] != stem]
        if len(self.rooms) != before:
            self.save()
