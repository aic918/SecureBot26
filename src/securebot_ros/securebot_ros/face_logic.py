"""
Face identity logic for securebot_live_node, kept free of ROS, cameras and dlib.

The node finds people with YOLO + ByteTrack, looks for a face in the head area of
each tracked person, and feeds every face read into that person's TrackIdentity.
A person only counts as authorized or unknown after several consistent reads, and
the overall status treats any confirmed unknown as UNAUTHORIZED, even when an
authorized person is also in view.
"""
from collections import deque

import numpy as np

# Labels a single face read can produce (besides an enrolled person's name).
UNKNOWN = "Unknown"
UNCERTAIN = "Uncertain"

# Per-track states.
SCANNING = "SCANNING"
AUTHORIZED = "AUTHORIZED"
UNAUTHORIZED = "UNAUTHORIZED"

# Overall status published on /face_status.
NO_PERSON = "NO_PERSON"


def head_region(box, frame_shape, height_frac=0.45, pad_frac=0.15):
    """
    Return (x1, y1, x2, y2) of the upper part of a person box, padded and clamped.

    For a standing person the face is in the top part of the box. When only head and
    shoulders are visible (close up, or cut off by the frame) the box is about as wide
    as it is tall and the face fills much more of it, so the region is never shorter
    than the box is wide. A little padding keeps the whole face when the box is tight.
    """
    x1, y1, x2, y2 = box
    frame_h, frame_w = frame_shape[:2]
    box_w, box_h = x2 - x1, y2 - y1
    pad_x = box_w * pad_frac
    pad_y = box_h * 0.05
    rx1 = int(max(0, x1 - pad_x))
    ry1 = int(max(0, y1 - pad_y))
    rx2 = int(min(frame_w, x2 + pad_x))
    ry2 = int(min(frame_h, y2, y1 + max(box_h * height_frac, box_w)))
    return rx1, ry1, rx2, ry2


def crop_scale(region_height, region_width, target_height=480, max_side=640,
               max_scale=3.0):
    """
    Resize factor that brings a head region to about target_height pixels.

    Small (distant) regions are enlarged up to max_scale so faces are big enough to
    detect; the long side is capped at max_side because the CNN detector's GPU
    memory grows with image area (about 1200px with upsampling exhausts a Jetson).
    """
    if region_height <= 0 or region_width <= 0:
        return 1.0
    scale = min(max_scale, target_height / region_height)
    return float(min(scale, max_side / max(region_height, region_width)))


def largest_face(locations):
    """Index of the largest (top, right, bottom, left) box, or None if there are none."""
    if not locations:
        return None
    areas = [(b - t) * (r - l) for t, r, b, l in locations]
    return int(np.argmax(areas))


def face_quality_ok(face_size, sharpness, min_size=40, min_sharpness=30.0):
    """Return whether a face is big and sharp enough to vote (else no vote, not a wrong one)."""
    return face_size >= min_size and sharpness >= min_sharpness


class FaceDB:
    """Enrolled encodings grouped by person."""

    def __init__(self, names, encodings):
        grouped = {}
        for name, enc in zip(names, encodings):
            grouped.setdefault(name, []).append(np.asarray(enc, dtype=np.float64))
        self.people = {name: np.vstack(encs) for name, encs in grouped.items()}

    def __len__(self):
        return sum(len(e) for e in self.people.values())

    def distances(self, encoding):
        """Closest distance from encoding to each person's enrolled photos."""
        encoding = np.asarray(encoding, dtype=np.float64)
        return {
            name: float(np.min(np.linalg.norm(encs - encoding, axis=1)))
            for name, encs in self.people.items()
        }

    def match(self, encoding, tolerance=0.45, unknown_distance=0.55, margin=0.06):
        """
        Label one face read: a name, UNKNOWN, or UNCERTAIN, plus the best distance.

        - name:      closest person is under tolerance and clearly closer than anyone else
        - UNKNOWN:   nobody enrolled is within unknown_distance
        - UNCERTAIN: anything in between (grey zone, or two people too close to call);
                     it casts no vote either way
        """
        dists = self.distances(encoding)
        if not dists:
            return UNKNOWN, float("inf")
        ranked = sorted(dists.items(), key=lambda kv: kv[1])
        best_name, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else float("inf")
        if best >= unknown_distance:
            return UNKNOWN, best
        if best < tolerance and second - best >= margin:
            return best_name, best
        return UNCERTAIN, best


class TrackIdentity:
    """
    Identity of one tracked person, decided by a streak of consistent face reads.

    A track is decided when its last `needed` definite reads (names or UNKNOWN;
    UNCERTAIN reads are skipped) within the last `window` reads all agree. A decided
    track keeps its decision through a single odd read, and is re-checked every
    `recheck_s` seconds; after a disagreeing read it is checked every frame until
    the read is confirmed or overturned.
    """

    def __init__(self, window=7, needed=3, recheck_s=2.0):
        self.window = window
        self.needed = needed
        self.recheck_s = recheck_s
        self.votes = deque(maxlen=window)
        self.state = SCANNING
        self.name = None
        self.dist = None
        self.last_read = None
        self.disputed = False

    def _decided_label(self):
        return self.name if self.state == AUTHORIZED else UNKNOWN

    def _streak(self):
        definite = [v for v in self.votes if v != UNCERTAIN]
        if len(definite) < self.needed:
            return None
        tail = definite[-self.needed:]
        return tail[0] if all(v == tail[0] for v in tail) else None

    def add(self, label, dist, now):
        """Record one face read taken at time `now` (seconds)."""
        self.votes.append(label)
        self.last_read = now
        if label != UNCERTAIN:
            self.dist = dist
        winner = self._streak()
        if winner == UNKNOWN:
            self.state, self.name = UNAUTHORIZED, None
        elif winner is not None:
            self.state, self.name = AUTHORIZED, winner
        if winner is not None or self.state == SCANNING:
            self.disputed = False
        elif label != UNCERTAIN:
            # One odd read doesn't overturn a decision, but keeps the track under
            # close watch until a read confirms it or a streak overturns it.
            self.disputed = label != self._decided_label()

    def needs_face(self, now):
        """Whether this track should get a face read this frame."""
        if self.state == SCANNING or self.disputed or self.last_read is None:
            return True
        return now - self.last_read >= self.recheck_s

    def priority(self, now):
        """Sort key: undecided and disputed tracks first, then the longest unread."""
        urgent = 0 if (self.state == SCANNING or self.disputed) else 1
        waited = float("inf") if self.last_read is None else now - self.last_read
        return (urgent, -waited)


def overall_status(states):
    """Combine per-person states. Any confirmed unknown wins over everything else."""
    states = list(states)
    if not states:
        return NO_PERSON
    if UNAUTHORIZED in states:
        return UNAUTHORIZED
    if SCANNING in states:
        return SCANNING
    return AUTHORIZED


def recognized_names(tracks):
    """Comma-separated names of authorized people in view, plus Unknown if any; or NONE."""
    names = sorted({t.name for t in tracks if t.state == AUTHORIZED})
    if any(t.state == UNAUTHORIZED for t in tracks):
        names.append(UNKNOWN)
    return ",".join(names) if names else "NONE"
