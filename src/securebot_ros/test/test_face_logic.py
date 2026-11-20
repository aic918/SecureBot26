import numpy as np
from securebot_ros.face_logic import (
    AUTHORIZED, box_iou, crop_scale, face_quality_ok, FaceDB, head_region, largest_face,
    NO_PERSON, overall_status, own_face, recognized_names, review_photos, SCANNING,
    touches_edge,
    TrackIdentity, UNAUTHORIZED, UNCERTAIN, UNKNOWN,
)


def enc(*head):
    """Build a 128-d encoding whose first values are `head` and the rest zero."""
    v = np.zeros(128)
    v[:len(head)] = head
    return v


DB = FaceDB(["alexis", "alexis", "daniel"], [enc(0.0), enc(0.1), enc(1.0)])


# ---- matching ----

def test_clear_match_uses_closest_photo_of_that_person():
    label, dist = DB.match(enc(0.12))
    assert label == "alexis" and abs(dist - 0.02) < 1e-9


def test_far_from_everyone_is_unknown():
    assert DB.match(enc(0.0, 0.8))[0] == UNKNOWN


def test_grey_zone_between_tolerance_and_unknown_is_uncertain():
    # 0.5 from alexis: past the 0.45 tolerance, not yet the 0.55 unknown distance
    assert DB.match(enc(0.0, 0.5))[0] == UNCERTAIN


def test_two_people_too_close_to_call_is_uncertain():
    db = FaceDB(["a", "b"], [enc(0.0), enc(0.1)])
    assert db.match(enc(0.05, 0.3))[0] == UNCERTAIN  # ~0.304 vs ~0.304


def test_empty_database_treats_everyone_as_unknown():
    assert FaceDB([], []).match(enc(0.0))[0] == UNKNOWN


def test_database_groups_photos_by_person():
    assert len(DB) == 3 and sorted(DB.people) == ["alexis", "daniel"]


# ---- track voting ----

def test_track_waits_for_a_streak_before_deciding():
    t = TrackIdentity(needed=3)
    t.add("alexis", 0.2, 0.0)
    t.add("alexis", 0.2, 0.1)
    assert t.state == SCANNING
    t.add("alexis", 0.2, 0.2)
    assert t.state == AUTHORIZED and t.name == "alexis"


def test_uncertain_reads_do_not_count_or_break_a_streak():
    t = TrackIdentity(needed=3)
    for label in ["alexis", UNCERTAIN, "alexis", UNCERTAIN, "alexis"]:
        t.add(label, 0.3, 0.0)
    assert t.state == AUTHORIZED


def test_mixed_reads_stay_scanning():
    t = TrackIdentity(needed=3)
    for label in ["alexis", UNKNOWN, "alexis", UNKNOWN]:
        t.add(label, 0.3, 0.0)
    assert t.state == SCANNING


def test_unknown_streak_marks_the_track_unauthorized():
    t = TrackIdentity(needed=3)
    for _ in range(3):
        t.add(UNKNOWN, 0.7, 0.0)
    assert t.state == UNAUTHORIZED and t.name is None


def test_one_odd_read_does_not_flip_a_decision_but_puts_it_under_watch():
    t = TrackIdentity(needed=3, recheck_s=2.0)
    for _ in range(3):
        t.add("alexis", 0.2, 0.0)
    t.add(UNKNOWN, 0.7, 0.5)
    assert t.state == AUTHORIZED and t.disputed
    assert t.needs_face(0.6)  # checked every frame while disputed
    t.add(UNCERTAIN, 0.5, 0.6)
    assert t.disputed  # an uncertain read settles nothing
    t.add("alexis", 0.2, 0.7)
    assert t.state == AUTHORIZED and not t.disputed
    assert not t.needs_face(0.8)


def test_a_decision_is_overturned_by_an_opposite_streak():
    t = TrackIdentity(needed=3)
    for _ in range(3):
        t.add("alexis", 0.2, 0.0)
    for _ in range(3):
        t.add(UNKNOWN, 0.7, 1.0)
    assert t.state == UNAUTHORIZED


def test_decided_track_is_rechecked_after_the_interval():
    t = TrackIdentity(needed=3, recheck_s=2.0)
    for _ in range(3):
        t.add("alexis", 0.2, 10.0)
    assert not t.needs_face(11.0)
    assert t.needs_face(12.0)


def test_priority_puts_undecided_tracks_first():
    decided = TrackIdentity(needed=1)
    decided.add("alexis", 0.2, 0.0)
    new = TrackIdentity()
    assert sorted([decided, new], key=lambda t: t.priority(5.0))[0] is new


# ---- overall status ----

def _track(state, name=None):
    t = TrackIdentity()
    t.state, t.name = state, name
    return t


def test_unknown_next_to_authorized_is_unauthorized():
    assert overall_status([AUTHORIZED, UNAUTHORIZED]) == UNAUTHORIZED


def test_authorized_next_to_unscanned_person_is_still_scanning():
    assert overall_status([AUTHORIZED, SCANNING]) == SCANNING


def test_status_with_everyone_authorized_or_nobody():
    assert overall_status([AUTHORIZED, AUTHORIZED]) == AUTHORIZED
    assert overall_status([]) == NO_PERSON


def test_recognized_names_lists_everyone_decided():
    tracks = [_track(AUTHORIZED, "daniel"), _track(AUTHORIZED, "alexis"),
              _track(UNAUTHORIZED), _track(SCANNING)]
    assert recognized_names(tracks) == "alexis,daniel,Unknown"
    assert recognized_names([_track(SCANNING)]) == "NONE"


# ---- geometry and quality ----

def test_head_region_is_the_padded_top_of_the_box():
    assert head_region((100, 100, 200, 500), (720, 1280)) == (85, 80, 215, 280)


def test_head_region_covers_a_close_up_head_and_shoulders_box():
    # 529 wide x 449 tall: the face fills most of it, so take the whole box height
    assert head_region((397, 257, 926, 706), (720, 1280))[3] == 706


def test_head_region_clamps_to_the_frame():
    assert head_region((0, 0, 1280, 720), (720, 1280)) == (0, 0, 1280, 720)


def test_crop_scale_enlarges_small_regions_within_limits():
    assert crop_scale(80, 60) == 3.0          # distant person: capped enlargement
    assert crop_scale(240, 200) == 2.0        # brings height to 480
    assert crop_scale(320, 1280) == 0.5       # wide close-up: long side capped at 640
    assert crop_scale(4000, 3000) == 0.12
    assert crop_scale(0, 100) == 1.0


def test_largest_face_and_quality_gate():
    assert largest_face([]) is None
    assert largest_face([(0, 10, 10, 0), (0, 50, 50, 0)]) == 1
    assert face_quality_ok(60, 100.0)
    assert not face_quality_ok(30, 100.0)
    assert not face_quality_ok(60, 5.0)


# ---- whose face, cut-off faces, box jumps ----

def test_own_face_ignores_a_neighbour_seen_through_the_padding():
    person = (100, 0, 300, 600)
    faces = [(50, 250, 150, 150), (50, 340, 150, 280)]   # second centre x=310: outside
    assert own_face(faces, person) == 0


def test_own_face_is_none_when_someone_is_in_front_or_behind():
    person = (100, 0, 300, 600)
    faces = [(50, 200, 150, 120), (60, 290, 140, 220)]   # both centres inside the box
    assert own_face(faces, person) is None


def test_own_face_with_no_faces():
    assert own_face([], (0, 0, 10, 10)) is None


def test_touches_edge_flags_cut_off_faces_only():
    region = (100, 50, 400, 350)
    assert not touches_edge((100, 300, 200, 200), region)
    assert touches_edge((50, 300, 200, 200), region)     # cut at the top
    assert touches_edge((100, 300, 200, 101), region)    # cut at the left
    assert touches_edge((100, 400, 200, 200), region)    # cut at the right


def test_box_iou():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 0, 30, 10)) == 0.0
    assert abs(box_iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_own_face_is_none_when_the_face_is_inside_someone_elses_box():
    # Alexis walks in front of Balde: only her face is visible, inside both boxes
    balde, alexis = (100, 0, 400, 700), (150, 40, 330, 700)
    faces = [(80, 280, 200, 180)]
    assert own_face(faces, balde, [alexis]) is None
    assert own_face(faces, alexis, [balde]) is None


def test_own_face_with_others_nearby_but_not_overlapping():
    faces = [(80, 280, 200, 180)]
    assert own_face(faces, (100, 0, 400, 700), [(600, 0, 900, 700)]) == 0


# ---- reviewing new enrollment photos ----

def photo(name, *head, note=""):
    return {"file": name, "encoding": None if head == (None,) else enc(*head), "note": note}


def test_review_keeps_a_consistent_set_far_from_everyone():
    photos = [photo(f"{i}.jpg", 3.0, 0.05 * i) for i in range(5)]
    out = review_photos(photos, "dan", {"alexis": [enc(0.0)]})
    assert all(p["keep"] for p in out) and out[0]["closest"] == ["alexis", 3.0]


def test_review_unchecks_photos_that_look_like_someone_else():
    photos = [photo(f"{i}.jpg", 3.0, 0.05 * i) for i in range(4)] + [photo("x.jpg", 0.4)]
    out = review_photos(photos, "dan", {"balde": [enc(0.0)]}, rest_max=9.0)
    assert not out[-1]["keep"] and out[-1]["reasons"] == ["looks like balde (0.40)"]
    assert all(p["keep"] for p in out[:-1])


def test_review_only_notes_someone_close_but_not_too_close():
    out = review_photos([photo("a.jpg", 0.52)], "dan", {"alexis": [enc(0.0)]})
    assert out[0]["keep"] and out[0]["notes"] == ["close to alexis (0.52)"]


def test_review_unchecks_the_odd_one_out():
    photos = [photo(f"{i}.jpg", 3.0, 0.05 * i) for i in range(5)] + [photo("side.jpg", 3.0, 1.0)]
    out = review_photos(photos, "dan", {})
    assert not out[-1]["keep"] and out[-1]["reasons"][0].startswith("unlike the other photos")
    assert all(p["keep"] for p in out[:-1])


def test_review_of_an_enrolled_person_checks_their_own_photos():
    photos = [photo("me.jpg", 0.1), photo("other.jpg", 2.0)]
    out = review_photos(photos, "alexis", {}, own=[enc(0.0)])
    assert out[0]["keep"] and out[0]["own"] == 0.1
    assert not out[1]["keep"] and "doesn't match alexis" in out[1]["reasons"][0]


def test_review_unchecks_photos_without_a_face():
    out = review_photos([photo("blank.jpg", None)], "dan", {})
    assert out[0]["reasons"] == ["no face found"] and not out[0]["keep"]
