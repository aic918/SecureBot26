"""
Build the authorized-face encodings database from photos.

Expected layout: <faces_dir>/<person_name>/*.jpg|jpeg|png
Output: a pickle {"names": [...], "encodings": [...], "meta": {...}} read by
securebot_live_node. "meta" records how the encodings were made, so the node can
warn if it encodes faces differently (encodings from different landmark models
don't compare well).

Photos are turned upright using their EXIF orientation (phone photos are often
stored sideways), downscaled to 1200px, and searched with the GPU CNN face detector. Each
face is encoded with several jitters, which is slow once here but gives steadier
encodings. Aim for 8-15 photos per person: different angles, lighting, glasses.
"""
import argparse
import itertools
import pickle
from pathlib import Path

import face_recognition
import numpy as np
from PIL import Image, ImageOps

from securebot_ros.face_logic import largest_face

DEFAULT_BASE = Path("~/yolo/securebot_face_id").expanduser()
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MIN_GOOD_PHOTOS = 5


def load_upright(path, max_side):
    """Load an image as RGB, rotated per EXIF and shrunk so its long side <= max_side."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((max_side, max_side))
        return np.array(img)


def find_faces(path, detector, max_side):
    """
    Locate faces, trying progressively more sensitive (and memory-hungry) passes.

    The CNN detector upsamples internally; on a ~1200px photo that runs a Jetson out
    of GPU memory, so large images are searched without upsampling, and only a
    smaller copy is upsampled. An out-of-memory error falls back to HOG.
    """
    passes = [(max_side, 0), (800, 1)]
    for side, upsample in passes:
        image = load_upright(path, side)
        try:
            locations = face_recognition.face_locations(
                image, number_of_times_to_upsample=upsample, model=detector)
        except RuntimeError as e:
            if "out of memory" not in str(e):
                raise
            print(f"[WARN] GPU out of memory on {path.name} at {side}px; using HOG")
            locations = face_recognition.face_locations(
                image, number_of_times_to_upsample=upsample, model="hog")
        if locations:
            return image, locations
    return None, []


def encode_photo(path, detector, model, jitters, max_side):
    """Return (encoding, note) for the largest face in a photo, or (None, reason)."""
    image, locations = find_faces(path, detector, max_side)
    if not locations:
        return None, "no face found"
    idx = largest_face(locations)
    top, right, bottom, left = locations[idx]
    encoding = face_recognition.face_encodings(
        image, [locations[idx]], num_jitters=jitters, model=model)[0]
    note = f"face {right - left}px"
    if len(locations) > 1:
        note += f", {len(locations)} faces (used the largest)"
    return encoding, note


def summarize(names, encodings, tolerance):
    """Print how consistent each person's photos are and whether people could be confused."""
    people = {}
    for name, e in zip(names, encodings):
        people.setdefault(name, []).append(e)
    print("\n[SUMMARY]")
    for name, encs in sorted(people.items()):
        encs = np.array(encs)
        pairs = [np.linalg.norm(a - b) for a, b in itertools.combinations(encs, 2)]
        spread = f"self-distance mean {np.mean(pairs):.3f}, max {np.max(pairs):.3f}" \
            if pairs else "only one photo"
        print(f"  {name}: {len(encs)} photos, {spread}")
        if len(encs) < MIN_GOOD_PHOTOS:
            print(f"  [WARN] {name} has {len(encs)} usable photos; "
                  f"{MIN_GOOD_PHOTOS}+ (ideally 8-15) gives more reliable recognition")
    for (a, ea), (b, eb) in itertools.combinations(sorted(people.items()), 2):
        closest = min(np.linalg.norm(x - y) for x in ea for y in eb)
        flag = "  [WARN] could be confused" if closest < tolerance + 0.1 else ""
        print(f"  {a} vs {b}: closest {closest:.3f}{flag}")


def build_database(authorized_dir, output_file, detector="cnn", model="large",
                   jitters=10, max_side=1200, tolerance=0.45):
    known_names = []
    known_encodings = []

    if not authorized_dir.exists():
        raise FileNotFoundError(f"Missing folder: {authorized_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    for person_dir in sorted(authorized_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        image_files = sorted(
            p for p in person_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)

        if not image_files:
            print(f"[WARN] No images found for {person_name}")
            continue

        for image_path in image_files:
            encoding, note = encode_photo(image_path, detector, model, jitters, max_side)
            if encoding is None:
                print(f"[SKIP] {image_path}: {note}")
                continue
            print(f"[OK]   {image_path}: {note}")
            known_names.append(person_name)
            known_encodings.append(encoding)

    if not known_encodings:
        raise ValueError("No valid face encodings were created.")

    meta = {"detector": detector, "model": model, "jitters": jitters}
    with open(output_file, "wb") as f:
        pickle.dump({"names": known_names, "encodings": known_encodings, "meta": meta}, f)

    summarize(known_names, known_encodings, tolerance)
    print(f"\n[DONE] Saved {len(known_encodings)} encodings to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--faces-dir", type=Path, default=DEFAULT_BASE / "authorized_faces")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_BASE / "encodings" / "authorized_faces.pkl"
    )
    parser.add_argument("--detector", choices=["cnn", "hog"], default="cnn")
    parser.add_argument("--model", choices=["large", "small"], default="large",
                        help="landmark model; must match the node's face_model")
    parser.add_argument("--jitters", type=int, default=10)
    parser.add_argument("--max-side", type=int, default=1200)
    parser.add_argument("--tolerance", type=float, default=0.45,
                        help="the node's face_tolerance, for the confusion check")
    args = parser.parse_args()
    build_database(args.faces_dir.expanduser(), args.output.expanduser(), args.detector,
                   args.model, args.jitters, args.max_side, args.tolerance)


if __name__ == "__main__":
    main()
