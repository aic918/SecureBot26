"""
Check captured enrollment photos before they are added (face_logic.review_photos).

Encodes each photo in a folder, compares it with the other photos in the folder and
with everyone already in the face database, and writes the verdicts as JSON for the
web panel: photos that could be mistaken for someone else, that are unlike the rest
(side views, heavy blur, a different person) or that don't match the person's
existing enrollment come back unchecked, each with its reason.

  python -m securebot_ros.review_faces --name Daniel --dir <staging>/Daniel --out review.json
"""
import argparse
import json
import pickle
from pathlib import Path

import cv2
import face_recognition

from securebot_ros.capture_faces import find_faces
from securebot_ros.enroll_faces import IMAGE_SUFFIXES
from securebot_ros.face_logic import FaceDB, largest_face, review_photos

DEFAULT_ENCODINGS = Path("~/yolo/securebot_face_id/encodings/authorized_faces.pkl")


def encode(path):
    """Return (encoding or None, note) for the largest face in a photo."""
    frame = cv2.imread(str(path))
    if frame is None:
        return None, "cannot read the file"
    if frame.shape[1] < 640:      # captures are face crops: give the detector room
        scale = 640 / frame.shape[1]
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    boxes = find_faces(frame)
    if not boxes:
        return None, ""
    idx = largest_face(boxes)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    enc = face_recognition.face_encodings(rgb, [boxes[idx]], model="large")[0]
    note = f"{len(boxes)} faces in the photo; checked the largest" if len(boxes) > 1 else ""
    return enc, note


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dir", type=Path, required=True, help="folder of photos to check")
    parser.add_argument("--out", type=Path, required=True, help="write the verdicts here")
    parser.add_argument("--encodings", type=Path, default=DEFAULT_ENCODINGS)
    args = parser.parse_args()

    others, own = {}, None
    db_path = args.encodings.expanduser()
    if db_path.exists():
        with open(db_path, "rb") as f:
            data = pickle.load(f)
        db = FaceDB(data["names"], data["encodings"])
        others = {n: e for n, e in db.people.items() if n.lower() != args.name.lower()}
        own = next((e for n, e in db.people.items() if n.lower() == args.name.lower()), None)

    photos = []
    for path in sorted(p for p in args.dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        enc, note = encode(path)
        photos.append({"file": path.name, "encoding": enc, "note": note})
        print(f"[CHECK] {path.name}")

    verdicts = review_photos(photos, args.name, others, own)
    args.out.write_text(json.dumps({"name": args.name, "photos": verdicts}))
    kept = sum(v["keep"] for v in verdicts)
    print(f"[DONE] {kept} of {len(verdicts)} photos look good; verdicts in {args.out}")


if __name__ == "__main__":
    main()
