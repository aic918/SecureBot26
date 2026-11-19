"""
Capture enrollment photos of one person from the robot's own camera.

Phone photos often don't look like what the robot's camera sees (lens, height,
lighting), which makes live matches weak. This watches the camera and saves a
padded face crop into <faces_dir>/<name>/ whenever the face is sharp, big enough,
and a new look compared with what was already saved this session (so turning the
head slowly collects different angles instead of many copies of one pose).

Only the person being enrolled should be in view. Each frame's distance to the
person's current enrollment is shown, so poses the database doesn't cover yet
stand out (high numbers). Afterwards, re-run enroll_faces.

  python -m securebot_ros.capture_faces --name alexis
"""
import argparse
import pickle
import time
from pathlib import Path

import cv2
import face_recognition
import numpy as np

from securebot_ros.face_logic import FaceDB, largest_face

DEFAULT_BASE = Path("~/yolo/securebot_face_id").expanduser()


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index} (is the live node still running?)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def find_face(frame, detect_width=640):
    """Return the largest face as (top, right, bottom, left) in frame pixels, or None."""
    scale = detect_width / frame.shape[1]
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, number_of_times_to_upsample=1, model="cnn")
    idx = largest_face(locations)
    if idx is None:
        return None
    return tuple(int(v / scale) for v in locations[idx])


def padded_crop(frame, box, pad=0.8):
    """Face box grown by `pad` of its size on each side, clamped to the frame."""
    top, right, bottom, left = box
    h, w = bottom - top, right - left
    y1, y2 = max(0, int(top - pad * h)), min(frame.shape[0], int(bottom + pad * h))
    x1, x2 = max(0, int(left - pad * w)), min(frame.shape[1], int(right + pad * w))
    return frame[y1:y2, x1:x2]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="person folder name, e.g. alexis")
    parser.add_argument("--faces-dir", type=Path, default=DEFAULT_BASE / "authorized_faces")
    parser.add_argument(
        "--encodings", type=Path, default=DEFAULT_BASE / "encodings" / "authorized_faces.pkl")
    parser.add_argument("--count", type=int, default=15, help="photos to save")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--min-size", type=int, default=80, help="min face size in pixels")
    parser.add_argument("--min-sharpness", type=float, default=60.0)
    parser.add_argument("--novelty", type=float, default=0.12,
                        help="min encoding distance from photos already saved this session")
    parser.add_argument("--min-gap", type=float, default=0.7, help="seconds between saves")
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    out_dir = args.faces_dir.expanduser() / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    db = None
    if args.encodings.expanduser().exists():
        with open(args.encodings.expanduser(), "rb") as f:
            data = pickle.load(f)
        db = FaceDB(data["names"], data["encodings"])
        if args.name not in db.people:
            print(f"[INFO] {args.name} is not enrolled yet; distances will show as n/a")

    cap = open_camera(args.camera)
    saved = []
    last_save = 0.0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    print(f"[INFO] Saving up to {args.count} photos to {out_dir}. Turn your head slowly: "
          "left, right, up, down, and step closer/farther. Q to quit.")

    try:
        while len(saved) < args.count:
            ok, frame = cap.read()
            if not ok:
                continue
            shown = frame.copy()
            box = find_face(frame)
            status = "no face"
            if box is not None:
                top, right, bottom, left = box
                size = min(right - left, bottom - top)
                gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
                sharp = cv2.Laplacian(gray, cv2.CV_64F).var() if gray.size else 0.0
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                enc = face_recognition.face_encodings(rgb, [box], model="large")[0]
                dist = db.distances(enc).get(args.name) if db else None
                novelty = min((np.linalg.norm(enc - e) for e in saved), default=1.0)
                dist_txt = "n/a" if dist is None else f"{dist:.2f}"
                status = f"dist {dist_txt} size {size} sharp {sharp:.0f} new {novelty:.2f}"
                good = size >= args.min_size and sharp >= args.min_sharpness
                due = time.monotonic() - last_save > args.min_gap
                if good and novelty >= args.novelty and due:
                    path = out_dir / f"cam_{stamp}_{len(saved) + 1:02d}.jpg"
                    cv2.imwrite(str(path), padded_crop(frame, box))
                    saved.append(enc)
                    last_save = time.monotonic()
                    print(f"[SAVE] {path.name}: {status}")
                color = (0, 200, 0) if good else (0, 165, 255)
                cv2.rectangle(shown, (left, top), (right, bottom), color, 2)
            if not args.no_window:
                cv2.putText(shown, f"{len(saved)}/{args.count}  {status}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("SecureBot face capture", shown)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[DONE] Saved {len(saved)} photos to {out_dir}. Now re-run: "
          "python -m securebot_ros.enroll_faces")


if __name__ == "__main__":
    main()
