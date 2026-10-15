"""Build the authorized-face encodings database from photos.

Expected layout: <faces_dir>/<person_name>/*.jpg|jpeg|png
Output: a pickle {"names": [...], "encodings": [...]} read by securebot_live_node.
"""
import argparse
import pickle
from pathlib import Path

import face_recognition

DEFAULT_BASE = Path("~/yolo/securebot_face_id").expanduser()


def build_database(authorized_dir, output_file):
    known_names = []
    known_encodings = []

    if not authorized_dir.exists():
        raise FileNotFoundError(f"Missing folder: {authorized_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    for person_dir in sorted(authorized_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        image_files = (
            list(person_dir.glob("*.jpg"))
            + list(person_dir.glob("*.jpeg"))
            + list(person_dir.glob("*.png"))
        )

        if not image_files:
            print(f"[WARN] No images found for {person_name}")
            continue

        for image_path in image_files:
            print(f"[INFO] Processing {image_path}")
            image = face_recognition.load_image_file(str(image_path))
            encodings = face_recognition.face_encodings(image)

            if len(encodings) == 0:
                print(f"[WARN] No face found in {image_path}")
                continue

            if len(encodings) > 1:
                print(f"[WARN] More than one face found in {image_path}; using the first face")

            known_names.append(person_name)
            known_encodings.append(encodings[0])

    if not known_encodings:
        raise ValueError("No valid face encodings were created.")

    with open(output_file, "wb") as f:
        pickle.dump({"names": known_names, "encodings": known_encodings}, f)

    print(f"[DONE] Saved {len(known_encodings)} encodings to {output_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faces-dir", type=Path, default=DEFAULT_BASE / "authorized_faces")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_BASE / "encodings" / "authorized_faces.pkl"
    )
    args = parser.parse_args()
    build_database(args.faces_dir.expanduser(), args.output.expanduser())


if __name__ == "__main__":
    main()
