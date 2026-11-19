"""
Live person detection and face identification for SecureBot.

Pipeline per frame:
  1. YOLO + ByteTrack find people and give each a stable track ID.
  2. For tracked people whose identity is undecided (or due a re-check), the head
     area of the person box is cropped from the full-resolution frame, enlarged if
     the person is far away, and searched with dlib's CNN face detector (GPU).
  3. Each face read (name / Unknown / Uncertain) is a vote for that track; a track is
     decided after a streak of consistent votes (see face_logic.TrackIdentity).
  4. /face_status: UNAUTHORIZED if anyone is a confirmed unknown, else SCANNING if
     anyone is undecided, else AUTHORIZED (NO_PERSON when nobody is in view).
"""
import json
import pickle
import threading
import time
from pathlib import Path

import cv2
import face_recognition
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

from securebot_ros.face_logic import (
    AUTHORIZED, crop_scale, face_quality_ok, FaceDB, head_region, largest_face,
    overall_status, recognized_names, SCANNING, TrackIdentity, UNAUTHORIZED, UNKNOWN,
)

# BGR colours per track state
STATE_COLORS = {AUTHORIZED: (0, 200, 0), UNAUTHORIZED: (0, 0, 255), SCANNING: (0, 220, 255)}


class LatestFrameGrabber:
    """
    Reads the camera in a background thread and keeps only the newest frame.

    Processing a frame can take longer than the camera's frame interval; reading in
    the processing loop would then hand out stale buffered frames.
    """

    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame
                self.seq += 1

    def latest(self):
        """Return (seq, frame); seq increases with every new frame."""
        with self.lock:
            return self.seq, self.frame

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)


class SecureBotLiveNode(Node):
    def __init__(self):
        super().__init__("securebot_live_node")

        self.person_pub = self.create_publisher(Bool, "/person_detected", 10)
        self.face_status_pub = self.create_publisher(String, "/face_status", 10)
        self.recognized_name_pub = self.create_publisher(String, "/recognized_name", 10)
        # JSON list of {id, state, name, dist} per tracked person, for debugging/UI
        self.face_tracks_pub = self.create_publisher(String, "/face_tracks", 10)
        # stop | patrol: the security decision; motion_manager_node turns it into wheel
        # commands and esp32_serial_bridge (the only program on the ESP32 port) sends them.
        self.motion_pub = self.create_publisher(String, "/security/motion", 10)

        params = {
            "encodings_file": "~/yolo/securebot_face_id/encodings/authorized_faces.pkl",
            "model_path": "~/SecureBot/yolo11n.pt",
            "camera_index": 0,
            # a track counts as a person once YOLO has been this confident about it
            "yolo_confidence": 0.80,
            # YOLO input size; 640 is more confident on distant people, 416 on groups
            "yolo_imgsz": 416,
            # face read distance: < tolerance (and clear of others by margin) = match,
            # >= unknown_distance = Unknown, in between = Uncertain (no vote)
            "face_tolerance": 0.45,
            "face_unknown_distance": 0.55,
            "face_margin": 0.06,
            "vote_window": 7,
            "votes_needed": 3,
            "face_recheck_s": 2.0,
            "face_detector": "cnn",
            "face_model": "large",
            "face_upsample": 1,
            "face_min_size": 40,
            "face_min_sharpness": 30.0,
            "max_face_crops": 2,
            "show_camera": True,
            # when set, save each face read (crop) and a full frame at every status
            # change here, to trace exactly which image caused a decision
            "debug_dir": "",
        }
        for name, default in params.items():
            self.declare_parameter(name, default)
        p = {name: self.get_parameter(name).value for name in params}

        self.encodings_file = Path(p["encodings_file"]).expanduser()
        self.model_path = str(Path(p["model_path"]).expanduser())
        self.camera_index = p["camera_index"]
        self.yolo_confidence = p["yolo_confidence"]
        self.yolo_imgsz = p["yolo_imgsz"]
        self.face_tolerance = p["face_tolerance"]
        self.face_unknown_distance = p["face_unknown_distance"]
        self.face_margin = p["face_margin"]
        self.vote_window = p["vote_window"]
        self.votes_needed = p["votes_needed"]
        self.face_recheck_s = p["face_recheck_s"]
        self.face_detector = p["face_detector"]
        self.face_model = p["face_model"]
        self.face_upsample = p["face_upsample"]
        self.face_min_size = p["face_min_size"]
        self.face_min_sharpness = p["face_min_sharpness"]
        self.max_face_crops = p["max_face_crops"]
        self.show_camera = p["show_camera"]
        self.debug_dir = None
        if p["debug_dir"]:
            self.debug_dir = Path(p["debug_dir"]).expanduser() / time.strftime("%Y%m%d_%H%M%S")
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.debug_saved = 0
        self.debug_last = {}  # track id -> last time a read crop was saved

        self.last_motor_command = None
        self.demo_running = False

        self.get_logger().info("[INFO] Loading YOLO...")
        self.model = YOLO(self.model_path)

        self.get_logger().info("[INFO] Loading known faces...")
        self.db = self.load_known_faces()

        self.get_logger().info("[INFO] Opening camera...")
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.grabber = LatestFrameGrabber(self.cap)
        self.last_seq = 0

        # track id -> TrackIdentity, plus bookkeeping the logic module doesn't need
        self.tracks = {}
        self.track_seen = {}      # id -> last time the tracker reported it
        self.track_face = {}      # id -> (time, (top, right, bottom, left)) last face box
        self.confirmed_ids = set()  # ids that reached yolo_confidence at least once
        self.track_forget_s = 3.0

        self.fps_frames = 0
        self.fps_since = time.monotonic()

        self.get_logger().info("SecureBot live vision node started.")
        self.get_logger().info("Controls: S=start demo, X=stop robot, Q=quit")
        self.get_logger().info(
            f"Person threshold {self.yolo_confidence:.2f}; face tolerance "
            f"{self.face_tolerance:.2f}, unknown at {self.face_unknown_distance:.2f}, "
            f"{self.votes_needed} votes to decide; detector {self.face_detector}")

        self.timer = self.create_timer(0.01, self.process_frame)

    def send_motor_command(self, command):
        """
        Publish the security decision (stop | patrol).

        Sent every frame, so the motion manager can treat silence (this node crashed
        or hung) as stop.
        """
        self.motion_pub.publish(String(data=command))
        if command != self.last_motor_command:
            self.get_logger().info(f"Security motion: {command}")
            self.last_motor_command = command

    def load_known_faces(self):
        if not self.encodings_file.exists():
            raise FileNotFoundError(
                f"Missing encodings file: {self.encodings_file}. Run enroll_faces first."
            )

        with open(self.encodings_file, "rb") as f:
            data = pickle.load(f)

        db = FaceDB(data["names"], data["encodings"])
        counts = ", ".join(f"{n}: {len(e)}" for n, e in sorted(db.people.items()))
        self.get_logger().info(f"Loaded {len(db)} authorized face encodings ({counts}).")

        enrolled_model = data.get("meta", {}).get("model", "small")
        if enrolled_model != self.face_model:
            self.get_logger().warn(
                f"Encodings were made with the '{enrolled_model}' landmark model but "
                f"face_model is '{self.face_model}'; re-run enroll_faces or set "
                f"face_model:={enrolled_model}, or matching will be less accurate.")
        return db

    def publish_string(self, publisher, value):
        publisher.publish(String(data=str(value)))

    def draw_text_with_bg(
        self,
        frame,
        text,
        org,
        text_color,
        bg_color=(0, 0, 0),
        font_scale=0.7,
        thickness=2,
        padding=6
    ):
        font = cv2.FONT_HERSHEY_SIMPLEX
        x, y = org
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(
            frame,
            (x - padding, y - text_h - padding),
            (x + text_w + padding, y + baseline + padding),
            bg_color,
            cv2.FILLED
        )
        cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)

    # ------------------------------------------------------------------ faces

    def detect_faces(self, rgb):
        try:
            return face_recognition.face_locations(
                rgb, number_of_times_to_upsample=self.face_upsample, model=self.face_detector)
        except RuntimeError as e:
            if self.face_detector != "cnn" or "out of memory" not in str(e):
                raise
            self.get_logger().error(
                "GPU out of memory in the CNN face detector; switching to HOG "
                "(less accurate). Free GPU memory or lower face_upsample.")
            self.face_detector = "hog"
            return face_recognition.face_locations(
                rgb, number_of_times_to_upsample=self.face_upsample, model="hog")

    def read_face(self, frame, box):
        """
        Look for one face in a person's head area.

        Returns (label, distance, face_box) where label is a name, Unknown or
        Uncertain; label is None when a face was seen but was too small or blurry to
        vote. Returns None when no face was found (person turned away, etc.).
        face_box is (top, right, bottom, left) in frame pixels.
        """
        x1, y1, x2, y2 = head_region(box, frame.shape)
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return None
        scale = crop_scale(y2 - y1, x2 - x1)
        interp = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
        crop = cv2.resize(region, None, fx=scale, fy=scale, interpolation=interp)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        locations = self.detect_faces(rgb)
        idx = largest_face(locations)
        if idx is None:
            return None
        top, right, bottom, left = locations[idx]
        face_box = (
            int(y1 + top / scale), int(x1 + right / scale),
            int(y1 + bottom / scale), int(x1 + left / scale),
        )

        # Judge size and sharpness on the camera's own pixels: enlarging a small face
        # adds no detail, and the enlargement itself would read as blur.
        f_top, f_right, f_bottom, f_left = face_box
        size = min(f_right - f_left, f_bottom - f_top)
        face_px = frame[max(0, f_top):f_bottom, max(0, f_left):f_right]
        gray = cv2.cvtColor(face_px, cv2.COLOR_BGR2GRAY) if face_px.size else None
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var() if gray is not None else 0.0
        if not face_quality_ok(size, sharpness, self.face_min_size, self.face_min_sharpness):
            return None, None, face_box

        encoding = face_recognition.face_encodings(
            rgb, [locations[idx]], model=self.face_model)[0]
        label, dist = self.db.match(
            encoding, self.face_tolerance, self.face_unknown_distance, self.face_margin)
        return label, dist, face_box

    # ------------------------------------------------------------------ frame loop

    def track_people(self, frame, now):
        """Run YOLO + ByteTrack; return {track_id: (x1, y1, x2, y2)} of confirmed people."""
        # A low detector threshold lets ByteTrack keep following a person through
        # frames where YOLO is less sure; yolo_confidence decides who counts as a person.
        result = self.model.track(
            frame, imgsz=self.yolo_imgsz, conf=0.25, classes=[0], persist=True,
            tracker="bytetrack.yaml", verbose=False)[0]
        people = {}
        if result.boxes is None or result.boxes.id is None:
            return people
        for box, tid, conf in zip(result.boxes.xyxy.tolist(),
                                  result.boxes.id.int().tolist(),
                                  result.boxes.conf.tolist()):
            if conf >= self.yolo_confidence:
                self.confirmed_ids.add(tid)
            if tid in self.confirmed_ids:
                people[tid] = tuple(int(v) for v in box)
                self.track_seen[tid] = now
        return people

    def forget_old_tracks(self, now):
        for tid in [t for t, seen in self.track_seen.items() if now - seen > self.track_forget_s]:
            for store in (self.tracks, self.track_seen, self.track_face):
                store.pop(tid, None)
            self.confirmed_ids.discard(tid)

    def update_identities(self, frame, people, now):
        for tid in people:
            if tid not in self.tracks:
                self.tracks[tid] = TrackIdentity(
                    self.vote_window, self.votes_needed, self.face_recheck_s)
        due = [tid for tid in people if self.tracks[tid].needs_face(now)]
        due.sort(key=lambda tid: self.tracks[tid].priority(now))
        for tid in due[:self.max_face_crops]:
            read = self.read_face(frame, people[tid])
            if read is None:
                continue
            label, dist, face_box = read
            self.track_face[tid] = (now, face_box)
            self.save_debug_read(frame, tid, label, dist, face_box, now)
            if label is not None:
                before = self.tracks[tid].state
                self.tracks[tid].add(label, dist, now)
                after = self.tracks[tid]
                if after.state != before:
                    who = after.name or UNKNOWN
                    self.get_logger().info(
                        f"Person #{tid}: {before} -> {after.state} ({who}, dist {dist:.3f})")
                    self.save_debug_change(frame, tid, people[tid], before, after.state)

    # ------------------------------------------------------------------ debug

    DEBUG_MAX_FILES = 3000
    DEBUG_READ_INTERVAL_S = 0.3

    def _debug_path(self, suffix):
        if self.debug_dir is None or self.debug_saved >= self.DEBUG_MAX_FILES:
            return None
        self.debug_saved += 1
        if self.debug_saved == self.DEBUG_MAX_FILES:
            self.get_logger().warn(f"debug_dir: reached {self.DEBUG_MAX_FILES} files, stopping")
        t = time.time()
        stamp = time.strftime("%H%M%S", time.localtime(t)) + f".{int(t % 1 * 1000):03d}"
        return str(self.debug_dir / f"{stamp}_{suffix}.jpg")

    def save_debug_read(self, frame, tid, label, dist, face_box, now):
        """Save the face (with margin) behind one read: <time>_t<id>_<label>_<dist>.jpg."""
        if self.debug_dir is None:
            return
        if now - self.debug_last.get(tid, -1.0) < self.DEBUG_READ_INTERVAL_S:
            return
        self.debug_last[tid] = now
        top, right, bottom, left = face_box
        h, w = bottom - top, right - left
        crop = frame[max(0, top - h // 2):bottom + h // 2, max(0, left - w // 2):right + w // 2]
        result = "lowquality" if label is None else f"{label}_{dist:.2f}"
        path = self._debug_path(f"t{tid}_{result}")
        if path and crop.size:
            cv2.imwrite(path, crop)

    def save_debug_change(self, frame, tid, box, before, after):
        """Save the full frame, with the person's box, when their status changes."""
        path = self._debug_path(f"t{tid}_CHANGE_{before}-{after}")
        if path is None:
            return
        marked = frame.copy()
        x1, y1, x2, y2 = box
        cv2.rectangle(marked, (x1, y1), (x2, y2), STATE_COLORS[after], 3)
        cv2.imwrite(path, marked)

    def process_frame(self):
        seq, frame = self.grabber.latest()
        if frame is None or seq == self.last_seq:
            return
        self.last_seq = seq
        frame = frame.copy()
        now = time.monotonic()

        people = self.track_people(frame, now)
        self.forget_old_tracks(now)
        self.update_identities(frame, people, now)

        visible = [self.tracks[tid] for tid in people]
        face_status = overall_status(t.state for t in visible)
        self.person_pub.publish(Bool(data=bool(people)))
        self.publish_string(self.face_status_pub, face_status)
        self.publish_string(self.recognized_name_pub, recognized_names(visible))
        self.publish_string(self.face_tracks_pub, json.dumps([
            {"id": tid, "state": self.tracks[tid].state, "name": self.tracks[tid].name,
             "dist": None if self.tracks[tid].dist is None else round(self.tracks[tid].dist, 3)}
            for tid in sorted(people)
        ]))

        # ==================== MOTOR DECISION ====================
        if not self.demo_running:
            self.send_motor_command("stop")
            motor_status = "STOPPED"
        elif face_status in (UNAUTHORIZED, SCANNING):
            self.send_motor_command("stop")
            motor_status = "STOP"
        else:
            self.send_motor_command("patrol")
            motor_status = "PATROL"

        self.fps_frames += 1
        if now - self.fps_since >= 10.0:
            fps = self.fps_frames / (now - self.fps_since)
            self.get_logger().info(f"{fps:.1f} fps, {len(people)} people, {face_status}")
            self.fps_frames, self.fps_since = 0, now

        if self.show_camera:
            self.draw(frame, people, now, motor_status, face_status)

    # ------------------------------------------------------------------ display

    def draw(self, frame, people, now, motor_status, face_status):
        for tid, (x1, y1, x2, y2) in people.items():
            track = self.tracks[tid]
            color = STATE_COLORS[track.state]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if track.state == AUTHORIZED:
                label = f"#{tid} {track.name}"
            elif track.state == UNAUTHORIZED:
                label = f"#{tid} UNKNOWN"
            else:
                label = f"#{tid} scanning"
            if track.dist is not None and track.state != SCANNING:
                label += f" {track.dist:.2f}"
            self.draw_text_with_bg(frame, label, (x1 + 4, max(25, y1 - 10)),
                                   text_color=(255, 255, 255), bg_color=color,
                                   font_scale=0.6, thickness=2, padding=4)
            seen = self.track_face.get(tid)
            if seen and now - seen[0] < 1.0:
                top, right, bottom, left = seen[1]
                cv2.rectangle(frame, (left, top), (right, bottom), color, 1)

        if people:
            top_text = f"{motor_status} - {face_status}"
            text_color = {AUTHORIZED: (0, 255, 0), UNAUTHORIZED: (0, 0, 255)}.get(
                face_status, (0, 255, 255))
        else:
            top_text = f"{motor_status} - NO PERSON"
            text_color = (0, 255, 255) if self.demo_running else (0, 0, 255)

        self.draw_text_with_bg(frame, top_text, (20, 40), text_color=text_color,
                               font_scale=0.8, thickness=2, padding=6)
        self.draw_text_with_bg(frame, "Controls: S=start  X=stop  Q=quit", (20, 80),
                               text_color=(255, 255, 255), font_scale=0.65, padding=5)

        cv2.imshow("SecureBot Live ROS", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            self.demo_running = True
            self.get_logger().info("Demo STARTED")
        elif key == ord("x"):
            self.demo_running = False
            self.send_motor_command("stop")
            self.get_logger().info("Demo STOPPED")
        elif key == ord("q"):
            self.get_logger().info("Quit command received")
            self.demo_running = False
            self.send_motor_command("stop")
            raise SystemExit

    def destroy_node(self):
        # Stop the camera thread first: if it is still inside cap.read() when the
        # interpreter exits, the process aborts ("FATAL: exception not rethrown").
        if hasattr(self, "grabber"):
            self.grabber.stop()
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()

        # After Ctrl-C rclpy has already shut the context down and publishing fails;
        # the motion manager then treats the silence as stop anyway.
        if rclpy.ok():
            self.send_motor_command("stop")

        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SecureBotLiveNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass

    node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
