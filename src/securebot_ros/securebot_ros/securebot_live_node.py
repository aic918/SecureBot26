import pickle
from pathlib import Path
import time

import cv2
import face_recognition
import numpy as np
from ultralytics import YOLO
import serial

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String


class SecureBotLiveNode(Node):
    def __init__(self):
        super().__init__("securebot_live_node")

        self.person_pub = self.create_publisher(Bool, "/person_detected", 10)
        self.face_status_pub = self.create_publisher(String, "/face_status", 10)
        self.recognized_name_pub = self.create_publisher(String, "/recognized_name", 10)

        self.declare_parameter(
            "encodings_file",
            "~/yolo/securebot_face_id/encodings/authorized_faces.pkl"
        )
        self.declare_parameter("model_path", "~/SecureBot/yolo11n.pt")
        self.declare_parameter("camera_index", 0)

        # Person detection must be 80% or higher
        self.declare_parameter("yolo_confidence", 0.80)

        self.declare_parameter("face_tolerance", 0.45)
        self.declare_parameter("face_scale", 0.25)
        self.declare_parameter("face_interval", 5)
        self.declare_parameter("show_camera", True)

        # ESP32 settings
        self.declare_parameter("esp32_port", "/dev/ttyUSB0")
        self.declare_parameter("esp32_baud", 115200)
        self.declare_parameter("enable_motors", True)
        self.declare_parameter("motor_speed", 88)

        self.encodings_file = Path(self.get_parameter("encodings_file").value).expanduser()
        self.model_path = str(Path(self.get_parameter("model_path").value).expanduser())
        self.camera_index = self.get_parameter("camera_index").value
        self.yolo_confidence = self.get_parameter("yolo_confidence").value
        self.face_tolerance = self.get_parameter("face_tolerance").value
        self.face_scale = self.get_parameter("face_scale").value
        self.face_interval = self.get_parameter("face_interval").value
        self.show_camera = self.get_parameter("show_camera").value

        self.esp32_port = self.get_parameter("esp32_port").value
        self.esp32_baud = self.get_parameter("esp32_baud").value
        self.enable_motors = self.get_parameter("enable_motors").value
        self.motor_speed = int(self.get_parameter("motor_speed").value)

        self.esp = None
        self.last_motor_command = None
        self.last_motor_send = 0.0
        # ESP32 watchdog stops the robot if no command arrives for 500 ms,
        # so the current command is resent at this period.
        self.heartbeat_period = 0.2
        self.demo_running = False

        if self.enable_motors:
            self.connect_esp32()

        self.get_logger().info("[INFO] Loading YOLO...")
        self.model = YOLO(self.model_path)

        self.get_logger().info("[INFO] Loading known faces...")
        self.known_names, self.known_encodings = self.load_known_faces()

        self.get_logger().info("[INFO] Opening camera...")
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self.frame_count = 0
        self.last_face_results = []

        self.get_logger().info("SecureBot live vision node started.")
        self.get_logger().info("Controls: S=start demo, X=stop robot, Q=quit")
        self.get_logger().info(f"Person detection threshold: {self.yolo_confidence:.2f}")
        self.get_logger().info(f"Motor speed: {self.motor_speed}")

        self.timer = self.create_timer(0.03, self.process_frame)

    def connect_esp32(self):
        try:
            self.esp = serial.Serial(self.esp32_port, self.esp32_baud, timeout=1)
            time.sleep(2)
            self.esp.reset_input_buffer()

            self.send_motor_command("stop")

            self.send_motor_command(f"speed {self.motor_speed}")

            self.get_logger().info(
                f"[INFO] Connected to ESP32 on {self.esp32_port} at {self.esp32_baud}"
            )
        except Exception as e:
            self.esp = None
            self.get_logger().error(f"[ERROR] Could not connect to ESP32: {e}")

    def send_motor_command(self, command):
        if not self.enable_motors or self.esp is None:
            return

        try:
            now = time.monotonic()
            changed = command != self.last_motor_command
            heartbeat_due = now - self.last_motor_send >= self.heartbeat_period

            if changed or heartbeat_due or command.startswith("speed"):
                self.esp.write((command + "\n").encode())
                if changed or command.startswith("speed"):
                    self.get_logger().info(f"Sent to ESP32: {command}")
                self.last_motor_command = command
                self.last_motor_send = now
        except Exception as e:
            self.get_logger().error(f"[ERROR] Failed to send command to ESP32: {e}")

    def load_known_faces(self):
        if not self.encodings_file.exists():
            raise FileNotFoundError(
                f"Missing encodings file: {self.encodings_file}. Run enroll_faces.py first."
            )

        with open(self.encodings_file, "rb") as f:
            data = pickle.load(f)

        self.get_logger().info(
            f"Loaded {len(data['encodings'])} authorized face encodings."
        )

        return data["names"], data["encodings"]

    def publish_bool(self, publisher, value):
        msg = Bool()
        msg.data = bool(value)
        publisher.publish(msg)

    def publish_string(self, publisher, value):
        msg = String()
        msg.data = str(value)
        publisher.publish(msg)

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

        (text_w, text_h), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness
        )

        cv2.rectangle(
            frame,
            (x - padding, y - text_h - padding),
            (x + text_w + padding, y + baseline + padding),
            bg_color,
            cv2.FILLED
        )

        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            font_scale,
            text_color,
            thickness
        )

    def recognize_faces_fast(self, frame):
        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=self.face_scale,
            fy=self.face_scale
        )

        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        results = []
        inv_scale = int(round(1 / self.face_scale))

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"

            if len(self.known_encodings) > 0:
                distances = face_recognition.face_distance(
                    self.known_encodings,
                    face_encoding
                )

                best_index = np.argmin(distances)

                if distances[best_index] < self.face_tolerance:
                    name = self.known_names[best_index]

            top *= inv_scale
            right *= inv_scale
            bottom *= inv_scale
            left *= inv_scale

            results.append((name, (top, right, bottom, left)))

        return results

    def process_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warn("[WARN] Failed to read frame")
            return

        self.frame_count += 1

        yolo_results = self.model(frame, imgsz=416, verbose=False)[0]

        person_found = False

        if yolo_results.boxes is not None:
            for box in yolo_results.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                # COCO class 0 = person
                # Only detect person if confidence is 0.80 or higher
                if cls_id == 0 and conf >= self.yolo_confidence:
                    person_found = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    if self.show_camera:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        self.draw_text_with_bg(
                            frame,
                            f"person {conf:.2f}",
                            (x1, max(25, y1 - 10)),
                            text_color=(0, 255, 0),
                            bg_color=(0, 0, 0),
                            font_scale=0.6,
                            thickness=2,
                            padding=4
                        )

        self.publish_bool(self.person_pub, person_found)

        face_status = "NO_PERSON"
        recognized_name = "NONE"

        if person_found:
            face_status = "SCANNING"

            if self.frame_count % self.face_interval == 0:
                self.last_face_results = self.recognize_faces_fast(frame)

            if len(self.last_face_results) > 0:
                authorized_names = [
                    name for name, _ in self.last_face_results
                    if name != "Unknown"
                ]

                if len(authorized_names) > 0:
                    face_status = "AUTHORIZED"
                    recognized_name = authorized_names[0]
                else:
                    face_status = "UNAUTHORIZED"
                    recognized_name = "Unknown"
            else:
                face_status = "SCANNING"
                recognized_name = "NONE"

        else:
            self.last_face_results = []
            face_status = "NO_PERSON"
            recognized_name = "NONE"

        self.publish_string(self.face_status_pub, face_status)
        self.publish_string(self.recognized_name_pub, recognized_name)

        # ==================== MOTOR DECISION ====================
        if not self.demo_running:
            self.send_motor_command("stop")
            motor_status = "STOPPED"
        else:
            if face_status == "UNAUTHORIZED":
                self.send_motor_command("stop")
                motor_status = "STOP"
            elif face_status == "SCANNING":
                self.send_motor_command("stop")
                motor_status = "STOP"
            else:
                self.send_motor_command("patrol")
                motor_status = "PATROL"

        if self.show_camera:
            for name, (top, right, bottom, left) in self.last_face_results:
                box_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

                cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)

                label_bg = (0, 120, 0) if name != "Unknown" else (0, 0, 180)

                cv2.rectangle(
                    frame,
                    (left, bottom - 34),
                    (right, bottom),
                    label_bg,
                    cv2.FILLED
                )

                cv2.putText(
                    frame,
                    name,
                    (left + 6, bottom - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            if person_found:
                top_text = f"{motor_status} - {face_status}: {recognized_name}"

                if face_status == "AUTHORIZED":
                    text_color = (0, 255, 0)
                elif face_status == "UNAUTHORIZED":
                    text_color = (0, 0, 255)
                else:
                    text_color = (0, 255, 255)
            else:
                top_text = f"{motor_status} - NO PERSON"

                if self.demo_running:
                    text_color = (0, 255, 255)
                else:
                    text_color = (0, 0, 255)

            self.draw_text_with_bg(
                frame,
                top_text,
                (20, 40),
                text_color=text_color,
                bg_color=(0, 0, 0),
                font_scale=0.8,
                thickness=2,
                padding=6
            )

            control_text = "Controls: S=start  X=stop  Q=quit"

            self.draw_text_with_bg(
                frame,
                control_text,
                (20, 80),
                text_color=(255, 255, 255),
                bg_color=(0, 0, 0),
                font_scale=0.65,
                thickness=2,
                padding=5
            )

            threshold_text = (
                f"Person confidence required: {self.yolo_confidence:.2f}+ | "
                f"Speed: {self.motor_speed}"
            )

            self.draw_text_with_bg(
                frame,
                threshold_text,
                (20, 120),
                text_color=(255, 255, 255),
                bg_color=(0, 0, 0),
                font_scale=0.6,
                thickness=2,
                padding=5
            )

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
                self.cap.release()
                cv2.destroyAllWindows()
                raise SystemExit

    def destroy_node(self):
        self.send_motor_command("stop")

        if self.esp is not None:
            self.esp.close()

        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()

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