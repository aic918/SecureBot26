import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from ultralytics import YOLO


class YoloPersonNode(Node):
    def __init__(self):
        super().__init__('yolo_person_node')

        self.publisher = self.create_publisher(Bool, '/person_detected', 10)

        self.declare_parameter('camera_index', 0)
        self.declare_parameter('model_path', 'yolo11n.pt')
        self.declare_parameter('confidence', 0.55)
        self.declare_parameter('required_detection_frames', 3)
        self.declare_parameter('required_lost_frames', 10)
        self.declare_parameter('show_camera', True)

        self.camera_index = self.get_parameter('camera_index').value
        self.model_path = self.get_parameter('model_path').value
        self.confidence = self.get_parameter('confidence').value
        self.required_detection_frames = self.get_parameter('required_detection_frames').value
        self.required_lost_frames = self.get_parameter('required_lost_frames').value
        self.show_camera = self.get_parameter('show_camera').value

        self.get_logger().info(f'Loading YOLO model: {self.model_path}')
        self.model = YOLO(self.model_path)

        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            self.get_logger().error('Could not open camera.')
            raise RuntimeError('Camera failed to open')

        self.detect_count = 0
        self.lost_count = 0
        self.person_state = False

        self.timer = self.create_timer(0.1, self.process_frame)

    def process_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warn('Failed to read camera frame.')
            return

        results = self.model.predict(
            frame,
            imgsz=640,
            conf=self.confidence,
            verbose=False
        )

        person_found_this_frame = False

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])

                # COCO class 0 = person
                if class_id == 0:
                    person_found_this_frame = True

                    if self.show_camera:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f'Person {conf:.2f}',
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

        if person_found_this_frame:
            self.detect_count += 1
            self.lost_count = 0
        else:
            self.lost_count += 1
            self.detect_count = 0

        # Require multiple frames before changing state.
        if self.detect_count >= self.required_detection_frames:
            self.person_state = True

        if self.lost_count >= self.required_lost_frames:
            self.person_state = False

        msg = Bool()
        msg.data = self.person_state
        self.publisher.publish(msg)

        if self.show_camera:
            status = 'PERSON DETECTED - STOP' if self.person_state else 'NO PERSON - PATROL'
            cv2.putText(
                frame,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255) if self.person_state else (255, 255, 255),
                2
            )
            cv2.imshow('SecureBot YOLO Person Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                rclpy.shutdown()

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloPersonNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


