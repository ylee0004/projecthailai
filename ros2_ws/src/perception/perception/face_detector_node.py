"""
Perception Node — Project Hail AI
Detects faces using MediaPipe and publishes normalized offset from frame center.

Input:  image file / video file / webcam (configurable via ROS2 parameter)
Output: /face/offset topic (FaceOffset message)

Phase 1: static image or video file (no webcam required)
Phase 2: live webcam feed
"""

import rclpy
from rclpy.node import Node

import cv2
import mediapipe as mp
import numpy as np

from face_tracker_msgs.msg import FaceOffset


class FaceDetectorNode(Node):

    def __init__(self):
        super().__init__('face_detector_node')

        # Parameters
        self.declare_parameter('source', 'image')        # 'image', 'video', 'webcam'
        self.declare_parameter('file_path', '')          # path to image or video file
        self.declare_parameter('publish_rate', 30.0)     # Hz
        self.declare_parameter('show_preview', True)     # display OpenCV window

        self.source = self.get_parameter('source').value
        self.file_path = self.get_parameter('file_path').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.show_preview = self.get_parameter('show_preview').value

        # Publisher
        self.publisher = self.create_publisher(FaceOffset, '/face/offset', 10)

        # MediaPipe face detection
        self.mp_face = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.detector = self.mp_face.FaceDetection(
            model_selection=0,       # 0 = short range (< 2m), 1 = full range
            min_detection_confidence=0.6
        )

        # OpenCV capture / image load
        self.cap = None
        self.static_frame = None
        self._init_source()

        # Timer
        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.process_frame
        )

        self.get_logger().info(
            f'FaceDetectorNode started | source={self.source} | rate={self.publish_rate}Hz'
        )

    def _init_source(self):
        if self.source == 'webcam':
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError('Cannot open webcam')
            self.get_logger().info('Using webcam (device 0)')

        elif self.source == 'image':
            if not self.file_path:
                raise ValueError('file_path is required for image source')
            self.static_frame = cv2.imread(self.file_path)
            if self.static_frame is None:
                raise RuntimeError(f'Cannot read image: {self.file_path}')
            self.get_logger().info(f'Loaded image: {self.file_path}')

        elif self.source == 'video':
            if not self.file_path:
                raise ValueError('file_path is required for video source')
            self.cap = cv2.VideoCapture(self.file_path)
            if not self.cap.isOpened():
                raise RuntimeError(f'Cannot open video: {self.file_path}')
            self.get_logger().info(f'Using video: {self.file_path}')

        else:
            raise ValueError(f'Unknown source: {self.source}')

    def process_frame(self):
        # Get frame based on source type
        if self.source == 'image':
            frame = self.static_frame.copy()
        else:
            ret, frame = self.cap.read()
            if not ret:
                if self.source == 'video':
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                if not ret:
                    self.get_logger().warn('Cannot read frame')
                    return

        h, w = frame.shape[:2]
        frame_center_x = w / 2
        frame_center_y = h / 2

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        msg = FaceOffset()
        msg.detected = False
        msg.x = 0.0
        msg.y = 0.0
        msg.confidence = 0.0

        if results.detections:
            # Use the highest-confidence detection
            best = max(results.detections, key=lambda d: d.score[0])
            bbox = best.location_data.relative_bounding_box

            # Face center in pixel coordinates
            face_cx = (bbox.xmin + bbox.width / 2) * w
            face_cy = (bbox.ymin + bbox.height / 2) * h

            # Normalized offset from frame center: -1.0 to 1.0
            msg.x = (face_cx - frame_center_x) / frame_center_x
            msg.y = -((face_cy - frame_center_y) / frame_center_y)  # invert Y (up = positive)
            msg.confidence = float(best.score[0])
            msg.detected = True

            if self.show_preview:
                self.mp_draw.draw_detection(frame, best)
                cv2.circle(frame, (int(face_cx), int(face_cy)), 5, (0, 255, 0), -1)

        self.publisher.publish(msg)

        if self.show_preview:
            status = f'x={msg.x:.2f} y={msg.y:.2f} conf={msg.confidence:.2f}' \
                     if msg.detected else 'No face'
            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Face Detector', frame)
            cv2.waitKey(1)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
