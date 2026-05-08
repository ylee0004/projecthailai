"""
Perception Node — Project Hail AI
====================================
이 노드의 역할:
  - 입력 소스(이미지/비디오/웹캠)에서 프레임을 읽어옴
  - MediaPipe Face Mesh를 사용해 프레임 안에서 얼굴을 감지
  - 감지된 얼굴의 위치를 화면 중앙 기준 정규화된 오프셋(x, y)으로 계산
  - 계산된 오프셋을 ROS2 topic '/face/offset'으로 publish
  - solvePnP로 yaw 각도를 계산해 '/face/direction'으로 publish
  - Control 노드가 이 topic을 subscribe해서 서보 모터를 제어

아키텍처에서의 위치:
  [웹캠] → [FaceDetectorNode] → /face/offset topic → [ServoControllerNode] → [서보]
                              → /face/direction topic

입력 파라미터 (ros2 run 실행 시 --ros-args로 지정):
  source          : 'image' | 'video' | 'webcam'  (기본값: 'image')
  file_path       : 이미지 또는 비디오 파일 경로   (source가 image/video일 때 필수)
  publish_rate    : topic publish 주기 (Hz)        (기본값: 30.0)
  show_preview    : OpenCV 미리보기 창 표시 여부   (기본값: True)
  device_index    : 웹캠 장치 번호                 (source=webcam 전용, 기본값: 0)
  width           : 웹캠 캡처 해상도 너비 (px)     (source=webcam 전용, 기본값: 1280)
  height          : 웹캠 캡처 해상도 높이 (px)     (source=webcam 전용, 기본값: 720)
  fps             : 웹캠 목표 프레임레이트          (source=webcam 전용, 기본값: 30)
  yaw_threshold_deg: LEFT/RIGHT 판정 임계값 (°)   (기본값: 25.0)
  yaw_ema_alpha   : EMA 필터 계수                  (기본값: 0.3)
"""

import math
import time
from collections import deque

import rclpy
from rclpy.node import Node

import cv2
import mediapipe as mp
import numpy as np

from face_tracker_msgs.msg import FaceDirection, FaceOffset
from sensor_msgs.msg import CompressedImage
from perception.yaw_estimator import YawSmoother, classify_direction, estimate_yaw_from_landmarks


class FaceDetectorNode(Node):
    """
    ROS2 노드: 얼굴 감지 및 오프셋/방향 publish

    Node 클래스를 상속받아 ROS2 시스템에 등록되는 노드.
    타이머로 주기적으로 process_frame()을 호출하며,
    감지 결과를 FaceOffset + FaceDirection 메시지로 publish한다.
    """

    def __init__(self):
        """
        노드 초기화
        - ROS2 파라미터 선언 및 로드
        - Publisher 생성 (/face/offset, /face/direction topic)
        - MediaPipe Face Mesh 모델 초기화
        - 입력 소스(이미지/비디오/웹캠) 초기화
        - 주기적 처리를 위한 타이머 생성
        """
        super().__init__('face_detector_node')

        # ── ROS2 파라미터 선언 ──────────────────────────────────────────
        self.declare_parameter('source', 'image')
        self.declare_parameter('file_path', '')
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('show_preview', True)
        # webcam 전용 파라미터
        self.declare_parameter('device_index', 0)
        self.declare_parameter('width', 1280)
        self.declare_parameter('height', 720)
        self.declare_parameter('fps', 30)
        # Phase B 파라미터
        self.declare_parameter('yaw_threshold_deg', 25.0)
        self.declare_parameter('yaw_ema_alpha', 0.3)
        # Preview (Foxglove) 파라미터
        self.declare_parameter('preview_enable', True)
        self.declare_parameter('preview_rate_hz', 5.0)
        self.declare_parameter('preview_jpeg_quality', 70)

        self.source = self.get_parameter('source').value
        self.file_path = self.get_parameter('file_path').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.show_preview = self.get_parameter('show_preview').value
        self._device_index = self.get_parameter('device_index').value
        self._width = self.get_parameter('width').value
        self._height = self.get_parameter('height').value
        self._fps = self.get_parameter('fps').value
        self._yaw_threshold_deg = self.get_parameter('yaw_threshold_deg').value
        self._yaw_ema_alpha = self.get_parameter('yaw_ema_alpha').value
        self._preview_enable = self.get_parameter('preview_enable').value
        self._preview_rate_hz = self.get_parameter('preview_rate_hz').value
        self._preview_jpeg_quality = self.get_parameter('preview_jpeg_quality').value

        # ── ROS2 Publisher 생성 ─────────────────────────────────────────
        self.publisher = self.create_publisher(FaceOffset, '/face/offset', 10)
        self.direction_pub = self.create_publisher(FaceDirection, '/face/direction', 10)

        # ── Preview publisher (Foxglove 시각화용, preview_enable=True일 때만) ──
        self.preview_pub = None
        if self._preview_enable:
            self.preview_pub = self.create_publisher(
                CompressedImage, '/camera/preview/compressed', 10
            )

        # ── Phase B: YawSmoother 초기화 ─────────────────────────────────
        self.yaw_smoother = YawSmoother(alpha=self._yaw_ema_alpha)

        # ── Preview 상태 변수 ────────────────────────────────────────────
        self.last_frame = None          # 가장 최근 BGR frame
        self.last_detection = None      # detect_face_offset 결과 dict
        self.last_yaw_smoothed = float('nan')
        self.last_direction = 'CENTER'
        self.last_main_fps_estimate = 0.0
        self._frame_times = deque(maxlen=30)  # rolling FPS 측정용

        # ── MediaPipe Face Mesh 초기화 ───────────────────────────────────
        # FaceDetection → FaceMesh 로 업그레이드 (landmark 468점 제공)
        # refine_landmarks=False: iris 추가 랜드마크 불필요 (속도 절약)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils
        self.detector = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

        # ── 입력 소스 초기화 ────────────────────────────────────────────
        self.cap = None
        self.static_frame = None
        self._init_source()

        # ── 주기적 타이머 생성 ──────────────────────────────────────────
        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.process_frame,
        )

        # ── Preview timer (preview_enable=True일 때만) ───────────────────
        if self._preview_enable:
            self.create_timer(
                1.0 / self._preview_rate_hz,
                self.publish_preview,
            )

        self.get_logger().info(
            f'FaceDetectorNode started | source={self.source} | rate={self.publish_rate}Hz'
            f' | yaw_threshold={self._yaw_threshold_deg}° | ema_alpha={self._yaw_ema_alpha}'
            f' | preview={self._preview_enable} @ {self._preview_rate_hz}Hz'
        )

    def _validate_device_index(self):
        """device_index 타입/범위 사전 검증 — 음수나 비정수는 V4L2 미정의 동작 유발"""
        if not isinstance(self._device_index, int):
            raise TypeError(f'device_index must be int, got {type(self._device_index).__name__}')
        if self._device_index < 0:
            raise ValueError(f'device_index must be >= 0, got {self._device_index}')

    def _init_source(self):
        """
        입력 소스 초기화

        source 파라미터 값에 따라 세 가지 방식으로 초기화:
          - 'webcam': OpenCV VideoCapture로 USB 웹캠 연결 (장치 번호 0)
          - 'image' : cv2.imread()로 JPEG/PNG 정적 이미지 로드
          - 'video' : OpenCV VideoCapture로 MP4/AVI 등 비디오 파일 열기
        """
        if self.source == 'webcam':
            self._validate_device_index()
            self.cap = cv2.VideoCapture(self._device_index, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                raise RuntimeError(f'Cannot open webcam (device {self._device_index})')
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self.cap.set(cv2.CAP_PROP_FPS, self._fps)
            for _ in range(5):
                self.cap.read()
            self.get_logger().info(
                f'Using webcam (device {self._device_index}) '
                f'MJPG {self._width}x{self._height}@{self._fps}fps'
            )

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

    @staticmethod
    def detect_face_offset(frame, detector):
        """
        프레임에서 얼굴을 감지하고 오프셋을 계산해 dict로 반환.

        MediaPipe FaceMesh 기반. 468 landmark에서 bbox를 min/max로 계산한다.

        Args:
            frame   : BGR numpy 배열
            detector: MediaPipe FaceMesh 인스턴스

        Returns:
            dict with keys:
              detected(bool), x(float), y(float), confidence(float),
              face_cx(float), face_cy(float), landmarks(landmark_list or None)
              x: -1.0(좌) ~ +1.0(우), y: -1.0(하) ~ +1.0(상)
              confidence: FaceMesh는 스코어를 직접 제공하지 않으므로 감지 시 1.0 고정
        """
        h, w = frame.shape[:2]
        frame_center_x = w / 2
        frame_center_y = h / 2

        # BGR → RGB (MediaPipe 요구 포맷)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        result = {
            'detected': False,
            'x': 0.0,
            'y': 0.0,
            'confidence': 0.0,
            'face_cx': 0.0,
            'face_cy': 0.0,
            'landmarks': None,
        }

        if not results.multi_face_landmarks:
            return result

        # 첫 번째 얼굴의 landmark list
        face_landmarks = results.multi_face_landmarks[0]
        lm_list = face_landmarks.landmark  # normalized x/y/z, 0~1

        # 468 landmark 전체에서 bbox 계산 (min/max)
        xs = [lm.x for lm in lm_list]
        ys = [lm.y for lm in lm_list]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # bbox 중심 → 픽셀 좌표
        face_cx = ((x_min + x_max) / 2.0) * w
        face_cy = ((y_min + y_max) / 2.0) * h

        # 화면 중앙 대비 -1.0~+1.0 정규화
        # OpenCV Y축은 아래로 증가하므로 y는 부호 반전해 위쪽을 양수로
        result['x'] = (face_cx - frame_center_x) / frame_center_x
        result['y'] = -((face_cy - frame_center_y) / frame_center_y)
        # FaceMesh는 confidence 스코어를 직접 노출하지 않음; 임계값 통과 = 1.0
        result['confidence'] = 1.0
        result['detected'] = True
        result['face_cx'] = face_cx
        result['face_cy'] = face_cy
        result['landmarks'] = lm_list  # yaw 계산에 사용

        return result

    def _acquire_frame(self):
        """
        현재 source 설정에 따라 프레임 한 장을 획득해 반환.
        실패하면 None 반환.
        """
        if self.source == 'image':
            return self.static_frame.copy()

        ret, frame = self.cap.read()
        if not ret:
            if self.source == 'video':
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            if not ret:
                return None
        return frame

    def process_frame(self):
        """
        핵심 처리 루프 — 타이머에 의해 매 주기마다 호출됨

        처리 순서:
          1. 입력 소스에서 프레임 획득 (_acquire_frame)
          2. MediaPipe 얼굴 감지 + 오프셋 계산 (detect_face_offset)
          3. FaceOffset 메시지 publish (/face/offset)
          4. yaw 추정 + EMA 필터 + 방향 분류
          5. FaceDirection 메시지 publish (/face/direction)
          6. (선택) OpenCV 미리보기 창에 결과 시각화
        """
        # ── FPS 측정 (rolling 1초 window) ──────────────────────────────
        self._frame_times.append(time.time())
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                self.last_main_fps_estimate = (len(self._frame_times) - 1) / elapsed

        # ── Step 1: 프레임 획득 ─────────────────────────────────────────
        frame = self._acquire_frame()
        if frame is None:
            self.get_logger().warn('Cannot read frame')
            return

        frame_h, frame_w = frame.shape[:2]

        # ── Step 2: 얼굴 감지 + 오프셋 계산 ────────────────────────────
        detection = self.detect_face_offset(frame, self.detector)

        # ── 상태 변수 업데이트 (publish_preview에서 사용) ────────────────
        self.last_frame = frame
        self.last_detection = detection

        # ── Step 3: FaceOffset publish ───────────────────────────────────
        now = self.get_clock().now().to_msg()

        offset_msg = FaceOffset()
        offset_msg.detected = detection['detected']
        offset_msg.x = detection['x']
        offset_msg.y = detection['y']
        offset_msg.confidence = detection['confidence']
        self.publisher.publish(offset_msg)

        # ── Step 4: yaw 추정 + EMA + 분류 ───────────────────────────────
        if detection['detected'] and detection['landmarks'] is not None:
            yaw_raw = estimate_yaw_from_landmarks(
                detection['landmarks'], frame_w, frame_h
            )
        else:
            yaw_raw = float('nan')

        yaw_smoothed = self.yaw_smoother.update(yaw_raw)
        direction = classify_direction(yaw_smoothed, self._yaw_threshold_deg)

        # 상태 변수 업데이트 (publish_preview에서 사용)
        self.last_yaw_smoothed = yaw_smoothed
        self.last_direction = direction

        # ── Step 5: FaceDirection publish ────────────────────────────────
        dir_msg = FaceDirection()
        dir_msg.yaw_deg = float(yaw_smoothed) if not math.isnan(yaw_smoothed) else 0.0
        dir_msg.direction = direction
        dir_msg.confidence = float(detection['confidence'])
        dir_msg.detected = bool(detection['detected'])
        dir_msg.stamp = now
        self.direction_pub.publish(dir_msg)

        # ── Step 6: 미리보기 창 (선택) ──────────────────────────────────
        if self.show_preview:
            if detection['detected']:
                cv2.circle(
                    frame,
                    (int(detection['face_cx']), int(detection['face_cy'])),
                    5, (0, 255, 0), -1,
                )
                status = (
                    f'x={offset_msg.x:.2f} y={offset_msg.y:.2f} '
                    f'yaw={dir_msg.yaw_deg:.1f} {direction}'
                )
            else:
                status = 'No face'
            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Face Detector', frame)
            cv2.waitKey(1)

    def publish_preview(self):
        """
        5Hz throttle preview publisher — Foxglove Studio 시각화용.

        last_frame에 yaw/detection 정보를 오버레이한 뒤
        /camera/preview/compressed (CompressedImage, JPEG)로 publish.
        warm-up 중(last_frame=None)이면 skip.
        """
        if self.last_frame is None:
            return

        detection = self.last_detection or {'detected': False, 'confidence': 0.0, 'landmarks': None}
        detected = detection['detected']
        confidence = detection['confidence']
        yaw = self.last_yaw_smoothed
        direction = self.last_direction
        main_fps = self.last_main_fps_estimate

        annotated = self.last_frame.copy()
        frame_h, frame_w = annotated.shape[:2]

        if detected:
            # 얼굴 박스 (landmark bbox, green)
            landmarks = detection.get('landmarks')
            if landmarks is not None:
                xs = [lm.x for lm in landmarks]
                ys = [lm.y for lm in landmarks]
                x1 = int(min(xs) * frame_w)
                y1 = int(min(ys) * frame_h)
                x2 = int(max(xs) * frame_w)
                y2 = int(max(ys) * frame_h)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 6개 yaw landmark 점 (yellow)
                for idx in [1, 152, 33, 263, 61, 291]:
                    lm = landmarks[idx]
                    px, py = int(lm.x * frame_w), int(lm.y * frame_h)
                    cv2.circle(annotated, (px, py), 3, (0, 255, 255), -1)

            # 화면 중심 → yaw 방향 화살표
            cx, cy = frame_w // 2, frame_h // 2
            arrow_len = int(abs(yaw) * 4) if not math.isnan(yaw) else 0
            if direction == 'LEFT':
                arrow_color = (0, 0, 255)    # red (BGR)
                cv2.arrowedLine(annotated, (cx, cy), (cx - arrow_len, cy), arrow_color, 2)
            elif direction == 'RIGHT':
                arrow_color = (255, 0, 0)    # blue (BGR)
                cv2.arrowedLine(annotated, (cx, cy), (cx + arrow_len, cy), arrow_color, 2)
            else:
                arrow_color = (0, 255, 0)    # green
                cv2.arrowedLine(annotated, (cx, cy), (cx, cy), arrow_color, 2)

        # 텍스트 오버레이 (왼쪽 위, white with black outline)
        yaw_str = f'{yaw:+.1f}' if not math.isnan(yaw) else 'nan'
        lines = [
            f'yaw: {yaw_str} deg',
            f'dir: {direction}',
            f'det: {detected}  conf: {confidence:.2f}',
            f'fps: {main_fps:.1f}',
        ]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2
        for i, line in enumerate(lines):
            org = (10, 28 + i * 26)
            # black outline
            cv2.putText(annotated, line, org, font, font_scale, (0, 0, 0), thickness + 1)
            # white text
            cv2.putText(annotated, line, org, font, font_scale, (255, 255, 255), thickness)

        # JPEG encode
        _, buf = cv2.imencode(
            '.jpg', annotated,
            [cv2.IMWRITE_JPEG_QUALITY, self._preview_jpeg_quality],
        )

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.format = 'jpeg'
        msg.data = buf.tobytes()
        self.preview_pub.publish(msg)

    def destroy_node(self):
        """
        노드 종료 시 리소스 해제
        VideoCapture 객체와 OpenCV 창을 닫아 메모리 누수 방지
        """
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    """
    노드 진입점
    rclpy 초기화 → 노드 생성 → spin(이벤트 루프) → 종료 처리
    """
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
