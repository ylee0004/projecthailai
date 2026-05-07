"""
Perception Node — Project Hail AI
====================================
이 노드의 역할:
  - 입력 소스(이미지/비디오/웹캠)에서 프레임을 읽어옴
  - MediaPipe를 사용해 프레임 안에서 얼굴을 감지
  - 감지된 얼굴의 위치를 화면 중앙 기준 정규화된 오프셋(x, y)으로 계산
  - 계산된 오프셋을 ROS2 topic '/face/offset'으로 publish
  - Control 노드가 이 topic을 subscribe해서 서보 모터를 제어

아키텍처에서의 위치:
  [웹캠] → [FaceDetectorNode] → /face/offset topic → [ServoControllerNode] → [서보]

입력 파라미터 (ros2 run 실행 시 --ros-args로 지정):
  source      : 'image' | 'video' | 'webcam'  (기본값: 'image')
  file_path   : 이미지 또는 비디오 파일 경로   (source가 image/video일 때 필수)
  publish_rate: topic publish 주기 (Hz)        (기본값: 30.0)
  show_preview: OpenCV 미리보기 창 표시 여부   (기본값: True)
  device_index: 웹캠 장치 번호                 (source=webcam 전용, 기본값: 0)
  width       : 웹캠 캡처 해상도 너비 (px)     (source=webcam 전용, 기본값: 1280)
  height      : 웹캠 캡처 해상도 높이 (px)     (source=webcam 전용, 기본값: 720)
  fps         : 웹캠 목표 프레임레이트          (source=webcam 전용, 기본값: 30)

Phase 1: 정적 이미지/비디오 파일로 테스트 (웹캠 불필요)
Phase 2: 실시간 웹캠 입력으로 전환 (source:=webcam 파라미터만 변경)
"""

import rclpy
from rclpy.node import Node

import cv2
import mediapipe as mp
import numpy as np

from face_tracker_msgs.msg import FaceOffset


class FaceDetectorNode(Node):
    """
    ROS2 노드: 얼굴 감지 및 오프셋 publish

    Node 클래스를 상속받아 ROS2 시스템에 등록되는 노드.
    타이머로 주기적으로 process_frame()을 호출하며,
    감지 결과를 FaceOffset 메시지로 publish한다.
    """

    def __init__(self):
        """
        노드 초기화
        - ROS2 파라미터 선언 및 로드
        - Publisher 생성 (/face/offset topic)
        - MediaPipe 얼굴 감지 모델 초기화
        - 입력 소스(이미지/비디오/웹캠) 초기화
        - 주기적 처리를 위한 타이머 생성
        """
        super().__init__('face_detector_node')

        # ── ROS2 파라미터 선언 ──────────────────────────────────────────
        # 실행 시 --ros-args -p <name>:=<value> 로 외부에서 변경 가능
        self.declare_parameter('source', 'image')       # 입력 소스 종류
        self.declare_parameter('file_path', '')         # 이미지/비디오 파일 경로
        self.declare_parameter('publish_rate', 30.0)    # publish 빈도 (Hz)
        self.declare_parameter('show_preview', True)    # OpenCV 미리보기 창 여부
        # webcam 전용 파라미터
        self.declare_parameter('device_index', 0)       # /dev/videoN 장치 번호
        self.declare_parameter('width', 1280)           # 캡처 해상도 너비
        self.declare_parameter('height', 720)           # 캡처 해상도 높이
        self.declare_parameter('fps', 30)               # 목표 프레임레이트

        self.source = self.get_parameter('source').value
        self.file_path = self.get_parameter('file_path').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.show_preview = self.get_parameter('show_preview').value
        self._device_index = self.get_parameter('device_index').value
        self._width = self.get_parameter('width').value
        self._height = self.get_parameter('height').value
        self._fps = self.get_parameter('fps').value

        # ── ROS2 Publisher 생성 ─────────────────────────────────────────
        # Control 노드가 subscribe할 topic '/face/offset'에 FaceOffset 메시지를 발행
        # queue size 10: 처리가 늦어질 경우 최대 10개 메시지를 버퍼에 보관
        self.publisher = self.create_publisher(FaceOffset, '/face/offset', 10)

        # ── MediaPipe 얼굴 감지 초기화 ──────────────────────────────────
        # MediaPipe FaceDetection: 경량 ML 모델로 실시간 얼굴 감지
        # model_selection=0: 2m 이내 단거리 모델 (웹캠 용도에 적합)
        # model_selection=1: 5m 이내 장거리 모델
        self.mp_face = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils  # 미리보기용 시각화 유틸리티
        self.detector = self.mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6  # 60% 미만 확신도는 감지 무시
        )

        # ── 입력 소스 초기화 ────────────────────────────────────────────
        # source 파라미터에 따라 이미지/비디오/웹캠을 다르게 초기화
        self.cap = None           # VideoCapture 객체 (webcam/video용)
        self.static_frame = None  # 정적 이미지 배열 (image용)
        self._init_source()

        # ── 주기적 타이머 생성 ──────────────────────────────────────────
        # publish_rate Hz로 process_frame()을 반복 호출
        # 예: 30Hz → 1/30초(약 33ms)마다 한 번씩 실행
        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.process_frame
        )

        self.get_logger().info(
            f'FaceDetectorNode started | source={self.source} | rate={self.publish_rate}Hz'
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

        VideoCapture를 image에 쓰지 않는 이유:
          VideoCapture는 일부 시스템에서 JPEG 정적 파일을 읽지 못하는 문제가 있음.
          이미지는 imread()가 더 안정적.
        """
        if self.source == 'webcam':
            self._validate_device_index()
            # Jetson OpenCV는 GStreamer 빌드라 default backend가 CSI(nvarguscamerasrc)부터
            # 시도해 USB 웹캠 열기에 실패함. CAP_V4L2 backend를 명시해야 함.
            self.cap = cv2.VideoCapture(self._device_index, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                raise RuntimeError(f'Cannot open webcam (device {self._device_index})')
            # fourcc를 width/height/fps보다 먼저 설정해야 함.
            # YUYV(기본) 대신 MJPG를 사용해야 USB 2.0 대역폭 제한 내에서 30 fps 가능.
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self.cap.set(cv2.CAP_PROP_FPS, self._fps)
            # 첫 몇 프레임은 하드웨어 초기화 중이라 실패할 수 있으므로 warm-up
            for _ in range(5):
                self.cap.read()
            self.get_logger().info(
                f'Using webcam (device {self._device_index}) '
                f'MJPG {self._width}x{self._height}@{self._fps}fps'
            )

        elif self.source == 'image':
            if not self.file_path:
                raise ValueError('file_path is required for image source')
            # imread: 이미지를 numpy 배열(BGR 포맷)로 메모리에 로드
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

        image/webcam 양쪽 분기에서 공유하는 순수 함수.
        ROS2·FaceOffset 메시지에 의존하지 않으므로 pytest에서 직접 호출 가능.

        Args:
            frame   : BGR numpy 배열
            detector: MediaPipe FaceDetection 인스턴스

        Returns:
            dict with keys: detected(bool), x(float), y(float), confidence(float)
              x: -1.0(좌) ~ +1.0(우), y: -1.0(하) ~ +1.0(상)
        """
        h, w = frame.shape[:2]
        frame_center_x = w / 2
        frame_center_y = h / 2

        # BGR → RGB (MediaPipe 요구 포맷)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        result = {'detected': False, 'x': 0.0, 'y': 0.0, 'confidence': 0.0}

        if results.detections:
            # 여러 얼굴 중 confidence 최고인 것 선택
            best = max(results.detections, key=lambda d: d.score[0])
            bbox = best.location_data.relative_bounding_box

            # bounding box 비율값(0~1) → 픽셀 좌표 변환
            face_cx = (bbox.xmin + bbox.width / 2) * w
            face_cy = (bbox.ymin + bbox.height / 2) * h

            # 화면 중앙 대비 -1.0~+1.0 정규화
            # OpenCV Y축은 아래로 증가하므로 y는 부호 반전해 위쪽을 양수로
            result['x'] = (face_cx - frame_center_x) / frame_center_x
            result['y'] = -((face_cy - frame_center_y) / frame_center_y)
            result['confidence'] = float(best.score[0])
            result['detected'] = True

        return result

    def _acquire_frame(self):
        """
        현재 source 설정에 따라 프레임 한 장을 획득해 반환.
        실패하면 None 반환.
        """
        if self.source == 'image':
            # 정적 이미지: 매번 복사본 사용 (원본 보존)
            return self.static_frame.copy()

        ret, frame = self.cap.read()
        if not ret:
            if self.source == 'video':
                # 비디오 끝에 도달하면 처음으로 되감기 (루프 재생)
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
          3. FaceOffset 메시지 구성 후 /face/offset topic에 publish
          4. (선택) OpenCV 미리보기 창에 결과 시각화
        """
        # ── Step 1: 프레임 획득 ─────────────────────────────────────────
        frame = self._acquire_frame()
        if frame is None:
            self.get_logger().warn('Cannot read frame')
            return

        # ── Step 2: 얼굴 감지 + 오프셋 계산 (공유 함수) ────────────────
        detection = self.detect_face_offset(frame, self.detector)

        # ── Step 3: FaceOffset 메시지 구성 후 publish ───────────────────
        msg = FaceOffset()
        msg.detected = detection['detected']
        msg.x = detection['x']
        msg.y = detection['y']
        msg.confidence = detection['confidence']

        # topic에 메시지 발행 — Control 노드가 이것을 받아 서보를 제어
        self.publisher.publish(msg)

        # ── Step 4: 미리보기 창 (선택) ──────────────────────────────────
        if self.show_preview:
            if msg.detected:
                h, w = frame.shape[:2]
                # 오프셋 역산으로 픽셀 좌표 복원 (detect_face_offset와 역연산)
                face_cx = (msg.x * (w / 2)) + (w / 2)
                face_cy = (-msg.y * (h / 2)) + (h / 2)
                cv2.circle(frame, (int(face_cx), int(face_cy)), 5, (0, 255, 0), -1)
                status = f'x={msg.x:.2f} y={msg.y:.2f} conf={msg.confidence:.2f}'
            else:
                status = 'No face'
            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Face Detector', frame)
            cv2.waitKey(1)

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
    spin()은 Ctrl+C가 눌릴 때까지 타이머/콜백을 계속 실행
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
