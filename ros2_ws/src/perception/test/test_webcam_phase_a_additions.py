# test_webcam_phase_a_additions.py
#
# Phase A 보강 테스트 — QA 추가분
# Review agent 권장 항목 기반 PC(WSL) pytest 실행 가능 테스트.
#
# 실행: pytest ros2_ws/src/perception/test/ -v
#
# 테스트 범위:
#   TC-A1.  MediaPipe smoke test — PC(WSL) 환경에서 FaceDetection 인스턴스 생성 가능 확인
#   TC-A2.  device_index 경계값 — 음수 입력 시 ValueError 발생 확인
#   TC-A3.  device_index 경계값 — 문자열 입력 시 TypeError 발생 확인
#   TC-A4.  device_index 경계값 — 0은 정상 (ValueError 없음)
#   TC-A5.  detect_face_offset Y축 부호 — 화면 상단 얼굴: y > 0 (mock)
#   TC-A6.  detect_face_offset Y축 부호 — 화면 하단 얼굴: y < 0 (mock)
#   TC-A7.  detect_face_offset Y축 부호 — 화면 좌측 얼굴: x < 0 (mock)
#   TC-A8.  detect_face_offset Y축 부호 — 화면 우측 얼굴: x > 0 (mock)
#   TC-A9.  image 모드 회귀 — file 없을 때 RuntimeError 발생
#   TC-A10. image 모드 회귀 — file 있을 때 frame 정상 반환
#   TC-A11. _acquire_frame webcam 분기 — cap.read() 성공 시 frame 반환 (mock)
#   TC-A12. _acquire_frame webcam 분기 — cap.read() 실패 시 None 반환 (mock)
#   TC-A13. _acquire_frame video 분기 — 비디오 끝에서 seek 후 재시도 (mock)

import numpy as np
import cv2
import pytest
import types
from unittest.mock import MagicMock, patch

# conftest.py가 sys.path + stub 설정을 먼저 처리
from perception.face_detector_node import FaceDetectorNode


# ─────────────────────────────────────────────────────────────────────────────
# 공용 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _blank_bgr(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_mock_detection(rel_xmin, rel_ymin, rel_w, rel_h, score=0.9):
    """
    MediaPipe detection 결과를 흉내 내는 Mock 객체 생성.

    rel_xmin, rel_ymin, rel_w, rel_h : 0~1 비율값 (relative bounding box)
    반환 객체: results.detections 리스트 원소와 동일한 인터페이스.
    """
    bbox = MagicMock()
    bbox.xmin = rel_xmin
    bbox.ymin = rel_ymin
    bbox.width = rel_w
    bbox.height = rel_h

    detection = MagicMock()
    detection.score = [score]
    detection.location_data.relative_bounding_box = bbox

    results = MagicMock()
    results.detections = [detection]
    return results


# ─────────────────────────────────────────────────────────────────────────────
# TC-A1: MediaPipe smoke test
# ─────────────────────────────────────────────────────────────────────────────

def test_mediapipe_smoke_test_on_wsl_pc():
    """
    PC(WSL) 환경에서 MediaPipe FaceDetection 인스턴스가 정상 생성되는지 확인.
    ARM64 wheel 이슈와 분리해 PC 환경 이상 여부를 탐지.
    """
    import mediapipe as mp
    det = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    )
    assert det is not None
    det.close()


# ─────────────────────────────────────────────────────────────────────────────
# TC-A2 ~ TC-A4: device_index 경계값 검증
# ─────────────────────────────────────────────────────────────────────────────

def _node_with_device_index(device_index):
    """
    FaceDetectorNode.__init__ 없이 duck-typing으로 속성만 주입해
    _init_source의 webcam 분기를 검증하는 보조 함수.
    실제 VideoCapture 호출은 patch로 차단.
    """
    node = object.__new__(FaceDetectorNode)
    node.source = 'webcam'
    node._device_index = device_index
    node._width = 640
    node._height = 480
    node._fps = 30
    return node


def test_device_index_negative_raises_value_error():
    """
    device_index가 음수이면 ValueError를 발생시켜야 함.
    음수 장치 인덱스는 V4L2에서 미정의 동작을 유발하므로 사전 차단.
    """
    node = _node_with_device_index(-1)
    with pytest.raises(ValueError, match='device_index'):
        # VideoCapture 실제 호출 없이 검증 가능하도록 _init_source 내 guard에 의존
        # guard가 없으면 이 테스트가 실패 → Implementation에 guard 추가 요청
        node._validate_device_index()


def test_device_index_string_raises_type_error():
    """
    device_index가 문자열이면 TypeError를 발생시켜야 함.
    """
    node = _node_with_device_index("video0")
    with pytest.raises(TypeError, match='device_index'):
        node._validate_device_index()


def test_device_index_zero_is_valid():
    """
    device_index=0 은 정상 값 — 예외 없이 통과해야 함.
    """
    node = _node_with_device_index(0)
    # _validate_device_index 가 없으면 AttributeError → guard 추가 필요 신호
    node._validate_device_index()  # 예외 없음 확인


# ─────────────────────────────────────────────────────────────────────────────
# TC-A5 ~ TC-A8: detect_face_offset Y/X 부호 정확성 (Mock detection)
# ─────────────────────────────────────────────────────────────────────────────

def test_detect_face_offset_y_positive_when_face_at_top():
    """
    얼굴이 화면 상단에 위치하면 y > 0 이어야 함.
    (OpenCV Y축은 아래 방향 증가 → 상단 위치 = face_cy < frame_center_y → y 부호 반전 → 양수)
    """
    frame = _blank_bgr(h=480, w=640)
    # 상단: rel_ymin=0.0, rel_h=0.1 → face_cy = 0.05 * 480 = 24 < 240
    mock_results = _make_mock_detection(rel_xmin=0.45, rel_ymin=0.0, rel_w=0.1, rel_h=0.1)

    mock_detector = MagicMock()
    mock_detector.process.return_value = mock_results

    result = FaceDetectorNode.detect_face_offset(frame, mock_detector)

    assert result['detected'] is True
    assert result['y'] > 0, f'상단 얼굴에서 y가 양수여야 하는데 y={result["y"]}'


def test_detect_face_offset_y_negative_when_face_at_bottom():
    """
    얼굴이 화면 하단에 위치하면 y < 0 이어야 함.
    face_cy > frame_center_y → 부호 반전 → 음수
    """
    frame = _blank_bgr(h=480, w=640)
    # 하단: rel_ymin=0.9, rel_h=0.1 → face_cy = 0.95 * 480 = 456 > 240
    mock_results = _make_mock_detection(rel_xmin=0.45, rel_ymin=0.9, rel_w=0.1, rel_h=0.1)

    mock_detector = MagicMock()
    mock_detector.process.return_value = mock_results

    result = FaceDetectorNode.detect_face_offset(frame, mock_detector)

    assert result['detected'] is True
    assert result['y'] < 0, f'하단 얼굴에서 y가 음수여야 하는데 y={result["y"]}'


def test_detect_face_offset_x_negative_when_face_at_left():
    """
    얼굴이 화면 좌측에 위치하면 x < 0 이어야 함.
    face_cx < frame_center_x → x 음수
    """
    frame = _blank_bgr(h=480, w=640)
    # 좌측: rel_xmin=0.0, rel_w=0.1 → face_cx = 0.05 * 640 = 32 < 320
    mock_results = _make_mock_detection(rel_xmin=0.0, rel_ymin=0.45, rel_w=0.1, rel_h=0.1)

    mock_detector = MagicMock()
    mock_detector.process.return_value = mock_results

    result = FaceDetectorNode.detect_face_offset(frame, mock_detector)

    assert result['detected'] is True
    assert result['x'] < 0, f'좌측 얼굴에서 x가 음수여야 하는데 x={result["x"]}'


def test_detect_face_offset_x_positive_when_face_at_right():
    """
    얼굴이 화면 우측에 위치하면 x > 0 이어야 함.
    face_cx > frame_center_x → x 양수
    """
    frame = _blank_bgr(h=480, w=640)
    # 우측: rel_xmin=0.9, rel_w=0.1 → face_cx = 0.95 * 640 = 608 > 320
    mock_results = _make_mock_detection(rel_xmin=0.9, rel_ymin=0.45, rel_w=0.1, rel_h=0.1)

    mock_detector = MagicMock()
    mock_detector.process.return_value = mock_results

    result = FaceDetectorNode.detect_face_offset(frame, mock_detector)

    assert result['detected'] is True
    assert result['x'] > 0, f'우측 얼굴에서 x가 양수여야 하는데 x={result["x"]}'


# ─────────────────────────────────────────────────────────────────────────────
# TC-A9 ~ TC-A10: image 모드 회귀 — file 없을 때/있을 때
# ─────────────────────────────────────────────────────────────────────────────

def test_init_source_image_missing_file_raises_runtime_error(tmp_path):
    """
    image 모드에서 존재하지 않는 파일 경로를 주면 RuntimeError가 발생해야 함.
    (cv2.imread가 None 반환 → _init_source에서 RuntimeError)
    """
    node = object.__new__(FaceDetectorNode)
    node.source = 'image'
    node.file_path = str(tmp_path / 'nonexistent.jpg')
    node.cap = None
    node.static_frame = None

    # get_logger stub
    logger_stub = MagicMock()
    node.get_logger = lambda: logger_stub

    with pytest.raises(RuntimeError, match='Cannot read image'):
        node._init_source()


def test_init_source_image_existing_file_loads_frame(tmp_path):
    """
    image 모드에서 유효한 이미지 파일을 주면 static_frame에 numpy 배열이 로드되어야 함.
    """
    # 1x1 흰색 BGR JPEG 생성
    img_path = tmp_path / 'test_img.png'
    cv2.imwrite(str(img_path), np.ones((10, 10, 3), dtype=np.uint8) * 200)

    node = object.__new__(FaceDetectorNode)
    node.source = 'image'
    node.file_path = str(img_path)
    node.cap = None
    node.static_frame = None

    logger_stub = MagicMock()
    node.get_logger = lambda: logger_stub

    node._init_source()

    assert node.static_frame is not None
    assert isinstance(node.static_frame, np.ndarray)
    assert node.static_frame.shape == (10, 10, 3)


def test_init_source_image_empty_file_path_raises_value_error():
    """
    image 모드에서 file_path가 비어있으면 ValueError가 발생해야 함.
    """
    node = object.__new__(FaceDetectorNode)
    node.source = 'image'
    node.file_path = ''
    node.cap = None
    node.static_frame = None

    logger_stub = MagicMock()
    node.get_logger = lambda: logger_stub

    with pytest.raises(ValueError, match='file_path'):
        node._init_source()


# ─────────────────────────────────────────────────────────────────────────────
# TC-A11 ~ TC-A13: _acquire_frame webcam/video 분기 (mock cap)
# ─────────────────────────────────────────────────────────────────────────────

def test_acquire_frame_webcam_read_success_returns_frame():
    """
    webcam 분기: cap.read()가 성공하면 frame을 반환해야 함.
    """
    node = object.__new__(FaceDetectorNode)
    node.source = 'webcam'

    mock_frame = _blank_bgr()
    mock_cap = MagicMock()
    mock_cap.read.return_value = (True, mock_frame)
    node.cap = mock_cap

    result = node._acquire_frame()

    assert result is not None
    assert np.array_equal(result, mock_frame)


def test_acquire_frame_webcam_read_fail_returns_none():
    """
    webcam 분기: cap.read()가 실패(ret=False)하면 None을 반환해야 함.
    """
    node = object.__new__(FaceDetectorNode)
    node.source = 'webcam'

    mock_cap = MagicMock()
    mock_cap.read.return_value = (False, None)
    node.cap = mock_cap

    result = node._acquire_frame()

    assert result is None


def test_acquire_frame_video_seeks_to_start_on_eof():
    """
    video 분기: 첫 번째 cap.read()가 실패(비디오 끝)이면
    CAP_PROP_POS_FRAMES=0으로 seek 후 재시도해야 함.
    두 번째 read가 성공하면 frame을 반환.
    """
    node = object.__new__(FaceDetectorNode)
    node.source = 'video'

    mock_frame = _blank_bgr()
    mock_cap = MagicMock()
    # 첫 번째 read: EOF, 두 번째 read: 성공
    mock_cap.read.side_effect = [(False, None), (True, mock_frame)]
    node.cap = mock_cap

    result = node._acquire_frame()

    assert result is not None
    assert np.array_equal(result, mock_frame)
    # seek 호출 확인
    mock_cap.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 0)
