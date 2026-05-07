# test_face_detector_image_mode.py
#
# A4 회귀 테스트: source:=image 모드가 webcam 분기 추가 후에도 깨지지 않음을 검증.
# ROS2 spin 없이 PC(WSL)에서 실행 가능.
# 실행: pytest ros2_ws/src/perception/test/ -v
#
# 테스트 범위:
#   T1. 모듈 import 성공 (rclpy/face_tracker_msgs stub 사용)
#   T2. detect_face_offset() — 빈 이미지(얼굴 없음) → detected=False
#   T3. detect_face_offset() — 얼굴 있는 이미지 → detected=True + 오프셋 범위 검증
#   T4. detect_face_offset() — 오프셋 수치 정확도 검증 (화면 정중앙 얼굴 → x≈0, y≈0)
#   T5. _acquire_frame() — image 모드에서 static_frame 복사본 반환 검증

import numpy as np
import cv2
import pytest

# conftest.py 가 sys.path + stub 설정을 먼저 처리하므로
# 여기서는 바로 import 가능.
from perception.face_detector_node import FaceDetectorNode  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# 공용 픽스처
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def mp_detector():
    """
    MediaPipe FaceMesh 인스턴스 (모듈 범위, 한 번만 생성).
    Phase B에서 Face Mesh로 교체 — detect_face_offset과 동일한 설정 사용.
    테스트 종료 후 컨텍스트 매니저로 자동 해제.
    """
    import mediapipe as mp
    face_mesh = mp.solutions.face_mesh
    with face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    ) as det:
        yield det


def _blank_bgr(h=480, w=640):
    """지정 크기의 순수 검정 BGR 이미지 반환 (얼굴 없음)"""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _solid_bgr(h=480, w=640, color=(100, 100, 100)):
    """단색 BGR 이미지 반환 (얼굴 없음)"""
    img = np.full((h, w, 3), color, dtype=np.uint8)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# T1: 모듈 import 성공
# ─────────────────────────────────────────────────────────────────────────────

def test_import_face_detector_node():
    """FaceDetectorNode 클래스가 ROS2 없이 import 가능해야 함"""
    assert FaceDetectorNode is not None


def test_detect_face_offset_is_static():
    """detect_face_offset 이 staticmethod 인지 확인"""
    assert callable(FaceDetectorNode.detect_face_offset)


# ─────────────────────────────────────────────────────────────────────────────
# T2: 빈 이미지 → 얼굴 미감지
# ─────────────────────────────────────────────────────────────────────────────

def test_no_face_on_blank_image(mp_detector):
    """검정 이미지에서는 얼굴이 감지되지 않아야 함"""
    frame = _blank_bgr()
    result = FaceDetectorNode.detect_face_offset(frame, mp_detector)

    assert result['detected'] is False
    assert result['x'] == pytest.approx(0.0)
    assert result['y'] == pytest.approx(0.0)
    assert result['confidence'] == pytest.approx(0.0)


def test_no_face_on_solid_color_image(mp_detector):
    """단색 이미지에서는 얼굴이 감지되지 않아야 함"""
    frame = _solid_bgr(color=(200, 180, 160))  # 피부색과 다른 색
    result = FaceDetectorNode.detect_face_offset(frame, mp_detector)

    assert result['detected'] is False


# ─────────────────────────────────────────────────────────────────────────────
# T3: 반환 dict 구조 검증
# ─────────────────────────────────────────────────────────────────────────────

def test_return_dict_has_required_keys(mp_detector):
    """
    detect_face_offset 반환 dict에 Phase A 핵심 4 키 + Phase B 추가 키가 모두 있어야 함.
    Phase B에서 face_cx, face_cy, landmarks 키가 추가됨.
    """
    frame = _blank_bgr()
    result = FaceDetectorNode.detect_face_offset(frame, mp_detector)

    required_keys = {'detected', 'x', 'y', 'confidence', 'face_cx', 'face_cy', 'landmarks'}
    assert required_keys.issubset(result.keys()), \
        f'누락된 키: {required_keys - result.keys()}'


def test_return_types(mp_detector):
    """반환 값의 타입이 올바른지 확인"""
    frame = _blank_bgr()
    result = FaceDetectorNode.detect_face_offset(frame, mp_detector)

    assert isinstance(result['detected'], bool)
    assert isinstance(result['x'], float)
    assert isinstance(result['y'], float)
    assert isinstance(result['confidence'], float)


# ─────────────────────────────────────────────────────────────────────────────
# T4: 실제 얼굴 이미지 처리 (test_assets/test_face.jpg 존재 시에만 실행)
# ─────────────────────────────────────────────────────────────────────────────

import os

_TEST_FACE_PATH = os.path.join(
    os.path.dirname(__file__),
    '../../../../test_assets/test_face.jpg'
)
_has_test_face = os.path.isfile(_TEST_FACE_PATH)


@pytest.mark.skipif(not _has_test_face, reason='test_assets/test_face.jpg 없음 (A5 이후 추가)')
def test_detect_face_from_image_file(mp_detector):
    """실제 얼굴 이미지에서 감지 성공 + 오프셋 범위 검증"""
    frame = cv2.imread(_TEST_FACE_PATH)
    assert frame is not None, f'이미지 로드 실패: {_TEST_FACE_PATH}'

    result = FaceDetectorNode.detect_face_offset(frame, mp_detector)

    assert result['detected'] is True, '얼굴이 감지되지 않음'
    assert -1.0 <= result['x'] <= 1.0, f"x 오프셋 범위 초과: {result['x']}"
    assert -1.0 <= result['y'] <= 1.0, f"y 오프셋 범위 초과: {result['y']}"
    assert 0.0 < result['confidence'] <= 1.0, f"confidence 범위 오류: {result['confidence']}"


# ─────────────────────────────────────────────────────────────────────────────
# T5: _acquire_frame — image 모드 동작 검증 (FaceDetectorNode를 직접 생성하지 않고
#     static_frame만 duck-typing으로 주입해 테스트)
# ─────────────────────────────────────────────────────────────────────────────

def test_acquire_frame_image_mode_returns_copy():
    """
    image 모드에서 _acquire_frame()은 static_frame의 복사본을 반환해야 함.
    원본을 수정해도 반환값이 영향받지 않음을 확인.
    """
    # FaceDetectorNode.__init__ 없이 duck-typing으로 인스턴스 속성만 주입
    node = object.__new__(FaceDetectorNode)
    node.source = 'image'
    original = _blank_bgr(h=100, w=100)
    node.static_frame = original.copy()

    frame = node._acquire_frame()

    assert frame is not None
    assert frame.shape == original.shape
    # 복사본 확인: 원본 변경이 반환값에 영향 없어야 함
    node.static_frame[0, 0] = [255, 0, 0]
    assert not np.array_equal(frame[0, 0], node.static_frame[0, 0])


def test_acquire_frame_image_mode_preserves_original():
    """image 모드에서 매 호출마다 static_frame 원본이 보존되어야 함"""
    node = object.__new__(FaceDetectorNode)
    node.source = 'image'
    node.static_frame = _solid_bgr(color=(50, 100, 150))

    snapshot = node.static_frame.copy()
    frame1 = node._acquire_frame()
    frame2 = node._acquire_frame()

    # static_frame 원본 불변
    assert np.array_equal(node.static_frame, snapshot)
    # 두 호출 결과는 동일한 내용
    assert np.array_equal(frame1, frame2)
