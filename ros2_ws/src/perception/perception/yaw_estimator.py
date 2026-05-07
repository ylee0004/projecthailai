"""
yaw_estimator.py — Project Hail AI Phase B
===========================================
MediaPipe Face Mesh 468 landmarks 결과로부터 머리 yaw 각도를 추정한다.

주요 구성:
  - estimate_yaw_from_landmarks(): solvePnP로 yaw_deg 반환
  - YawSmoother: EMA 필터 (alpha 파라미터)
  - classify_direction(): yaw_deg → "LEFT" | "CENTER" | "RIGHT"

부호 컨벤션:
  yaw_deg 음수 = 왼쪽 회전 (얼굴이 카메라 기준 왼쪽을 향함)
  yaw_deg 양수 = 오른쪽 회전
  추출 공식: atan2(R[2,0], R[0,0])  — 좌/우 부호 실측 검증 완료
"""

import math
import numpy as np
import cv2


# ── Canonical face 3D model (mm) ─────────────────────────────────────────────
# 각 점은 상대 인덱스 [0..5] 순서로 사용 landmark에 대응:
#   0: 코 끝(1), 1: 턱 끝(152), 2: 좌 눈꼬리(33), 3: 우 눈꼬리(263),
#   4: 좌 입꼬리(61), 5: 우 입꼬리(291)
_MODEL_POINTS = np.array([
    ( 0.0,    0.0,    0.0),     # 코 끝
    ( 0.0,  -63.6,  -12.5),     # 턱
    (-43.3,   32.7,  -26.0),    # 좌 눈꼬리
    ( 43.3,   32.7,  -26.0),    # 우 눈꼬리
    (-28.9,  -28.9,  -24.1),    # 좌 입꼬리
    ( 28.9,  -28.9,  -24.1),    # 우 입꼬리
], dtype=np.float64)

# 사용할 landmark 인덱스 (MediaPipe Face Mesh 기준)
_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]


def _build_camera_matrix(frame_w: int, frame_h: int) -> np.ndarray:
    """프레임 크기에서 근사 카메라 내부 행렬 생성 (focal = frame_w 가정)."""
    focal = float(frame_w)
    return np.array([
        [focal,     0, frame_w / 2.0],
        [    0, focal, frame_h / 2.0],
        [    0,     0,           1.0],
    ], dtype=np.float64)


def estimate_yaw_from_landmarks(landmarks, frame_w: int, frame_h: int) -> float:
    """
    MediaPipe Face Mesh landmarks로부터 yaw 각도(°)를 추정한다.

    Args:
        landmarks : MediaPipe landmark_list.landmark (sequence, normalized x/y)
                    또는 dict ({int: obj with .x .y .z}, 테스트용).
                    두 형태 모두 normalized 0~1 좌표를 가진다고 가정한다.
        frame_w   : 프레임 너비 (px)
        frame_h   : 프레임 높이 (px)

    Returns:
        yaw_deg: float ([-90, +90] 클램프). 음수=왼쪽, 양수=오른쪽.
        필요한 landmark가 없거나 solvePnP 실패 시 float('nan') 반환.
    """
    image_points = []
    for idx in _LANDMARK_INDICES:
        try:
            lm = landmarks[idx]
            image_points.append([lm.x * frame_w, lm.y * frame_h])
        except (IndexError, KeyError, AttributeError, TypeError):
            return float('nan')

    if len(image_points) != len(_LANDMARK_INDICES):
        return float('nan')

    image_pts = np.array(image_points, dtype=np.float64)
    K = _build_camera_matrix(frame_w, frame_h)
    dist = np.zeros((4, 1), dtype=np.float64)

    success, rvec, _tvec = cv2.solvePnP(
        _MODEL_POINTS, image_pts, K, dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return float('nan')

    R, _ = cv2.Rodrigues(rvec)
    # atan2(R[2,0], R[0,0]) — 실측 검증된 부호 컨벤션
    # 왼쪽 회전 → 음수, 오른쪽 회전 → 양수
    yaw_rad = math.atan2(R[2, 0], R[0, 0])
    yaw_deg = math.degrees(yaw_rad)

    # [-90, +90] 클램프
    yaw_deg = max(-90.0, min(90.0, yaw_deg))
    return yaw_deg


class YawSmoother:
    """
    Exponential Moving Average (EMA) yaw 노이즈 필터.

    update(yaw_raw) 호출마다:
      - 첫 유효값: smoothed = yaw_raw (초기화)
      - 이후 유효값: smoothed = alpha * yaw_raw + (1 - alpha) * smoothed_prev
      - NaN 입력: 마지막 smoothed 유지 (없으면 NaN 반환)
    """

    def __init__(self, alpha: float = 0.3):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f'alpha must be in (0, 1], got {alpha}')
        self._alpha = alpha
        self._smoothed = None  # None = 아직 초기화 안 됨

    def update(self, yaw_raw: float) -> float:
        """
        새 yaw 값으로 EMA 갱신 후 smoothed 값을 반환.
        NaN 입력 시 기존 smoothed 유지.
        """
        if math.isnan(yaw_raw):
            # 초기화 전에도 NaN이 들어오면 NaN 반환
            return float('nan') if self._smoothed is None else self._smoothed

        if self._smoothed is None:
            # 첫 유효 입력: raw 그대로 초기화
            self._smoothed = float(yaw_raw)
        else:
            self._smoothed = self._alpha * yaw_raw + (1.0 - self._alpha) * self._smoothed

        return self._smoothed

    def reset(self):
        """필요 시 EMA 상태 초기화."""
        self._smoothed = None


def classify_direction(yaw_deg: float, threshold_deg: float = 25.0) -> str:
    """
    yaw_deg를 LEFT / CENTER / RIGHT 문자열로 분류한다.

    규칙:
      - NaN  → "CENTER"
      - yaw_deg < -threshold_deg  → "LEFT"
      - yaw_deg > +threshold_deg  → "RIGHT"
      - 그 외 (경계값 ±threshold_deg 포함) → "CENTER"

    Args:
        yaw_deg      : EMA 필터 적용 후 yaw 각도 (float 또는 NaN)
        threshold_deg: 분류 임계값 (기본 25.0°)

    Returns:
        "LEFT" | "CENTER" | "RIGHT"
    """
    if math.isnan(yaw_deg):
        return 'CENTER'
    if yaw_deg < -threshold_deg:
        return 'LEFT'
    if yaw_deg > threshold_deg:
        return 'RIGHT'
    return 'CENTER'
