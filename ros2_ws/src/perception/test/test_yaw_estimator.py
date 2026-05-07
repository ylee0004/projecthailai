# test_yaw_estimator.py — Phase B pytest
#
# yaw_estimator 모듈의 세 함수를 검증한다:
#   - estimate_yaw_from_landmarks(): solvePnP 기반 yaw 계산
#   - YawSmoother: EMA 필터 동작
#   - classify_direction(): LEFT/CENTER/RIGHT 분류
#
# 실행: pytest ros2_ws/src/perception/test/ -v

import math
import pytest

# conftest.py가 sys.path를 설정하므로 바로 import 가능
from perception.yaw_estimator import (
    YawSmoother,
    classify_direction,
    estimate_yaw_from_landmarks,
)


# ─────────────────────────────────────────────────────────────────────────────
# 합성 landmark 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

class FakeLandmark:
    """MediaPipe landmark를 흉내 내는 최소 stub (normalized x/y/z)."""

    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


def make_front_landmarks() -> dict:
    """
    정면을 향한 대칭 합성 6점 landmarks.
    solvePnP에서 |yaw| < 5° 가 나와야 한다.
    """
    return {
        1:   FakeLandmark(0.50, 0.50),   # 코 끝 — 중앙
        152: FakeLandmark(0.50, 0.65),   # 턱
        33:  FakeLandmark(0.42, 0.45),   # 좌 눈꼬리
        263: FakeLandmark(0.58, 0.45),   # 우 눈꼬리
        61:  FakeLandmark(0.45, 0.58),   # 좌 입꼬리
        291: FakeLandmark(0.55, 0.58),   # 우 입꼬리
    }


def make_left_turn_landmarks() -> dict:
    """
    왼쪽을 향한 합성 landmarks.
    코/입꼬리를 왼쪽으로 시프트 → yaw < -15° 가 나와야 한다.
    solvePnP 실측: 이 시프트(±0.08)로 약 -17° 추정.
    """
    lm = make_front_landmarks()
    lm[1]   = FakeLandmark(0.42, 0.50)  # 코 — 왼쪽으로
    lm[61]  = FakeLandmark(0.37, 0.58)  # 좌 입꼬리 — 왼쪽으로
    lm[291] = FakeLandmark(0.47, 0.58)  # 우 입꼬리 — 왼쪽으로
    return lm


def make_right_turn_landmarks() -> dict:
    """
    오른쪽을 향한 합성 landmarks (make_left_turn의 좌우 대칭).
    코/입꼬리를 오른쪽으로 시프트 → yaw > +15° 가 나와야 한다.
    solvePnP 실측: 이 시프트(±0.08)로 약 +17° 추정.
    """
    lm = make_front_landmarks()
    lm[1]   = FakeLandmark(0.58, 0.50)  # 코 — 오른쪽으로
    lm[61]  = FakeLandmark(0.53, 0.58)  # 좌 입꼬리 — 오른쪽으로
    lm[291] = FakeLandmark(0.63, 0.58)  # 우 입꼬리 — 오른쪽으로
    return lm


# ─────────────────────────────────────────────────────────────────────────────
# estimate_yaw_from_landmarks() 테스트
# ─────────────────────────────────────────────────────────────────────────────

_FRAME_W = 640
_FRAME_H = 480


def test_yaw_front_face_near_zero():
    """정면 합성 landmarks → |yaw_deg| < 5°"""
    lms = make_front_landmarks()
    yaw = estimate_yaw_from_landmarks(lms, _FRAME_W, _FRAME_H)

    assert not math.isnan(yaw), 'solvePnP 실패 — NaN 반환'
    assert abs(yaw) < 5.0, f'정면에서 |yaw|={abs(yaw):.2f}° — 5° 미만이어야 함'


def test_yaw_left_turn_negative():
    """
    왼쪽 회전 합성 landmarks → yaw_deg < -15°.
    solvePnP 실측: 0.08 normalized 단위 시프트 → 약 -17° 추정.
    """
    lms = make_left_turn_landmarks()
    yaw = estimate_yaw_from_landmarks(lms, _FRAME_W, _FRAME_H)

    assert not math.isnan(yaw), 'solvePnP 실패 — NaN 반환'
    assert yaw < -15.0, f'좌회전에서 yaw={yaw:.2f}° — -15° 미만이어야 함 (음수 = 왼쪽)'


def test_yaw_right_turn_positive():
    """
    오른쪽 회전 합성 landmarks → yaw_deg > +15°.
    solvePnP 실측: 0.08 normalized 단위 시프트 → 약 +17° 추정.
    """
    lms = make_right_turn_landmarks()
    yaw = estimate_yaw_from_landmarks(lms, _FRAME_W, _FRAME_H)

    assert not math.isnan(yaw), 'solvePnP 실패 — NaN 반환'
    assert yaw > 15.0, f'우회전에서 yaw={yaw:.2f}° — +15° 초과이어야 함 (양수 = 오른쪽)'


def test_yaw_empty_landmarks_returns_nan():
    """빈 dict → solvePnP 입력 부족 → NaN 반환"""
    yaw = estimate_yaw_from_landmarks({}, _FRAME_W, _FRAME_H)
    assert math.isnan(yaw)


def test_yaw_partial_landmarks_returns_nan():
    """일부 landmark 누락 (4점만 있음) → NaN 반환"""
    lms = {1: FakeLandmark(0.5, 0.5), 152: FakeLandmark(0.5, 0.65),
           33: FakeLandmark(0.42, 0.45), 263: FakeLandmark(0.58, 0.45)}
    yaw = estimate_yaw_from_landmarks(lms, _FRAME_W, _FRAME_H)
    assert math.isnan(yaw)


def test_yaw_clamped_to_90():
    """
    반환값은 [-90, +90] 클램프 내에 있어야 함.
    정상 합성 케이스 3가지 확인.
    """
    for lm_fn in [make_front_landmarks, make_left_turn_landmarks, make_right_turn_landmarks]:
        yaw = estimate_yaw_from_landmarks(lm_fn(), _FRAME_W, _FRAME_H)
        assert -90.0 <= yaw <= 90.0, f'yaw={yaw}° — [-90, +90] 범위 초과'


# ─────────────────────────────────────────────────────────────────────────────
# YawSmoother 테스트
# ─────────────────────────────────────────────────────────────────────────────

def test_yaw_smoother_first_update_returns_raw():
    """
    첫 번째 유효 update → raw 그대로 반환 (EMA 초기화).
    alpha=0.3: 첫 값은 초기화라 raw와 동일.
    """
    smoother = YawSmoother(alpha=0.3)
    result = smoother.update(10.0)
    assert result == pytest.approx(10.0)


def test_yaw_smoother_second_update_applies_ema():
    """
    두 번째 update: smoothed = 0.3 * 20.0 + 0.7 * 10.0 = 13.0
    """
    smoother = YawSmoother(alpha=0.3)
    smoother.update(10.0)
    result = smoother.update(20.0)
    assert result == pytest.approx(13.0, abs=1e-6)


def test_yaw_smoother_nan_input_preserves_last():
    """
    NaN 입력 → 마지막 smoothed 값 유지 (업데이트 없음).
    """
    smoother = YawSmoother(alpha=0.3)
    smoother.update(10.0)
    result = smoother.update(float('nan'))
    assert result == pytest.approx(10.0)


def test_yaw_smoother_nan_before_init_returns_nan():
    """
    초기화 전 NaN 입력 → NaN 반환 (smoothed 없음).
    """
    smoother = YawSmoother(alpha=0.3)
    result = smoother.update(float('nan'))
    assert math.isnan(result)


def test_yaw_smoother_invalid_alpha_raises():
    """
    alpha <= 0 또는 alpha > 1 이면 ValueError 발생.
    """
    with pytest.raises(ValueError):
        YawSmoother(alpha=0.0)
    with pytest.raises(ValueError):
        YawSmoother(alpha=1.1)


def test_yaw_smoother_reset_clears_state():
    """
    reset() 후 첫 update는 다시 raw 값으로 초기화.
    """
    smoother = YawSmoother(alpha=0.3)
    smoother.update(50.0)
    smoother.reset()
    result = smoother.update(10.0)
    assert result == pytest.approx(10.0)


# ─────────────────────────────────────────────────────────────────────────────
# classify_direction() 테스트
# ─────────────────────────────────────────────────────────────────────────────

def test_classify_negative_30_is_left():
    """-30° → LEFT"""
    assert classify_direction(-30.0) == 'LEFT'


def test_classify_zero_is_center():
    """0° → CENTER"""
    assert classify_direction(0.0) == 'CENTER'


def test_classify_positive_30_is_right():
    """+30° → RIGHT"""
    assert classify_direction(30.0) == 'RIGHT'


def test_classify_boundary_positive_25_is_center():
    """정확히 +25.0° → CENTER (경계값 포함)"""
    assert classify_direction(25.0) == 'CENTER'


def test_classify_boundary_negative_25_is_center():
    """정확히 -25.0° → CENTER (경계값 포함)"""
    assert classify_direction(-25.0) == 'CENTER'


def test_classify_just_above_threshold_is_right():
    """+25.1° → RIGHT (임계값 초과)"""
    assert classify_direction(25.1) == 'RIGHT'


def test_classify_just_below_threshold_is_left():
    """-25.1° → LEFT (임계값 초과)"""
    assert classify_direction(-25.1) == 'LEFT'


def test_classify_nan_is_center():
    """NaN → CENTER"""
    assert classify_direction(float('nan')) == 'CENTER'


def test_classify_custom_threshold():
    """커스텀 임계값 threshold_deg=10.0 적용 검증"""
    assert classify_direction(-15.0, threshold_deg=10.0) == 'LEFT'
    assert classify_direction(5.0, threshold_deg=10.0) == 'CENTER'
    assert classify_direction(10.0, threshold_deg=10.0) == 'CENTER'  # 경계
    assert classify_direction(10.1, threshold_deg=10.0) == 'RIGHT'
