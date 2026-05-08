# conftest.py — perception 패키지 pytest 설정
#
# ROS2 및 face_tracker_msgs 없이 perception 모듈을 import 할 수 있게
# sys.path와 stub 모듈을 설정한다.
# pytest는 WSL PC에서 실행 (Nano SSH 불필요).

import sys
import os
import types

# ── 1. perception 패키지 경로를 sys.path에 추가 ──────────────────────────────
# 이 파일의 위치: ros2_ws/src/perception/test/
# perception 패키지 위치: ros2_ws/src/perception/
_package_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(_package_root))

# ── 2. ROS2 / face_tracker_msgs stub 생성 ────────────────────────────────────
# ROS2가 설치되지 않은 PC에서 import 오류 없이 테스트 가능하게 최소 stub 제공.
# FaceDetectorNode.__init__은 호출하지 않으므로 rclpy stub은 껍데기만 있으면 됨.

def _make_stub(name):
    """존재하지 않는 모듈을 빈 stub으로 등록"""
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod

# rclpy 계열
if 'rclpy' not in sys.modules:
    rclpy_stub = _make_stub('rclpy')
    rclpy_stub.init = lambda *a, **kw: None
    rclpy_stub.shutdown = lambda *a, **kw: None
    rclpy_stub.spin = lambda *a, **kw: None

    node_mod = _make_stub('rclpy.node')

    class _NodeStub:
        def __init__(self, *a, **kw):
            pass
    node_mod.Node = _NodeStub

# face_tracker_msgs 계열
if 'face_tracker_msgs' not in sys.modules:
    _make_stub('face_tracker_msgs')
    msg_mod = _make_stub('face_tracker_msgs.msg')

    class _FaceOffsetStub:
        detected: bool = False
        x: float = 0.0
        y: float = 0.0
        confidence: float = 0.0
    msg_mod.FaceOffset = _FaceOffsetStub

    class _FaceDirectionStub:
        yaw_deg: float = 0.0
        direction: str = 'CENTER'
        confidence: float = 0.0
        detected: bool = False
        stamp = None
    msg_mod.FaceDirection = _FaceDirectionStub

# sensor_msgs 계열 (CompressedImage — preview publisher용)
if 'sensor_msgs' not in sys.modules:
    _make_stub('sensor_msgs')
    sensor_msg_mod = _make_stub('sensor_msgs.msg')

    class _HeaderStub:
        stamp = None
        frame_id: str = ''

    class _CompressedImageStub:
        def __init__(self):
            self.header = _HeaderStub()
            self.format: str = ''
            self.data: bytes = b''
    sensor_msg_mod.CompressedImage = _CompressedImageStub
