"""
Control Node — Project Hail AI
====================================
이 노드의 역할:
  - /face/offset topic을 subscribe해서 얼굴의 x, y 오프셋을 받아옴
  - PID 컨트롤러로 오프셋을 서보 각도 보정값으로 변환
  - PCA9685 I2C PWM 드라이버를 통해 pan/tilt 서보 모터를 제어
  - 얼굴이 화면 중앙에 오도록 서보를 지속적으로 추적

아키텍처에서의 위치:
  [FaceDetectorNode] → /face/offset topic → [ServoControllerNode] → [PCA9685] → [서보]

PID 제어란:
  P (비례): 현재 오차에 비례해서 보정 → 오차가 클수록 빠르게 반응
  I (적분): 누적 오차를 보정 → 미세한 오차가 계속 남는 것을 방지
  D (미분): 오차 변화율에 반응 → 과도한 진동(오버슈트)을 억제

파라미터 (ros2 run 실행 시 --ros-args로 지정):
  dry_run   : True이면 하드웨어 없이 로그만 출력 (기본값: True)
  pan_kp/ki/kd  : 좌우(pan) 서보 PID 게인값
  tilt_kp/ki/kd : 상하(tilt) 서보 PID 게인값
  deadzone  : 이 값보다 작은 오프셋은 무시 (카메라 노이즈 제거)

Phase 1: dry_run=true — 하드웨어 없이 PID 계산 결과만 로그 출력
Phase 2: dry_run=false — PCA9685를 통해 실제 서보 제어
"""

import rclpy
from rclpy.node import Node
from face_tracker_msgs.msg import FaceOffset


class PID:
    """
    범용 PID 컨트롤러

    PID 제어는 로봇공학에서 가장 널리 쓰이는 제어 알고리즘.
    목표값과 현재값의 차이(오차)를 입력으로 받아
    출력(제어 명령)을 계산한다.

    사용 예:
      pid = PID(kp=50.0, ki=0.1, kd=5.0, output_min=-30.0, output_max=30.0)
      correction = pid.compute(error=0.3, dt=0.033)  # 33ms 주기
    """

    def __init__(self, kp, ki, kd, output_min=-1.0, output_max=1.0):
        """
        Args:
          kp: 비례 게인 — 값이 클수록 오차에 빠르게 반응 (너무 크면 진동)
          ki: 적분 게인 — 누적 오차 제거 (너무 크면 오버슈트)
          kd: 미분 게인 — 진동 억제, 부드러운 움직임
          output_min/max: 출력값 클리핑 범위 (서보 각도 제한)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self._integral = 0.0     # 누적 오차 (I 항)
        self._prev_error = 0.0   # 이전 오차 (D 항 계산용)

    def compute(self, error, dt):
        """
        PID 출력값 계산

        Args:
          error: 현재 오차 (목표값 - 현재값), 여기서는 face offset (x 또는 y)
          dt   : 이전 호출 이후 경과 시간 (초), 적분/미분 계산에 필요

        Returns:
          서보 각도 보정값 (output_min ~ output_max 범위로 클리핑됨)
        """
        # I 항: 오차를 시간에 걸쳐 누적 (지속적인 편향 제거)
        self._integral += error * dt

        # D 항: 오차의 변화율 (급격한 변화에 제동)
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error

        # PID 합산
        output = self.kp * error + self.ki * self._integral + self.kd * derivative

        # 출력 범위 제한 (서보가 물리적으로 움직일 수 있는 범위 초과 방지)
        return max(self.output_min, min(self.output_max, output))

    def reset(self):
        """
        적분항과 이전 오차 초기화
        얼굴이 사라졌을 때 호출 — 재감지 시 이전 누적값이 영향 주지 않도록
        """
        self._integral = 0.0
        self._prev_error = 0.0


class ServoControllerNode(Node):
    """
    ROS2 노드: 서보 모터 제어

    /face/offset topic을 받아 PID 컨트롤러로 서보 각도를 계산하고
    PCA9685 PWM 드라이버를 통해 pan/tilt 서보를 제어한다.

    서보 구성:
      - Pan  서보 (channel 0): 좌우 회전, ±90도
      - Tilt 서보 (channel 1): 상하 회전, ±45도
    """

    # 서보 물리적 회전 한계 (도)
    # 이 범위를 넘으면 서보가 손상될 수 있으므로 소프트웨어적으로 제한
    PAN_MIN = -90.0   # 왼쪽 최대
    PAN_MAX = 90.0    # 오른쪽 최대
    TILT_MIN = -45.0  # 아래쪽 최대
    TILT_MAX = 45.0   # 위쪽 최대

    def __init__(self):
        """
        노드 초기화
        - ROS2 파라미터 선언 및 로드
        - Pan/Tilt PID 컨트롤러 생성
        - 하드웨어(PCA9685) 초기화 (dry_run=false일 때만)
        - /face/offset topic subscriber 생성
        """
        super().__init__('servo_controller_node')

        # ── ROS2 파라미터 선언 ──────────────────────────────────────────
        self.declare_parameter('dry_run', True)     # 하드웨어 없이 테스트할 때 True
        self.declare_parameter('pan_kp', 50.0)      # Pan PID 비례 게인
        self.declare_parameter('pan_ki', 0.1)       # Pan PID 적분 게인
        self.declare_parameter('pan_kd', 5.0)       # Pan PID 미분 게인
        self.declare_parameter('tilt_kp', 40.0)     # Tilt PID 비례 게인 (Pan보다 작게)
        self.declare_parameter('tilt_ki', 0.1)      # Tilt PID 적분 게인
        self.declare_parameter('tilt_kd', 5.0)      # Tilt PID 미분 게인
        self.declare_parameter('deadzone', 0.05)    # 무시할 최소 오프셋 (5%)

        self.dry_run = self.get_parameter('dry_run').value
        self.deadzone = self.get_parameter('deadzone').value

        # ── PID 컨트롤러 생성 ───────────────────────────────────────────
        # output_min/max: 한 번의 제어 주기에 최대로 움직일 수 있는 각도 (도)
        # 너무 크면 서보가 급격하게 움직여 불안정해짐
        self.pan_pid = PID(
            self.get_parameter('pan_kp').value,
            self.get_parameter('pan_ki').value,
            self.get_parameter('pan_kd').value,
            output_min=-30.0, output_max=30.0  # 한 번에 최대 ±30도 이동
        )
        self.tilt_pid = PID(
            self.get_parameter('tilt_kp').value,
            self.get_parameter('tilt_ki').value,
            self.get_parameter('tilt_kd').value,
            output_min=-20.0, output_max=20.0  # Tilt는 범위가 좁으므로 ±20도로 제한
        )

        # 현재 서보 각도 (0 = 중앙)
        self.pan_angle = 0.0   # 좌우 현재 위치
        self.tilt_angle = 0.0  # 상하 현재 위치

        # ── 하드웨어 초기화 ─────────────────────────────────────────────
        self.pca = None  # PCA9685 드라이버 객체
        if not self.dry_run:
            self._init_hardware()

        # ── ROS2 Subscriber 생성 ────────────────────────────────────────
        # FaceDetectorNode가 publish하는 /face/offset을 수신
        # 메시지가 도착할 때마다 face_offset_callback() 호출
        self.subscription = self.create_subscription(
            FaceOffset,
            '/face/offset',
            self.face_offset_callback,
            10  # queue size
        )

        # 마지막 콜백 시각 (dt 계산용)
        self._last_time = self.get_clock().now()

        mode = 'DRY RUN (no hardware)' if self.dry_run else 'HARDWARE MODE'
        self.get_logger().info(f'ServoControllerNode started | {mode}')

    def _init_hardware(self):
        """
        PCA9685 I2C PWM 드라이버 초기화

        PCA9685란:
          - I2C 통신으로 제어하는 16채널 PWM 드라이버 보드
          - Jetson Nano의 GPIO에서 직접 여러 서보를 제어하기 어려워 사용
          - I2C 주소: 0x40 (기본값)
          - Jetson Nano I2C 핀: SDA=핀3, SCL=핀5

        초기화 실패 시 자동으로 dry_run 모드로 전환 (안전장치)
        """
        try:
            from adafruit_pca9685 import PCA9685
            import board
            import busio

            # I2C 버스 연결 (Jetson Nano 핀 3, 5)
            i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(i2c)

            # 서보 모터 표준 주파수: 50Hz (20ms 주기)
            # SG90 서보: 1ms 펄스 = 0도, 2ms 펄스 = 180도
            self.pca.frequency = 50
            self.get_logger().info('PCA9685 initialized on I2C')

        except Exception as e:
            self.get_logger().error(f'Hardware init failed: {e}')
            self.get_logger().warn('Falling back to dry run mode')
            self.dry_run = True

    def face_offset_callback(self, msg: FaceOffset):
        """
        /face/offset 메시지 수신 콜백

        FaceDetectorNode로부터 얼굴 위치 오프셋을 받아 서보를 제어.
        얼굴이 감지되지 않으면 서보를 현재 위치에 고정하고 PID를 리셋.

        Args:
          msg: FaceOffset 메시지
               msg.x        — 좌우 오프셋 (-1.0 ~ 1.0)
               msg.y        — 상하 오프셋 (-1.0 ~ 1.0)
               msg.detected — 얼굴 감지 여부
        """
        if not msg.detected:
            # 얼굴이 없으면 서보 유지 + PID 누적값 초기화
            # 리셋하지 않으면 재감지 시 누적된 적분값 때문에 서보가 급격히 움직임
            self.pan_pid.reset()
            self.tilt_pid.reset()
            return

        # ── dt 계산 ─────────────────────────────────────────────────────
        # 이전 콜백 이후 경과 시간 (초) — PID 적분/미분 계산에 필요
        now = self.get_clock().now()
        dt = (now - self._last_time).nanoseconds / 1e9
        self._last_time = now

        # ── 데드존 적용 ─────────────────────────────────────────────────
        # 카메라 노이즈, 미세 떨림으로 인한 작은 오프셋은 무시
        # deadzone=0.05: 화면의 5% 이내 움직임은 무반응
        x_error = msg.x if abs(msg.x) > self.deadzone else 0.0
        y_error = msg.y if abs(msg.y) > self.deadzone else 0.0

        # ── PID 계산 ────────────────────────────────────────────────────
        # 오프셋(오차)을 서보 각도 보정값으로 변환
        # x_error > 0 (얼굴이 오른쪽) → pan_delta > 0 → 카메라를 오른쪽으로 회전
        pan_delta = self.pan_pid.compute(x_error, dt)
        tilt_delta = self.tilt_pid.compute(y_error, dt)

        # ── 서보 각도 업데이트 ──────────────────────────────────────────
        # 현재 각도에 보정값을 더하되, 물리적 한계 범위로 클리핑
        self.pan_angle = max(self.PAN_MIN, min(self.PAN_MAX, self.pan_angle + pan_delta))
        self.tilt_angle = max(self.TILT_MIN, min(self.TILT_MAX, self.tilt_angle + tilt_delta))

        self._apply_servos()

        self.get_logger().debug(
            f'face=({msg.x:.2f},{msg.y:.2f}) '
            f'pan={self.pan_angle:.1f}° tilt={self.tilt_angle:.1f}°'
        )

    def _apply_servos(self):
        """
        계산된 각도를 실제 서보에 적용

        dry_run=True: 로그만 출력 (하드웨어 불필요, 테스트용)
        dry_run=False: PCA9685 채널에 PWM 듀티사이클 값 전송

        서보 채널 배치:
          channel 0 → Pan  서보 (좌우)
          channel 1 → Tilt 서보 (상하)
        """
        if self.dry_run:
            # 하드웨어 없이 계산 결과만 출력
            self.get_logger().info(
                f'[DRY RUN] pan={self.pan_angle:.1f}° tilt={self.tilt_angle:.1f}°'
            )
            return

        # 우리 코드의 각도 범위: -90~90도 (0 = 중앙)
        # PCA9685 각도 입력: 0~180도 (90 = 중앙)
        # → +90 오프셋으로 변환
        pan_duty = self._angle_to_duty(self.pan_angle + 90)
        tilt_duty = self._angle_to_duty(self.tilt_angle + 90)

        if self.pca:
            self.pca.channels[0].duty_cycle = pan_duty   # Pan 서보
            self.pca.channels[1].duty_cycle = tilt_duty  # Tilt 서보

    @staticmethod
    def _angle_to_duty(angle_deg, min_pulse_ms=1.0, max_pulse_ms=2.0, period_ms=20.0):
        """
        서보 각도(도)를 PCA9685 16비트 듀티사이클 값으로 변환

        서보 모터 PWM 신호 규격 (SG90 기준):
          - 주기: 20ms (50Hz)
          - 0도  → 펄스 1.0ms → 듀티사이클 5%
          - 90도 → 펄스 1.5ms → 듀티사이클 7.5%
          - 180도 → 펄스 2.0ms → 듀티사이클 10%

        PCA9685는 16비트(0~65535)로 듀티사이클을 표현:
          duty = (펄스시간 / 주기) × 65535

        Args:
          angle_deg    : 서보 각도 (0 ~ 180도)
          min_pulse_ms : 0도에 해당하는 펄스 폭 (기본 1.0ms)
          max_pulse_ms : 180도에 해당하는 펄스 폭 (기본 2.0ms)
          period_ms    : PWM 주기 (기본 20ms = 50Hz)

        Returns:
          PCA9685에 전달할 16비트 듀티사이클 값 (0 ~ 65535)
        """
        pulse_ms = min_pulse_ms + (angle_deg / 180.0) * (max_pulse_ms - min_pulse_ms)
        duty = int((pulse_ms / period_ms) * 65535)
        return max(0, min(65535, duty))  # 0~65535 범위 클리핑

    def destroy_node(self):
        """
        노드 종료 시 하드웨어 리소스 해제
        PCA9685 I2C 연결을 정상적으로 닫아 다음 실행 시 충돌 방지
        """
        if self.pca:
            self.pca.deinit()
        super().destroy_node()


def main(args=None):
    """
    노드 진입점
    rclpy 초기화 → 노드 생성 → spin(이벤트 루프) → 종료 처리
    spin()은 Ctrl+C가 눌릴 때까지 subscriber 콜백을 계속 대기
    """
    rclpy.init(args=args)
    node = ServoControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
