"""
Control Node — Project Hail AI
Subscribes to /face/offset and drives pan/tilt servos via PCA9685 (I2C).

PID controller converts face offset → servo angle correction.
When no face is detected, servos hold last known position.

Phase 1: dry-run mode (no hardware) — logs commands only
Phase 2: real servo control via PCA9685
"""

import rclpy
from rclpy.node import Node
from face_tracker_msgs.msg import FaceOffset


class PID:
    """Simple PID controller."""

    def __init__(self, kp, ki, kd, output_min=-1.0, output_max=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self._integral = 0.0
        self._prev_error = 0.0

    def compute(self, error, dt):
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(self.output_min, min(self.output_max, output))

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0


class ServoControllerNode(Node):

    # Servo angle limits (degrees)
    PAN_MIN = -90.0
    PAN_MAX = 90.0
    TILT_MIN = -45.0
    TILT_MAX = 45.0

    def __init__(self):
        super().__init__('servo_controller_node')

        # Parameters
        self.declare_parameter('dry_run', True)    # True = no hardware, log only
        self.declare_parameter('pan_kp', 50.0)
        self.declare_parameter('pan_ki', 0.1)
        self.declare_parameter('pan_kd', 5.0)
        self.declare_parameter('tilt_kp', 40.0)
        self.declare_parameter('tilt_ki', 0.1)
        self.declare_parameter('tilt_kd', 5.0)
        self.declare_parameter('deadzone', 0.05)   # ignore offset smaller than this

        self.dry_run = self.get_parameter('dry_run').value
        self.deadzone = self.get_parameter('deadzone').value

        # PID controllers (output = degrees to move)
        self.pan_pid = PID(
            self.get_parameter('pan_kp').value,
            self.get_parameter('pan_ki').value,
            self.get_parameter('pan_kd').value,
            output_min=-30.0, output_max=30.0
        )
        self.tilt_pid = PID(
            self.get_parameter('tilt_kp').value,
            self.get_parameter('tilt_ki').value,
            self.get_parameter('tilt_kd').value,
            output_min=-20.0, output_max=20.0
        )

        # Current servo positions (degrees, 0 = center)
        self.pan_angle = 0.0
        self.tilt_angle = 0.0

        # Hardware init
        self.pca = None
        if not self.dry_run:
            self._init_hardware()

        # Subscriber
        self.subscription = self.create_subscription(
            FaceOffset,
            '/face/offset',
            self.face_offset_callback,
            10
        )

        self._last_time = self.get_clock().now()

        mode = 'DRY RUN (no hardware)' if self.dry_run else 'HARDWARE MODE'
        self.get_logger().info(f'ServoControllerNode started | {mode}')

    def _init_hardware(self):
        try:
            from adafruit_pca9685 import PCA9685
            import board
            import busio
            i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(i2c)
            self.pca.frequency = 50  # 50Hz for servos
            self.get_logger().info('PCA9685 initialized on I2C')
        except Exception as e:
            self.get_logger().error(f'Hardware init failed: {e}')
            self.get_logger().warn('Falling back to dry run mode')
            self.dry_run = True

    def face_offset_callback(self, msg: FaceOffset):
        if not msg.detected:
            # No face — hold position, reset integrators
            self.pan_pid.reset()
            self.tilt_pid.reset()
            return

        now = self.get_clock().now()
        dt = (now - self._last_time).nanoseconds / 1e9
        self._last_time = now

        # Apply deadzone — ignore tiny offsets (camera noise)
        x_error = msg.x if abs(msg.x) > self.deadzone else 0.0
        y_error = msg.y if abs(msg.y) > self.deadzone else 0.0

        # PID: error is face offset, output is angle correction
        pan_delta = self.pan_pid.compute(x_error, dt)
        tilt_delta = self.tilt_pid.compute(y_error, dt)

        # Update angles with limits
        self.pan_angle = max(self.PAN_MIN, min(self.PAN_MAX, self.pan_angle + pan_delta))
        self.tilt_angle = max(self.TILT_MIN, min(self.TILT_MAX, self.tilt_angle + tilt_delta))

        self._apply_servos()

        self.get_logger().debug(
            f'face=({msg.x:.2f},{msg.y:.2f}) '
            f'pan={self.pan_angle:.1f}° tilt={self.tilt_angle:.1f}°'
        )

    def _apply_servos(self):
        if self.dry_run:
            self.get_logger().info(
                f'[DRY RUN] pan={self.pan_angle:.1f}° tilt={self.tilt_angle:.1f}°'
            )
            return

        # Convert angle to PCA9685 duty cycle
        # SG90: 1ms (0°) to 2ms (180°) pulse, at 50Hz (20ms period)
        # duty = angle_to_duty(angle, min_pulse=1.0, max_pulse=2.0, period=20.0)
        pan_duty = self._angle_to_duty(self.pan_angle + 90)   # shift 0° center to 90°
        tilt_duty = self._angle_to_duty(self.tilt_angle + 90)

        if self.pca:
            self.pca.channels[0].duty_cycle = pan_duty   # channel 0 = pan
            self.pca.channels[1].duty_cycle = tilt_duty  # channel 1 = tilt

    @staticmethod
    def _angle_to_duty(angle_deg, min_pulse_ms=1.0, max_pulse_ms=2.0, period_ms=20.0):
        """Convert servo angle (0-180°) to PCA9685 16-bit duty cycle."""
        pulse_ms = min_pulse_ms + (angle_deg / 180.0) * (max_pulse_ms - min_pulse_ms)
        duty = int((pulse_ms / period_ms) * 65535)
        return max(0, min(65535, duty))

    def destroy_node(self):
        if self.pca:
            self.pca.deinit()
        super().destroy_node()


def main(args=None):
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
