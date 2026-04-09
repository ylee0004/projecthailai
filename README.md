# Project Hail AI

Real-time pan-tilt face tracking on NVIDIA Jetson Orin Nano.  
Built with ROS2 Humble + Docker — same architecture as NVIDIA Isaac ROS and production robotics systems.

## Architecture

```
USB Webcam → [perception container] → /face/offset topic → [control container] → Pan/Tilt Servos
```

- **perception**: MediaPipe face detection → publishes normalized (x, y) offset from frame center
- **control**: PID controller → drives pan/tilt servos via PCA9685

## Hardware

| Component | Detail |
|-----------|--------|
| SBC | NVIDIA Jetson Orin Nano |
| OS | Ubuntu 22.04 (JetPack 6.2) |
| Camera | USB Webcam |
| Servos | 2x SG90 |
| PWM Driver | PCA9685 (I2C) |

## Phases

- [x] Phase 1 — Face detection on image/video (MediaPipe, no webcam)
- [ ] Phase 2 — Live servo tracking with PCA9685
- [ ] Phase 3 — TensorRT optimization
- [ ] Phase 4 — C++ control node (rclcpp)
- [ ] Phase 5 — Multi-object recognition

## Quick Start

```bash
# 1. Build workspace
cd ros2_ws
colcon build
source install/setup.bash

# 2. Run perception node (test image)
ros2 run perception face_detector_node \
  --ros-args -p source:=image -p file_path:=/path/to/test_face.jpg

# 3. In another terminal — check topic output
ros2 topic echo /face/offset

# 4. Run control node (dry run — no hardware needed)
ros2 run control servo_controller_node \
  --ros-args -p dry_run:=true
```

## Author
[@ylee0004](https://github.com/ylee0004) | [Project Hail AI Blog](https://velog.io/@ylee0004)
