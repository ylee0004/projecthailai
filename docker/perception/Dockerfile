# Dockerfile — perception 컨테이너
#
# 역할: 얼굴 감지 노드 실행 환경
# 베이스: ROS2 Humble (Ubuntu 22.04)
# 실행: face_detector_node (MediaPipe → /face/offset publish)

FROM ros:humble-ros-base-jammy

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    # OpenCV 의존성
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
# NumPy는 1.x 고정 (MediaPipe 0.10.x 호환)
RUN pip3 install \
    "numpy<2" \
    opencv-contrib-python==4.11.0.86 \
    mediapipe==0.10.18

# ROS2 워크스페이스 복사 및 빌드
WORKDIR /ros2_ws
COPY ros2_ws/src ./src

RUN . /opt/ros/humble/setup.sh && \
    colcon build --packages-select face_tracker_msgs perception && \
    rm -rf build log

# 환경변수 설정 스크립트
COPY docker/perception/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "run", "perception", "face_detector_node", \
     "--ros-args", "-p", "source:=webcam", "-p", "show_preview:=false"]
