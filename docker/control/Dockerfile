# Dockerfile — control 컨테이너
#
# 역할: 서보 제어 노드 실행 환경
# 베이스: ROS2 Humble (Ubuntu 22.04)
# 실행: servo_controller_node (/face/offset subscribe → PCA9685 → 서보)

FROM ros:humble-ros-base-jammy

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    # I2C 통신 도구 (PCA9685 디버깅용)
    i2c-tools \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
# adafruit: PCA9685 I2C PWM 드라이버
RUN pip3 install \
    adafruit-circuitpython-pca9685 \
    adafruit-blinka

# ROS2 워크스페이스 복사 및 빌드
WORKDIR /ros2_ws
COPY ros2_ws/src ./src

RUN . /opt/ros/humble/setup.sh && \
    colcon build --packages-select face_tracker_msgs control && \
    rm -rf build log

# 환경변수 설정 스크립트
COPY docker/control/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "run", "control", "servo_controller_node", \
     "--ros-args", "-p", "dry_run:=false"]
