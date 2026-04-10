#!/bin/bash
# entrypoint.sh — perception 컨테이너 진입점
#
# 컨테이너 시작 시 ROS2 환경변수를 source한 뒤 CMD 실행
# Docker ENTRYPOINT에서 source 명령이 동작하려면 별도 스크립트 필요

set -e

# ROS2 기본 환경
source /opt/ros/humble/setup.bash

# 빌드된 워크스페이스 환경
source /ros2_ws/install/setup.bash

# CMD로 넘어온 명령어 실행
exec "$@"
