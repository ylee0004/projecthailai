#!/bin/bash
# entrypoint.sh — control 컨테이너 진입점
#
# 컨테이너 시작 시 ROS2 환경변수를 source한 뒤 CMD 실행

set -e

source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

exec "$@"
