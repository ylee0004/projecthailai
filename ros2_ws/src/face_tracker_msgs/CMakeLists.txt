# CMakeLists.txt — face_tracker_msgs
#
# 이 파일의 역할:
#   ROS2 커스텀 메시지 패키지 빌드 설정.
#   FaceOffset.msg 파일을 읽어 Python/C++ 코드에서 import 가능한
#   메시지 클래스를 자동 생성한다.
#
# colcon build 실행 시 이 파일이 처리되어:
#   - Python: from face_tracker_msgs.msg import FaceOffset
#   - C++:    #include "face_tracker_msgs/msg/face_offset.hpp"
#   위와 같이 사용할 수 있게 된다.

cmake_minimum_required(VERSION 3.8)
project(face_tracker_msgs)

# ament_cmake: ROS2 빌드 시스템 기본 도구
find_package(ament_cmake REQUIRED)

# rosidl_default_generators: .msg 파일을 각 언어 코드로 변환하는 도구
find_package(rosidl_default_generators REQUIRED)

# 생성할 메시지 파일 목록
# 새로운 메시지 추가 시 이 블록에 경로를 추가
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FaceOffset.msg"
)

ament_package()
