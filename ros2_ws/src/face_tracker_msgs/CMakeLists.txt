cmake_minimum_required(VERSION 3.8)
project(face_tracker_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FaceOffset.msg"
)

ament_package()
