# setup.py — perception 패키지
#
# 이 파일의 역할:
#   ROS2 Python 패키지 설치 설정.
#   colcon build 시 이 파일을 읽어 패키지를 설치하고
#   'ros2 run' 명령어로 실행할 수 있는 진입점(entry point)을 등록한다.
#
# 실행 방법:
#   ros2 run perception face_detector_node
#   → entry_points의 'face_detector_node' 항목을 찾아
#     perception/face_detector_node.py의 main() 함수를 실행

from setuptools import setup

package_name = 'perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],  # perception/ 디렉토리를 Python 패키지로 등록
    data_files=[
        # ROS2 패키지 인덱스에 등록 (ros2 pkg list에서 보이게 함)
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        # package.xml을 설치 디렉토리에 복사
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ylee0004',
    maintainer_email='ylee0004@github.com',
    description='Face detection node for Project Hail AI',
    license='MIT',
    entry_points={
        'console_scripts': [
            # 'ros2 run perception face_detector_node' 명령어 등록
            # 형식: '<실행명> = <패키지>.<모듈>:<함수>'
            'face_detector_node = perception.face_detector_node:main',
        ],
    },
)
