# FaceDirection.msg — Project Hail AI
#
# 이 메시지의 역할:
#   FaceDetectorNode가 /face/direction topic으로 publish하는 머리 방향 정보.
#   solvePnP로 계산한 yaw 각도와 EMA 필터 + 임계값 분류 결과를 전달한다.
#
# 좌표계 컨벤션:
#   yaw_deg 음수 = 왼쪽 회전 (카메라 기준 피사체가 왼쪽을 봄)
#   yaw_deg 양수 = 오른쪽 회전
#   범위: -90.0 ~ +90.0
#
# direction 값:
#   "LEFT"   — yaw_deg < -25.0
#   "CENTER" — -25.0 <= yaw_deg <= +25.0  (경계값 포함)
#   "RIGHT"  — yaw_deg > +25.0

float32 yaw_deg          # 머리 yaw 각도: -90 ~ +90 (음수=왼쪽, 양수=오른쪽)
string direction         # 방향 분류: "LEFT" | "CENTER" | "RIGHT"
float32 confidence       # 감지 확신도: 0.0 ~ 1.0
bool detected            # 얼굴 감지 여부: true=감지됨, false=감지 안됨
builtin_interfaces/Time stamp  # 프레임 타임스탬프 (/face/offset와 동일)
