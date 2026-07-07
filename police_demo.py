import cv2
import mediapipe as mp
import numpy as np

# 1. 初始化 MediaPipe 的 Pose（姿态）模块
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# 2. 创建姿态识别对象（置信度调低一点，提高检出率）
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 3. 连接你的 RTSP 视频流（和你的 stream_core.py 推流地址保持一致）
#    特别注意：如果你的 stream_core.py 正在运行，这里就能读到画面！
rtsp_url = "rtsp://127.0.0.1:8554/test"
cap = cv2.VideoCapture(rtsp_url)

# 如果上面 RTSP 连不上，可以临时换成本地视频文件测试（把上面注释掉，取消下面这行的注释）
# cap = cv2.VideoCapture("police.mp4")  # 前提是你有 police.mp4 文件

if not cap.isOpened():
    print("❌ 错误：无法打开视频流！请确认 MediaMTX 和 FFmpeg 正在推流。")
    exit()

print("✅ 视频流打开成功！正在尝试识别交警骨架...")

# 4. 主循环：逐帧读取并识别
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 视频帧读取失败，可能推流已中断")
        break

    # 将 BGR 格式转为 RGB（MediaPipe 要求 RGB）
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # 5. 如果检测到人体骨架
    if results.pose_landmarks:
        # 在画面上画出 33 个关键点和骨骼连线（这就是同学代码干的活）
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
        # 左上角显示绿色大字：“检测到交警（姿态）”
        cv2.putText(frame, "Police Gesture Detected!", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        # 如果没检测到人，左上角显示红色大字：“未检测到人体”
        cv2.putText(frame, "No Human Body Detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # 6. 显示画面
    cv2.imshow("Police Gesture Recognition - Local Test", frame)

    # 按 Q 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理资源
cap.release()
cv2.destroyAllWindows()
pose.close()
print("🛑 识别已结束")