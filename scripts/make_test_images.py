"""生成测试用图像（无需外部素材）。

说明：纯色/几何图形无法被 MediaPipe 识别为真实手部或人体，
因此接口会返回 "未检测到手部/人体"（pipeline 正常，仅无目标）。
要验证"检测到 N 只手/人"，请用手边真实含手/人体的照片替换。
"""
import numpy as np
import cv2

HEIGHT, WIDTH = 480, 640

# 一张通用底图（模拟车身/场景背景）
base = np.full((HEIGHT, WIDTH, 3), (30, 30, 30), dtype=np.uint8)

# 模拟人体轮廓（矩形 + 头）
pose = base.copy()
cv2.rectangle(pose, (260, 140), (380, 420), (200, 200, 200), -1)
cv2.circle(pose, (320, 110), 35, (200, 200, 200), -1)

# 模拟手部区域（亮色块）
hand = base.copy()
cv2.rectangle(hand, (280, 200), (360, 320), (220, 220, 220), -1)

cv2.imwrite("test_pose_frame.jpg", pose)
cv2.imwrite("test_hand_frame.jpg", hand)
print("已生成 test_pose_frame.jpg / test_hand_frame.jpg")
