"""
车主手势识别 — 验证脚本（后端拉流模式）

用法:
  # 1) 图片单帧测试
  python verify_owner_gesture.py --image test_hand.jpg

  # 2) 视频文件测试
  python verify_owner_gesture.py --video test.avi

  # 3) RTSP 流测试（需先启动 stream_core.py 推流）
  python verify_owner_gesture.py --rtsp rtsp://127.0.0.1:8554/test

  # 4) 摄像头测试
  python verify_owner_gesture.py --camera 0

依赖:
  pip install opencv-python mediapipe numpy
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# 将 backend 加入 sys.path
backend_dir = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.models_infer.gesture_classifier import GestureClassifier
from app.models_infer.mediapipe_hands import MediaPipeHands

# ----------------------------------------------------------------
# 手势 → 动作映射（与服务端一致）
# ----------------------------------------------------------------

GESTURE_ACTION = {
    "open_palm":   "wake         ← 唤醒",
    "palm":        "wake         ← 唤醒",
    "fist":        "confirm      ← 确认",
    "index_circle":"volume      ← 音量调节",
    "circle_cw":   "volume_down  ← 音量-",
    "circle_ccw":  "volume_up    ← 音量+",
    "swipe_left":  "prev_func    ← 上一个功能",
    "swipe_right": "next_func    ← 下一个功能",
    "thumbs_up":   "call_answer  ← 接听",
    "thumb_up":    "call_answer  ← 接听",
    "thumbs_down": "call_hangup  ← 挂断",
    "thumb_down":  "call_hangup  ← 挂断",
    "wave":        "home         ← 主页",
    "point":       "idle         ← 食指追踪中",
    "pointing":    "idle         ← 食指追踪中",
    "idle":        "idle",
    "unknown":     "idle",
    "未检测到手部":"idle",
}


def gesture_label(gesture: str) -> str:
    return GESTURE_ACTION.get(gesture, gesture)


# ----------------------------------------------------------------
# 绘制工具
# ----------------------------------------------------------------


def draw_result(frame: np.ndarray, gesture: str, action: str, conf: float,
                hand_count: int, kp: list | None = None) -> np.ndarray:
    h, w = frame.shape[:2]

    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 95), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    # 手势 + 动作
    cv2.putText(frame, f"Gesture: {gesture}  |  Action: {action}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {conf:.3f}  |  Hands: {hand_count}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    # 关键点
    if kp and len(kp) == 21:
        for i, pt in enumerate(kp):
            px = int(pt["x"] * w)
            py = int(pt["y"] * h)
            color = (0, 255, 255) if i in (4, 8, 12, 16, 20) else (255, 255, 255)
            cv2.circle(frame, (px, py), 4, color, -1)

        # 连线（简化版）
        connections = [
            (0,1),(1,2),(2,3),(3,4),      # thumb
            (0,5),(5,6),(6,7),(7,8),      # index
            (0,9),(9,10),(10,11),(11,12), # middle
            (0,13),(13,14),(14,15),(15,16),# ring
            (0,17),(17,18),(18,19),(19,20),# pinky
            (5,9),(9,13),(13,17),          # palm base
        ]
        for a, b in connections:
            p1 = (int(kp[a]["x"] * w), int(kp[a]["y"] * h))
            p2 = (int(kp[b]["x"] * w), int(kp[b]["y"] * h))
            cv2.line(frame, p1, p2, (255, 255, 255), 1)

    return frame


# ----------------------------------------------------------------
# 推理核心
# ----------------------------------------------------------------


def process_frame(frame_bgr: np.ndarray, hands_model: MediaPipeHands, classifier: GestureClassifier):
    infer_result = hands_model.infer(frame_bgr)
    raw_keypoints = infer_result["keypoints"]
    num_hands = infer_result.get("num_hands_detected", 0)
    hands = [raw_keypoints[index:index + 21] for index in range(0, len(raw_keypoints), 21)]
    hand_kp = hands[0] if hands else None

    gesture, conf = classifier.classify_frame(hand_kp)
    action = gesture_label(gesture)
    return gesture, action, conf, num_hands, hand_kp


def process_image(frame_bgr: np.ndarray, hands_model: MediaPipeHands, classifier: GestureClassifier):
    infer_result = hands_model.infer(frame_bgr)
    raw_keypoints = infer_result["keypoints"]
    num_hands = infer_result.get("num_hands_detected", 0)
    if not raw_keypoints or num_hands == 0:
        return "未检测到手部", "idle", 0.0, 0, None

    gesture, conf = classifier.classify_static(raw_keypoints[:21])
    action = gesture_label(gesture)
    return gesture, action, conf, num_hands, raw_keypoints[:21]


def main():
    parser = argparse.ArgumentParser(description="车主手势识别验证")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="单张图片路径")
    group.add_argument("--video", help="视频文件路径")
    group.add_argument("--rtsp", help="RTSP 地址")
    group.add_argument("--camera", type=int, help="摄像头索引 (0,1...)")
    args = parser.parse_args()

    # 初始化模型
    model_path = os.path.join(settings.models_dir, settings.hand_landmarker_model)
    if not os.path.exists(model_path):
        print(f"❌ 模型未找到: {model_path}")
        return

    MediaPipeHands.configure(model_path=model_path)
    hands_model = MediaPipeHands(model_path=model_path)
    classifier = GestureClassifier(domain="owner")

    # ---- 图片模式 ----
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"❌ 无法读取图片: {args.image}")
            return
        gesture, action, conf, hc, kp = process_image(frame, hands_model, classifier)
        frame = draw_result(frame, gesture, action, conf, hc, kp)
        print(f"Gesture: {gesture} | Action: {action} | Conf: {conf:.3f}")
        cv2.imshow("Owner Gesture — Image", frame)
        print("按任意键退出...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ---- 视频 / RTSP / 摄像头模式 ----
    if args.video:
        source = args.video
    elif args.rtsp:
        source = args.rtsp
    else:
        source = args.camera

    print(f"📹 打开视频源: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ 无法打开视频源")
        return

    print("✅ 流已打开！按 Q 退出")
    fps_timer = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gesture, action, conf, hc, kp = process_frame(frame, hands_model, classifier)
        frame = draw_result(frame, gesture, action, conf, hc, kp)

        # FPS 显示
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_timer
            fps = 30 / elapsed
            fps_timer = time.time()
            print(f"  FPS: {fps:.1f}  |  Gesture: {gesture}  →  {action}  (conf={conf:.2f})")

        cv2.imshow("Owner Gesture — Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 识别结束")


if __name__ == "__main__":
    main()
