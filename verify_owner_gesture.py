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
from police.visualization import draw_chinese_text, draw_chinese_text_lines

# ----------------------------------------------------------------
# 手势 → 动作映射（与服务端一致）
# ----------------------------------------------------------------

GESTURE_ACTION = {
    "palm": "唤醒",
    "open_palm": "唤醒",
    "fist": "确认",
    "circle_ccw": "降低音量",
    "circle_cw": "升高音量",
    "swipe_left": "上一个功能",
    "swipe_right": "下一个功能",
    "thumb_up": "接听电话",
    "thumbs_up": "接听电话",
    "thumb_down": "挂断电话",
    "thumbs_down": "挂断电话",
    "thunb_index": "返回主页",
    "thumb_index": "返回主页",
    "wave": "等待动作",
    "pointing": "追踪中",
    "point": "追踪中",
    "idle": "等待动作",
    "unknown": "等待识别",
}

GESTURE_NAME = {
    "palm": "张开手掌",
    "open_palm": "张开手掌",
    "fist": "握拳",
    "circle_ccw": "逆时针画圈",
    "circle_cw": "顺时针画圈",
    "swipe_left": "张开拳头",
    "swipe_right": "收回拳头",
    "thumb_up": "拇指向上",
    "thumbs_up": "拇指向上",
    "thumb_down": "拇指向下",
    "thumbs_down": "拇指向下",
    "thunb_index": "捏指",
    "thumb_index": "捏指",
    "wave": "挥手",
    "pointing": "食指指向",
    "point": "食指指向",
    "idle": "识别中",
    "unknown": "未识别",
}

# 音量类手势：允许在手未离开屏幕时连续 +/- 切换（连续调音）
VOLUME_GESTURES = {"circle_cw", "circle_ccw"}
SWITCH_GESTURES = {"swipe_left", "swipe_right"}
SWITCH_ENDPOINT_GESTURES = {"fist", "palm", "open_palm"}


def gesture_label(gesture: str) -> str:
    return GESTURE_ACTION.get(gesture, gesture)


# ----------------------------------------------------------------
# 绘制工具
# ----------------------------------------------------------------


def draw_result(frame: np.ndarray, gesture: str, action: str, conf: float,
                hand_count: int, kp: list | None = None, *, locked: bool = False) -> np.ndarray:
    h, w = frame.shape[:2]

    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 104), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    text_lines = [
        (
            f"手势：{GESTURE_NAME.get(gesture, gesture)}  |  动作：{action}",
            (10, 12),
            (0, 255, 0),
            28,
        ),
        (
            f"置信度：{conf:.3f}  |  检测手数：{hand_count}",
            (10, 57),
            (210, 210, 210),
            23,
        ),
    ]
    if locked:
        text_lines.append(
            ("已锁定", (max(10, w - 110), 15), (0, 165, 255), 24)
        )
    frame = draw_chinese_text_lines(frame, text_lines)

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


def draw_locked_marker(frame: np.ndarray) -> np.ndarray:
    return draw_chinese_text(
        frame,
        "已锁定",
        (max(10, frame.shape[1] - 110), 15),
        (0, 165, 255),
        24,
    )


# ----------------------------------------------------------------
# 推理核心
# ----------------------------------------------------------------


def process_frame(frame_bgr: np.ndarray, classifier: GestureClassifier):
    hands = MediaPipeHands.infer_video(frame_bgr)
    hand_kp = hands[0] if hands else None

    gesture, conf = classifier.classify_frame(hand_kp)
    action = gesture_label(gesture)
    return gesture, action, conf, len(hands), hand_kp


def process_image(frame_bgr: np.ndarray, classifier: GestureClassifier):
    """单张图片推理，绕过 classify_frame 的时序去抖逻辑，直接调用静态分类。"""
    hands = MediaPipeHands.infer_video(frame_bgr)
    hand_kp = hands[0] if hands else None

    if hand_kp:
        gesture, conf = classifier.classify_static(hand_kp)
    else:
        gesture, conf = "unknown", 0.0
    action = gesture_label(gesture)
    return gesture, action, conf, len(hands), hand_kp


def main():
    parser = argparse.ArgumentParser(description="车主手势识别验证")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="单张图片路径")
    group.add_argument("--video", help="视频文件路径")
    group.add_argument("--rtsp", help="RTSP 地址")
    group.add_argument("--camera", type=int, help="摄像头索引 (0,1...)")
    args = parser.parse_args()

    # 初始化模型
    model_path = settings.resolved_hand_model_path
    if not os.path.exists(model_path):
        print(f"❌ 模型未找到: {model_path}")
        return

    MediaPipeHands.configure(model_path=model_path)
    classifier = GestureClassifier(domain="owner")

    # ---- 图片模式 ----
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"❌ 无法读取图片: {args.image}")
            return
        gesture, action, conf, hc, kp = process_image(frame, classifier)
        frame = draw_result(frame, gesture, action, conf, hc, kp)
        print(f"Gesture: {gesture} | Action: {action} | Conf: {conf:.3f}")
        cv2.imshow("Owner Gesture — Image", frame)
        print("按任意键退出...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        MediaPipeHands.reset()
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

    # ---- 手势结果锁定状态（简化去抖：进手/撤手过渡窗口不记录）----
    GRACE_FRAMES = 8             # 翻转后宽限帧数（期间不记录识别结果）
    grace_count = 0              # 剩余宽限帧
    prev_hand = False            # 上一帧手是否在屏幕内
    result_locked = False
    locked_gesture = None
    locked_action = None
    locked_conf = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gesture, action, conf, hc, kp = process_frame(frame, classifier)
        hand_present = hc > 0

        # ---- 进手/撤手翻转检测：触发过渡宽限期（简化去抖）----
        # 手状态翻转（进手或撤手）的那一刻起，宽限 GRACE_FRAMES 帧；
        # 期间不记录/不更新任何识别结果，统一输出 idle，避免翻转瞬间
        # 噪声被锁定、或撤手后沿用旧手势导致前端误触发（如多切一次功能）。
        if hand_present != prev_hand:
            grace_count = GRACE_FRAMES
            # 任何翻转都先解锁，开始新一轮，不沿用旧结果
            result_locked = False
            locked_gesture = locked_action = None
            locked_conf = 0.0
        prev_hand = hand_present

        if grace_count > 0:
            # 过渡窗口：不记录，输出 idle（不沿用旧结果）
            grace_count -= 1
            out_g, out_a, out_c = "idle", "idle", 0.0
        elif hand_present:
            # 稳定有手：正常识别 + 锁定
            if result_locked:
                # 已锁定起点手势后，允许张开/收回的切换结果完成本轮动作。
                if (locked_gesture in SWITCH_ENDPOINT_GESTURES
                        and gesture in SWITCH_GESTURES):
                    locked_gesture = gesture
                    locked_action = action
                    locked_conf = conf
                # 音量手势：允许 +/- 方向切换（连续调音），仅稳定态生效
                if (locked_gesture in VOLUME_GESTURES
                        and gesture in VOLUME_GESTURES
                        and gesture != locked_gesture):
                    locked_gesture = gesture
                    locked_action = action
                    locked_conf = conf
                out_g, out_a, out_c = locked_gesture, locked_action, locked_conf
            else:
                if gesture not in ("unknown", "idle") and conf > 0.0:
                    result_locked = True
                    locked_gesture = gesture
                    locked_action = action
                    locked_conf = conf
                out_g, out_a, out_c = (locked_gesture, locked_action, locked_conf) if result_locked else (gesture, action, conf)
        else:
            # 稳定无手：不沿用旧结果，输出 idle
            out_g, out_a, out_c = "idle", "idle", 0.0

        # 文字用锁定结果（不再变化），关键点/手数用当前帧（手的位置仍跟随）
        frame = draw_result(frame, out_g, out_a, out_c, hc, kp, locked=result_locked)

        # FPS 显示
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_timer
            fps = 30 / elapsed
            fps_timer = time.time()
            print(f"  FPS: {fps:.1f}  |  Gesture: {out_g}  →  {out_a}  (conf={out_c:.2f}){' [LOCKED]' if result_locked else ''}")

        cv2.imshow("Owner Gesture — Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    MediaPipeHands.reset()
    print("🛑 识别结束")


if __name__ == "__main__":
    main()
