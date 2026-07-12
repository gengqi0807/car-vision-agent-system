"""
recognize_image.py — 单张图片交警手势识别

使用规则模型（MediaPipe Pose + 手部区域划分）+ 深度学习模型（CTPGREngine）
分别识别单张图片中的手势，不依赖状态机和帧累积。

Usage:
    python recognize_image.py
    python recognize_image.py --image try.png
    python recognize_image.py --image C:/path/to/photo.jpg --no-dl
"""

import os
import sys
import argparse
import cv2
import numpy as np

# ---- 确保项目根目录在 sys.path 中 ----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---- 导入 police 包 ----
from police import config as police_cfg
from police.models import (
    create_pose_detector, create_hand_detector,
    detect_pose, detect_hand,
)
from police.features import extract_features
from police.visualization import draw_pose_landmarks, draw_hand_landmarks, draw_chinese_text
from police.geometry import setup_local_frame

# ---- 导入 CTPGREngine ----
try:
    from ctpgr_engine import CTPGREngine
    _DL_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] CTPGREngine 导入失败: {e}，将仅使用规则模型")
    _DL_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="单张图片交警手势识别")
    parser.add_argument(
        "--image", type=str, default="try.jpg",
        help="输入图片路径（默认: try.jpg）"
    )
    parser.add_argument(
        "--pose-model", default=police_cfg.POSE_MODEL_PATH,
        help=f"Pose 模型路径（默认: {police_cfg.POSE_MODEL_PATH}）"
    )
    parser.add_argument(
        "--hand-model", default=police_cfg.HAND_MODEL_PATH,
        help=f"Hand 模型路径（默认: {police_cfg.HAND_MODEL_PATH}）"
    )
    parser.add_argument(
        "--no-hand", action="store_true",
        help="不使用 Hand Landmarker"
    )
    parser.add_argument(
        "--no-dl", action="store_true",
        help="禁用深度学习模型"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="不显示窗口，仅打印终端结果"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="保存结果图片的路径（可选）"
    )
    parser.add_argument(
        "--scale", type=float, default=0.7,
        help="显示缩放比（默认: 0.7）"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 图片路径
    image_path = args.image
    if not os.path.isabs(image_path):
        image_path = os.path.join(PROJECT_ROOT, image_path)

    if not os.path.exists(image_path):
        print(f"[ERROR] 图片不存在 -> {image_path}")
        return

    # ================================================================
    # 1. 加载模型
    # ================================================================
    print(f"[LOAD] Pose 模型: {args.pose_model}")
    pose_detector = create_pose_detector(args.pose_model)
    print("[ OK ] PoseLandmarker 就绪")

    hand_detector = None
    if not args.no_hand and os.path.exists(args.hand_model):
        print(f"[LOAD] Hand 模型: {args.hand_model}")
        hand_detector = create_hand_detector(args.hand_model)
        print("[ OK ] HandLandmarker 就绪")

    # DL 模型
    dl_engine = None
    if _DL_AVAILABLE and not args.no_dl:
        print("[LOAD] CTPGREngine 深度学习模型...")
        try:
            dl_engine = CTPGREngine()
            print("[ OK ] CTPGREngine 就绪")
        except Exception as e:
            import traceback
            print(f"[WARN] CTPGREngine 初始化失败: {e}")
            traceback.print_exc()

    # ================================================================
    # 2. 读取图片
    # ================================================================
    print(f"\n[READ] 图片: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] 无法读取图片 -> {image_path}")
        return
    h, w = frame.shape[:2]
    print(f"       尺寸: {w}x{h}")

    # ================================================================
    # 3. 推理
    # ================================================================
    print("\n--- 推理中 ---")

    # ---- 3a. MediaPipe Pose ----
    pose_result = detect_pose(pose_detector, frame)
    hand_result = detect_hand(hand_detector, frame) if hand_detector else None

    if not pose_result or not pose_result.pose_landmarks:
        print("[WARN] 未检测到人体姿态")
        cv2.putText(frame, "No person detected", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        if not args.no_display:
            display_frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
            cv2.imshow("Image Gesture Recognition", display_frame)
            cv2.waitKey(0)
        return

    landmarks = pose_result.pose_landmarks[0]
    world_landmarks = pose_result.pose_world_landmarks[0] if pose_result.pose_world_landmarks else None

    # ---- 3b. 绘制骨架 ----
    draw_pose_landmarks(frame, landmarks, h, w)

    # ---- 3c. 规则模型特征提取 ----
    left_region = "?"
    right_region = "?"
    left_raise = 0.0
    right_raise = 0.0

    if world_landmarks is not None:
        feat = extract_features(world_landmarks, landmarks, None, None)
        left_region = feat.get("left_region", "?")
        right_region = feat.get("right_region", "?")
        left_raise = feat.get("left_raise", 0.0)
        right_raise = feat.get("right_raise", 0.0)

        # ---- 图片侧身补偿：根据手臂朝向修正手指高度 ----
        # features.py 中统一用 wrist_y + 0.08 估算指尖（面向视频识别），
        # 但对于举起的手臂（raise<0），指尖应在手腕上方（y 更小），
        # 统一 +0.08 会把指尖估低一个档位。图片识别无帧累积补偿，
        # 此处用方向感知的偏移重算 hand_region。
        lw3 = world_landmarks[15]       # 左手腕
        rw3 = world_landmarks[16]       # 右手腕
        shoulder_y_avg = (world_landmarks[11].y + world_landmarks[12].y) / 2.0

        # 手臂下垂 → 指尖在手腕下方 (+0.08)；手臂上举 → 指尖在手腕上方 (-0.08)
        left_hand_y_corrected  = lw3.y + (0.08 if left_raise >= 0 else -0.08)
        right_hand_y_corrected = rw3.y + (0.08 if right_raise >= 0 else -0.08)

        from police.geometry import get_hand_region
        left_region  = get_hand_region(left_hand_y_corrected,  shoulder_y_avg, 0, 0)
        right_region = get_hand_region(right_hand_y_corrected, shoulder_y_avg, 0, 0)

        # 打印规则模型结果
        print(f"  [规则模型] 左手区域: {left_region}  (raise={left_raise:+.2f})")
        print(f"  [规则模型] 右手区域: {right_region} (raise={right_raise:+.2f})")

    # ---- 3d. 深度学习模型 ----
    dl_gesture = "N/A"
    dl_confidence = 0.0
    if dl_engine is not None:
        try:
            dl_frame = cv2.resize(frame, (512, 512))
            # LSTM 需要时序上下文：训练时 LABEL_DELAY=15，前15帧标签为"无手势"
            # 单帧从 h0/c0 推理会偏向"无手势"，因此喂入同一帧 20 次预热 LSTM 隐藏状态
            WARMUP_FRAMES = 20
            for _ in range(WARMUP_FRAMES):
                dl_result = dl_engine.predict_frame(dl_frame)
            dl_gesture = dl_result["gesture"]
            dl_confidence = dl_result["confidence"]
            print(f"  [DL模型]  手势: {dl_gesture}  (置信度: {dl_confidence:.0%})"
                  f"  (预热{WARMUP_FRAMES}帧)")
        except Exception as e:
            print(f"  [DL-ERROR] 推理异常: {e}")
            dl_gesture = "推理失败"

    # ================================================================
    # 4. 画面叠加
    # ================================================================
    # ---- 标题 ----
    title_bar_h = 40
    cv2.rectangle(frame, (0, 0), (w, title_bar_h), (30, 30, 30), -1)
    cv2.putText(frame, "Image Gesture Recognition",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # ---- 左上角：规则模型结果 ----
    rule_x, rule_y = 10, title_bar_h + 15
    box_w, box_h = 280, 90

    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (rule_x, rule_y), (rule_x + box_w, rule_y + box_h),
                  (40, 40, 40), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    cv2.putText(frame, "Rule Model", (rule_x + 10, rule_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, f"L: {left_region}  ({left_raise:+.2f})",
                (rule_x + 10, rule_y + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
    cv2.putText(frame, f"R: {right_region} ({right_raise:+.2f})",
                (rule_x + 10, rule_y + 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 255), 2)

    # ---- 左下角：深度学习模型结果 ----
    if dl_engine is not None:
        dl_x, dl_y = 10, h - 100
        dl_box_w, dl_box_h = 280, 80

        overlay = frame.copy()
        cv2.rectangle(overlay, (dl_x, dl_y), (dl_x + dl_box_w, dl_y + dl_box_h),
                      (25, 25, 50), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(frame, "DL Model", (dl_x + 10, dl_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
        # 手势名（中文，用 PIL）
        frame = draw_chinese_text(frame, dl_gesture,
                                  (dl_x + 10, dl_y + 42), (255, 200, 0), 24)
        cv2.putText(frame, f"{dl_confidence:.0%}",
                    (dl_x + 10, dl_y + 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 160, 80), 2)
    else:
        dl_x, dl_y = 10, h - 80
        cv2.putText(frame, "DL: N/A", (dl_x + 10, dl_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (128, 128, 128), 2)

    # ---- 右上角：图片信息 ----
    info_x = w - 200
    cv2.putText(frame, f"Size: {w}x{h}", (info_x, title_bar_h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(frame, f"Hands: L+R" if left_region != "?" else "Hands: ?",
                (info_x, title_bar_h + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # ================================================================
    # 5. 显示 / 保存
    # ================================================================
    print("\n--- 完成 ---")
    print(f"  规则模型: L={left_region}  R={right_region}")
    if dl_engine:
        print(f"  DL模型:   {dl_gesture}  ({dl_confidence:.0%})")

    if args.output:
        cv2.imwrite(args.output, frame)
        print(f"  结果已保存: {args.output}")

    if not args.no_display:
        display_frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
        cv2.imshow("Image Gesture Recognition", display_frame)
        print("\n按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 清理
    pose_detector.close()
    if hand_detector:
        hand_detector.close()


if __name__ == "__main__":
    main()
