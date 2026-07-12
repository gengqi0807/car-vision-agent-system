"""
try_gesture.py — 交警手势识别测试脚本

读取本地视频文件，运行手势识别并打印每帧结果。
适合在无摄像头环境下手动校验分类逻辑。

Usage:
    python try_gesture.py                          # 使用默认 test.mp4
    python try_gesture.py --source test.mp4         # 指定视频文件
    python try_gesture.py --verbose                 # 打印每帧特征
    python try_gesture.py --no-mirror               # 非镜像摄像头
"""

import os
import sys
import time
import argparse

import cv2
import numpy as np

from police import config
from police.models import (
    create_pose_detector, create_hand_detector,
    detect_pose, detect_hand,
)
from police.features import extract_features, classify_palm_orientation, associate_hands
from police.gesture_classifier import GestureStateMachine
from police.geometry import setup_local_frame
from police.visualization import draw_pose_landmarks, draw_hand_landmarks, draw_chinese_text, draw_wrist_marker


def parse_args():
    parser = argparse.ArgumentParser(
        description="交警手势识别测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python try_gesture.py                             # 默认 test.mp4
  python try_gesture.py --source my_video.mp4       # 指定视频
  python try_gesture.py --verbose                    # 打印每帧特征
  python try_gesture.py --no-mirror                  # 非镜像摄像头
  python try_gesture.py --skip 1                     # 逐帧推理（最精确）
        """
    )
    parser.add_argument(
        "--source", "-s",
        default="test.mp4",
        help="本地视频文件路径（默认: test.mp4）"
    )
    parser.add_argument(
        "--pose-model",
        default=config.POSE_MODEL_PATH,
        help="Pose 模型路径"
    )
    parser.add_argument(
        "--hand-model",
        default=config.HAND_MODEL_PATH,
        help="Hand 模型路径"
    )
    parser.add_argument(
        "--no-mirror", action="store_true",
        help="禁用摄像头镜像"
    )
    parser.add_argument(
        "--skip", type=int, default=config.SKIP_FRAMES,
        help="跳帧数（默认: 2，设为 1 逐帧推理）"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="打印每帧的详细特征值"
    )
    parser.add_argument(
        "--no-hand", action="store_true",
        help="不使用 Hand Landmarker"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="不显示画面（纯终端输出模式）"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.no_mirror:
        config.CAMERA_MIRRORED = False
    config.SKIP_FRAMES = args.skip

    # ---- 检查模型 ----
    if not os.path.exists(args.pose_model):
        print(f"❌ Pose 模型不存在 → {args.pose_model}")
        return

    # ---- 检查视频 ----
    if not os.path.exists(args.source):
        print(f"❌ 视频文件不存在 → {args.source}")
        return

    # ---- 加载模型 ----
    print(f"📦 加载 Pose 模型: {args.pose_model}")
    pose_detector = create_pose_detector(args.pose_model)
    print("✅ PoseLandmarker 就绪")

    hand_detector = None
    if not args.no_hand:
        if os.path.exists(args.hand_model):
            print(f"📦 加载 Hand 模型: {args.hand_model}")
            hand_detector = create_hand_detector(args.hand_model)
            print("✅ HandLandmarker 就绪")
        else:
            print(f"⚠️ Hand 模型未找到 → {args.hand_model}")

    # ---- 打开视频 ----
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"❌ 无法打开视频 → {args.source}")
        pose_detector.close()
        if hand_detector:
            hand_detector.close()
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    print(f"📹 视频: {args.source}")
    print(f"   总帧数: {total_frames}, FPS: {fps_video:.1f}")
    print(f"   SKIP={config.SKIP_FRAMES}, 镜像={'开' if config.CAMERA_MIRRORED else '关'}")
    print(f"{'='*60}")

    # ---- 状态机 ----
    state_machine = GestureStateMachine()
    frame_counter = 0
    global_frame = 0

    last_landmarks = None
    last_world_landmarks = None
    last_feat = None
    last_hand_left = None
    last_hand_right = None

    # 识别结果显示计时器
    display_result = None
    display_confidence = 0.0
    result_display_timer = 0

    results_log = []  # 记录所有判定结果

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n📼 视频播放完毕")
            break
        frame_counter += 1
        should_infer = (frame_counter % config.SKIP_FRAMES == 0)

        global_frame += 1
        h, w = frame.shape[:2]

        # ---- 推理 ----
        if should_infer:
            pose_result = detect_pose(pose_detector, frame)
            hand_result = detect_hand(hand_detector, frame)
        else:
            pose_result = None
            hand_result = None

        if should_infer and pose_result and pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks[0]
            last_landmarks = landmarks

            world_landmarks = None
            if pose_result.pose_world_landmarks:
                world_landmarks = pose_result.pose_world_landmarks[0]
                last_world_landmarks = world_landmarks
            else:
                world_landmarks = last_world_landmarks

            # 像素关键点
            def px(idx):
                lm = landmarks[idx]
                return (lm.x * w, lm.y * h)

            # 局部坐标系
            sm, body_right_2d, body_up_2d = setup_local_frame(
                px(11), px(12), px(0), px(23), px(24)
            )

            # 特征
            if world_landmarks is not None:
                feat = extract_features(world_landmarks, landmarks,
                                        last_hand_left, last_hand_right)
                last_feat = feat
            else:
                feat = last_feat

            # Hand 关联（始终更新，未检测到则置 None，防止滞留旧骨架）
            hands = associate_hands(
                hand_result,
                (landmarks[15].x, landmarks[15].y),
                (landmarks[16].x, landmarks[16].y),
            )
            last_hand_left  = hands["left"]
            last_hand_right = hands["right"]

            # 手掌朝向
            left_palm_ori  = classify_palm_orientation(last_hand_left, body_right_2d, body_up_2d)
            right_palm_ori = classify_palm_orientation(last_hand_right, body_right_2d, body_up_2d)

            # 详细特征输出
            if args.verbose and feat is not None:
                print(f"\n--- 帧 {global_frame} ---")
                print(f"  L_Region={feat['left_region']}, R_Region={feat['right_region']}")
                print(f"  L_Palm={left_palm_ori}, R_Palm={right_palm_ori}")
                print(f"  L_Raise={feat['left_raise']:.3f}, R_Raise={feat['right_raise']:.3f}")
                print(f"  L_Stretch={feat['left_stretch']:.3f}, R_Stretch={feat['right_stretch']:.3f}")
                print(f"  L_Angle={feat['left_arm_angle']:.0f}, R_Angle={feat['right_arm_angle']:.0f}")
                print(f"  L_ZDiff={feat['left_z_diff']:.3f}, R_ZDiff={feat['right_z_diff']:.3f}")
                print(f"  L_Orient={feat['left_orient']}, R_Orient={feat['right_orient']}")

            # 状态机
            if feat is not None:
                result = state_machine.update(
                    feat, left_palm_ori, right_palm_ori,
                    global_frame, feat.get("shoulder_width", 0.35)
                )
                if result is not None:
                    name, conf = result
                    print(f"\n{'='*40}")
                    print(f"📍 帧 {global_frame} | 判定: {name} (置信度: {conf:.0%})")
                    print(f"{'='*40}")
                    results_log.append((global_frame, name, conf))
                    display_result = name
                    display_confidence = conf
                    result_display_timer = config.RESULT_DISPLAY_FRAMES

            # 画面绘制
            if not args.no_display:
                frame_copy = frame.copy()
                draw_pose_landmarks(frame, landmarks, h, w)
                if last_hand_left:
                    draw_hand_landmarks(frame, last_hand_left, h, w)
                else:
                    draw_wrist_marker(frame, landmarks, 15, h, w, side='L')
                if last_hand_right:
                    draw_hand_landmarks(frame, last_hand_right, h, w)
                else:
                    draw_wrist_marker(frame, landmarks, 16, h, w, side='R')

        elif should_infer:
            state_machine.cancel_action(global_frame)
            last_landmarks = None
            last_world_landmarks = None
            last_feat = None
            last_hand_left = None
            last_hand_right = None

        # 非推理帧：绘制缓存骨架
        if not should_infer and last_landmarks is not None and not args.no_display:
            draw_pose_landmarks(frame, last_landmarks, h, w)
            if last_hand_left:
                draw_hand_landmarks(frame, last_hand_left, h, w)
            else:
                draw_wrist_marker(frame, last_landmarks, 15, h, w, side='L')
            if last_hand_right:
                draw_hand_landmarks(frame, last_hand_right, h, w)
            else:
                draw_wrist_marker(frame, last_landmarks, 16, h, w, side='R')

        # 显示
        if not args.no_display:
            # --- 结果展示计时器：超时自动清除 ---
            if result_display_timer > 0:
                result_display_timer -= 1
            else:
                display_result = None
                display_confidence = 0.0

            # --- 按优先级生成状态提示 ---
            status_y = 5
            if last_landmarks is None:
                frame = draw_chinese_text(frame, "未检测到人体",
                                          (10, status_y), (0, 0, 255), 28)
            elif state_machine.state == config.STATE_ACTIVE:
                frame = draw_chinese_text(frame, "⏳ 动作识别中...",
                                          (10, status_y), (0, 255, 255), 28)
            elif state_machine.cooldown_counter > 0:
                frame = draw_chinese_text(frame, f"⏸ 冷却中 {state_machine.cooldown_counter}",
                                          (10, status_y), (128, 128, 128), 28)
            elif display_result is not None:
                text = f"交警手势: {display_result}"
                frame = draw_chinese_text(frame, text,
                                          (10, status_y), (0, 255, 0), 28)
                if display_confidence > 0:
                    frame = draw_chinese_text(frame,
                                              f"置信度: {display_confidence:.0%}",
                                              (10, 38), (0, 200, 0), 20)
            else:
                frame = draw_chinese_text(frame, "等待动作...",
                                          (10, status_y), (255, 255, 255), 28)

            # 帧号（纯英文 → cv2.putText 即可）
            cv2.putText(frame, f"Frame: {global_frame}", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Test - Police Gesture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n用户按下 q 键，退出。")
                break

    # ---- 结果汇总 ----
    print(f"\n{'='*60}")
    print(f"📊 识别汇总（共 {len(results_log)} 次判定）:")
    for gf, name, conf in results_log:
        print(f"  帧 {gf:>5d}: {name} (置信度: {conf:.0%})")
    print(f"{'='*60}")

    # 清理
    cap.release()
    cv2.destroyAllWindows()
    pose_detector.close()
    if hand_detector:
        hand_detector.close()
    print("🛑 测试结束")


if __name__ == "__main__":
    main()
