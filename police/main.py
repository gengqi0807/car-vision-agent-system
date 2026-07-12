"""
main.py — 交警手势识别主程序

整合所有模块，实现完整的识别管线：
  1. 读取视频源（RTSP 流或本地文件）
  2. 跳帧推理（每 SKIP_FRAMES 帧推理一次）
  3. 特征提取 → 状态机 → 手势分类
  4. 画面叠加（骨架 + 调试信息）
  5. 显示结果，按 q 退出

Usage:
    python -m police.main                           # 使用默认 RTSP
    python -m police.main --source test.mp4          # 本地视频
    python -m police.main --source rtsp://...        # 指定 RTSP
    python -m police.main --no-mirror                # 非镜像摄像头
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

from . import config
from .models import (
    create_pose_detector, create_hand_detector,
    detect_pose, detect_hand,
)
from .features import extract_features, classify_palm_orientation, associate_hands
from .gesture_classifier import GestureStateMachine, mode_str
from .geometry import setup_local_frame, calc_dist
from .visualization import (
    draw_pose_landmarks, draw_hand_landmarks, draw_chinese_text,
    draw_wrist_marker,
)


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="交警手势识别系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m police.main                           # 默认 RTSP 流
  python -m police.main --source test.mp4          # 本地视频文件
  python -m police.main --source 0                 # 摄像头
  python -m police.main --no-mirror                # 非镜像摄像头
  python -m police.main --skip 3 --scale 0.6       # 自定义跳帧和缩放
        """
    )
    parser.add_argument(
        "--source", "-s",
        default=config.DEFAULT_RTSP_URL,
        help=f"视频源：RTSP URL、本地文件路径或摄像头索引（默认: {config.DEFAULT_RTSP_URL}）"
    )
    parser.add_argument(
        "--pose-model",
        default=config.POSE_MODEL_PATH,
        help=f"Pose 模型路径（默认: {config.POSE_MODEL_PATH}）"
    )
    parser.add_argument(
        "--hand-model",
        default=config.HAND_MODEL_PATH,
        help=f"Hand 模型路径（默认: {config.HAND_MODEL_PATH}）"
    )
    parser.add_argument(
        "--no-mirror", action="store_true",
        help="禁用摄像头镜像（默认启用镜像）"
    )
    parser.add_argument(
        "--skip", type=int, default=config.SKIP_FRAMES,
        help=f"跳帧数（默认: {config.SKIP_FRAMES}）"
    )
    parser.add_argument(
        "--scale", type=float, default=config.INFER_SCALE,
        help=f"推理缩放比（默认: {config.INFER_SCALE}）"
    )
    parser.add_argument(
        "--no-hand", action="store_true",
        help="不使用 Hand Landmarker（仅 Pose 识别）"
    )
    return parser.parse_args()


def main():
    """主入口：加载模型 → 读取视频 → 推理循环 → 显示结果。"""
    args = parse_args()

    # 应用命令行覆盖
    if args.no_mirror:
        config.CAMERA_MIRRORED = False
    config.SKIP_FRAMES = args.skip
    config.INFER_SCALE = args.scale

    # ---- 1. 加载模型 ----
    if not os.path.exists(args.pose_model):
        print(f"❌ Pose 模型不存在 → {args.pose_model}")
        return

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
            print(f"⚠️ Hand 模型未找到 ({args.hand_model})，将仅使用 Pose 识别")

    # ---- 2. 连接视频源 ----
    source = args.source
    if source.isdigit():
        source = int(source)  # 摄像头索引

    cap = cv2.VideoCapture(source)
    if source == config.DEFAULT_RTSP_URL:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"❌ 无法打开视频源 → {source}")
        pose_detector.close()
        if hand_detector:
            hand_detector.close()
        return

    print(f"✅ 视频源打开！SKIP={config.SKIP_FRAMES}, SCALE={config.INFER_SCALE:.2f}, "
          f"镜像={'开' if config.CAMERA_MIRRORED else '关'}")
    print(f"   触发确认={config.START_CONFIRM}帧, 结束确认={config.HOLD_FRAMES}帧, "
          f"冷却={config.COOLDOWN_FRAMES}帧。等待动作...")

    # ---- 3. 状态变量初始化 ----
    state_machine = GestureStateMachine()

    frame_counter = 0
    global_frame = 0
    fps = 0
    last_time = time.time()

    # 缓存上一帧的推理结果
    last_landmarks = None
    last_world_landmarks = None
    last_feat = None
    last_hand_left = None
    last_hand_right = None
    left_palm_ori = '?'
    right_palm_ori = '?'

    # 识别结果显示计时器（结果展示 N 帧后自动清除）
    display_result = None
    display_confidence = 0.0
    result_display_timer = 0

    while True:
        # ---- 读取帧 ----
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 读取帧失败")
            break
        frame_counter += 1
        should_infer = (frame_counter % config.SKIP_FRAMES == 0)

        # FPS
        now = time.time()
        elapsed = now - last_time
        if elapsed > 0:
            fps = int(1.0 / elapsed)
        last_time = now

        global_frame += 1
        h, w = frame.shape[:2]

        # ============================================================
        # 4. AI 推理（仅跳帧时执行）
        # ============================================================
        if should_infer:
            pose_result = detect_pose(pose_detector, frame)
            hand_result = detect_hand(hand_detector, frame)
        else:
            pose_result = None
            hand_result = None

        if should_infer and pose_result and pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks[0]
            last_landmarks = landmarks

            # 世界坐标
            world_landmarks = None
            if pose_result.pose_world_landmarks:
                world_landmarks = pose_result.pose_world_landmarks[0]
                last_world_landmarks = world_landmarks
            else:
                world_landmarks = last_world_landmarks

            # ---- 绘制 Pose 骨架 ----
            draw_pose_landmarks(frame, landmarks, h, w)

            # ---- 提取像素关键点 ----
            def px(idx):
                lm = landmarks[idx]
                return (lm.x * w, lm.y * h)

            left_wrist    = px(15); right_wrist    = px(16)
            left_shoulder = px(11); right_shoulder = px(12)
            left_elbow    = px(13); right_elbow    = px(14)
            nose          = px(0)
            left_hip      = px(23); right_hip      = px(24)

            # 肩部标签
            cv2.putText(frame, "L", (int(left_shoulder[0]) - 15, int(left_shoulder[1]) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            cv2.putText(frame, "R", (int(right_shoulder[0]) + 5, int(right_shoulder[1]) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3)

            shoulder_width = calc_dist(left_shoulder, right_shoulder)

            # ---- 局部坐标系（用于手掌朝向检测） ----
            sm, body_right_2d, body_up_2d = setup_local_frame(
                left_shoulder, right_shoulder, nose, left_hip, right_hip
            )

            # ---- 特征提取 ----
            if world_landmarks is not None:
                feat = extract_features(world_landmarks, landmarks,
                                        last_hand_left, last_hand_right)

                # ---- 被遮挡手臂特征冻结（手腕 visibility < 0.3 时保持上一帧值）----
                if last_feat is not None:
                    lw_vis = landmarks[15].visibility if landmarks else 0.0
                    rw_vis = landmarks[16].visibility if landmarks else 0.0

                    _left_keys = ("left_raise", "left_stretch", "left_z_diff",
                                  "left_wx", "left_wy", "left_sx", "left_sy",
                                  "left_orient", "left_pose", "left_region",
                                  "left_fwd", "left_lat", "left_dir_raw",
                                  "left_arm_angle")
                    _right_keys = ("right_raise", "right_stretch", "right_z_diff",
                                   "right_wx", "right_wy", "right_sx", "right_sy",
                                   "right_orient", "right_pose", "right_region",
                                   "right_fwd", "right_lat", "right_dir_raw",
                                   "right_arm_angle")

                    if lw_vis < 0.3:
                        for k in _left_keys:
                            if k in feat and k in last_feat:
                                feat[k] = last_feat[k]
                        feat["left_visible"] = False
                    else:
                        feat["left_visible"] = True

                    if rw_vis < 0.3:
                        for k in _right_keys:
                            if k in feat and k in last_feat:
                                feat[k] = last_feat[k]
                        feat["right_visible"] = False
                    else:
                        feat["right_visible"] = True
                else:
                    feat["left_visible"] = True
                    feat["right_visible"] = True

                last_feat = feat
            else:
                feat = last_feat

            # ---- Hand 关联（始终更新，未检测到则置 None，防止滞留旧骨架）----
            hands = associate_hands(
                hand_result,
                (landmarks[15].x, landmarks[15].y),
                (landmarks[16].x, landmarks[16].y),
            )
            last_hand_left  = hands["left"]
            last_hand_right = hands["right"]

            # ---- 绘制 Hand 骨架 ----
            if last_hand_left:
                draw_hand_landmarks(frame, last_hand_left, h, w)
            else:
                draw_wrist_marker(frame, landmarks, 15, h, w, side='L')
            if last_hand_right:
                draw_hand_landmarks(frame, last_hand_right, h, w)
            else:
                draw_wrist_marker(frame, landmarks, 16, h, w, side='R')

            # ---- 手掌朝向 ----
            left_palm_ori  = classify_palm_orientation(last_hand_left, body_right_2d, body_up_2d)
            right_palm_ori = classify_palm_orientation(last_hand_right, body_right_2d, body_up_2d)

            # ---- 状态机更新 ----
            if feat is not None:
                result = state_machine.update(
                    feat, left_palm_ori, right_palm_ori,
                    global_frame, feat.get("shoulder_width", 0.35)
                )
                if result is not None:
                    display_result, display_confidence = result
                    result_display_timer = config.RESULT_DISPLAY_FRAMES

        elif should_infer:
            # 未检测到人体
            state_machine.cancel_action(global_frame)
            last_landmarks = None
            last_world_landmarks = None
            last_feat = None
            last_hand_left = None
            last_hand_right = None

        # ---- 非推理帧：绘制缓存的骨架（防止闪烁）----
        if not should_infer and last_landmarks is not None:
            draw_pose_landmarks(frame, last_landmarks, h, w)
            # 只绘制当前帧有检测到的手部骨架
            if last_hand_left:
                draw_hand_landmarks(frame, last_hand_left, h, w)
            else:
                # 左手未检测到 → 用手腕位置画标记点（索引15）
                draw_wrist_marker(frame, last_landmarks, 15, h, w, side='L')
            if last_hand_right:
                draw_hand_landmarks(frame, last_hand_right, h, w)
            else:
                # 右手未检测到 → 用手腕位置画标记点（索引16）
                draw_wrist_marker(frame, last_landmarks, 16, h, w, side='R')

        # ============================================================
        # 5. 画面文字叠加（按优先级从上到下）
        # ============================================================

        # --- 结果展示计时器：超时自动清除 ---
        if result_display_timer > 0:
            result_display_timer -= 1
        else:
            display_result = None
            display_confidence = 0.0

        # --- 按优先级生成左上角状态提示 ---
        status_y = 30

        if last_landmarks is None:
            # 最低优先级：未检测到人体
            frame = draw_chinese_text(frame, "未检测到人体",
                                      (10, status_y), (0, 0, 255), 36)

        elif state_machine.state == config.STATE_ACTIVE:
            # 动作识别中（黄色）
            frame = draw_chinese_text(frame, "⏳ 动作识别中...",
                                      (10, status_y), (0, 255, 255), 30)

        elif state_machine.cooldown_counter > 0:
            # 冷却中（灰色）
            frame = draw_chinese_text(
                frame, f"⏸ 冷却中 {state_machine.cooldown_counter}",
                (10, status_y), (128, 128, 128), 28)

        elif display_result is not None:
            # 展示刚完成的识别结果（绿色）
            text = f"交警手势: {display_result}"
            frame = draw_chinese_text(frame, text, (10, status_y), (0, 255, 0), 36)
            if display_confidence > 0:
                conf_text = f"置信度: {display_confidence:.0%}"
                frame = draw_chinese_text(frame, conf_text, (10, 75),
                                          (0, 200, 0), 22)

        else:
            # IDLE + 有人 + 无结果 → 等待动作（白色）
            frame = draw_chinese_text(frame, "等待动作...",
                                      (10, status_y), (255, 255, 255), 30)

        # ---- 胯部停留帧数可视化 ----
        if state_machine.action_data is not None:
            ad = state_machine.action_data
            hold = ad.get("hip_hold_count", 0)
            active = ad.get("active_arm", "?")
            cv2.putText(frame,
                        f"HOLD: {hold}/{config.HOLD_FRAMES}  active={active}",
                        (10, h - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100), 2)

        # ---- 左上角：实时区域/朝向调试 ----
        if last_feat is not None:
            l_region = last_feat["left_region"]
            r_region = last_feat["right_region"]
            lcy = 110
            cv2.putText(frame, f"L_Region: {l_region}  R_Region: {r_region}",
                        (10, lcy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 2)
            cv2.putText(frame,
                        f"L_Palm: {left_palm_ori}  R_Palm: {right_palm_ori}",
                        (10, lcy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
            cv2.putText(frame,
                        f"L_Raise={last_feat['left_raise']:.2f}  R_Raise={last_feat['right_raise']:.2f}",
                        (10, lcy + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
            cv2.putText(frame,
                        f"L_Angle={last_feat['left_arm_angle']:.0f}  R_Angle={last_feat['right_arm_angle']:.0f}",
                        (10, lcy + 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
            # 触发计数器
            if (state_machine.state == config.STATE_IDLE
                    and state_machine.cooldown_counter == 0):
                cv2.putText(frame,
                            f"Trigger: {state_machine.trigger_count}/{config.START_CONFIRM}",
                            (10, lcy + 72),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 200, 100), 1)

        # ---- 右上角：状态调试 ----
        if last_feat is not None:
            rxx = w - 310
            cv2.putText(frame, f"State: {state_machine.state_name}",
                        (rxx, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame,
                        f"L_Region: {last_feat['left_region']}  R_Region: {last_feat['right_region']}",
                        (rxx, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            cv2.putText(frame,
                        f"L_Palm: {left_palm_ori}  R_Palm: {right_palm_ori}",
                        (rxx, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
            cv2.putText(frame,
                        f"L_Raise={last_feat['left_raise']:.2f}  R_Raise={last_feat['right_raise']:.2f}",
                        (rxx, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(frame,
                        f"L_Angle={last_feat['left_arm_angle']:.0f}  R_Angle={last_feat['right_arm_angle']:.0f}",
                        (rxx, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(frame,
                        f"L_Stretch={last_feat['left_stretch']:.2f}  R_Stretch={last_feat['right_stretch']:.2f}",
                        (rxx, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            cv2.putText(frame,
                        f"L_Orient={last_feat['left_orient']}  R_Ori={last_feat['right_orient']}",
                        (rxx, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1)

        # ---- FPS ----
        cv2.putText(frame, f"FPS: {fps}", (w - 100, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ---- 显示 ----
        cv2.imshow("Police Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n用户按下 q 键，退出。")
            break

    # ---- 6. 清理资源 ----
    cap.release()
    cv2.destroyAllWindows()
    pose_detector.close()
    if hand_detector:
        hand_detector.close()
    print("🛑 识别已结束")


if __name__ == "__main__":
    main()
