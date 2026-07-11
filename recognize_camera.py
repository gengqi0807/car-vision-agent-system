"""
recognize_camera.py — 摄像头实时交警手势识别

使用规则模型（MediaPipe Pose + Hand + 状态机）+ 深度学习模型（CTPGREngine）
实时识别摄像头画面中的交警手势。

- 规则模型：左上角展示（含状态机状态 + 双手区域 + 识别结果）
- DL 模型：左下角展示（手势名 + 置信度）
- 整体逻辑与视频识别（run_recognition.py）保持一致，区分仅在于摄像头实时输入

Usage:
    python recognize_camera.py
    python recognize_camera.py --camera 1
    python recognize_camera.py --camera 0 --no-dl
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

# ---- 确保项目根目录在 sys.path 中 ----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---- 导入 police 包（MediaPipe 骨架 + 状态机手势分类） ----
from police import config as police_cfg
from police.models import (
    create_pose_detector, create_hand_detector,
    detect_pose, detect_hand,
)
from police.features import extract_features, classify_palm_orientation, associate_hands
from police.gesture_classifier import GestureStateMachine
from police.geometry import setup_local_frame, calc_dist
from police.visualization import (
    draw_pose_landmarks, draw_hand_landmarks, draw_chinese_text,
    draw_wrist_marker,
)

# ---- 导入 CTPGREngine（PyTorch pose_model.pt + lstm.pt RNN 深度学习模型） ----
try:
    from ctpgr_engine import CTPGREngine
    _DL_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] CTPGREngine 导入失败: {e}，将仅使用规则模型")
    _DL_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="摄像头实时交警手势识别"
    )
    parser.add_argument(
        "--camera", "-c", type=int, default=0,
        help="摄像头索引（默认: 0）"
    )
    parser.add_argument(
        "--pose-model",
        default=police_cfg.POSE_MODEL_PATH,
        help=f"MediaPipe Pose 模型路径（默认: {police_cfg.POSE_MODEL_PATH}）"
    )
    parser.add_argument(
        "--hand-model",
        default=police_cfg.HAND_MODEL_PATH,
        help=f"MediaPipe Hand 模型路径（默认: {police_cfg.HAND_MODEL_PATH}）"
    )
    parser.add_argument(
        "--skip", type=int, default=police_cfg.SKIP_FRAMES,
        help=f"跳帧数（默认: {police_cfg.SKIP_FRAMES}）"
    )
    parser.add_argument(
        "--scale", type=float, default=police_cfg.INFER_SCALE,
        help=f"推理缩放比（默认: {police_cfg.INFER_SCALE}）"
    )
    parser.add_argument(
        "--no-hand", action="store_true",
        help="不使用 Hand Landmarker（仅 Pose）"
    )
    parser.add_argument(
        "--no-dl", action="store_true",
        help="禁用深度学习模型（仅规则）"
    )
    parser.add_argument(
        "--dl-skip", type=int, default=3,
        help="深度学习模型跳帧数（默认: 3，性能优化）"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 应用命令行覆盖
    police_cfg.SKIP_FRAMES = args.skip
    police_cfg.INFER_SCALE = args.scale

    # ================================================================
    # 1. 加载 MediaPipe 模型
    # ================================================================
    if not os.path.exists(args.pose_model):
        print(f"[ERROR] Pose 模型不存在 -> {args.pose_model}")
        return

    print(f"[LOAD] Pose 模型: {args.pose_model}")
    pose_detector = create_pose_detector(args.pose_model)
    print("[ OK ] PoseLandmarker 就绪（33 个身体关键点）")

    hand_detector = None
    if not args.no_hand:
        if os.path.exists(args.hand_model):
            print(f"[LOAD] Hand 模型: {args.hand_model}")
            hand_detector = create_hand_detector(args.hand_model)
            print("[ OK ] HandLandmarker 就绪（21 个手部关键点）")
        else:
            print(f"[WARN] Hand 模型未找到 ({args.hand_model})，仅使用 Pose")

    # ================================================================
    # 2. 加载 CTPGREngine 深度学习模型
    # ================================================================
    dl_engine = None
    if _DL_AVAILABLE and not args.no_dl:
        print("[LOAD] CTPGREngine 深度学习模型（pose_model.pt + lstm.pt RNN）...")
        try:
            dl_engine = CTPGREngine()
            print("[ OK ] CTPGREngine 就绪（14 关键点 + LSTM 9 分类）")
        except Exception as e:
            import traceback
            print(f"[WARN] CTPGREngine 初始化失败: {e}")
            traceback.print_exc()
            dl_engine = None
    else:
        print(f"[INFO] 深度学习模型未启用")

    # ================================================================
    # 3. 打开摄像头
    # ================================================================
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 -> 索引 {args.camera}")
        pose_detector.close()
        if hand_detector:
            hand_detector.close()
        return

    fps_cam = cap.get(cv2.CAP_PROP_FPS)
    if fps_cam <= 0:
        fps_cam = 30
    print(f"[ OK ] 摄像头已打开  index={args.camera}  "
          f"FPS≈{fps_cam:.0f}  SKIP={police_cfg.SKIP_FRAMES}  "
          f"SCALE={police_cfg.INFER_SCALE:.2f}")
    print("       按 'q' 键退出\n")

    # ================================================================
    # 4. 初始化状态机 + 状态变量
    # ================================================================
    state_machine = GestureStateMachine()
    frame_counter = 0
    global_frame = 0
    fps = 0
    last_time = time.time()

    # 缓存上一帧推理结果
    last_landmarks = None
    last_world_landmarks = None
    last_feat = None
    last_hand_left = None
    last_hand_right = None
    left_palm_ori = '?'
    right_palm_ori = '?'

    # 识别结果显示计时器
    display_result = None
    display_confidence = 0.0
    result_display_timer = 0

    # 深度学习模型状态
    dl_gesture = "loading..."
    dl_confidence = 0.0
    dl_counter = 0
    dl_error_once = False

    DISPLAY_SCALE = 0.7

    # ---- 主循环 ----
    while True:
        t0 = time.time()

        ret, frame = cap.read()
        if not ret:
            print("\n摄像头读取失败，退出。")
            break

        frame_counter += 1
        should_infer = (frame_counter % police_cfg.SKIP_FRAMES == 0)

        # FPS 计算
        now = time.time()
        elapsed = now - last_time
        if elapsed > 0:
            fps = 30 if elapsed <= 0 else int(1.0 / elapsed)
        last_time = now

        global_frame += 1
        h, w = frame.shape[:2]

        # ============================================================
        # 5. AI 推理（跳帧执行）
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

            # 肩部标签 L / R
            cv2.putText(frame, "L", (int(left_shoulder[0]) - 15, int(left_shoulder[1]) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            cv2.putText(frame, "R", (int(right_shoulder[0]) + 5, int(right_shoulder[1]) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3)

            shoulder_width = calc_dist(left_shoulder, right_shoulder)

            # ---- 局部坐标系（用于手掌朝向） ----
            sm, body_right_2d, body_up_2d = setup_local_frame(
                left_shoulder, right_shoulder, nose, left_hip, right_hip
            )

            # ---- 特征提取 ----
            if world_landmarks is not None:
                feat = extract_features(world_landmarks, landmarks,
                                        last_hand_left, last_hand_right)

                # 被遮挡手臂特征冻结
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

            # ---- Hand 关联 ----
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
                    result_display_timer = police_cfg.RESULT_DISPLAY_FRAMES

        elif should_infer:
            # 未检测到人体
            state_machine.cancel_action(global_frame)
            last_landmarks = None
            last_world_landmarks = None
            last_feat = None
            last_hand_left = None
            last_hand_right = None

        # ---- 深度学习模型推理 ----
        if dl_engine is not None and should_infer and frame_counter % args.dl_skip == 0:
            dl_counter += 1
            dl_h, dl_w = 256, 256
            dl_frame = cv2.resize(frame, (dl_w, dl_h))
            try:
                dl_result = dl_engine.predict_frame(dl_frame)
                dl_gesture = dl_result["gesture"]
                dl_confidence = dl_result["confidence"]
            except Exception as e:
                if not dl_error_once:
                    print(f"\n[DL-ERROR] 推理异常: {e}")
                    dl_error_once = True
                dl_gesture = "DL error"
                dl_confidence = 0.0

        # ---- 非推理帧：绘制缓存骨架（防止闪烁） ----
        if not should_infer and last_landmarks is not None:
            draw_pose_landmarks(frame, last_landmarks, h, w)
            if last_hand_left:
                draw_hand_landmarks(frame, last_hand_left, h, w)
            else:
                draw_wrist_marker(frame, last_landmarks, 15, h, w, side='L')
            if last_hand_right:
                draw_hand_landmarks(frame, last_hand_right, h, w)
            else:
                draw_wrist_marker(frame, last_landmarks, 16, h, w, side='R')

        # ============================================================
        # 6. 画面文字叠加
        # ============================================================
        # 结果展示计时器
        if result_display_timer > 0:
            result_display_timer -= 1
        else:
            display_result = None
            display_confidence = 0.0

        # ---- 顶部标题栏 ----
        title_bar_h = 36
        cv2.rectangle(frame, (0, 0), (w, title_bar_h), (30, 30, 30), -1)
        cv2.putText(frame, "Camera Gesture Recognition",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

        # ---- 左上角：规则模型结果 ----
        rule_x, rule_y = 10, title_bar_h + 12
        box_w, box_h = 280, 120

        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (rule_x, rule_y), (rule_x + box_w, rule_y + box_h),
                      (40, 40, 40), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        # 标题
        cv2.putText(frame, "Rule Model", (rule_x + 10, rule_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        line_y = rule_y + 40

        # 状态
        if last_landmarks is None:
            status_text = "No person"
            status_color = (0, 0, 255)
        elif state_machine.state == police_cfg.STATE_ACTIVE:
            status_text = "识别中..."
            status_color = (0, 255, 255)
        elif state_machine.cooldown_counter > 0:
            status_text = f"冷却 {state_machine.cooldown_counter}"
            status_color = (128, 128, 128)
        elif display_result is not None:
            status_text = display_result
            status_color = (0, 255, 0)
        else:
            status_text = "等待动作"
            status_color = (255, 255, 255)

        frame = draw_chinese_text(frame, status_text,
                                  (rule_x + 10, line_y), status_color, 22)
        line_y += 24

        # 置信度（规则模型）
        if display_result is not None and display_confidence > 0:
            cv2.putText(frame, f"{display_confidence:.0%}",
                        (rule_x + 10, line_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 2)
        else:
            cv2.putText(frame, "---",
                        (rule_x + 10, line_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 128), 2)
        line_y += 22

        # 双手区域
        if last_feat is not None:
            left_region  = last_feat.get("left_region", "?")
            right_region = last_feat.get("right_region", "?")
            left_raise   = last_feat.get("left_raise", 0.0)
            right_raise  = last_feat.get("right_raise", 0.0)
        else:
            left_region = right_region = "?"
            left_raise = right_raise = 0.0

        cv2.putText(frame, f"L: {left_region}  ({left_raise:+.2f})",
                    (rule_x + 10, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 2)
        line_y += 20
        cv2.putText(frame, f"R: {right_region} ({right_raise:+.2f})",
                    (rule_x + 10, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 100, 255), 2)

        # ---- 左下角：深度学习模型结果 ----
        if dl_engine is not None:
            dl_x, dl_y = 10, h - 85
            dl_box_w, dl_box_h = 280, 72

            overlay = frame.copy()
            cv2.rectangle(overlay, (dl_x, dl_y), (dl_x + dl_box_w, dl_y + dl_box_h),
                          (25, 25, 50), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            cv2.putText(frame, "DL Model", (dl_x + 10, dl_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
            frame = draw_chinese_text(frame, dl_gesture,
                                      (dl_x + 10, dl_y + 40), (255, 200, 0), 22)
            cv2.putText(frame, f"{dl_confidence:.0%}",
                        (dl_x + 10, dl_y + 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 160, 80), 2)
        else:
            dl_x, dl_y = 10, h - 60
            cv2.putText(frame, "DL: N/A", (dl_x + 10, dl_y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)

        # ---- 右上角：信息 ----
        info_x = w - 210
        cv2.putText(frame, f"FPS: {fps}  Frame: {global_frame}",
                    (info_x, title_bar_h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)
        cv2.putText(frame, f"State: {state_machine.state_name}",
                    (info_x, title_bar_h + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 2)
        cv2.putText(frame, f"Cam: {args.camera}",
                    (info_x, title_bar_h + 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 2)

        # ---- 动作开始文字提示 ----
        if last_feat is not None and state_machine.state == police_cfg.STATE_ACTIVE:
            fc = state_machine.action_data.get("frame_count", 0) if state_machine.action_data else 0
            if fc == 1:
                frame = draw_chinese_text(frame, "动作开始！",
                                          (10, title_bar_h + 140), (0, 255, 0), 28)

        # ---- 显示 ----
        display_frame = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        cv2.imshow("Camera Gesture Recognition", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n用户按下 'q' 键，退出。")
            break

    # ================================================================
    # 7. 清理资源
    # ================================================================
    cap.release()
    cv2.destroyAllWindows()
    pose_detector.close()
    if hand_detector:
        hand_detector.close()
    print("[DONE] 识别结束")


if __name__ == "__main__":
    main()
