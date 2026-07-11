"""
run_recognition.py — 交警手势识别主程序

使用 police/ 包（MediaPipe Pose + Hand + 状态机）进行：
  - 人体骨架检测（面部关键点 + 身体骨骼 + 手部骨架连线）
  - 8 种标准交警手势实时分类：停止/直行/左转弯/左待转/右转弯/变道/减速/靠边停车
  - 画面叠加：骨架 + "动作开始"/"动作识别中"/识别结果/置信度
  - 可选：集成 CTPGREngine（PyTorch pose_model.pt + lstm.pt RNN）做二次验证

Usage:
    python run_recognition.py --source test.mp4
    python run_recognition.py --source 0                  # 摄像头
    python run_recognition.py --source rtsp://...         # RTSP 流
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
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="交警手势识别系统 — 骨架可视化 + 8 种手势分类",
    )
    parser.add_argument(
        "--source", "-s",
        default="test.mp4",
        help="视频源：本地文件路径、RTSP URL 或摄像头索引（默认: test.mp4）"
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
        "--no-mirror", action="store_true",
        help="禁用摄像头镜像"
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
        help="禁用 CTPGREngine 深度学习模型（仅使用规则模型）"
    )
    parser.add_argument(
        "--dl-skip", type=int, default=3,
        help="深度学习模型跳帧数（默认: 3，性能优化）"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 应用命令行覆盖
    police_cfg.CAMERA_MIRRORED = not args.no_mirror
    police_cfg.SKIP_FRAMES = args.skip
    police_cfg.INFER_SCALE = args.scale

    # ================================================================
    # 1. 加载 MediaPipe 模型
    # ================================================================
    if not os.path.exists(args.pose_model):
        print(f"[ERROR] Pose 模型不存在 -> {args.pose_model}")
        print("  请确认 backend/pose_landmarker_lite.task 文件存在")
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
    # 1.5 加载 CTPGREngine 深度学习模型（pose_model.pt + lstm.pt RNN）
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
            print("[WARN] 将仅使用规则模型，DL 模型不可用")
            dl_engine = None
    else:
        print(f"[INFO] 深度学习模型未启用  (_DL_AVAILABLE={_DL_AVAILABLE}, no_dl={args.no_dl})")

    # ================================================================
    # 2. 打开视频源
    # ================================================================
    source = args.source
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频源 -> {source}")
        pose_detector.close()
        if hand_detector:
            hand_detector.close()
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_delay = 1.0 / (fps_video * 0.5) if fps_video > 0 else 1.0 / 15  # 放慢到原始速度的 50%
    print(f"[ OK ] 视频源已打开  SKIP={police_cfg.SKIP_FRAMES}  "
          f"SCALE={police_cfg.INFER_SCALE:.2f}  "
          f"视频FPS={fps_video:.0f}  总帧={total_frames}  播放延迟={frame_delay:.1f}ms")
    print("       按 'q' 键退出\n")

    # ================================================================
    # 3. 初始化状态机 + 状态变量
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
    dl_error_once = False  # DL 错误只打印一次

    # 显示缩放（视频窗口缩小）
    DISPLAY_SCALE = 0.65  # 画面缩小到 65%

    # ---- 主循环 ----
    while True:
        t0 = time.time()  # 帧开始时间

        ret, frame = cap.read()
        if not ret:
            print("\n视频播放结束。")
            break

        frame_counter += 1
        should_infer = (frame_counter % police_cfg.SKIP_FRAMES == 0)

        # FPS 计算
        now = time.time()
        elapsed = now - last_time
        if elapsed > 0:
            fps = int(1.0 / elapsed)
        last_time = now

        global_frame += 1
        h, w = frame.shape[:2]

        # ============================================================
        # 4. AI 推理（跳帧执行）
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

            # ---- 绘制 Pose 骨架（33 个关键点 + 连线） ----
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

            # ---- 状态机更新（8 种交警手势分类） ----
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

        # ---- 深度学习模型推理（CTPGREngine: pose_model.pt + lstm.pt RNN） ----
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
            if dl_counter % 30 == 0:
                print(f"  [DL] 第{dl_counter}次推理 → {dl_gesture} ({dl_confidence:.0%})", end="\r")

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
        # 5. 画面文字叠加
        # ============================================================
        # 结果展示计时器
        if result_display_timer > 0:
            result_display_timer -= 1
        else:
            display_result = None
            display_confidence = 0.0

        status_y = 30

        if last_landmarks is None:
            # 未检测到人体
            frame = draw_chinese_text(frame, "未检测到人体",
                                      (10, status_y), (0, 0, 255), 36)

        elif state_machine.state == police_cfg.STATE_ACTIVE:
            # 动作识别中（黄色）
            frame = draw_chinese_text(frame, "动作识别中...",
                                      (10, status_y), (0, 255, 255), 30)

        elif state_machine.cooldown_counter > 0:
            # 冷却中
            frame = draw_chinese_text(
                frame, f"冷却中 {state_machine.cooldown_counter}",
                (10, status_y), (128, 128, 128), 28)

        elif display_result is not None:
            # 刚完成的识别结果（绿色）
            text = f"交警手势: {display_result}"
            frame = draw_chinese_text(frame, text, (10, status_y), (0, 255, 0), 36)
            if display_confidence > 0:
                conf_text = f"置信度: {display_confidence:.0%}"
                frame = draw_chinese_text(frame, conf_text, (10, 75),
                                          (0, 200, 0), 22)

        else:
            # 等待动作
            frame = draw_chinese_text(frame, "等待动作...",
                                      (10, status_y), (255, 255, 255), 30)

        # ---- 动作开始时文字提示 ----
        if last_feat is not None and state_machine.state == police_cfg.STATE_ACTIVE:
            fc = state_machine.action_data.get("frame_count", 0) if state_machine.action_data else 0
            if fc == 1:
                frame = draw_chinese_text(frame, "动作开始！",
                                          (10, status_y + 45), (0, 255, 0), 28)

        # ---- FPS 和帧号 ----
        cv2.putText(frame, f"FPS: {fps}  Frame: {global_frame}",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # ---- 右上角：状态 + 双手高度区域 ----
        if last_feat is not None:
            rxx = w - 220
            cv2.putText(frame, f"State: {state_machine.state_name}",
                        (rxx, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            left_region = last_feat.get("left_region", "?")
            right_region = last_feat.get("right_region", "?")
            cv2.putText(frame, f"L: {left_region}",
                        (rxx, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 2)
            cv2.putText(frame, f"R: {right_region}",
                        (rxx, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 100, 255), 2)

        # ---- 底部：深度学习模型结果（CTPGREngine: pose_model.pt + lstm.pt RNN） ----
        if dl_engine is not None:
            dl_bottom_y = h - 45
            # 分隔线
            cv2.line(frame, (10, dl_bottom_y - 5), (w - 10, dl_bottom_y - 5),
                     (80, 80, 80), 1)
            cv2.putText(frame, "DL:", (10, dl_bottom_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 2)
            # 手势名
            frame = draw_chinese_text(frame, dl_gesture,
                                      (50, dl_bottom_y - 8), (255, 200, 0), 24)
            # 置信度
            cv2.putText(frame, f"{dl_confidence:.0%}",
                        (50, dl_bottom_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 160, 80), 2)

        # ---- 显示 ----
        display_frame = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        cv2.imshow("Police Gesture Recognition", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n用户按下 'q' 键，退出。")
            break

        # ---- 帧率控制（限制播放速度） ----
        elapsed_frame = time.time() - t0
        sleep_ms = int((frame_delay - elapsed_frame) * 1000)
        if sleep_ms > 1:
            cv2.waitKey(sleep_ms)

    # ================================================================
    # 6. 清理资源
    # ================================================================
    cap.release()
    cv2.destroyAllWindows()
    pose_detector.close()
    if hand_detector:
        hand_detector.close()
    print("[DONE] 识别结束")


if __name__ == "__main__":
    main()
