"""
recognize_camera.py — 摄像头实时交警手势识别

管线完全对标 run_recognition.py（视频识别），唯一区别是输入源为摄像头。
  - MediaPipe Pose + Hand 骨架检测与绘制
  - CTPGREngine（PyTorch LSTM）深度学习模型唯一判定手势
  - 预热期显示倒计时提示，预热完毕后开始正常识别

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

# ---- 确保项目根目录在 sys.path 中 ----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---- 导入 police 包（MediaPipe 骨架 + 可视化） ----
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
    from ctpgr_engine import CTPGREngine, mediapipe_to_aic14
    _DL_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] CTPGREngine 导入失败: {e}，将仅使用规则模型")
    _DL_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="摄像头实时交警手势识别")
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
        help="禁用深度学习模型（仅规则）"
    )
    return parser.parse_args()



def main():
    args = parse_args()

    # 应用命令行覆盖
    if args.no_mirror:
        police_cfg.CAMERA_MIRRORED = False
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
    # 2. 加载 CTPGREngine 深度学习模型
    # ================================================================
    dl_engine = None
    if _DL_AVAILABLE and not args.no_dl:
        print("[LOAD] CTPGREngine 深度学习模型（仅 LSTM，复用 MediaPipe 关键点）...")
        try:
            dl_engine = CTPGREngine(load_pose_model=False)
            print("[ OK ] CTPGREngine 就绪（MediaPipe→AIC14 映射 + LSTM 9 分类）")
        except Exception as e:
            import traceback
            print(f"[WARN] CTPGREngine 初始化失败: {e}")
            traceback.print_exc()
            print("[WARN] 将仅使用规则模型，DL 模型不可用")
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
    # 4. 初始化状态（完全对标 run_recognition.py）
    # ================================================================
    state_machine = GestureStateMachine(verbose=False)
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

    # 深度学习模型状态
    dl_gesture = "预热中..."
    dl_confidence = 0.0
    dl_counter = 0
    dl_error_once = False

    # ---- 显示级帧持久化（摄像头关键点抖动导致置信度在 0.70 门槛附近振荡） ----
    #   保持已确认的手势显示，直到连续 N 个推理帧都为"无手势"才真正切换。
    #   SKIP_FRAMES=2 → 每秒约 15 次推理 → PERSIST=8 ≈ 0.5 秒容错窗口。
    dl_persist_gesture = "无手势"
    dl_persist_confidence = 0.0
    dl_persist_counter = 0
    PERSIST_MAX = 8

    # ---- 预热提示（对标 run_recognition.py：LSTM 从 h0/c0 零状态启动） ----
    # ★ 预热期间 SKIP 所有 DL 推理，LSTM 保持零状态，不做预喂帧。
    #    倒计时消失后 LSTM 才从零开始吃真实手势帧——和视频版行为完全一致。
    dl_warmed_up = False
    warmup_start_time = time.time()   # 预热提示开始时间
    WARMUP_SECONDS = 2.0              # 提示持续时间（秒）

    # 显示缩放
    DISPLAY_SCALE = 1.5

    print("[INFO] 预热 2 秒（期间不推理），请保持站立...")
    print("       倒计时消失后即可开始做手势\n")

    # ---- 主循环 ----
    while True:
        t0 = time.time()

        ret, frame = cap.read()
        if not ret:
            print("\n摄像头读取失败，退出。")
            break

        # 摄像头镜像（默认开启）
        if police_cfg.CAMERA_MIRRORED:
            frame = cv2.flip(frame, 1)

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
        # 5. AI 推理（跳帧执行）—— 对标 run_recognition.py
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

            # ---- 状态机更新（仅追踪内部状态，不作为识别依据） ----
            if feat is not None:
                state_machine.update(
                    feat, left_palm_ori, right_palm_ori,
                    global_frame, feat.get("shoulder_width", 0.35)
                )

            # ---- 深度学习模型推理（复用 MediaPipe 关键点，无需 VGG+PAFs） ----
            if dl_engine is not None:
                # 预热检查
                if not dl_warmed_up:
                    if (time.time() - warmup_start_time) >= WARMUP_SECONDS:
                        dl_warmed_up = True
                        dl_engine.reset_state()
                        print(f"\n[DL-OK] 预热完成，LSTM 状态已重置为 h0/c0，开始实时识别")
                        print("        现在可以开始做手势了！\n")

                # 仅预热完成后才推理
                if dl_warmed_up:
                    dl_counter += 1
                    try:
                        coord_aic = mediapipe_to_aic14(landmarks)
                        dl_result = dl_engine.predict_from_keypoints(coord_aic)
                        raw_gesture = dl_result["gesture"]
                        raw_confidence = dl_result["confidence"]
                    except Exception as e:
                        if not dl_error_once:
                            print(f"\n[DL-ERROR] 推理异常: {e}")
                            dl_error_once = True
                        raw_gesture = "DL error"
                        raw_confidence = 0.0

                    # ---- 帧持久化：防止置信度在 0.70 门槛附近抖动导致闪烁 ----
                    if raw_gesture not in ("无手势", "DL error", "预热中..."):
                        # 有效手势 → 更新持久化缓存（同手势续命，新手势切换）
                        dl_persist_gesture = raw_gesture
                        dl_persist_confidence = raw_confidence
                        dl_persist_counter = PERSIST_MAX
                        dl_gesture = raw_gesture
                        dl_confidence = raw_confidence
                    elif dl_persist_counter > 0:
                        # "无手势"但还在持久窗口内 → 保持上一手势，置信度衰减
                        dl_gesture = dl_persist_gesture
                        decay = dl_persist_counter / PERSIST_MAX
                        dl_confidence = dl_persist_confidence * max(0.55, decay)
                        dl_persist_counter -= 1
                    else:
                        # 持久化窗口耗尽 → 真正显示"无手势"
                        dl_gesture = raw_gesture
                        dl_confidence = raw_confidence

                    if dl_counter % 30 == 0:
                        left_r = (last_feat or {}).get("left_region", "?")
                        right_r = (last_feat or {}).get("right_region", "?")
                        persist_mark = "*" if raw_gesture == "无手势" and dl_persist_counter > 0 else ""
                        print(f"  [DL] {dl_gesture}{persist_mark} (置信度:{dl_confidence:.0%})  "
                              f"| 左手: {left_r}  右手: {right_r}")

        elif should_infer:
            # 未检测到人体
            state_machine.cancel_action(global_frame)
            last_landmarks = None
            last_world_landmarks = None
            last_feat = None
            last_hand_left = None
            last_hand_right = None
            dl_persist_counter = 0   # 人体丢失 → 立即清空持久化

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
        # 6. 画面文字叠加（对标 run_recognition.py）
        # ============================================================

        # ---- 预热提示（画面中央文字，2 秒后自动消失） ----
        if dl_engine is not None and not dl_warmed_up:
            remaining = WARMUP_SECONDS - (time.time() - warmup_start_time)
            if remaining > 0:
                # 半透明遮罩
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

                cv2.putText(frame, f"WARMING UP... {remaining:.0f}s",
                            (w // 2 - 160, h // 2 - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 4)
                frame = draw_chinese_text(frame, "请保持站立，倒计时结束后开始识别",
                                          (w // 2 - 190, h // 2 + 5),
                                          (0, 255, 255), 26)
                frame = draw_chinese_text(frame, "LSTM 将从零状态启动，直接识别手势",
                                          (w // 2 - 190, h // 2 + 42),
                                          (200, 200, 200), 22)

        # ---- 未检测到人体 ----
        if last_landmarks is None:
            frame = draw_chinese_text(frame, "未检测到人体",
                                      (10, 110), (0, 0, 255), 36)

        # ---- 右上角：双手高度区域 ----
        if last_feat is not None:
            rxx = w - 200
            left_region = last_feat.get("left_region", "?")
            right_region = last_feat.get("right_region", "?")
            frame = draw_chinese_text(frame, f"左手: {left_region}",
                                      (rxx, 15), (0, 200, 255), 20)
            frame = draw_chinese_text(frame, f"右手: {right_region}",
                                      (rxx, 42), (200, 100, 255), 20)

        # ---- 左上角：深度学习模型结果（唯一识别依据） ----
        if dl_engine is not None:
            panel_w, panel_h = 260, 80
            panel_x, panel_y = 15, 15

            overlay = frame.copy()
            cv2.rectangle(overlay, (panel_x, panel_y),
                          (panel_x + panel_w, panel_y + panel_h), (20, 20, 50), -1)
            cv2.rectangle(overlay, (panel_x, panel_y),
                          (panel_x + panel_w, panel_y + panel_h), (0, 200, 255), 2)
            frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

            frame = draw_chinese_text(frame, "交警手势识别",
                                      (panel_x + 10, panel_y + 5),
                                      (0, 255, 255), 22)
            frame = draw_chinese_text(frame, dl_gesture,
                                      (panel_x + 10, panel_y + 28),
                                      (0, 215, 255), 28)
            frame = draw_chinese_text(frame, f"置信度: {dl_confidence:.1%}",
                                      (panel_x + 10, panel_y + 60),
                                      (255, 255, 255), 18)

        # ---- 底部：FPS 和帧号 ----
        cv2.putText(frame, f"FPS: {fps}  Frame: {global_frame}",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # ---- 显示 ----
        display_frame = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        cv2.imshow("Police Gesture Recognition", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n用户按下 'q' 键，退出。")
            break

    # ================================================================
    # 7. 清理
    # ================================================================
    cap.release()
    cv2.destroyAllWindows()
    pose_detector.close()
    if hand_detector:
        hand_detector.close()
    print("[DONE] 识别结束")


if __name__ == "__main__":
    main()
