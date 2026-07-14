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
    parser.add_argument(
        "--num-poses", type=int, default=police_cfg.NUM_POSES_MULTI,
        help=f"同时检测的最大人数，用于排除其他人物的干扰（默认: {police_cfg.NUM_POSES_MULTI}）"
    )
    parser.add_argument(
        "--no-lock", action="store_true",
        help="禁用目标人物锁定（使用所有人物的第一个检测结果）"
    )
    return parser.parse_args()



def _select_target_person(pose_landmarks_list, frame_w, frame_h, prefer_center=True):
    """
    从多个检测到的人体中选出目标人物。

    优先策略：
      1) 选取肩宽最大的（通常离摄像头最近、最可能是测试者）
      2) 若 prefer_center=True，在肩宽相近时优先选靠近画面中心的

    Args:
        pose_landmarks_list: MediaPipe Pose 归一化关键点列表
        frame_w, frame_h: 画面尺寸
        prefer_center: 是否优先画面中心

    Returns:
        (best_idx, person_info_dict) 或 (0, None)
    """
    if not pose_landmarks_list:
        return 0, None

    best_idx = 0
    best_score = -1.0
    best_info = None

    for i, lm in enumerate(pose_landmarks_list):
        # 肩宽（归一化坐标）
        ls = (lm[11].x, lm[11].y)
        rs = (lm[12].x, lm[12].y)
        sw = ((ls[0] - rs[0]) ** 2 + (ls[1] - rs[1]) ** 2) ** 0.5

        # 人体中心坐标（归一化）
        cx = (lm[11].x + lm[12].x + lm[23].x + lm[24].x) / 4.0
        cy = (lm[11].y + lm[12].y + lm[23].y + lm[24].y) / 4.0

        # 距画面中心的距离
        dist_to_center = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5

        # 综合评分：肩宽大 + 靠近中心
        score = sw * 3.0 - dist_to_center * 0.5
        if score > best_score:
            best_score = score
            best_idx = i
            best_info = {"sw": sw, "cx": cx, "cy": cy, "dist_center": dist_to_center}

    return best_idx, best_info


def _track_target_person(pose_landmarks_list, last_target_info, track_thresh=0.35):
    """
    帧间追踪目标人物：找到与上一帧目标位置最接近的人体。

    Args:
        pose_landmarks_list: 当前帧所有检测到的人体关键点
        last_target_info: 上一帧目标人物的信息 {"cx", "cy", ...}
        track_thresh: 匹配距离阈值（归一化坐标）

    Returns:
        (matched_idx, matched_info) 或 (0, None) 如果匹配失败
    """
    if not pose_landmarks_list or last_target_info is None:
        return 0, None

    best_idx = 0
    best_dist = float('inf')
    best_info = None

    lcx, lcy = last_target_info.get("cx", 0.5), last_target_info.get("cy", 0.5)

    for i, lm in enumerate(pose_landmarks_list):
        cx = (lm[11].x + lm[12].x + lm[23].x + lm[24].x) / 4.0
        cy = (lm[11].y + lm[12].y + lm[23].y + lm[24].y) / 4.0
        dist = ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_idx = i
            best_info = {"cx": cx, "cy": cy}

    if best_dist > track_thresh:
        # 距离太远，可能目标人物丢失，回退到选择最靠近中心的
        return _select_target_person(pose_landmarks_list, 1.0, 1.0, prefer_center=True)

    return best_idx, best_info


def main():
    args = parse_args()

    # 应用命令行覆盖
    if args.no_mirror:
        police_cfg.CAMERA_MIRRORED = False
    police_cfg.SKIP_FRAMES = args.skip
    police_cfg.INFER_SCALE = args.scale

    # 是否启用目标人物锁定
    enable_target_lock = not args.no_lock

    # ================================================================
    # 1. 加载 MediaPipe 模型
    # ================================================================
    if not os.path.exists(args.pose_model):
        print(f"[ERROR] Pose 模型不存在 -> {args.pose_model}")
        print("  请确认 backend/pose_landmarker_lite.task 文件存在")
        return

    # 临时覆盖 NUM_POSES，使 pose_detector 支持多人检测
    _saved_num_poses = police_cfg.NUM_POSES
    police_cfg.NUM_POSES = args.num_poses

    print(f"[LOAD] Pose 模型: {args.pose_model}")
    print(f"        多人物检测: 最多 {args.num_poses} 人")
    pose_detector = create_pose_detector(args.pose_model)
    print("[ OK ] PoseLandmarker 就绪（33 个身体关键点）")

    # 恢复原始配置
    police_cfg.NUM_POSES = _saved_num_poses

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

    # ================================================================
    # 动作起止状态机（基于手部区域 + DL 结果滤波）
    # ================================================================
    #   动作开始：任意一只手离开 hip 区域 → 立即进入 ACTIVE
    #   动作停止：双手回到 hip 区域连续 3 帧 → 立即回到 IDLE
    #   IDLE 状态：始终显示"无手势"
    #   ACTIVE 状态：过滤爆点，选取置信度 >60% 且持续 ≥3 帧的手势
    action_state = "idle"           # "idle" | "active"
    action_frame_count = 0          # 动作期间的推理帧数
    hip_both_frames = 0             # 双手在 hip 的连续帧数
    HIP_STOP_THRESHOLD = 8          # 连续 8 帧 → 动作停止

    # DL 结果滑动窗口（用于 ACTIVE 期间的滤波）
    dl_window = []                  # [(gesture, confidence), ...]
    dl_filtered_gesture = "无手势"
    dl_filtered_confidence = 0.0
    GESTURE_MIN_CONF = 0.60         # 最低置信度阈值
    GESTURE_MIN_RUN = 3             # 同一手势最少连续帧数
    FIRST_GUESS_FRAMES = 5          # 动作开始后 5 帧内必须给出最佳判定动作

    # 动作切换画面提示（避免乱码：使用 draw_chinese_text 走 PIL 渲染）
    action_flash_text = ""           # 当前显示的切换提示文字（"动作开始"/"动作结束"）
    action_flash_remaining = 0       # 剩余显示帧数
    ACTION_FLASH_FRAMES = 30         # 提示持续帧数（约 1 秒 @30fps）

    # ---- 预热提示（对标 run_recognition.py：LSTM 从 h0/c0 零状态启动） ----
    # ★ 预热期间 SKIP 所有 DL 推理，LSTM 保持零状态，不做预喂帧。
    #    倒计时消失后 LSTM 才从零开始吃真实手势帧——和视频版行为完全一致。
    dl_warmed_up = False
    warmup_start_time = time.time()   # 预热提示开始时间
    WARMUP_SECONDS = 2.0              # 提示持续时间（秒）

    # ---- 目标人物锁定与追踪（排除其他人物走动干扰） ----
    #   预热完成后前 TARGET_LOCK_FRAMES 帧锁定目标人物
    #   之后通过帧间位置匹配持续追踪该人物
    target_locked = False
    target_info = None              # {"cx", "cy", "sw", ...} 目标人物位置/体型信息
    target_lock_counter = 0
    TARGET_LOCK_FRAMES = police_cfg.TARGET_LOCK_FRAMES
    TARGET_TRACK_THRESH = police_cfg.TARGET_TRACK_THRESH
    multi_person_warned = False     # 多人检测提示仅打印一次

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
            num_detected = len(pose_result.pose_landmarks)

            # ---- 目标人物选择（排除其他人物的干扰） ----
            if enable_target_lock:
                if not target_locked:
                    # 预热完成后锁定目标人物：选肩宽最大 + 最靠近中心的
                    target_idx, person_info = _select_target_person(
                        pose_result.pose_landmarks, w, h
                    )
                    target_lock_counter += 1
                    if target_lock_counter >= TARGET_LOCK_FRAMES:
                        target_locked = True
                        target_info = person_info
                        if num_detected > 1:
                            print(f"\n[TARGET] 已锁定目标人物 (共检测到 {num_detected} 人)")
                            print(f"         目标位置: 中心({target_info['cx']:.2f}, {target_info['cy']:.2f})")
                            multi_person_warned = True
                else:
                    # 追踪已锁定的目标人物
                    target_idx, person_info = _track_target_person(
                        pose_result.pose_landmarks, target_info, TARGET_TRACK_THRESH
                    )
                    if person_info:
                        # EMA 平滑更新目标位置
                        alpha = 0.7
                        target_info["cx"] = alpha * person_info["cx"] + (1 - alpha) * target_info["cx"]
                        target_info["cy"] = alpha * person_info["cy"] + (1 - alpha) * target_info["cy"]

                    if num_detected > 1 and not multi_person_warned:
                        print(f"\n[INFO] 检测到 {num_detected} 人，已锁定目标人物进行追踪")
                        multi_person_warned = True
            else:
                # 未启用目标锁定时，使用默认的第一个检测结果
                target_idx = 0
                if num_detected > 1 and not multi_person_warned:
                    print(f"\n[WARN] 检测到 {num_detected} 人，但目标锁定已禁用，可能受其他人物干扰")
                    multi_person_warned = True

            landmarks = pose_result.pose_landmarks[target_idx]
            last_landmarks = landmarks

            # ---- 多人物可视化：目标人物绿色骨架，其他人物灰色骨架 ----
            if enable_target_lock and num_detected > 1:
                for i, other_lm in enumerate(pose_result.pose_landmarks):
                    if i == target_idx:
                        continue
                    # 灰色半透明绘制其他人物
                    draw_pose_landmarks(frame, other_lm, h, w,
                                        point_color=(100, 100, 100),
                                        line_color=(80, 80, 80))

            world_landmarks = None
            if (pose_result.pose_world_landmarks and
                    target_idx < len(pose_result.pose_world_landmarks)):
                world_landmarks = pose_result.pose_world_landmarks[target_idx]
                last_world_landmarks = world_landmarks
            else:
                world_landmarks = last_world_landmarks

            # ---- 绘制 Pose 骨架（33 个关键点 + 连线） ----
            draw_pose_landmarks(frame, landmarks, h, w)

            # ---- 目标人物边界框（多人检测时用绿色标注追踪目标） ----
            if enable_target_lock and num_detected > 1:
                xs = [int(lm.x * w) for lm in landmarks]
                ys = [int(lm.y * h) for lm in landmarks]
                if xs and ys:
                    x1, y1 = max(min(xs) - 20, 0), max(min(ys) - 30, 0)
                    x2, y2 = min(max(xs) + 20, w), min(max(ys) + 20, h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "ME", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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

            # ============================================================
            # 动作起止检测（基于手部区域）
            #   开始：任意手离开 hip → 立即 ACTIVE
            #   停止：双手在 hip 连续 3 帧 → 立即 IDLE
            # ============================================================
            if feat is not None:
                lr = feat.get("left_region", "?")
                rr = feat.get("right_region", "?")
                both_hip = (lr == "hip" and rr == "hip")

                if action_state == "idle":
                    # 空闲状态：检测是否有手离开 hip → 动作开始
                    if not both_hip:
                        action_state = "active"
                        action_frame_count = 0
                        dl_window.clear()
                        dl_filtered_gesture = "无手势"
                        dl_filtered_confidence = 0.0
                        hip_both_frames = 0
                        # 重置 LSTM 状态，开启新一轮独立判定
                        if dl_engine is not None:
                            dl_engine.reset_state()
                        # 画面提示
                        action_flash_text = "动作开始"
                        action_flash_remaining = ACTION_FLASH_FRAMES
                else:  # active
                    action_frame_count += 1
                    if both_hip:
                        hip_both_frames += 1
                        if hip_both_frames >= HIP_STOP_THRESHOLD:
                            # 动作停止 → 立即切回 idle，清空所有判定状态
                            action_state = "idle"
                            dl_window.clear()
                            dl_filtered_gesture = "无手势"
                            dl_filtered_confidence = 0.0
                            dl_gesture = "无手势"
                            dl_confidence = 0.0
                            # 重置 LSTM 状态，确保下一个动作不受影响
                            if dl_engine is not None:
                                dl_engine.reset_state()
                            # 画面提示
                            action_flash_text = "动作结束"
                            action_flash_remaining = ACTION_FLASH_FRAMES
                    else:
                        hip_both_frames = 0

            # ---- 深度学习模型推理（复用 MediaPipe 关键点，无需 VGG+PAFs） ----
            if dl_engine is not None:
                # 预热检查
                if not dl_warmed_up:
                    if (time.time() - warmup_start_time) >= WARMUP_SECONDS:
                        dl_warmed_up = True
                        dl_engine.reset_state()
                        # 预热完成 → 启动目标人物锁定流程
                        target_locked = False
                        target_lock_counter = 0
                        target_info = None
                        multi_person_warned = False
                        print(f"\n[DL-OK] 预热完成，LSTM 状态已重置为 h0/c0，开始实时识别")
                        print("        正在锁定目标人物...\n")

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

                    # ========================================================
                    # 动作期间 DL 结果滤波
                    #   - 动作开始后 5 帧内必须给出手势判定（不可以是"未知动作"）
                    #   - 之后要求置信度 >60% + 同一手势连续 ≥3 帧
                    #   - 去除爆点（不满足条件的单帧孤立预测）
                    # ========================================================
                    if action_state == "active":
                        # 将当前帧原始结果加入滑动窗口
                        dl_window.append((raw_gesture, raw_confidence))
                        if len(dl_window) > 60:
                            dl_window = dl_window[-40:]

                        if action_frame_count <= FIRST_GUESS_FRAMES:
                            # 前 5 帧：找最好的非"无手势"结果作为初始判定
                            best = None
                            for g, c in dl_window:
                                if g not in ("无手势", "DL error", "预热中...") and c > 0:
                                    if best is None or c > best[1]:
                                        best = (g, c)
                            if best:
                                dl_filtered_gesture, dl_filtered_confidence = best
                            # 若无有效手势 → 保持上一帧判定，绝不输出"未知动作"
                        else:
                            # > 3 帧：分析最近帧的连续手势段
                            recent = dl_window[-15:]

                            # 统计连续手势段
                            runs = []  # [(gesture, count, avg_conf), ...]
                            if recent:
                                cur_g, cur_confs = recent[0][0], [recent[0][1]]
                                for i in range(1, len(recent)):
                                    g, c = recent[i]
                                    if g == cur_g:
                                        cur_confs.append(c)
                                    else:
                                        avg = sum(cur_confs) / len(cur_confs)
                                        runs.append((cur_g, len(cur_confs), avg))
                                        cur_g, cur_confs = g, [c]
                                if cur_g:
                                    avg = sum(cur_confs) / len(cur_confs)
                                    runs.append((cur_g, len(cur_confs), avg))

                            # 筛选有效手势：置信度 >60% 且连续 ≥3 帧
                            valid = [
                                (g, cnt, conf) for g, cnt, conf in runs
                                if g not in ("无手势", "DL error", "预热中...")
                                and conf >= GESTURE_MIN_CONF and cnt >= GESTURE_MIN_RUN
                            ]

                        if valid:
                            # 选取最近的有效段作为当前判定
                            best = valid[-1]  # (gesture, count, avg_confidence)
                            dl_filtered_gesture = best[0]
                            dl_filtered_confidence = best[2]  # ← 置信度，非帧数
                            # 若无有效手势段，则保持上一帧的 dl_filtered_gesture

                        dl_gesture = dl_filtered_gesture
                        dl_confidence = dl_filtered_confidence
                    else:
                        # idle 状态：始终显示"无手势"
                        dl_gesture = "无手势"
                        dl_confidence = 0.0
                        dl_window.clear()

                    if dl_counter % 30 == 0:
                        left_r = (last_feat or {}).get("left_region", "?")
                        right_r = (last_feat or {}).get("right_region", "?")
                        state_mark = "[A]" if action_state == "active" else "[I]"
                        print(f"  [DL] {state_mark} {dl_gesture} (置信度:{dl_confidence:.0%})  "
                              f"| 左手: {left_r}  右手: {right_r}")

        elif should_infer:
            # 未检测到人体
            state_machine.cancel_action(global_frame)
            last_landmarks = None
            last_world_landmarks = None
            last_feat = None
            last_hand_left = None
            last_hand_right = None
            # 人体丢失 → 立即重置动作状态
            action_state = "idle"
            action_frame_count = 0
            hip_both_frames = 0
            dl_window.clear()
            dl_filtered_gesture = "无手势"
            dl_filtered_confidence = 0.0
            # 人体丢失 → 重新锁定目标人物
            if target_locked:
                target_locked = False
                target_lock_counter = 0
                target_info = None

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

        # ---- 多人检测 / 目标锁定状态（左上角面板下方） ----
        if enable_target_lock and last_landmarks is not None and dl_warmed_up:
            # 获取当前帧检测到的人数（从缓存的 pose_result 推断）
            multi_info_y = 105
            if target_locked:
                frame = draw_chinese_text(frame, "已锁定目标",
                                          (25, multi_info_y),
                                          (0, 255, 0), 18)
                if target_info:
                    frame = draw_chinese_text(
                        frame,
                        f"位置: ({target_info.get('cx', 0):.2f}, {target_info.get('cy', 0):.2f})",
                        (25, multi_info_y + 22),
                        (180, 180, 180), 16
                    )
            elif target_lock_counter > 0:
                remaining = TARGET_LOCK_FRAMES - target_lock_counter
                frame = draw_chinese_text(
                    frame,
                    f"锁定目标中... {remaining}/{TARGET_LOCK_FRAMES}",
                    (25, multi_info_y),
                    (0, 200, 255), 18
                )

        # ---- 动作切换提示（画面中央偏上） ----
        if action_flash_remaining > 0:
            # 半透明横幅背景
            banner_w, banner_h = 300, 50
            banner_x = (w - banner_w) // 2
            banner_y = h // 8
            overlay = frame.copy()
            cv2.rectangle(overlay,
                          (banner_x, banner_y),
                          (banner_x + banner_w, banner_y + banner_h),
                          (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

            if action_flash_text == "动作开始":
                flash_color = (0, 255, 0)   # 绿色
            else:
                flash_color = (0, 140, 255)  # 橙色
            frame = draw_chinese_text(frame, action_flash_text,
                                      (banner_x + 65, banner_y + 8),
                                      flash_color, 32)
            action_flash_remaining -= 1

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
