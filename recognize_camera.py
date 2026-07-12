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
import numpy as np

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
    from ctpgr_engine import CTPGREngine
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
        "--dl-skip", type=int, default=3,
        help="深度学习模型跳帧数（默认: 3，性能优化）"
    )
    return parser.parse_args()


# ---- DL 模型 14 关键点骨架绘制（对标 aic_bones） ----
# 0-based 索引（AIChallenger: 1~14 → 0~13）
_DL_BONES = [
    (0, 1),   # 右大臂 (shoulder→elbow)
    (1, 2),   # 右小臂 (elbow→wrist)
    (3, 4),   # 左大臂
    (4, 5),   # 左小臂
    (13, 0),  # 右肩 (neck→shoulder)
    (13, 3),  # 左肩
    (0, 6),   # 右侧躯干 (shoulder→hip)
    (3, 9),   # 左侧躯干
    (6, 7),   # 右大腿 (hip→knee)
    (7, 8),   # 右小腿 (knee→ankle)
    (9, 10),  # 左大腿
    (10, 11), # 左小腿
    (12, 13), # 头 (head→neck)
]

# 下半身骨骼索引（6-11），用于检测是否拍到全身
_LOWER_BODY_BONES = {6, 7, 8, 9, 10, 11}

def _crop_person(frame, landmarks, target_size=512, padding_ratio=0.25):
    """根据 MediaPipe 33 关键点裁剪出人体区域并 resize 到 target_size×target_size"""
    h, w = frame.shape[:2]
    xs, ys = [], []
    for lm in landmarks:
        xs.append(int(lm.x * w))
        ys.append(int(lm.y * h))
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    # 加 padding
    bw, bh = x2 - x1, y2 - y1
    pad_w, pad_h = int(bw * padding_ratio), int(bh * padding_ratio)
    x1 = max(0, x1 - pad_w)
    x2 = min(w, x2 + pad_w)
    y1 = max(0, y1 - pad_h)
    y2 = min(h, y2 + pad_h)

    # 保持正方形（用较长边）
    box_w, box_h = x2 - x1, y2 - y1
    side = max(box_w, box_h)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = side // 2
    x1 = max(0, cx - half)
    x2 = min(w, cx + half)
    y1 = max(0, cy - half)
    y2 = min(h, cy + half)

    # 再次确保是正方形
    crop_w, crop_h = x2 - x1, y2 - y1
    side = min(crop_w, crop_h)
    crop = frame[y1:y1 + side, x1:x1 + side]
    crop = cv2.resize(crop, (target_size, target_size))
    return crop

def _draw_dl_skeleton(frame, kps):
    """在 frame 上绘制 DL 模型的 14 个关键点 + 骨骼连线"""
    h, w = frame.shape[:2]
    pts = []
    for kp in kps:
        px = int(kp["x"] * w)
        py = int(kp["y"] * h)
        pts.append((px, py))

    # 检测下半身是否在画面内（髋/膝/踝关键点 6-11）
    lower_pts = pts[6:12]  # right hip, right knee, right ankle, left hip, left knee, left ankle
    lower_y = [p[1] for p in lower_pts]
    lower_x = [p[0] for p in lower_pts]
    # 如果大部分下半身点位 y > 0.9h（接近底部）或 y < 0.1h（在顶部误检），说明没拍到腿
    bottom_frac = sum(1 for y in lower_y if y > h * 0.9) / len(lower_y)
    if bottom_frac > 0.5:
        cv2.putText(frame, "WARN: Lower body not visible!", (int(w * 0.02), int(h * 0.92)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, "Stand back so full body is in frame", (int(w * 0.02), int(h * 0.96)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)

    for bi, (i, j) in enumerate(_DL_BONES):
        if i >= len(pts) or j >= len(pts):
            continue
        pi, pj = pts[i], pts[j]
        # 下半身用红色（警告），上半身用绿色
        color = (0, 0, 255) if bi in _LOWER_BODY_BONES else (0, 255, 0)
        cv2.line(frame, pi, pj, color, 2)

    # 关键点：上半身蓝绿色，下半身橙色
    for idx, (px, py) in enumerate(pts):
        is_lower = any(idx in [b1, b2] for b1, b2 in [
            (6, 7), (7, 8), (9, 10), (10, 11)
        ])
        color = (0, 165, 255) if is_lower else (255, 200, 0)
        cv2.circle(frame, (px, py), 5, color, -1)
        cv2.circle(frame, (px, py), 6, (0, 0, 0), 1)


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

        elif should_infer:
            # 未检测到人体
            state_machine.cancel_action(global_frame)
            last_landmarks = None
            last_world_landmarks = None
            last_feat = None
            last_hand_left = None
            last_hand_right = None

        # ---- 深度学习模型推理（CTPGREngine: pose_model.pt + lstm.pt RNN） ----
        # 完全对标 run_recognition.py：LSTM 从 h0/c0 零状态启动，直接吃真实帧。
        # ★ 关键修复：预热期间 SKIP predict_frame，不往 LSTM 灌"站立不动"的帧。
        #    否则 2 秒 × 10 帧/秒 = 20 帧站立特征会锁死 LSTM 隐藏状态在 class 0。
        #    预热结束后 LSTM 从零状态接触手势帧，和视频版行为完全一致。
        if dl_engine is not None and frame_counter % args.dl_skip == 0:
            # ★ 关键优化：用 MediaPipe 骨架裁剪人体区域后再送入 DL 模型
            #    避免站太远时人影太小导致关键点检测失效
            if last_landmarks is not None:
                dl_frame = _crop_person(frame, last_landmarks, target_size=512)
            else:
                dl_frame = cv2.resize(frame, (512, 512))

            # ---- 预热：计时到 2 秒后标记 ready，期间不推理 ----
            if not dl_warmed_up:
                if (time.time() - warmup_start_time) >= WARMUP_SECONDS:
                    dl_warmed_up = True
                    # 预热结束，重置 LSTM 状态——确保从零开始，不受预热帧污染
                    dl_engine.reset_state()
                    print(f"\n[DL-OK] 预热完成，LSTM 状态已重置为 h0/c0，开始实时识别")
                    print("        现在可以开始做手势了！\n")
                # 预热期间：不推理也不 continue，让显示代码正常执行

            # ---- 仅预热完成后才推理 ----
            if dl_warmed_up:
                dl_counter += 1

                try:
                    dl_result = dl_engine.predict_frame(dl_frame)
                    dl_gesture = dl_result["gesture"]
                    dl_confidence = dl_result["confidence"]
                    raw_logits = dl_result.get("raw_logits", [])
                    dl_kps = dl_result.get("keypoints", [])
                except Exception as e:
                    if not dl_error_once:
                        print(f"\n[DL-ERROR] 推理异常: {e}")
                        dl_error_once = True
                    dl_gesture = "DL error"
                    dl_confidence = 0.0
                    raw_logits = []
                    dl_kps = []

                # 每 15 次推理打印诊断：校准后手势 + 原始偏差值
                if dl_counter % 15 == 0 and raw_logits:
                    left_r = (last_feat or {}).get("left_region", "?")
                    right_r = (last_feat or {}).get("right_region", "?")
                    # 从原始 logits 计算 class-0 偏差（校准前）
                    logits = np.array(raw_logits)
                    non0_logit = max(logits[1:]) if len(logits) > 1 else -999
                    bias0 = logits[0] - non0_logit
                    # 校准标记：如果原始 class-0 异常偏高
                    calibrated = "🔧已校准" if bias0 > 2.0 else " 正常"
                    print(f"  [DL] → {dl_gesture} ({dl_confidence:.0%})  "
                          f"| 原始偏差: {bias0:+.1f}  {calibrated}")
                    print(f"       手动区域: 左手={left_r}  右手={right_r}")

                # ---- 绘制 DL 模型的 14 个关键点到画面上（诊断用） ----
                if dl_kps:
                    _draw_dl_skeleton(frame, dl_kps)

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
