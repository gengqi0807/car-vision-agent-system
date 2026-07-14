"""
recognize_image.py — 单张图片交警手势识别

管线完全对标 run_recognition.py / recognize_camera.py，唯一区别是输入源为单张图片。
  - MediaPipe Pose + Hand 骨架检测与绘制
  - CTPGREngine（PyTorch LSTM）深度学习模型唯一判定手势
  - 规则模型特征提取 + 状态机（用于调试辅助）

Usage:
    python recognize_image.py
    python recognize_image.py --image try.jpg
    python recognize_image.py --image C:/path/to/photo.jpg --no-dl
"""

import os
import sys
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
    parser = argparse.ArgumentParser(description="单张图片交警手势识别")
    parser.add_argument(
        "--image", type=str, default="try.jpg",
        help="输入图片路径（默认: try.jpg）"
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
        "--no-hand", action="store_true",
        help="不使用 Hand Landmarker（仅 Pose）"
    )
    parser.add_argument(
        "--no-dl", action="store_true",
        help="禁用深度学习模型（仅规则）"
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
    # 3. 读取图片
    # ================================================================
    print(f"\n[READ] 图片: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] 无法读取图片 -> {image_path}")
        pose_detector.close()
        if hand_detector:
            hand_detector.close()
        return
    h, w = frame.shape[:2]
    print(f"       尺寸: {w}x{h}")

    # ================================================================
    # 4. 推理（单帧，无跳帧，对标 run_recognition.py 管线）
    # ================================================================
    print("\n--- 推理中 ---")

    state_machine = GestureStateMachine(verbose=False)

    pose_result = detect_pose(pose_detector, frame)
    hand_result = detect_hand(hand_detector, frame) if hand_detector else None

    if not pose_result or not pose_result.pose_landmarks:
        print("[WARN] 未检测到人体姿态")
        frame = draw_chinese_text(frame, "未检测到人体",
                                  (10, 110), (0, 0, 255), 36)
    else:
        landmarks = pose_result.pose_landmarks[0]

        world_landmarks = None
        if pose_result.pose_world_landmarks:
            world_landmarks = pose_result.pose_world_landmarks[0]

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
        last_feat = None
        last_hand_left = None
        last_hand_right = None

        # Hand 关联
        hands = associate_hands(
            hand_result,
            (landmarks[15].x, landmarks[15].y),
            (landmarks[16].x, landmarks[16].y),
        )
        last_hand_left  = hands["left"]
        last_hand_right = hands["right"]

        if world_landmarks is not None:
            feat = extract_features(world_landmarks, landmarks,
                                    last_hand_left, last_hand_right)
            feat["left_visible"] = True
            feat["right_visible"] = True
            last_feat = feat

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
        if last_feat is not None:
            state_machine.update(
                last_feat, left_palm_ori, right_palm_ori,
                1, last_feat.get("shoulder_width", 0.35)
            )

        # ---- 深度学习模型推理（复用 MediaPipe 关键点，无需 VGG+PAFs） ----
        dl_gesture = "N/A"
        dl_confidence = 0.0
        if dl_engine is not None:
            try:
                # ★ 重要：单张图片没有时序上下文，LSTM 从 h0/c0 直接推理。
                #    推理前重置 LSTM 状态，得到单帧的判断结果。
                dl_engine.reset_state()
                coord_aic = mediapipe_to_aic14(landmarks)
                dl_result = dl_engine.predict_from_keypoints(coord_aic)
                dl_gesture = dl_result["gesture"]
                dl_confidence = dl_result["confidence"]
            except Exception as e:
                print(f"  [DL-ERROR] 推理异常: {e}")
                dl_gesture = "推理失败"

            left_r = (last_feat or {}).get("left_region", "?")
            right_r = (last_feat or {}).get("right_region", "?")
            print(f"  [DL] {dl_gesture} (置信度:{dl_confidence:.0%})  "
                  f"| 左手: {left_r}  右手: {right_r}")

    # ================================================================
    # 5. 画面文字叠加（对标 run_recognition.py）
    # ================================================================

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

        # 半透明背景
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

    # ---- 底部：图片信息 ----
    cv2.putText(frame, f"Image: {os.path.basename(image_path)}  {w}x{h}",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # ================================================================
    # 6. 显示 / 保存
    # ================================================================
    print("\n--- 完成 ---")
    if last_feat:
        left_r = last_feat.get("left_region", "?")
        right_r = last_feat.get("right_region", "?")
        print(f"  规则模型: L={left_r}  R={right_r}")
    if dl_engine:
        print(f"  DL模型:   {dl_gesture}  ({dl_confidence:.0%})")

    if args.output:
        cv2.imwrite(args.output, frame)
        print(f"  结果已保存: {args.output}")

    if not args.no_display:
        display_frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
        cv2.imshow("Police Gesture Recognition", display_frame)
        print("\n按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 清理
    pose_detector.close()
    if hand_detector:
        hand_detector.close()


if __name__ == "__main__":
    main()
