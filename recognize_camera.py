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

import argparse
import os
import sys
import time

import cv2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from police import config as police_cfg
from police.features import associate_hands, classify_palm_orientation, extract_features
from police.gesture_classifier import GestureStateMachine
from police.geometry import calc_dist, setup_local_frame
from police.models import create_hand_detector, create_pose_detector, detect_hand, detect_pose
from police.visualization import draw_chinese_text, draw_hand_landmarks, draw_pose_landmarks, draw_wrist_marker

try:
    from ctpgr_engine import CTPGREngine, mediapipe_to_aic14

    _DL_AVAILABLE = True
except ImportError as exc:
    print(f"[WARN] CTPGREngine 导入失败: {exc}，将仅使用规则模型")
    _DL_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="摄像头实时交警手势识别")
    parser.add_argument(
        "--source",
        "-s",
        default=None,
        help="输入源：本地视频路径、RTSP 地址或摄像头索引；未提供时使用 --camera",
    )
    parser.add_argument("--camera", "-c", type=int, default=0, help="摄像头索引（默认: 0）")
    parser.add_argument("--pose-model", default=police_cfg.POSE_MODEL_PATH, help="MediaPipe Pose 模型路径")
    parser.add_argument("--hand-model", default=police_cfg.HAND_MODEL_PATH, help="MediaPipe Hand 模型路径")
    parser.add_argument("--no-mirror", action="store_true", help="禁用摄像头镜像")
    parser.add_argument("--skip", type=int, default=police_cfg.SKIP_FRAMES, help="跳帧数")
    parser.add_argument("--scale", type=float, default=police_cfg.INFER_SCALE, help="推理缩放比")
    parser.add_argument("--no-hand", action="store_true", help="不使用 Hand Landmarker（仅 Pose）")
    parser.add_argument("--no-dl", action="store_true", help="禁用深度学习模型（仅规则）")
    parser.add_argument("--no-display", action="store_true", help="不弹出 OpenCV 窗口，仅在终端输出日志")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理多少帧，0 表示不限制")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.no_mirror:
        police_cfg.CAMERA_MIRRORED = False
    police_cfg.SKIP_FRAMES = args.skip
    police_cfg.INFER_SCALE = args.scale

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

    dl_engine = None
    if _DL_AVAILABLE and not args.no_dl:
        print("[LOAD] CTPGREngine 深度学习模型（仅 LSTM，复用 MediaPipe 关键点）...")
        try:
            dl_engine = CTPGREngine(load_pose_model=False)
            print("[ OK ] CTPGREngine 就绪（MediaPipe→AIC14 映射 + LSTM 9 分类）")
        except Exception as exc:
            import traceback

            print(f"[WARN] CTPGREngine 初始化失败: {exc}")
            traceback.print_exc()
            print("[WARN] 将仅使用规则模型，DL 模型不可用")
            dl_engine = None
    else:
        print("[INFO] 深度学习模型未启用")

    source = args.source if args.source is not None else args.camera
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"[ERROR] 无法打开输入源 -> {source}")
        pose_detector.close()
        if hand_detector:
            hand_detector.close()
        return

    fps_cam = capture.get(cv2.CAP_PROP_FPS)
    if fps_cam <= 0:
        fps_cam = 30
    source_kind = "camera" if args.source is None else "video"
    print(
        f"[ OK ] 输入源已打开  type={source_kind}  source={source}  "
        f"FPS≈{fps_cam:.0f}  SKIP={police_cfg.SKIP_FRAMES}  SCALE={police_cfg.INFER_SCALE:.2f}"
    )
    print("       按 'q' 键退出\n")

    state_machine = GestureStateMachine(verbose=False)
    frame_counter = 0
    global_frame = 0
    fps = 0
    last_time = time.time()

    last_landmarks = None
    last_world_landmarks = None
    last_feat = None
    last_hand_left = None
    last_hand_right = None

    dl_gesture = "预热中..."
    dl_confidence = 0.0
    dl_counter = 0
    dl_error_once = False
    dl_persist_gesture = "无手势"
    dl_persist_confidence = 0.0
    dl_persist_counter = 0
    persist_max = 8
    dl_warmed_up = False
    warmup_start_time = time.time()
    warmup_seconds = 2.0
    display_scale = 1.5

    print("[INFO] 预热 2 秒（期间不推理），请保持站立...")
    print("       倒计时消失后即可开始做手势\n")

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                print("\n摄像头读取失败，退出。")
                break

            if police_cfg.CAMERA_MIRRORED:
                frame = cv2.flip(frame, 1)

            frame_counter += 1
            should_infer = frame_counter % police_cfg.SKIP_FRAMES == 0

            now = time.time()
            elapsed = now - last_time
            if elapsed > 0:
                fps = int(1.0 / elapsed)
            last_time = now

            global_frame += 1
            height, width = frame.shape[:2]

            if should_infer:
                pose_result = detect_pose(pose_detector, frame)
                hand_result = detect_hand(hand_detector, frame) if hand_detector is not None else None
            else:
                pose_result = None
                hand_result = None

            if should_infer and pose_result and pose_result.pose_landmarks:
                landmarks = pose_result.pose_landmarks[0]
                last_landmarks = landmarks

                if pose_result.pose_world_landmarks:
                    world_landmarks = pose_result.pose_world_landmarks[0]
                    last_world_landmarks = world_landmarks
                else:
                    world_landmarks = last_world_landmarks

                draw_pose_landmarks(frame, landmarks, height, width)

                def px(index):
                    landmark = landmarks[index]
                    return (landmark.x * width, landmark.y * height)

                left_shoulder = px(11)
                right_shoulder = px(12)
                nose = px(0)
                left_hip = px(23)
                right_hip = px(24)

                cv2.putText(
                    frame,
                    "L",
                    (int(left_shoulder[0]) - 15, int(left_shoulder[1]) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    3,
                )
                cv2.putText(
                    frame,
                    "R",
                    (int(right_shoulder[0]) + 5, int(right_shoulder[1]) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    3,
                )

                _ = calc_dist(left_shoulder, right_shoulder)
                _, body_right_2d, body_up_2d = setup_local_frame(
                    left_shoulder,
                    right_shoulder,
                    nose,
                    left_hip,
                    right_hip,
                )

                if world_landmarks is not None:
                    feat = extract_features(world_landmarks, landmarks, last_hand_left, last_hand_right)
                    if last_feat is not None:
                        left_visibility = landmarks[15].visibility if landmarks else 0.0
                        right_visibility = landmarks[16].visibility if landmarks else 0.0
                        left_keys = (
                            "left_raise",
                            "left_stretch",
                            "left_z_diff",
                            "left_wx",
                            "left_wy",
                            "left_sx",
                            "left_sy",
                            "left_orient",
                            "left_pose",
                            "left_region",
                            "left_fwd",
                            "left_lat",
                            "left_dir_raw",
                            "left_arm_angle",
                        )
                        right_keys = (
                            "right_raise",
                            "right_stretch",
                            "right_z_diff",
                            "right_wx",
                            "right_wy",
                            "right_sx",
                            "right_sy",
                            "right_orient",
                            "right_pose",
                            "right_region",
                            "right_fwd",
                            "right_lat",
                            "right_dir_raw",
                            "right_arm_angle",
                        )
                        if left_visibility < 0.3:
                            for key in left_keys:
                                if key in feat and key in last_feat:
                                    feat[key] = last_feat[key]
                            feat["left_visible"] = False
                        else:
                            feat["left_visible"] = True
                        if right_visibility < 0.3:
                            for key in right_keys:
                                if key in feat and key in last_feat:
                                    feat[key] = last_feat[key]
                            feat["right_visible"] = False
                        else:
                            feat["right_visible"] = True
                    else:
                        feat["left_visible"] = True
                        feat["right_visible"] = True
                    last_feat = feat
                else:
                    feat = last_feat

                hands = associate_hands(
                    hand_result,
                    (landmarks[15].x, landmarks[15].y),
                    (landmarks[16].x, landmarks[16].y),
                )
                last_hand_left = hands["left"]
                last_hand_right = hands["right"]

                if last_hand_left:
                    draw_hand_landmarks(frame, last_hand_left, height, width)
                else:
                    draw_wrist_marker(frame, landmarks, 15, height, width, side="L")
                if last_hand_right:
                    draw_hand_landmarks(frame, last_hand_right, height, width)
                else:
                    draw_wrist_marker(frame, landmarks, 16, height, width, side="R")

                left_palm_ori = classify_palm_orientation(last_hand_left, body_right_2d, body_up_2d)
                right_palm_ori = classify_palm_orientation(last_hand_right, body_right_2d, body_up_2d)

                if feat is not None:
                    state_machine.update(
                        feat,
                        left_palm_ori,
                        right_palm_ori,
                        global_frame,
                        feat.get("shoulder_width", 0.35),
                    )

                if dl_engine is not None:
                    if not dl_warmed_up and (time.time() - warmup_start_time) >= warmup_seconds:
                        dl_warmed_up = True
                        dl_engine.reset_state()
                        print("\n[DL-OK] 预热完成，LSTM 状态已重置为 h0/c0，开始实时识别")
                        print("        现在可以开始做手势了！\n")

                    if dl_warmed_up:
                        dl_counter += 1
                        try:
                            coord_aic = mediapipe_to_aic14(landmarks)
                            dl_result = dl_engine.predict_from_keypoints(coord_aic)
                            raw_gesture = dl_result["gesture"]
                            raw_confidence = dl_result["confidence"]
                        except Exception as exc:
                            if not dl_error_once:
                                print(f"\n[DL-ERROR] 推理异常: {exc}")
                                dl_error_once = True
                            raw_gesture = "DL error"
                            raw_confidence = 0.0

                        if raw_gesture not in {"无手势", "DL error", "预热中..."}:
                            dl_persist_gesture = raw_gesture
                            dl_persist_confidence = raw_confidence
                            dl_persist_counter = persist_max
                            dl_gesture = raw_gesture
                            dl_confidence = raw_confidence
                        elif dl_persist_counter > 0:
                            decay = dl_persist_counter / persist_max
                            dl_gesture = dl_persist_gesture
                            dl_confidence = dl_persist_confidence * max(0.55, decay)
                            dl_persist_counter -= 1
                        else:
                            dl_gesture = raw_gesture
                            dl_confidence = raw_confidence

                        if dl_counter % 30 == 0:
                            left_region = (last_feat or {}).get("left_region", "?")
                            right_region = (last_feat or {}).get("right_region", "?")
                            persist_mark = "*" if raw_gesture == "无手势" and dl_persist_counter > 0 else ""
                            print(
                                f"  [DL] {dl_gesture}{persist_mark} (置信度:{dl_confidence:.0%})  "
                                f"| 左手: {left_region}  右手: {right_region}"
                            )
            elif should_infer:
                state_machine.cancel_action(global_frame)
                last_landmarks = None
                last_world_landmarks = None
                last_feat = None
                last_hand_left = None
                last_hand_right = None
                dl_persist_counter = 0

            if not should_infer and last_landmarks is not None:
                draw_pose_landmarks(frame, last_landmarks, height, width)
                if last_hand_left:
                    draw_hand_landmarks(frame, last_hand_left, height, width)
                else:
                    draw_wrist_marker(frame, last_landmarks, 15, height, width, side="L")
                if last_hand_right:
                    draw_hand_landmarks(frame, last_hand_right, height, width)
                else:
                    draw_wrist_marker(frame, last_landmarks, 16, height, width, side="R")

            if dl_engine is not None and not dl_warmed_up:
                remaining = warmup_seconds - (time.time() - warmup_start_time)
                if remaining > 0:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
                    frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
                    cv2.putText(
                        frame,
                        f"WARMING UP... {remaining:.0f}s",
                        (width // 2 - 160, height // 2 - 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.6,
                        (0, 255, 255),
                        4,
                    )
                    frame = draw_chinese_text(
                        frame,
                        "请保持站立，倒计时结束后开始识别",
                        (width // 2 - 190, height // 2 + 5),
                        (0, 255, 255),
                        26,
                    )
                    frame = draw_chinese_text(
                        frame,
                        "LSTM 将从零状态启动，直接识别手势",
                        (width // 2 - 190, height // 2 + 42),
                        (200, 200, 200),
                        22,
                    )

            if last_landmarks is None:
                frame = draw_chinese_text(frame, "未检测到人体", (10, 110), (0, 0, 255), 36)

            if last_feat is not None:
                info_x = width - 200
                left_region = last_feat.get("left_region", "?")
                right_region = last_feat.get("right_region", "?")
                frame = draw_chinese_text(frame, f"左手: {left_region}", (info_x, 15), (0, 200, 255), 20)
                frame = draw_chinese_text(frame, f"右手: {right_region}", (info_x, 42), (200, 100, 255), 20)

            if dl_engine is not None:
                panel_w, panel_h = 260, 80
                panel_x, panel_y = 15, 15
                overlay = frame.copy()
                cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 50), -1)
                cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 200, 255), 2)
                frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
                frame = draw_chinese_text(frame, "交警手势识别", (panel_x + 10, panel_y + 5), (0, 255, 255), 22)
                frame = draw_chinese_text(frame, dl_gesture, (panel_x + 10, panel_y + 28), (0, 215, 255), 28)
                frame = draw_chinese_text(
                    frame,
                    f"置信度: {dl_confidence:.1%}",
                    (panel_x + 10, panel_y + 60),
                    (255, 255, 255),
                    18,
                )

            cv2.putText(
                frame,
                f"FPS: {fps}  Frame: {global_frame}",
                (10, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
            )

            if not args.no_display:
                display_frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale)
                cv2.imshow("Camera Gesture Recognition", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n用户按下 'q' 键，退出。")
                    break

            if args.max_frames > 0 and global_frame >= args.max_frames:
                print(f"\n已达到 max_frames={args.max_frames}，结束测试。")
                break
    finally:
        capture.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        pose_detector.close()
        if hand_detector:
            hand_detector.close()
        print("[DONE] 识别结束")


if __name__ == "__main__":
    main()
