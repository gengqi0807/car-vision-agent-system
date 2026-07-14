"""
run_recognition.py

保留两个入口：
1. `backend-api`：读取 RTSP 流并轮询后端的车主/交警手势接口做叠加显示。
2. `local-model`：直接使用本地交警手势模型进行视频/摄像头/RTSP 识别。

Usage:
    python run_recognition.py --mode backend-api --rtsp-url rtsp://127.0.0.1:8554/test
    python run_recognition.py --mode local-model --source test.mp4
    python run_recognition.py --mode local-model --source 0
"""

import argparse
import os
import sys
import time

import cv2
import requests

OWNER_API_URL = "http://127.0.0.1:8000/api/v1/owner-gesture/current"
POLICE_API_URL = "http://127.0.0.1:8000/api/v1/police-gesture/current"

COLOR_HAND = (0, 255, 0)
COLOR_POSE = (0, 0, 255)

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


def call_gesture_api(image_bytes: bytes, api_url: str) -> dict | None:
    files = {"file": ("frame.jpg", image_bytes, "image/jpeg")}
    try:
        response = requests.post(api_url, files=files, timeout=10)
        if response.status_code == 200:
            return response.json()
        print(f"  [WARN] {api_url.split('/')[-2]} 返回状态码 {response.status_code}")
        return None
    except requests.exceptions.RequestException as exc:
        print(f"  [ERR] 请求 {api_url} 失败: {exc}")
        return None


def draw_keypoints(frame, keypoints: list[dict], color, radius: int = 5):
    height, width = frame.shape[:2]
    for keypoint in keypoints:
        if keypoint.get("score", 0) > 0.5:
            center_x = int(keypoint["x"] * width)
            center_y = int(keypoint["y"] * height)
            cv2.circle(frame, (center_x, center_y), radius, color, -1)


def draw_text(frame, text: str, position: tuple[int, int], color):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def parse_args():
    parser = argparse.ArgumentParser(description="交警/车主手势识别调试入口")
    parser.add_argument(
        "--mode",
        choices=("backend-api", "local-model"),
        default="local-model",
        help="选择调用后端 API 还是直接使用本地交警模型",
    )
    parser.add_argument("--source", "-s", default="test.mp4", help="视频源：本地文件路径、RTSP URL 或摄像头索引")
    parser.add_argument("--pose-model", default=police_cfg.POSE_MODEL_PATH, help="MediaPipe Pose 模型路径")
    parser.add_argument("--hand-model", default=police_cfg.HAND_MODEL_PATH, help="MediaPipe Hand 模型路径")
    parser.add_argument("--no-mirror", action="store_true", help="禁用摄像头镜像")
    parser.add_argument("--skip", type=int, default=police_cfg.SKIP_FRAMES, help="跳帧数")
    parser.add_argument("--scale", type=float, default=police_cfg.INFER_SCALE, help="推理缩放比")
    parser.add_argument("--no-hand", action="store_true", help="不使用 Hand Landmarker（仅 Pose）")
    parser.add_argument("--no-dl", action="store_true", help="禁用 CTPGREngine 深度学习模型（仅使用规则模型）")
    parser.add_argument("--rtsp-url", default="rtsp://127.0.0.1:8554/test", help="backend-api 模式下读取的 RTSP 地址")
    return parser.parse_args()


def run_backend_api(rtsp_url: str):
    print(f"正在连接 RTSP 流: {rtsp_url}")
    capture = cv2.VideoCapture(rtsp_url)
    if not capture.isOpened():
        print("错误：无法打开 RTSP 流，请确认 MediaMTX + FFmpeg 正在推流。")
        sys.exit(1)

    frame_count = 0
    print("开始识别，按 'q' 键退出...\n")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("警告：读取帧失败，尝试继续...")
                time.sleep(0.1)
                continue

            frame_count += 1
            if frame_count % 5 == 0:
                success, jpeg_bytes = cv2.imencode(".jpg", frame)
                if not success:
                    continue
                image_bytes = jpeg_bytes.tobytes()

                owner_result = call_gesture_api(image_bytes, OWNER_API_URL)
                if owner_result:
                    draw_keypoints(frame, owner_result.get("keypoints", []), COLOR_HAND, radius=5)
                    draw_text(frame, f"[Owner] {owner_result.get('gesture', '')}", (10, 35), COLOR_HAND)

                police_result = call_gesture_api(image_bytes, POLICE_API_URL)
                if police_result:
                    draw_keypoints(frame, police_result.get("keypoints", []), COLOR_POSE, radius=5)
                    draw_text(frame, f"[Police] {police_result.get('gesture', '')}", (10, 70), COLOR_POSE)

            cv2.putText(
                frame,
                f"Frame: {frame_count}",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n用户按下 'q'，退出识别。")
                break
    except KeyboardInterrupt:
        print("\n中断退出。")
    finally:
        capture.release()
        cv2.destroyAllWindows()
        print("资源已释放。")


def run_local_model(args):
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
        print(f"[INFO] 深度学习模型未启用  (_DL_AVAILABLE={_DL_AVAILABLE}, no_dl={args.no_dl})")

    source = int(args.source) if str(args.source).isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"[ERROR] 无法打开视频源 -> {source}")
        pose_detector.close()
        if hand_detector:
            hand_detector.close()
        return

    fps_video = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_delay = 1.0 / fps_video if fps_video > 0 else 1.0 / 30.0
    print(
        f"[ OK ] 视频源已打开  SKIP={police_cfg.SKIP_FRAMES}  "
        f"SCALE={police_cfg.INFER_SCALE:.2f}  视频FPS={fps_video:.0f}  "
        f"总帧={total_frames}  播放间隔={frame_delay * 1000:.1f}ms"
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
    dl_gesture = "loading..."
    dl_confidence = 0.0
    dl_counter = 0
    dl_error_once = False
    display_scale = 0.65

    try:
        while True:
            frame_started_at = time.time()
            ok, frame = capture.read()
            if not ok:
                print("\n视频播放结束。")
                break

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
                    dl_counter += 1
                    try:
                        coord_aic = mediapipe_to_aic14(landmarks)
                        dl_result = dl_engine.predict_from_keypoints(coord_aic)
                        dl_gesture = dl_result["gesture"]
                        dl_confidence = dl_result["confidence"]
                    except Exception as exc:
                        if not dl_error_once:
                            print(f"\n[DL-ERROR] 推理异常: {exc}")
                            dl_error_once = True
                        dl_gesture = "DL error"
                        dl_confidence = 0.0

                    if dl_counter % 30 == 0:
                        left_region = (last_feat or {}).get("left_region", "?")
                        right_region = (last_feat or {}).get("right_region", "?")
                        print(
                            f"  [DL] {dl_gesture} (置信度:{dl_confidence:.0%})  "
                            f"| 左手: {left_region}  右手: {right_region}"
                        )

            elif should_infer:
                state_machine.cancel_action(global_frame)
                last_landmarks = None
                last_world_landmarks = None
                last_feat = None
                last_hand_left = None
                last_hand_right = None

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

            if last_landmarks is None:
                frame = draw_chinese_text(frame, "未检测到人体", (10, 110), (0, 0, 255), 36)

            cv2.putText(
                frame,
                f"FPS: {fps}  Frame: {global_frame}",
                (10, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
            )

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

            display_frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale)
            cv2.imshow("Police Gesture Recognition", display_frame)

            if not isinstance(source, int):
                elapsed_frame = time.time() - frame_started_at
                wait_ms = max(1, int(max(frame_delay - elapsed_frame, 0.0) * 1000))
            else:
                wait_ms = 1

            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                print("\n用户按下 'q' 键，退出。")
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        pose_detector.close()
        if hand_detector:
            hand_detector.close()
        print("[DONE] 识别结束")


def main():
    args = parse_args()
    if args.mode == "backend-api":
        run_backend_api(args.rtsp_url)
        return
    run_local_model(args)


if __name__ == "__main__":
    main()
