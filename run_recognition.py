import cv2
import requests
import time
import sys
import os

# ------------------------------
# API 地址配置
# ------------------------------
OWNER_API_URL = "http://127.0.0.1:8000/api/v1/owner-gesture/current"   # 车主手势
POLICE_API_URL = "http://127.0.0.1:8000/api/v1/police-gesture/current"  # 交警手势

# 绘图颜色（BGR 格式）
COLOR_HAND = (0, 255, 0)    # 绿色 — 车主手势关键点
COLOR_POSE = (0, 0, 255)    # 红色 — 交警手势关键点


def call_gesture_api(image_bytes: bytes, api_url: str) -> dict | None:
    """将一帧 JPEG 字节发送到后端手势 API，返回解析后的 JSON 结果。"""
    files = {"file": ("frame.jpg", image_bytes, "image/jpeg")}
    try:
        resp = requests.post(api_url, files=files, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"  [WARN] {api_url.split('/')[-2]} 返回状态码 {resp.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"  [ERR] 请求 {api_url} 失败: {e}")
        return None


def draw_keypoints(frame, keypoints: list[dict], color, radius: int = 5):
    """在画面上绘制归一化关键点（x, y 已归一化为 0~1）。"""
    h, w = frame.shape[:2]
    for kp in keypoints:
        if kp.get("score", 0) > 0.5:       # 只绘制置信度较高的点
            cx = int(kp["x"] * w)
            cy = int(kp["y"] * h)
            cv2.circle(frame, (cx, cy), radius, color, -1)


def draw_text(frame, text: str, position: tuple, color):
    """在画面上叠加文字（带描边，增强可读性）。"""
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 0), 3, cv2.LINE_AA)       # 黑色描边
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2, cv2.LINE_AA)


def main():
    # 1. 连接 RTSP 流 ------------------------------------------------
    rtsp_url = "rtsp://127.0.0.1:8554/test"
    print(f"正在连接 RTSP 流: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("错误：无法打开 RTSP 流，请确认 MediaMTX + FFmpeg 正在推流。")
        sys.exit(1)

    frame_count = 0
    print("开始识别，按 'q' 键退出...\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("警告：读取帧失败，尝试继续...")
                time.sleep(0.1)
                continue

            frame_count += 1

            # 2. 每 5 帧调用一次 API ----------------------------------
            if frame_count % 5 == 0:
                # 编码为 JPEG 字节
                success, jpeg_bytes = cv2.imencode('.jpg', frame)
                if not success:
                    continue
                image_bytes = jpeg_bytes.tobytes()

                # ------ 车主手势 ------
                owner_result = call_gesture_api(image_bytes, OWNER_API_URL)
                if owner_result:
                    gesture_owner = owner_result.get("gesture", "")
                    kps_owner = owner_result.get("keypoints", [])
                    draw_keypoints(frame, kps_owner, COLOR_HAND, radius=5)
                    draw_text(frame, f"[Owner] {gesture_owner}",
                              (10, 35), COLOR_HAND)

                # ------ 交警手势 ------
                police_result = call_gesture_api(image_bytes, POLICE_API_URL)
                if police_result:
                    gesture_police = police_result.get("gesture", "")
                    kps_police = police_result.get("keypoints", [])
                    draw_keypoints(frame, kps_police, COLOR_POSE, radius=5)
                    draw_text(frame, f"[Police] {gesture_police}",
                              (10, 70), COLOR_POSE)

            # 3. 帧率提示（左上角） -----------------------------------
            cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Gesture Recognition", frame)

            # 4. 按 q 退出 --------------------------------------------
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n用户按下 'q'，退出识别。")
                break

    except KeyboardInterrupt:
        print("\n中断退出。")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("资源已释放。")


if __name__ == "__main__":
    main()
