"""
police_local.py — 本地交警手势识别（MediaPipe Tasks API + 状态机 + 归一化特征 + 轨迹画圈检测）

核心设计：
  - 使用 MediaPipe Tasks API（pose_landmarker_lite.task 模型）替代旧版 solutions API
  - 归一化距离和关节角度替代绝对坐标判定，支持 90° 侧身识别
  - STATE_IDLE (0)：空闲，等待任意手腕抬过肩膀触发动作
  - STATE_ACTIVE (1)：逐帧记录 6 个特征极值 + 交替历史 + 手腕轨迹
  - 双手落回肩膀以下 → 退出 ACTIVE → classify_action() 判定结果
  - ★ 画圈检测：记录手腕相对肩膀的 (dx, dy) 轨迹，用方差区分画圈/垂直摆动/静止
    → 左转弯/右转弯依靠轨迹画圈识别，不再仅依赖静态弯曲角度

性能优化（防卡顿）：
  - 跳帧推理：每 3 帧推理 1 次（SKIP_FRAMES=3），画面 30fps 流畅，识别约 10fps
  - 缩小推理：推理帧缩至原图 60%（INFER_SCALE=0.6），速度翻倍
  - 缓冲区限制：cap.set(CAP_PROP_BUFFERSIZE, 1) 防止 RTSP 帧积压延迟
  - 画面文字独立于推理，跳帧期间不清空上一次识别结果

支持手势：停止信号 / 靠边停车 / 左转弯待转 / 左转弯 / 右转弯 / 变道信号 / 直行信号

关键点索引（0~32）：
  0-鼻子  11-左肩  12-右肩  13-左肘  14-右肘
  15-左腕  16-右腕  23-左髋  24-右髋
"""

import os
import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
import math

# MediaPipe Tasks API（新版）
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# 状态机常量
# ============================================================

STATE_IDLE = 0
STATE_ACTIVE = 1

# 画圈检测阈值 — 归一化坐标下的方差阈值
# 值越小越敏感（0.015），值越大越严格（0.03）
CIRCULAR_THRESHOLD = 0.005


# ============================================================
# MediaPipe Pose 骨架连接（33 个关键点之间的连线定义）
# ============================================================

POSE_CONNECTIONS = [
    # 脸部
    (0, 1), (1, 2), (2, 3), (3, 7),   # 左眼-左耳
    (0, 4), (4, 5), (5, 6), (6, 8),   # 右眼-右耳
    (9, 10),                           # 嘴唇
    # 躯干
    (11, 12),                          # 肩膀
    (11, 23), (12, 24), (23, 24),     # 肩-髋
    # 左臂
    (11, 13), (13, 15),               # 左肩-左肘-左腕
    (15, 17), (15, 19), (15, 21),     # 左手
    (17, 19),
    # 右臂
    (12, 14), (14, 16),               # 右肩-右肘-右腕
    (16, 18), (16, 20), (16, 22),     # 右手
    (18, 20),
    # 左腿
    (23, 25), (25, 27),               # 左髋-左膝-左踝
    (27, 29), (27, 31), (29, 31),     # 左脚
    # 右腿
    (24, 26), (26, 28),               # 右髋-右膝-右踝
    (28, 30), (28, 32), (30, 32),     # 右脚
]


# ============================================================
# 几何特征计算函数
# ============================================================

def calc_angle(a: tuple[float, float], b: tuple[float, float],
               c: tuple[float, float]) -> float:
    """
    计算三点夹角 ∠ABC（以 b 为顶点，即 a-b-c），返回角度值（0~180°）。
    例如：calc_angle(肩, 肘, 腕) → 180° = 手臂完全伸直，<90° = 严重弯曲。
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(ba[0], ba[1])
    mag_bc = math.hypot(bc[0], bc[1])
    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return 180.0
    cos_val = dot / (mag_ba * mag_bc)
    cos_val = max(-1.0, min(1.0, cos_val))
    return math.degrees(math.acos(cos_val))


def calc_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    """两点欧几里得距离。"""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def detect_circular_motion(trail: list) -> str:
    """
    根据手腕轨迹判定运动类型。

    参数:
        trail: [(dx, dy), ...]  手腕相对于同侧肩膀的偏移坐标列表

    返回:
        "circular"  — X/Y 两个方向方差均大（画圈）
        "vertical"  — 仅 Y 方向方差大（垂直摆动）
        "static"    — 两个方向方差都不大（静止/小动作）
    """
    if len(trail) < 5:
        return "static"

    xs = [p[0] for p in trail]
    ys = [p[1] for p in trail]

    var_x = float(np.var(xs)) if len(xs) > 1 else 0.0
    var_y = float(np.var(ys)) if len(ys) > 1 else 0.0

    if var_x > CIRCULAR_THRESHOLD and var_y > CIRCULAR_THRESHOLD:
        return "circular"
    elif var_x < CIRCULAR_THRESHOLD and var_y > CIRCULAR_THRESHOLD:
        return "vertical"
    else:
        return "static"


# ============================================================
# 状态机 — 动作过程记录 + 结束判定
# ============================================================

def reset_action_data(sw: float) -> dict:
    """
    初始化/重置一次动作的记录字典。
    sw = 当前帧肩宽（用于归一化的基准距离）。
    """
    return {
        # 极值特征（归一化后）
        "min_left_raise": float('inf'),      # 左腕最小 raise（最负 = 抬最高）
        "min_right_raise": float('inf'),     # 右腕最小 raise
        "max_left_stretch": 0.0,            # 左臂最大伸展
        "max_right_stretch": 0.0,           # 右臂最大伸展
        "min_left_angle": 180.0,            # 左臂最小角度（最弯）
        "min_right_angle": 180.0,           # 右臂最小角度（最弯）
        # 交替检测（直行信号）
        "raise_history": [],                # [(帧序号, left_raise, right_raise), ...]
        "frame_count": 0,
        "shoulder_width": max(sw, 1e-6),   # 防止除零
        # 手腕轨迹（画圈检测）
        "left_wrist_trail": [],             # [(dx, dy), ...] 左腕相对左肩偏移
        "right_wrist_trail": [],            # [(dx, dy), ...] 右腕相对右肩偏移
    }


def compute_features(lw_x: float, lw_y: float, rw_x: float, rw_y: float,
                     ls_x: float, ls_y: float, rs_x: float, rs_y: float,
                     le_x: float, le_y: float, re_x: float, re_y: float,
                     sw: float) -> dict:
    """
    根据关键点像素坐标，计算 6 个归一化特征值。

    返回值字典键名：
      left_arm_angle   — 左肩-左肘-左腕夹角（180° = 伸直）
      right_arm_angle  — 右肩-右肘-右腕夹角
      left_raise       — (左腕.y - 左肩.y) / 肩宽（负值 = 高于肩）
      right_raise      — (右腕.y - 右肩.y) / 肩宽
      left_stretch     — 左腕→左肩欧氏距离 / 肩宽（手臂伸出多远）
      right_stretch    — 右腕→右肩欧氏距离 / 肩宽
    """
    sw_safe = max(sw, 1e-6)

    # 关节角度
    left_arm_angle = calc_angle((ls_x, ls_y), (le_x, le_y), (lw_x, lw_y))
    right_arm_angle = calc_angle((rs_x, rs_y), (re_x, re_y), (rw_x, rw_y))

    # 归一化高度差
    left_raise = (lw_y - ls_y) / sw_safe
    right_raise = (rw_y - rs_y) / sw_safe

    # 归一化伸展距离
    left_stretch = calc_dist((lw_x, lw_y), (ls_x, ls_y)) / sw_safe
    right_stretch = calc_dist((rw_x, rw_y), (rs_x, rs_y)) / sw_safe

    return {
        "left_arm_angle": left_arm_angle,
        "right_arm_angle": right_arm_angle,
        "left_raise": left_raise,
        "right_raise": right_raise,
        "left_stretch": left_stretch,
        "right_stretch": right_stretch,
    }


def classify_action(action_data: dict) -> str:
    """
    基于归一化特征极值 + 手腕轨迹，判定最终手势类别。
    所有阈值均基于肩宽归一化后的值，与人物远近、侧身角度无关。
    """
    lr = action_data["min_left_raise"]       # 最负 = 抬最高
    rr = action_data["min_right_raise"]
    ls = action_data["max_left_stretch"]
    rs = action_data["max_right_stretch"]
    la = action_data["min_left_angle"]       # 最小 = 最弯
    ra = action_data["min_right_angle"]
    history = action_data["raise_history"]
    fc = action_data["frame_count"]

    # 手腕轨迹判定
    left_trail  = action_data.get("left_wrist_trail", [])
    right_trail = action_data.get("right_wrist_trail", [])
    left_motion  = detect_circular_motion(left_trail)
    right_motion = detect_circular_motion(right_trail)

    # 辅助：判断手臂是否"自然下垂"
    def is_hanging(r_val: float, s_val: float, a_val: float) -> bool:
        """未明显抬高 + 未明显伸展 + 未明显弯曲 → 自然下垂。"""
        return r_val > -0.15 and s_val < 1.0 and a_val > 120

    left_hanging  = is_hanging(lr, ls, la)
    right_hanging = is_hanging(rr, rs, ra)

    # 辅助：判断手腕水平伸出幅度（相对肩膀的 dx 足够大）
    def is_extended(trail: list) -> bool:
        if len(trail) < 3:
            return False
        avg_dx = sum(abs(p[0]) for p in trail) / len(trail)
        return avg_dx > 0.12

    left_extended  = is_extended(left_trail)
    right_extended = is_extended(right_trail)

    # ----------------------------------------------------------------
    # 优先：靠边停车 — 单臂极高 + 伸直，另一臂自然下垂
    # ----------------------------------------------------------------
    if rr < -0.8 and ra > 160 and left_hanging:
        return "靠边停车信号"
    if lr < -0.8 and la > 160 and right_hanging:
        return "靠边停车信号"

    # ----------------------------------------------------------------
    # 优先：停止信号 — 双臂均高于肩 + 均伸直 + 均大幅伸展
    # ----------------------------------------------------------------
    if (lr < -0.3 and rr < -0.3
            and la > 150 and ra > 150
            and ls > 0.8 and rs > 0.8):
        return "停止信号"

    # ----------------------------------------------------------------
    # ★ 轨迹判定：画圈 → 左转弯 / 右转弯
    # ----------------------------------------------------------------
    # 左腕画圈 + 右腕水平伸展 → 左转弯信号
    if left_motion == "circular" and right_extended:
        return "左转弯信号"

    # 右腕画圈 + 左腕水平伸展 → 右转弯信号
    if right_motion == "circular" and left_extended:
        return "右转弯信号"

    # ----------------------------------------------------------------
    # ★ 轨迹判定：垂直摆动（不画圈）→ 直行信号
    # ----------------------------------------------------------------
    if (left_motion == "vertical" or right_motion == "vertical"):
        return "直行信号"

    # ----------------------------------------------------------------
    # 备用：交替抬高检测（轨迹不足时的直行信号回退方案）
    # ----------------------------------------------------------------
    if len(history) > 5:
        states = []
        for _, lr_i, rr_i in history:
            if lr_i < -0.25 and rr_i > -0.15:
                states.append("L")
            elif rr_i < -0.25 and lr_i > -0.15:
                states.append("R")
        if len(states) >= 2:
            alternations = sum(
                1 for i in range(1, len(states))
                if states[i] != states[i - 1]
            )
            if alternations >= 2:
                return "直行信号"

    # ----------------------------------------------------------------
    # 左转弯待转：左臂水平前伸 + 伸直，右臂下垂（静态姿态）
    # ----------------------------------------------------------------
    if (lr > -0.12 and lr < 0.12
            and ls > 1.2
            and la > 150
            and right_hanging):
        return "左转弯待转信号"

    # ----------------------------------------------------------------
    # 变道信号：右臂弯曲于胸前，左臂下垂（静态姿态）
    # ----------------------------------------------------------------
    if (ra < 90
            and rr > -0.25 and rr < 0.1
            and left_hanging):
        return "变道信号"

    # ----------------------------------------------------------------
    # 右转弯（旧式静态判定备用）：左臂弯曲于胸前，右臂下垂
    # ----------------------------------------------------------------
    if (la < 90
            and lr < -0.05
            and right_hanging):
        return "右转弯信号"

    # ----------------------------------------------------------------
    # 其他
    # ----------------------------------------------------------------
    return "其他手势"


# ============================================================
# 骨架绘制（手动绘制 33 个关键点 + 连线，替换旧版 mp_draw）
# ============================================================

def draw_pose_landmarks(frame: np.ndarray, landmarks,
                         h: int, w: int,
                         point_color=(0, 255, 0),
                         line_color=(0, 255, 0),
                         point_radius=3,
                         line_thickness=2):
    """
    在 frame 上手动绘制 MediaPipe Pose 骨架（33 个关键点 + 连线）。
    新版 Tasks API 返回的 landmarks[i] 包含 .x, .y, .z, .visibility 属性。
    """
    num_landmarks = len(landmarks) if landmarks else 0

    # 画连线
    for a, b in POSE_CONNECTIONS:
        if a < num_landmarks and b < num_landmarks:
            x1, y1 = int(landmarks[a].x * w), int(landmarks[a].y * h)
            x2, y2 = int(landmarks[b].x * w), int(landmarks[b].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), line_color, line_thickness)

    # 画关键点
    for i in range(num_landmarks):
        cx, cy = int(landmarks[i].x * w), int(landmarks[i].y * h)
        cv2.circle(frame, (cx, cy), point_radius, point_color, -1)


# ============================================================
# 中文绘制工具
# ============================================================

def draw_chinese_text(img: np.ndarray, text: str, pos: tuple[int, int],
                      color: tuple[int, int, int] = (0, 255, 0),
                      size: int = 30) -> np.ndarray:
    """在 OpenCV 图片上绘制中文文字。"""
    img_pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("simhei.ttf", size, encoding="utf-8")
    except Exception:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ============================================================
# 模型初始化（MediaPipe Tasks API）
# ============================================================

def create_detector(model_path: str) -> vision.PoseLandmarker:
    """
    使用 MediaPipe Tasks API 创建 PoseLandmarker 实例。
    running_mode=IMAGE 表示每帧独立推理（适合 while 循环逐帧处理）。
    """
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


# ============================================================
# 主程序
# ============================================================

def main():
    """主函数：RTSP 流 → MediaPipe Tasks 推理 → 状态机识别 → 动作结束后判定。"""

    # ---- 1. 初始化 PoseLandmarker（新版 Tasks API） ----
    model_path = os.path.join("backend", "pose_landmarker_lite.task")
    if not os.path.exists(model_path):
        print(f"❌ 错误：模型文件不存在 → {model_path}")
        return

    print(f"📦 加载模型: {model_path}")
    detector = create_detector(model_path)
    print("✅ PoseLandmarker 初始化成功！")

    # ---- 2. 连接 RTSP 流 ----
    rtsp_url = "rtsp://127.0.0.1:8554/test"
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # ★ 缓冲区只保留最新一帧，防止内存堆积
    if not cap.isOpened():
        print("❌ 错误：无法打开视频流！请确认 MediaMTX + FFmpeg 正在推流。")
        detector.close()
        return

    print("✅ 视频流打开成功！跳帧模式（每3帧推理1次，推理分辨率×0.6），等待动作开始...")

    # ---- 3. 跳帧 & 状态机变量 ----
    frame_skip_counter = 0
    SKIP_FRAMES = 3                         # ★ 每隔 3 帧推理一次（跳帧）
    INFER_SCALE = 0.6                       # ★ 推理时缩小到 60%，速度翻倍

    current_state = STATE_IDLE
    action_data: dict | None = None
    last_result: str | None = None
    global_frame = 0

    # 持久化绘制状态（跳帧期间不清空）
    last_landmarks = None                   # 上次推理的骨架
    last_feat = None                        # 上次推理的特征值
    last_shoulder_width = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 读取帧失败，可能推流中断")
            break

        frame_skip_counter += 1
        global_frame += 1
        h, w = frame.shape[:2]

        should_infer = (frame_skip_counter % SKIP_FRAMES == 0)

        # ---- 4. AI 推理（仅跳帧命中时执行） ----
        if should_infer:
            # ★ 缩小帧为原尺寸的 60%，加速推理（归一化坐标等比缩放，直接乘 w/h 即可）
            small_frame = cv2.resize(frame, (0, 0), fx=INFER_SCALE, fy=INFER_SCALE)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)
            detection_result = detector.detect(mp_image)

            if detection_result.pose_landmarks:
                # 提取第一个人的 33 个关键点（NormLandmark 列表，属性：.x .y .z .visibility）
                last_landmarks = detection_result.pose_landmarks[0]
                landmarks = last_landmarks

                # 手动绘制骨架（归一化坐标等比，直接乘 w/h 映射到原图）
                draw_pose_landmarks(frame, landmarks, h, w,
                                    point_color=(0, 255, 0),
                                    line_color=(0, 0, 255))

                # 反归一化：关键点 → 像素坐标
                def px(idx: int) -> tuple[float, float]:
                    lm = landmarks[idx]
                    return lm.x * w, lm.y * h

                lw_x, lw_y = px(15)   # 左腕
                rw_x, rw_y = px(16)   # 右腕
                ls_x, ls_y = px(11)   # 左肩
                rs_x, rs_y = px(12)   # 右肩
                le_x, le_y = px(13)   # 左肘
                re_x, re_y = px(14)   # 右肘

                # 肩宽（归一化基准）
                shoulder_width = calc_dist((ls_x, ls_y), (rs_x, rs_y))
                last_shoulder_width = shoulder_width

                # 当前帧 6 个特征值
                feat = compute_features(
                    lw_x, lw_y, rw_x, rw_y,
                    ls_x, ls_y, rs_x, rs_y,
                    le_x, le_y, re_x, re_y,
                    shoulder_width,
                )
                last_feat = feat

                # 腕是否高于同侧肩（触发条件）
                left_above  = lw_y < ls_y
                right_above = rw_y < rs_y
                both_below  = (lw_y > ls_y) and (rw_y > rs_y)

                # ---------- IDLE → ACTIVE：任意一只手抬过肩膀 ----------
                if current_state == STATE_IDLE and (left_above or right_above):
                    current_state = STATE_ACTIVE
                    action_data = reset_action_data(shoulder_width)
                    action_data["frame_count"] = 1
                    # 初始帧直接写入极值
                    action_data["min_left_raise"]    = feat["left_raise"]
                    action_data["min_right_raise"]   = feat["right_raise"]
                    action_data["max_left_stretch"]  = feat["left_stretch"]
                    action_data["max_right_stretch"] = feat["right_stretch"]
                    action_data["min_left_angle"]    = feat["left_arm_angle"]
                    action_data["min_right_angle"]   = feat["right_arm_angle"]
                    action_data["raise_history"].append(
                        (1, feat["left_raise"], feat["right_raise"])
                    )
                    # 初始帧写入手腕轨迹（相对肩膀的归一化偏移）
                    left_dx = landmarks[15].x - landmarks[11].x
                    left_dy = landmarks[15].y - landmarks[11].y
                    right_dx = landmarks[16].x - landmarks[12].x
                    right_dy = landmarks[16].y - landmarks[12].y
                    action_data["left_wrist_trail"].append((left_dx, left_dy))
                    action_data["right_wrist_trail"].append((right_dx, right_dy))
                    last_result = None
                    print(f"\n▶ [帧 {global_frame}] 动作开始！"
                          f"  L_Angle={feat['left_arm_angle']:.0f}"
                          f"  R_Angle={feat['right_arm_angle']:.0f}"
                          f"  L_Raise={feat['left_raise']:.2f}"
                          f"  R_Raise={feat['right_raise']:.2f}")

                # ---------- ACTIVE：持续记录极值 ----------
                elif current_state == STATE_ACTIVE:
                    fc = action_data["frame_count"] + 1
                    action_data["frame_count"] = fc

                    # 更新最小 raise（最负 = 最高点）
                    if feat["left_raise"] < action_data["min_left_raise"]:
                        action_data["min_left_raise"] = feat["left_raise"]
                    if feat["right_raise"] < action_data["min_right_raise"]:
                        action_data["min_right_raise"] = feat["right_raise"]

                    # 更新最大伸展
                    if feat["left_stretch"] > action_data["max_left_stretch"]:
                        action_data["max_left_stretch"] = feat["left_stretch"]
                    if feat["right_stretch"] > action_data["max_right_stretch"]:
                        action_data["max_right_stretch"] = feat["right_stretch"]

                    # 更新最小角度（最弯）
                    if feat["left_arm_angle"] < action_data["min_left_angle"]:
                        action_data["min_left_angle"] = feat["left_arm_angle"]
                    if feat["right_arm_angle"] < action_data["min_right_angle"]:
                        action_data["min_right_angle"] = feat["right_arm_angle"]

                    # 记录交替历史
                    action_data["raise_history"].append(
                        (fc, feat["left_raise"], feat["right_raise"])
                    )

                    # ★ 记录手腕轨迹（相对肩膀的归一化偏移，用于画圈检测）
                    left_dx = landmarks[15].x - landmarks[11].x
                    left_dy = landmarks[15].y - landmarks[11].y
                    right_dx = landmarks[16].x - landmarks[12].x
                    right_dy = landmarks[16].y - landmarks[12].y
                    action_data["left_wrist_trail"].append((left_dx, left_dy))
                    action_data["right_wrist_trail"].append((right_dx, right_dy))
                    # 只保留最近 15 帧，避免内存膨胀
                    if len(action_data["left_wrist_trail"]) > 15:
                        action_data["left_wrist_trail"].pop(0)
                        action_data["right_wrist_trail"].pop(0)

                    # ---------- ACTIVE → IDLE：双手都回到肩膀以下 ----------
                    if both_below:
                        print(f"◀ [帧 {global_frame}] 动作结束！"
                              f"  minLR={action_data['min_left_raise']:.2f}"
                              f"  minRR={action_data['min_right_raise']:.2f}"
                              f"  maxLS={action_data['max_left_stretch']:.2f}"
                              f"  maxRS={action_data['max_right_stretch']:.2f}"
                              f"  minLA={action_data['min_left_angle']:.0f}"
                              f"  minRA={action_data['min_right_angle']:.0f}"
                              f"  持续{action_data['frame_count']}帧")
                        current_state = STATE_IDLE
                        result = classify_action(action_data)
                        last_result = result
                        action_data = None
                        print(f"   判定结果: {result}\n")

            else:
                # 当前推理帧未检测到人体 → 重置状态机
                if current_state == STATE_ACTIVE:
                    print(f"⚠️ [帧 {global_frame}] 动作中断（未检测到人体）")
                    current_state = STATE_IDLE
                    action_data = None
                last_landmarks = None
                last_feat = None

        # ---- 5. 画面文字显示（每帧都画，跳帧时不清空上一次结果） ----
        if current_state == STATE_ACTIVE:
            # 动作进行中：黄色提示
            frame = draw_chinese_text(frame, "⏳ 动作识别中...",
                                      (10, 30), (0, 255, 255), 30)
            if last_feat is not None:
                # 调试：第 1 行 — 角度 + raise
                debug1 = (
                    f"L_Angle:{last_feat['left_arm_angle']:.0f}"
                    f"  R_Angle:{last_feat['right_arm_angle']:.0f}"
                    f"  L_Raise:{last_feat['left_raise']:.2f}"
                    f"  R_Raise:{last_feat['right_raise']:.2f}"
                )
                # 调试：第 2 行 — stretch + 肩宽
                debug2 = (
                    f"L_Str:{last_feat['left_stretch']:.2f}"
                    f"  R_Str:{last_feat['right_stretch']:.2f}"
                    f"  SW:{last_shoulder_width:.0f}"
                )
                frame = draw_chinese_text(frame, debug1, (10, h - 60),
                                          (200, 200, 200), 16)
                frame = draw_chinese_text(frame, debug2, (10, h - 35),
                                          (200, 200, 200), 16)

        elif current_state == STATE_IDLE and last_result is not None:
            # 已完成一次动作：绿色大字显示结果（跳帧期间持续显示）
            frame = draw_chinese_text(frame, f"交警手势: {last_result}",
                                      (10, 30), (0, 255, 0), 36)

        elif current_state == STATE_IDLE and last_landmarks is not None:
            # 有检测到人体但暂无结果
            frame = draw_chinese_text(frame, "等待动作...",
                                      (10, 30), (255, 255, 255), 30)

        else:
            # 未检测到人体
            frame = draw_chinese_text(frame, "未检测到人体",
                                      (10, 30), (0, 0, 255), 36)

        # ---- 6. 显示画面 ----
        cv2.imshow("Police Gesture - State Machine (Tasks API)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n用户按下 q 键，退出。")
            break

    # ---- 7. 清理 ----
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("🛑 识别已结束")


if __name__ == "__main__":
    main()
