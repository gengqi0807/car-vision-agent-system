"""
police_local.py — 本地交警手势识别（状态机模式 + 关节角度/归一化距离特征）

核心设计：
  - 使用归一化距离和关节角度替代绝对坐标判定，支持 90° 侧身识别
  - 所有阈值均除以肩宽归一化，与人物远近、侧身角度无关
  - STATE_IDLE (0)：空闲，等待任意手腕抬过肩膀触发动作
  - STATE_ACTIVE (1)：逐帧记录 6 个特征极值 + 交替历史
  - 双手落回肩膀以下 → 退出 ACTIVE → classify_action() 判定结果

支持手势：停止信号 / 靠边停车 / 左转弯待转 / 右转弯 / 变道信号 / 直行信号

关键点索引（0~32）：
  0-鼻子  11-左肩  12-右肩  13-左肘  14-右肘
  15-左腕  16-右腕  23-左髋  24-右髋
"""

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math


# ============================================================
# 状态机常量
# ============================================================

STATE_IDLE = 0
STATE_ACTIVE = 1


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
    基于归一化特征极值，判定最终手势类别。
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

    # 辅助：判断手臂是否"自然下垂"
    def is_hanging(r_val: float, s_val: float, a_val: float) -> bool:
        """未明显抬高 + 未明显伸展 + 未明显弯曲 → 自然下垂。"""
        return r_val > -0.15 and s_val < 1.0 and a_val > 120

    left_hanging  = is_hanging(lr, ls, la)
    right_hanging = is_hanging(rr, rs, ra)

    # ----------------------------------------------------------------
    # 靠边停车：单臂极高（>0.8肩宽） + 伸直，另一臂自然下垂
    # ----------------------------------------------------------------
    if rr < -0.8 and ra > 160 and left_hanging:
        return "靠边停车信号"
    if lr < -0.8 and la > 160 and right_hanging:
        return "靠边停车信号"

    # ----------------------------------------------------------------
    # 停止信号：双臂均高于肩 + 均伸直 + 均大幅向外/前伸展
    # ----------------------------------------------------------------
    if (lr < -0.3 and rr < -0.3
            and la > 150 and ra > 150
            and ls > 0.8 and rs > 0.8):
        return "停止信号"

    # ----------------------------------------------------------------
    # 左转弯待转：左臂水平前伸（raise≈0 且 stretch大 且 伸直），右臂下垂
    # ----------------------------------------------------------------
    if (lr > -0.12 and lr < 0.12       # 左腕 ≈ 肩高（水平伸出）
            and ls > 1.2               # 左臂大幅伸展
            and la > 150               # 左臂伸直
            and right_hanging):
        return "左转弯待转信号"

    # ----------------------------------------------------------------
    # 右转弯：左臂弯曲于胸前（angle<90°），左腕与肩同高或略高，右臂下垂
    # ----------------------------------------------------------------
    if (la < 90                         # 左臂严重弯曲
            and lr < -0.05              # 左腕不低于肩
            and right_hanging):
        return "右转弯信号"

    # ----------------------------------------------------------------
    # 变道信号：右臂弯曲于胸前（angle<90°），右腕接近肩高，左臂下垂
    # ----------------------------------------------------------------
    if (ra < 90                         # 右臂严重弯曲
            and rr > -0.25 and rr < 0.1  # 右腕接近肩高
            and left_hanging):
        return "变道信号"

    # ----------------------------------------------------------------
    # 直行信号：左右手腕交替抬高（交替次数 ≥ 2）
    # 不依赖手臂朝哪个方向，只看"哪只手高"的切换次数
    # ----------------------------------------------------------------
    if len(history) > 5:
        # 每帧归类："左高右低" 或 "右高左低"
        states = []
        for _, lr_i, rr_i in history:
            if lr_i < -0.25 and rr_i > -0.15:
                states.append("L")       # 左臂明显抬高，右臂未抬
            elif rr_i < -0.25 and lr_i > -0.15:
                states.append("R")       # 右臂明显抬高，左臂未抬

        # 统计相邻不同状态的切换次数
        if len(states) >= 2:
            alternations = sum(
                1 for i in range(1, len(states))
                if states[i] != states[i - 1]
            )
            if alternations >= 2:
                return "直行信号"

    # ----------------------------------------------------------------
    # 其他
    # ----------------------------------------------------------------
    return "其他手势"


# ============================================================
# 中文绘制工具
# ============================================================

def draw_chinese_text(img: np.ndarray, text: str, pos: tuple[int, int],
                      color: tuple[int, int, int] = (0, 255, 0),
                      size: int = 30) -> np.ndarray:
    """
    在 OpenCV 图片上绘制中文文字。
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("simhei.ttf", size, encoding="utf-8")
    except Exception:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ============================================================
# 主程序
# ============================================================

def main():
    """主函数：RTSP 流 → MediaPipe Pose 推理 → 状态机识别 → 动作结束后判定。"""

    # ---- 1. 初始化 MediaPipe Pose ----
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ---- 2. 连接 RTSP 流 ----
    rtsp_url = "rtsp://127.0.0.1:8554/test"
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("❌ 错误：无法打开视频流！请确认 MediaMTX + FFmpeg 正在推流。")
        return

    print("✅ 视频流打开成功！状态机模式（角度+归一化特征）：等待动作开始...")

    # ---- 3. 状态机变量 ----
    current_state = STATE_IDLE
    action_data: dict | None = None
    last_result: str | None = None
    global_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 读取帧失败，可能推流中断")
            break

        global_frame += 1
        h, w = frame.shape[:2]

        # BGR → RGB（MediaPipe 要求 RGB）
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # ---- 4. 骨骼检测 & 状态机 ----
        if results.pose_landmarks:
            # 画出骨骼连线
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2),
            )

            # 反归一化：关键点 → 像素坐标
            def px(idx: int) -> tuple[float, float]:
                lm = results.pose_landmarks.landmark[idx]
                return lm.x * w, lm.y * h

            lw_x, lw_y = px(15)   # 左腕
            rw_x, rw_y = px(16)   # 右腕
            ls_x, ls_y = px(11)   # 左肩
            rs_x, rs_y = px(12)   # 右肩
            le_x, le_y = px(13)   # 左肘
            re_x, re_y = px(14)   # 右肘

            # 肩宽（归一化基准）
            shoulder_width = calc_dist((ls_x, ls_y), (rs_x, rs_y))

            # 当前帧 6 个特征值
            feat = compute_features(
                lw_x, lw_y, rw_x, rw_y,
                ls_x, ls_y, rs_x, rs_y,
                le_x, le_y, re_x, re_y,
                shoulder_width,
            )

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

            # ---- 5. 画面文字显示 ----
            if current_state == STATE_ACTIVE:
                # 动作进行中：黄色提示
                frame = draw_chinese_text(frame, "⏳ 动作识别中...",
                                          (10, 30), (0, 255, 255), 30)
                # 调试：第 1 行 — 角度 + raise
                debug1 = (
                    f"L_Angle:{feat['left_arm_angle']:.0f}"
                    f"  R_Angle:{feat['right_arm_angle']:.0f}"
                    f"  L_Raise:{feat['left_raise']:.2f}"
                    f"  R_Raise:{feat['right_raise']:.2f}"
                )
                # 调试：第 2 行 — stretch + 肩宽
                debug2 = (
                    f"L_Str:{feat['left_stretch']:.2f}"
                    f"  R_Str:{feat['right_stretch']:.2f}"
                    f"  SW:{shoulder_width:.0f}"
                )
                frame = draw_chinese_text(frame, debug1, (10, h - 60),
                                          (200, 200, 200), 16)
                frame = draw_chinese_text(frame, debug2, (10, h - 35),
                                          (200, 200, 200), 16)

            elif current_state == STATE_IDLE and last_result is not None:
                # 已完成一次动作：绿色大字显示结果
                frame = draw_chinese_text(frame, f"交警手势: {last_result}",
                                          (10, 30), (0, 255, 0), 36)
            else:
                # IDLE 且无历史结果
                frame = draw_chinese_text(frame, "等待动作...",
                                          (10, 30), (255, 255, 255), 30)
        else:
            # 未检测到人体
            frame = draw_chinese_text(frame, "未检测到人体",
                                      (10, 30), (0, 0, 255), 36)

        # ---- 6. 显示画面 ----
        cv2.imshow("Police Gesture - State Machine", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n用户按下 q 键，退出。")
            break

    # ---- 7. 清理 ----
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("🛑 识别已结束")


if __name__ == "__main__":
    main()
