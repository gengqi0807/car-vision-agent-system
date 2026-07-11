"""
visualization.py — 骨架绘制与中文文字叠加

包含：
  - POSE_CONNECTIONS / HAND_CONNECTIONS : 关键点连线定义
  - draw_pose_landmarks  : 绘制 Pose 骨架（33 个关键点）
  - draw_hand_landmarks  : 绘制 Hand 骨架（21 个关键点）
  - draw_chinese_text    : 使用 PIL 绘制中文文字
"""

import os
import sys
import cv2
import numpy as np

# ---- Pillow（PIL）可用性检查 ----
try:
    from PIL import Image as PILImage, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    PILImage = ImageDraw = ImageFont = None


# ============================================================
# 骨架连线定义
# ============================================================

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22), (18, 20),
    (23, 25), (25, 27),
    (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28),
    (28, 30), (28, 32), (30, 32),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# 尝试加载中文字体（只加载一次）
_CHINESE_FONT = None
_FONT_WARNED = False      # 只警告一次


def _get_chinese_font(size: int):
    """
    获取支持中文的字体对象，带缓存。

    检测流程：
      1. 复用已缓存的字体路径
      2. 优先使用 config.FONT_PATH
      3. 回退到系统常见中文字体路径
      4. 全部失败则抛出异常并给出提示

    Args:
        size: 字体大小（像素）

    Returns:
        PIL ImageFont 对象

    Raises:
        RuntimeError: 找不到中文字体且 Pillow 不可用时
    """
    global _CHINESE_FONT, _FONT_WARNED

    if not _PIL_AVAILABLE:
        if not _FONT_WARNED:
            print("=" * 55, file=sys.stderr)
            print("⚠️  Pillow（PIL）未安装，无法绘制中文文字。", file=sys.stderr)
            print("   请运行: pip install Pillow", file=sys.stderr)
            print("=" * 55, file=sys.stderr)
            _FONT_WARNED = True
        raise RuntimeError("Pillow not installed — cannot render Chinese text")

    # 已缓存且字体路径仍有效 → 直接复用
    if _CHINESE_FONT is not None:
        try:
            return ImageFont.truetype(_CHINESE_FONT, size)
        except Exception:
            _CHINESE_FONT = None   # 缓存失效，重新探测

    # ---- 字体候选列表 ----
    font_paths = []

    # 1) 优先使用 config.FONT_PATH
    from . import config as _cfg
    font_paths.append(_cfg.FONT_PATH)

    # 2) 操作系统常见字体
    if sys.platform == "win32":
        font_paths += [
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simsun.ttc",
        ]
    elif sys.platform == "darwin":
        font_paths += [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
        ]
    else:
        font_paths += [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ]

    # 3) 项目本地字体
    font_paths.append("simhei.ttf")

    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, size)
                _CHINESE_FONT = fp
                return font
            except Exception:
                continue

    # ---- 全部失败：给出清晰提示 ----
    if not _FONT_WARNED:
        print("=" * 55, file=sys.stderr)
        print("❌ 找不到中文字体！中文文字将显示为方块。", file=sys.stderr)
        print(f"   尝试的路径: {font_paths}", file=sys.stderr)
        print("   解决方法:", file=sys.stderr)
        print("     Windows:  下载 simhei.ttf 放到 C:/Windows/Fonts/", file=sys.stderr)
        print("     Linux:    sudo apt install fonts-wqy-microhei", file=sys.stderr)
        print("     macOS:    系统自带 PingFang.ttc，无需额外安装", file=sys.stderr)
        print("     或者:     下载 simhei.ttf 放到项目根目录 fonts/ 文件夹下", file=sys.stderr)
        print("=" * 55, file=sys.stderr)
        _FONT_WARNED = True

    return ImageFont.load_default()


# ============================================================
# 骨架绘制
# ============================================================

def draw_pose_landmarks(frame: np.ndarray,
                        landmarks,
                        h: int,
                        w: int,
                        point_color=(0, 255, 0),
                        line_color=(0, 0, 255),
                        point_radius=3,
                        line_thickness=2):
    """
    绘制 Pose 骨架（33 个关键点及连线）。

    Args:
        frame:           BGR 图像
        landmarks:       MediaPipe Pose 归一化关键点列表（属性 .x .y）
        h, w:            图像高/宽
        point_color:     关键点颜色 (B, G, R)
        line_color:      连线颜色 (B, G, R)
        point_radius:    关键点半径
        line_thickness:  连线粗细
    """
    n = len(landmarks) if landmarks else 0
    for a, b in POSE_CONNECTIONS:
        if a < n and b < n:
            x1, y1 = int(landmarks[a].x * w), int(landmarks[a].y * h)
            x2, y2 = int(landmarks[b].x * w), int(landmarks[b].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), line_color, line_thickness)
    for i in range(n):
        cx, cy = int(landmarks[i].x * w), int(landmarks[i].y * h)
        cv2.circle(frame, (cx, cy), point_radius, point_color, -1)


def draw_hand_landmarks(frame: np.ndarray,
                        hand_lm,
                        h: int,
                        w: int,
                        point_color=(255, 0, 255),
                        line_color=(255, 100, 255),
                        point_radius=2,
                        line_thickness=1):
    """
    绘制 Hand 骨架（21 个关键点及连线）。

    若 hand_lm 为 None 或长度不足 21，则不绘制任何内容，
    确保遮挡/转身后手部骨架立即消失。

    Args:
        frame:           BGR 图像
        hand_lm:         Hand Landmarker 归一化关键点列表（属性 .x .y），或 None
        h, w:            图像高/宽
        point_color:     关键点颜色 (B, G, R)
        line_color:      连线颜色 (B, G, R)
        point_radius:    关键点半径
        line_thickness:  连线粗细
    """
    if hand_lm is None or len(hand_lm) < 21:
        return
    n = len(hand_lm)
    for a, b in HAND_CONNECTIONS:
        if a < n and b < n:
            x1, y1 = int(hand_lm[a].x * w), int(hand_lm[a].y * h)
            x2, y2 = int(hand_lm[b].x * w), int(hand_lm[b].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), line_color, line_thickness)
    for i in range(n):
        cx, cy = int(hand_lm[i].x * w), int(hand_lm[i].y * h)
        cv2.circle(frame, (cx, cy), point_radius, point_color, -1)


def draw_wrist_marker(frame: np.ndarray,
                      landmarks,
                      wrist_idx: int,
                      h: int,
                      w: int,
                      side: str = '?',
                      color=(255, 165, 0),
                      radius=6,
                      thickness=2):
    """
    在 Pose 手腕关键点位置绘制醒目标记，用于手部骨架不可用时的替代显示。

    使用双重圆圈（实心 + 空心）区别于普通骨架点，便于区分"近似位置"。

    Args:
        frame:      BGR 图像
        landmarks:  Pose 归一化关键点列表（属性 .x .y），或 None
        wrist_idx:  手腕关键点索引（15=左手腕, 16=右手腕）
        h, w:       图像高/宽
        side:       左右标签（'L'/'R'），用于调试文字
        color:      标记颜色 (B, G, R)，默认橙色
        radius:     外圈半径
        thickness:  外圈线条粗细
    """
    if landmarks is None or wrist_idx >= len(landmarks):
        return
    lm = landmarks[wrist_idx]
    cx, cy = int(lm.x * w), int(lm.y * h)
    # 外圈（空心）
    cv2.circle(frame, (cx, cy), radius, color, thickness)
    # 内圈（实心，更醒目）
    cv2.circle(frame, (cx, cy), max(2, radius // 3), color, -1)
    # 左右标签
    cv2.putText(frame, side, (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


# ============================================================
# 中文文字绘制
# ============================================================

def draw_chinese_text(img: np.ndarray,
                      text: str,
                      pos: tuple[int, int],
                      color=(0, 255, 0),
                      size=30) -> np.ndarray:
    """
    在 BGR 图像上绘制中文文字（使用 PIL）。

    若 Pillow 未安装或字体加载失败，将退化到 cv2.putText
    （中文会显示为乱码，但程序不会崩溃）。

    Args:
        img:   BGR 图像
        text:  要绘制的文字（支持中文）
        pos:   文字左上角坐标 (x, y)
        color: 文字颜色 (B, G, R)
        size:  字体大小

    Returns:
        绘制后的 BGR 图像（新数组，不修改原图）
    """
    try:
        img_pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = _get_chinese_font(size)
        # PIL 使用 RGB 格式
        draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        # 退化到 OpenCV putText（不复制图像以减少开销）
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    size / 30.0, color, 2, cv2.LINE_AA)
        return img
