"""
geometry.py — 几何计算与坐标系工具

包含：
  - 基础几何：calc_angle / calc_dist / normalize_vec / dot
  - 人物局部坐标系：setup_local_frame / compute_body_orientation
  - 手臂方位分类：classify_arm_orientation / get_arm_pose
  - 手部区域划分：get_hand_y / get_hand_region
  - 摆动检测：detect_circular_motion / detect_oscillation /
             detect_oscillation_stretch / detect_swing_direction
  - 稳定性检测：is_arm_stable

所有函数使用世界坐标（米制）或归一化坐标进行计算。
"""

import math
import numpy as np

from . import config


# ============================================================
# 基础几何工具
# ============================================================

def calc_angle(a: tuple[float, float, float],
               b: tuple[float, float, float],
               c: tuple[float, float, float]) -> float:
    """
    计算三点夹角 ∠ABC（b 为顶点）。

    Args:
        a, b, c: 三个 3D 点的世界坐标 (x, y, z)

    Returns:
        夹角角度，范围 0° ~ 180°
    """
    ba = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    bc = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
    d = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2 + ba[2]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2 + bc[2]**2)
    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return 180.0
    cos_val = max(-1.0, min(1.0, d / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_val))


def calc_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    """计算两点间的 2D 欧氏距离。"""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def normalize_vec(v: tuple[float, float]) -> tuple[float, float]:
    """将 2D 向量归一化为单位向量。"""
    mag = max(math.hypot(*v), 1e-6)
    return (v[0] / mag, v[1] / mag)


def dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    """计算两个 2D 向量的点积。"""
    return a[0] * b[0] + a[1] * b[1]


# ============================================================
# 人物局部坐标系（像素坐标系，2D）
# ============================================================

def setup_local_frame(ls: tuple[float, float],
                      rs: tuple[float, float],
                      nose: tuple[float, float],
                      lh: tuple[float, float],
                      rh: tuple[float, float]):
    """
    以肩膀为原点建立 2D 局部坐标系（用于手掌朝向检测）。

    Args:
        ls, rs: 左/右肩像素坐标
        nose:   鼻子像素坐标
        lh, rh: 左/右髋像素坐标

    Returns:
        shoulder_mid : 肩中点 (x, y)（像素）
        body_right   : 归一化肩膀连线方向（向右），单位向量
        body_up      : 归一化躯干向上方向，单位向量
    """
    shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    body_right = normalize_vec((rs[0] - ls[0], rs[1] - ls[1]))
    body_up = (-body_right[1], body_right[0])
    nose_dir = (nose[0] - shoulder_mid[0], nose[1] - shoulder_mid[1])
    if dot(body_up, nose_dir) < 0:
        body_up = (-body_up[0], -body_up[1])
    return shoulder_mid, body_right, body_up


# ============================================================
# 人物朝向（世界坐标，3D → XZ 平面）
# ============================================================

def compute_body_orientation(world_landmarks):
    """
    基于鼻子 + 肩膀中点（XZ 平面），计算人物自身局部坐标系。

    在 XZ 平面中：
      body_forward : 鼻子相对于肩中点的方向 → 人物正前方
      body_left    : body_forward 逆时针旋转 90° → 人物正左方

    Args:
        world_landmarks: MediaPipe Pose 世界坐标关键点列表

    Returns:
        (body_forward, body_left)
          body_forward : (x, z) 人物正前方向，XZ 平面 2D 归一化向量
          body_left    : (x, z) 人物正左方向，XZ 平面 2D 归一化向量
    """
    ls_w = (world_landmarks[11].x, world_landmarks[11].y, world_landmarks[11].z)
    rs_w = (world_landmarks[12].x, world_landmarks[12].y, world_landmarks[12].z)
    nose_w = (world_landmarks[0].x, world_landmarks[0].y, world_landmarks[0].z)

    shoulder_mid = (
        (ls_w[0] + rs_w[0]) / 2,
        (ls_w[1] + rs_w[1]) / 2,
        (ls_w[2] + rs_w[2]) / 2,
    )

    forward_candidate = (nose_w[0] - shoulder_mid[0], nose_w[2] - shoulder_mid[2])
    mag = math.sqrt(forward_candidate[0] ** 2 + forward_candidate[1] ** 2)
    if mag > 1e-6:
        body_forward = (forward_candidate[0] / mag, forward_candidate[1] / mag)
    else:
        body_forward = (0.0, 1.0)

    body_left = (-body_forward[1], body_forward[0])
    return body_forward, body_left


# ============================================================
# 手部位置获取
# ============================================================

def get_hand_y(hand_lm):
    """
    获取手的 Y 坐标（优先使用中指指尖）。

    注意：返回的是 Hand Landmarker 归一化图像坐标 (0-1)，
    不能直接与 Pose 世界坐标(米)比较。手部区域判定应使用
    Pose 手腕世界坐标 + 偏移量。

    Args:
        hand_lm: Hand Landmarker 返回的 21 个关键点列表，或 None

    Returns:
        中指指尖（index=12）的归一化图像 Y 值，若无 Hand 数据则返回 None
    """
    if hand_lm is None or len(hand_lm) < 13:
        return None
    return hand_lm[12].y   # 归一化图像坐标 Y


# ============================================================
# 手部区域划分
# ============================================================

def get_hand_region(hand_y: float,
                    shoulder_y: float,
                    hip_y: float,   # 保留但不使用（兼容调用方签名）
                    head_y: float) -> str:
    """
    基于手与肩的相对高度差划分区域，不依赖髋部坐标。

    MediaPipe 世界坐标 y 轴向下 → dy = shoulder_y - hand_y：
      - head    : dy > HEAD_THRESHOLD            （手高于肩 5cm 以上）
      - shoulder: SHOULDER_LOWER <= dy <= HEAD_THRESHOLD  （肩下15cm ~ 肩上5cm）
      - waist   : WAIST_LOWER <= dy < SHOULDER_LOWER      （肩下50cm ~ 肩下15cm）
      - hip     : dy < WAIST_LOWER               （手低于肩 50cm 以上）

    所有阈值从 config 读取，不依赖 hip_y 和 head_y。

    Args:
        hand_y:     手指/手腕的世界坐标 Y 值
        shoulder_y: 肩中点的世界坐标 Y 值
        hip_y:      髋中点的世界坐标 Y 值（保留兼容，当前未使用）
        head_y:     鼻子（头部）的世界坐标 Y 值（保留兼容，当前未使用）

    Returns:
        区域标签: 'head' / 'shoulder' / 'waist' / 'hip'
    """
    # MediaPipe 世界坐标 y 轴向下，shoulder_y < hand_y（下垂时）
    # 因此 dy = shoulder_y - hand_y：正值 = 手高于肩，负值 = 手低于肩
    dy = shoulder_y - hand_y

    if dy > config.HEAD_THRESHOLD:
        return "head"
    elif dy >= config.SHOULDER_LOWER:
        return "shoulder"
    elif dy >= config.WAIST_LOWER:
        return "waist"
    else:
        return "hip"


# ============================================================
# 手臂方位分类
# ============================================================

def classify_arm_orientation(fwd: float, lat: float, dy: float) -> str:
    """
    基于人物自身坐标系（XZ 平面）的投影分量判定手臂方位。

    Args:
        fwd : forward_comp — dot(腕→肩方向_XZ, body_forward)
        lat : left_comp    — dot(腕→肩方向_XZ, body_left)
        dy  : wrist.y - shoulder.y（世界坐标 Y，正值=高于肩）

    Returns:
        方向标签，如 'forward' / 'left' / 'right' / 'up' / 'forward_up'
               / 'left_up' / 'right_up' / 'forward_down' / 'down' / 'other'
    """
    # ---- 前向判定：手腕在人物正前方 ----
    if fwd > config.ORI_FWD_STRONG and abs(lat) < config.ORI_LAT_TIGHT:
        if dy > config.ORI_DY_FWD_CHECK:
            return "forward_up"
        elif dy < -config.ORI_DY_FWD_CHECK:
            return "forward_down"
        else:
            return "forward"

    # ---- 弱前向 ----
    if fwd > config.ORI_FWD_WEAK and abs(lat) < config.ORI_LAT_STRONG:
        if dy > config.ORI_DY_STRONG:
            return "forward_up"
        elif dy < -config.ORI_DY_STRONG:
            return "forward_down"
        else:
            return "forward"

    # ---- 侧向判定 ----
    if abs(lat) > config.ORI_LAT_STRONG:
        if dy > config.ORI_DY_STRONG:
            return "left_up" if lat > 0 else "right_up"
        elif lat > 0:
            return "left"
        else:
            return "right"

    # ---- 上下判定 ----
    if dy > config.ORI_DY_STRONG:
        return "up"
    if dy < -config.ORI_DY_STRONG:
        return "down"

    return "other"


def get_arm_pose(wrist: tuple[float, float, float],
                 shoulder: tuple[float, float, float],
                 body_forward: tuple[float, float],
                 body_left: tuple[float, float]) -> dict:
    """
    返回手臂姿态的详细描述（供分类决策使用）。

    Args:
        wrist, shoulder          : 世界坐标 (x, y, z)
        body_forward, body_left  : 人物局部坐标系（XZ 平面）

    Returns:
        字典，包含字段：
          fwd, lat, dy   : 投影分量
          is_level       : 是否接近水平（平伸）
          is_side        : 是否侧伸
          is_forward     : 是否前伸
          is_raised      : 是否上举
          is_lowered     : 是否下垂
          mag_horizontal : XZ 平面投影幅度
    """
    dir_raw = (
        wrist[0] - shoulder[0],
        wrist[1] - shoulder[1],
        wrist[2] - shoulder[2],
    )
    dy = dir_raw[1]
    fwd = dir_raw[0] * body_forward[0] + dir_raw[2] * body_forward[1]
    lat = dir_raw[0] * body_left[0]    + dir_raw[2] * body_left[1]

    mag_xy = math.sqrt(fwd**2 + lat**2) if (fwd**2 + lat**2) > 1e-12 else 0.0

    return {
        "fwd": fwd,
        "lat": lat,
        "dy":  dy,
        "is_level":   abs(dy) < config.ORI_DY_FWD_CHECK and mag_xy > 0.15,
        "is_side":    abs(lat) > config.ORI_LAT_STRONG,
        "is_forward": fwd > config.ORI_FWD_WEAK and abs(lat) < config.ORI_LAT_STRONG,
        "is_raised":  dy > config.ORI_DY_STRONG,
        "is_lowered": dy < -config.ORI_DY_STRONG,
        "mag_horizontal": mag_xy,
    }


# ============================================================
# 运动/摆动检测
# ============================================================

def detect_circular_motion(trail: list) -> str:
    """
    根据手腕 (raise, stretch) 轨迹判定运动类型。

    Args:
        trail: [(raise, stretch), ...] 轨迹点列表

    Returns:
        "circular" / "vertical" / "horizontal" / "static"
    """
    if len(trail) < 5:
        return "static"

    xs = [p[0] for p in trail]
    ys = [p[1] for p in trail]
    var_x = float(np.var(xs)) if len(xs) > 1 else 0.0
    var_y = float(np.var(ys)) if len(ys) > 1 else 0.0

    if var_x > config.CIRCULAR_VAR_THRESHOLD and var_y > config.CIRCULAR_VAR_THRESHOLD:
        return "circular"
    elif var_y > var_x * 1.5 and var_y > config.CIRCULAR_VAR_THRESHOLD:
        return "vertical"
    elif var_x > var_y * 1.5 and var_x > config.CIRCULAR_VAR_THRESHOLD:
        return "horizontal"
    else:
        return "static"


def detect_oscillation(trail: list) -> tuple[bool, float]:
    """
    检测轨迹 raise 分量是否存在往返振荡。

    通过检测极大值（峰值）和极小值（谷值）的数量判断。
    至少需要 OSCILLATION_MIN_PEAKS 个交替极值点才视为振荡。

    Args:
        trail: [(raise, stretch), ...]  对 raise（第0维）做检测

    Returns:
        (是否振荡, 峰谷总幅度)
    """
    if len(trail) < 6:
        return (False, 0.0)

    vals = [p[0] for p in trail]

    peaks = []
    valleys = []

    for i in range(1, len(vals) - 1):
        left, mid, right = vals[i - 1], vals[i], vals[i + 1]
        if mid > left and mid > right:
            peaks.append(mid)
        elif mid < left and mid < right:
            valleys.append(mid)

    if len(peaks) >= config.OSCILLATION_MIN_PEAKS and len(valleys) >= config.OSCILLATION_MIN_PEAKS:
        peak_mean = sum(peaks) / len(peaks)
        valley_mean = sum(valleys) / len(valleys)
        amplitude = abs(peak_mean - valley_mean)
        return (True, amplitude)

    return (False, 0.0)


def detect_oscillation_stretch(trail: list) -> tuple[bool, float]:
    """
    检测轨迹 stretch 分量是否存在往返振荡（用于水平摆动检测）。

    Args:
        trail: [(raise, stretch), ...]  对 stretch（第1维）做检测

    Returns:
        (是否振荡, 峰谷总幅度)
    """
    if len(trail) < 6:
        return (False, 0.0)

    vals = [p[1] for p in trail]
    peaks = []
    valleys = []

    for i in range(1, len(vals) - 1):
        left, mid, right = vals[i - 1], vals[i], vals[i + 1]
        if mid > left and mid > right:
            peaks.append(mid)
        elif mid < left and mid < right:
            valleys.append(mid)

    if len(peaks) >= config.OSCILLATION_MIN_PEAKS and len(valleys) >= config.OSCILLATION_MIN_PEAKS:
        peak_mean = sum(peaks) / len(peaks)
        valley_mean = sum(valleys) / len(valleys)
        amplitude = abs(peak_mean - valley_mean)
        return (True, amplitude)

    return (False, 0.0)


def detect_swing_direction(trail: list,
                           shoulder_width: float = 0.35) -> tuple[str, tuple[float, float]]:
    """
    累计轨迹位移（世界坐标米制），判定摆动方向。

    改进点：
      - 位移阈值基于肩宽自适应（SWING_DISP_THRESH_FACTOR * shoulder_width）
      - 新增振荡检测，区分"持续位移"和"往返摆动"
      - 区分水平摆动（左右）和垂直摆动（上下）

    Args:
        trail:         [(raise, stretch), ...] 轨迹点列表
        shoulder_width: 肩宽（米），用于自适应阈值

    Returns:
        (swing_type, (sum_dx, sum_dy))
          swing_type: 'circular' | 'horizontal_swing' | 'vertical_swing'
                     | 'left_swing' | 'right_swing'
                     | 'up_swing' | 'down_swing'
                     | 'static'
    """
    if len(trail) < 3:
        return ('static', (0.0, 0.0))

    sum_dx = 0.0
    sum_dy = 0.0
    for i in range(1, len(trail)):
        prev_r, prev_s = trail[i - 1]
        curr_r, curr_s = trail[i]
        sum_dx += curr_s - prev_s    # stretch → 水平伸缩
        sum_dy += curr_r - prev_r    # raise   → 垂直抬降

    abs_dx = abs(sum_dx)
    abs_dy = abs(sum_dy)

    disp_thresh = max(config.SWING_DISP_THRESH_FACTOR * shoulder_width, 0.10)

    # 先检查 circular（优先保留圆形运动）
    motion = detect_circular_motion(trail)
    if motion == "circular":
        # 但如果水平位移远大于垂直位移，即使轨迹有圆形特征，
        # 也降级为水平摆动（适应水平摆动中伴随的轻微上下浮动）
        if abs_dx > abs_dy * 0.6 and abs_dx > disp_thresh:
            if sum_dx < 0:
                return ('left_swing', (sum_dx, sum_dy))
            else:
                return ('right_swing', (sum_dx, sum_dy))
        # 同理，垂直位移远大于水平位移 → 降级为垂直摆动
        if abs_dy > abs_dx * 0.6 and abs_dy > disp_thresh:
            if sum_dy < 0:
                return ('up_swing', (sum_dx, sum_dy))
            else:
                return ('down_swing', (sum_dx, sum_dy))
        return ('circular', (sum_dx, sum_dy))

    # 振荡检测
    has_osc_raise, osc_amp_raise = detect_oscillation(trail)
    has_osc_stretch, osc_amp_stretch = detect_oscillation_stretch(trail)

    # 水平摆动（stretch 变化为主）
    # 放宽判定：允许垂直分量达到水平分量的 60%，适应真实动作中的轻微上下浮动
    if abs_dx > disp_thresh and abs_dx > abs_dy * 0.6:
        if has_osc_stretch and osc_amp_stretch > disp_thresh * 0.3:
            return ('horizontal_swing', (sum_dx, sum_dy))
        elif sum_dx < 0:
            return ('left_swing', (sum_dx, sum_dy))
        else:
            return ('right_swing', (sum_dx, sum_dy))

    # 垂直摆动（raise 变化为主）
    # 保持与水平一致的放宽比例
    if abs_dy > disp_thresh and abs_dy > abs_dx * 0.6:
        if has_osc_raise and osc_amp_raise > disp_thresh * 0.3:
            return ('vertical_swing', (sum_dx, sum_dy))
        elif sum_dy < 0:
            return ('up_swing', (sum_dx, sum_dy))
        else:
            return ('down_swing', (sum_dx, sum_dy))

    # 振荡幅度足够但累计位移不明显，仍判定为摆动
    if has_osc_stretch and osc_amp_stretch > disp_thresh * 0.5:
        return ('horizontal_swing', (sum_dx, sum_dy))
    if has_osc_raise and osc_amp_raise > disp_thresh * 0.5:
        return ('vertical_swing', (sum_dx, sum_dy))

    return ('static', (sum_dx, sum_dy))


# ============================================================
# 手臂稳定性检测
# ============================================================

def is_arm_stable(hand_trail: list, threshold: float = None) -> bool:
    """
    判断手臂是否固定不动，基于手腕轨迹的方差。

    Args:
        hand_trail: [(raise, stretch), ...] 或 [(wx, wy), ...]
        threshold:  方差阈值，默认使用 ARM_STABLE_THRESHOLD

    Returns:
        True 如果手臂稳定（轨迹方差小于阈值）
    """
    if threshold is None:
        threshold = config.ARM_STABLE_THRESHOLD
    if len(hand_trail) < 3:
        return True
    xs = [p[0] for p in hand_trail]
    ys = [p[1] for p in hand_trail]
    var_x = float(np.var(xs)) if len(xs) > 1 else 0.0
    var_y = float(np.var(ys)) if len(ys) > 1 else 0.0
    return (var_x + var_y) < threshold
