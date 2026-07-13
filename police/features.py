"""
features.py — 逐帧动作特征提取

包含：
  - extract_features       : 核心特征提取函数（替代原 compute_features）
  - classify_palm_orientation : 手掌朝向检测
  - analyze_hand           : 手部特征分析（手掌张开/手指向上）
  - associate_hands        : 将 Hand 检测结果与 Pose 左右腕关联

所有函数使用 MediaPipe 世界坐标（米制）。
"""

import math

from . import config
from .geometry import (
    calc_angle, calc_dist,
    compute_body_orientation, classify_arm_orientation, get_arm_pose,
    get_hand_y, get_hand_region,
)


# ============================================================
# 特征提取（核心）
# ============================================================

def extract_features(world_landmarks,
                     landmarks=None,
                     hand_left=None,
                     hand_right=None) -> dict:
    """
    使用 MediaPipe 世界坐标（米制）提取单帧动作特征。

    计算的特征包括：
      raise   = wrist.y - shoulder.y（MediaPipe y 向下，正值=手腕低于肩）
      stretch = 手腕→肩膀 3D 欧氏距离（米）
      z_diff  = wrist.z - shoulder.z（负值=前伸）
      arm_angle = 肘关节 3D 夹角（°）
      orient    = 手臂方位标签（基于人物坐标系）
      region    = 手部区域（基于手指 Y 坐标）

    手部区域判断：优先使用中指指尖（Hand Landmarker index=12），
    回退到手腕位置 + 偏移量。

    Args:
        world_landmarks: MediaPipe Pose 世界坐标关键点（33 个）
        landmarks:       MediaPipe Pose 归一化关键点（33 个），可选
        hand_left:       Hand Landmarker 左手关键点列表，或 None
        hand_right:      Hand Landmarker 右手关键点列表，或 None

    Returns:
        特征字典，包含 30+ 个字段（详见代码中的 result 构建部分）。
        CAMERA_MIRRORED=True 时会自动交换左右标签。
    """
    # ---- 提取关键点世界坐标 ----
    # 左臂：肩(11) 肘(13) 腕(15)
    ls3 = (world_landmarks[11].x, world_landmarks[11].y, world_landmarks[11].z)
    le3 = (world_landmarks[13].x, world_landmarks[13].y, world_landmarks[13].z)
    lw3 = (world_landmarks[15].x, world_landmarks[15].y, world_landmarks[15].z)

    # 右臂：肩(12) 肘(14) 腕(16)
    rs3 = (world_landmarks[12].x, world_landmarks[12].y, world_landmarks[12].z)
    re3 = (world_landmarks[14].x, world_landmarks[14].y, world_landmarks[14].z)
    rw3 = (world_landmarks[16].x, world_landmarks[16].y, world_landmarks[16].z)

    # ---- raise: wrist.y - shoulder.y ----
    left_raise  = lw3[1] - ls3[1]
    right_raise = rw3[1] - rs3[1]

    # ---- stretch: 手腕到肩膀 3D 距离 ----
    def dist3d(a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    left_stretch  = dist3d(lw3, ls3)
    right_stretch = dist3d(rw3, rs3)

    # ---- arm_angle: 肘关节夹角 ----
    left_arm_angle  = calc_angle(ls3, le3, lw3)
    right_arm_angle = calc_angle(rs3, re3, rw3)

    # ---- z_diff: wrist.z - shoulder.z ----
    left_z_diff  = lw3[2] - ls3[2]
    right_z_diff = rw3[2] - rs3[2]

    # ---- 人物朝向（局部坐标系） ----
    body_forward, body_left = compute_body_orientation(world_landmarks)

    # ---- 手臂方位投影（XZ 平面） ----
    left_dir_raw = (
        lw3[0] - ls3[0],
        lw3[1] - ls3[1],
        lw3[2] - ls3[2],
    )
    right_dir_raw = (
        rw3[0] - rs3[0],
        rw3[1] - rs3[1],
        rw3[2] - rs3[2],
    )

    left_fwd  = left_dir_raw[0] * body_forward[0] + left_dir_raw[2] * body_forward[1]
    left_lat  = left_dir_raw[0] * body_left[0]    + left_dir_raw[2] * body_left[1]
    right_fwd = right_dir_raw[0] * body_forward[0] + right_dir_raw[2] * body_forward[1]
    right_lat = right_dir_raw[0] * body_left[0]    + right_dir_raw[2] * body_left[1]

    # ---- 手臂方位标签 ----
    left_orient  = classify_arm_orientation(left_fwd,  left_lat,  left_raise)
    right_orient = classify_arm_orientation(right_fwd, right_lat, right_raise)

    # ---- 详细手臂姿态 ----
    left_pose  = get_arm_pose(lw3, ls3, body_forward, body_left)
    right_pose = get_arm_pose(rw3, rs3, body_forward, body_left)

    # ---- 身体关键点 Y 坐标及人体中心 ----
    hip_y      = (world_landmarks[23].y + world_landmarks[24].y) / 2.0
    shoulder_y = (world_landmarks[11].y + world_landmarks[12].y) / 2.0
    head_y     = world_landmarks[0].y
    # 人体髋部中心（世界坐标 X/Z，用于位移检测）
    body_cx = (world_landmarks[23].x + world_landmarks[24].x) / 2.0
    body_cz = (world_landmarks[23].z + world_landmarks[24].z) / 2.0
    # ---- 下半身关键点世界坐标（用于全身移动检测） ----
    left_knee_xz  = (world_landmarks[25].x, world_landmarks[25].z)
    right_knee_xz = (world_landmarks[26].x, world_landmarks[26].z)
    left_ankle_xz  = (world_landmarks[27].x, world_landmarks[27].z)
    right_ankle_xz = (world_landmarks[28].x, world_landmarks[28].z)
    left_hip_xz  = (world_landmarks[23].x, world_landmarks[23].z)
    right_hip_xz = (world_landmarks[24].x, world_landmarks[24].z)

    # ---- 手部区域（基于 Pose 手腕世界坐标 + 指尖偏移） ----
    # 注意：Hand Landmarker 的 hand_landmarks 是归一化图像坐标(0-1)，
    # 不能直接与 Pose 世界坐标(米)做运算，必须统一使用 Pose 坐标。
    # 手腕到中指指尖约 8cm，用 wrist_y + 0.08 近似手指高度。
    left_hand_y  = get_hand_y(hand_left)      # 仅用于调试显示
    right_hand_y = get_hand_y(hand_right)

    left_hand_y_for_region  = lw3[1] + 0.08   # Pose 手腕世界坐标 + 指尖偏移
    right_hand_y_for_region = rw3[1] + 0.08

    left_region  = get_hand_region(left_hand_y_for_region, shoulder_y, hip_y, head_y)
    right_region = get_hand_region(right_hand_y_for_region, shoulder_y, hip_y, head_y)

    # ---- 肩宽 ----
    shoulder_width = math.sqrt(
        (rs3[0] - ls3[0])**2 + (rs3[1] - ls3[1])**2 + (rs3[2] - ls3[2])**2
    )

    # ---- 构建特征字典 ----
    result = {
        "left_arm_angle":  left_arm_angle,
        "right_arm_angle": right_arm_angle,
        "left_raise":      left_raise,
        "right_raise":     right_raise,
        "left_stretch":    left_stretch,
        "right_stretch":   right_stretch,
        "left_z_diff":     left_z_diff,
        "right_z_diff":    right_z_diff,
        # 世界坐标 X/Y（用于方向/摆动判定）
        "left_wx":  lw3[0],  "left_sx":  ls3[0],
        "right_wx": rw3[0],  "right_sx": rs3[0],
        "left_wy":  lw3[1],  "left_sy":  ls3[1],
        "right_wy": rw3[1],  "right_sy": rs3[1],
        # 手臂方位
        "left_orient":  left_orient,
        "right_orient": right_orient,
        # 手臂详细姿态
        "left_pose":   left_pose,
        "right_pose":  right_pose,
        # 原始方向向量（世界坐标，调试用）
        "left_dir_raw":  left_dir_raw,
        "right_dir_raw": right_dir_raw,
        # 人物坐标系投影分量（调试用）
        "left_fwd":   left_fwd,
        "left_lat":   left_lat,
        "right_fwd":  right_fwd,
        "right_lat":  right_lat,
        # 人物朝向向量（调试用）
        "body_forward": body_forward,
        "body_left":    body_left,
        # 手部空间区域
        "left_region":  left_region,
        "right_region": right_region,
        # 指尖 Y 坐标
        "left_hand_y":   left_hand_y,
        "right_hand_y":  right_hand_y,
        # 身体关键点 Y 坐标（用于调试区域划分）
        "shoulder_y": shoulder_y,
        "hip_y":      hip_y,
        "head_y":     head_y,
        # 人体髋部中心世界坐标（用于行走检测）
        "body_cx": body_cx,
        "body_cz": body_cz,
        # 下半身关键点世界坐标 XZ（用于全身移动检测）
        "left_knee_xz":   left_knee_xz,
        "right_knee_xz":  right_knee_xz,
        "left_ankle_xz":  left_ankle_xz,
        "right_ankle_xz": right_ankle_xz,
        "left_hip_xz":   left_hip_xz,
        "right_hip_xz":  right_hip_xz,
        # 肩宽
        "shoulder_width": shoulder_width,
    }

    # ---- 镜像摄像机：交换左右标签 ----
    if config.CAMERA_MIRRORED:
        for key in ("arm_angle", "raise", "stretch", "z_diff",
                    "wx", "sx", "wy", "sy",
                    "region", "orient", "pose",
                    "dir_raw",
                    "fwd", "lat",
                    ):
            l_val = result[f"left_{key}"]
            r_val = result[f"right_{key}"]
            result[f"left_{key}"]  = r_val
            result[f"right_{key}"] = l_val

    return result


# ============================================================
# 手掌朝向检测
# ============================================================

def classify_palm_orientation(hand_landmarks,
                              body_right: tuple[float, float],
                              body_up: tuple[float, float]) -> str:
    """
    使用 Hand Landmarker 的 21 个关键点检测手掌朝向。

    取手腕(0)、食指根部(5)、小指根部(17) 构造法向量，
    投影到局部坐标系，判断掌心朝向。

    Args:
        hand_landmarks: Hand Landmarker 21 个关键点列表，或 None
        body_right:     人物右侧方向单位向量（图像坐标）
        body_up:        人物上方方向单位向量（图像坐标）

    Returns:
        朝向标签：'forward' | 'backward' | 'down' | 'left' | 'right' | 'unknown'
    """
    if hand_landmarks is None or len(hand_landmarks) < 21:
        return 'unknown'

    w0  = hand_landmarks[0]
    idx = hand_landmarks[5]
    pk  = hand_landmarks[17]

    v1 = (idx.x - w0.x, idx.y - w0.y, idx.z - w0.z)
    v2 = (pk.x - w0.x,   pk.y - w0.y,   pk.z - w0.z)

    nx = v1[1] * v2[2] - v1[2] * v2[1]
    ny = v1[2] * v2[0] - v1[0] * v2[2]
    nz = v1[0] * v2[1] - v1[1] * v2[0]

    mag = math.sqrt(nx * nx + ny * ny + nz * nz)
    if mag < 1e-6:
        return 'unknown'
    nx /= mag
    ny /= mag
    nz /= mag

    proj_right = nx * body_right[0] + ny * body_right[1]
    proj_up    = nx * body_up[0]    + ny * body_up[1]

    # 掌心向前（靠近相机）/ 向后（远离相机）
    if nz < config.PALM_FWD_NZ:
        return 'forward'
    if nz > config.PALM_BWD_NZ:
        return 'backward'

    # 掌心朝向（上→down, 右→left, 左→right）
    if proj_up > config.PALM_PROJ_THRESH:
        return 'down'
    if proj_right > config.PALM_PROJ_THRESH * 0.8:
        return 'left'
    if proj_right < -config.PALM_PROJ_THRESH * 0.8:
        return 'right'

    return 'forward'


# ============================================================
# 手部特征分析
# ============================================================

def analyze_hand(hand_landmarks, wrist_px=None) -> dict:
    """
    分析单手特征：手掌是否张开、手指是否向上。

    Args:
        hand_landmarks: Hand Landmarker 21 个关键点列表，或 None
        wrist_px:       手腕像素坐标，未使用（保留接口兼容性）

    Returns:
        {"palm_open": bool, "fingers_up": bool, "side": str}
    """
    if hand_landmarks is None or len(hand_landmarks) < 21:
        return {"palm_open": False, "fingers_up": False, "side": "unknown"}

    tips = [4, 8, 12, 16, 20]
    wrist = hand_landmarks[0]

    # 手指展开程度（尖端间距之和）
    spread_sum = 0.0
    for i in range(len(tips) - 1):
        a = hand_landmarks[tips[i]]
        b = hand_landmarks[tips[i + 1]]
        spread_sum += math.hypot(a.x - b.x, a.y - b.y)
    palm_open = spread_sum > 0.25

    # 中指指尖是否在手腕上方
    middle_tip = hand_landmarks[12]
    fingers_up = middle_tip.y < wrist.y

    return {"palm_open": palm_open, "fingers_up": fingers_up, "side": "unknown"}


# ============================================================
# Hand 关联
# ============================================================

def associate_hands(hand_result,
                    left_wrist_px: tuple[float, float],
                    right_wrist_px: tuple[float, float]) -> dict:
    """
    将 Hand Landmarker 检测结果与 Pose 的左右腕关联。

    使用最近邻匹配，匹配失败时返回 None（调用方应保留上一帧结果）。

    Args:
        hand_result:   HandLandmarkerResult，或 None
        left_wrist_px:  左手腕归一化坐标 (x, y)
        right_wrist_px: 右手腕归一化坐标 (x, y)

    Returns:
        {"left": hand_landmarks | None, "right": hand_landmarks | None}
    """
    left_hand = None
    right_hand = None

    if not hand_result or not hand_result.hand_landmarks:
        return {"left": None, "right": None}

    hands = hand_result.hand_landmarks
    handedness = hand_result.handedness if hand_result.handedness else []

    thresh = config.HAND_ASSOCIATE_THRESH

    # 有 handedness 标签：优先使用标签直接匹配，距离作为二次校验
    if len(handedness) == len(hands):
        for i, hl in enumerate(hands):
            hw = (hl[0].x, hl[0].y)
            label = handedness[i][0].category_name.lower()  # 'left' / 'right'

            if label == 'left':
                dist = calc_dist(hw, left_wrist_px) if left_wrist_px else float('inf')
                if dist < thresh and left_hand is None:
                    left_hand = hl
            elif label == 'right':
                dist = calc_dist(hw, right_wrist_px) if right_wrist_px else float('inf')
                if dist < thresh and right_hand is None:
                    right_hand = hl

        # 标签匹配后仍有未分配的 → 距离兜底
        if left_hand is None or right_hand is None:
            for i, hl in enumerate(hands):
                hw = (hl[0].x, hl[0].y)
                dist_l = calc_dist(hw, left_wrist_px) if left_wrist_px else float('inf')
                dist_r = calc_dist(hw, right_wrist_px) if right_wrist_px else float('inf')
                if left_hand is None and dist_l < thresh:
                    left_hand = hl
                elif right_hand is None and dist_r < thresh:
                    right_hand = hl
    else:
        # 无 handedness 标签，纯距离匹配
        for hl in hands:
            hw = (hl[0].x, hl[0].y)
            dist_l = calc_dist(hw, left_wrist_px) if left_wrist_px else float('inf')
            dist_r = calc_dist(hw, right_wrist_px) if right_wrist_px else float('inf')

            if dist_l < dist_r and dist_l < thresh:
                left_hand = hl
            elif dist_r < dist_l and dist_r < thresh:
                right_hand = hl

    return {"left": left_hand, "right": right_hand}
