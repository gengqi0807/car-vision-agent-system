"""
config.py — 集中管理所有可调参数

参数分组：
  1. 状态机参数
  2. 区域划分参数
  3. 手臂方位分类阈值
  4. 摆动检测阈值
  5. 手臂稳定性阈值
  6. 手掌朝向阈值
  7. 分类置信度
  8. 模型与摄像头参数
"""

import os
import sys


def _get_default_font_path() -> str:
    """
    自动检测操作系统并返回默认中文字体路径。

    检测优先级：
      Windows → simhei.ttf / msyh.ttc / simsun.ttc
      macOS   → PingFang.ttc / STHeiti Light.ttc
      Linux   → wqy-microhei.ttc / wqy-zenhei.ttc / NotoSansCJK
      Fallback → 项目目录 fonts/simhei.ttf

    Returns:
        最佳候选字体路径（文件不一定存在）
    """
    candidates = []
    if sys.platform == "win32":
        candidates = [
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simsun.ttc",
        ]
    elif sys.platform == "darwin":
        candidates = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/STHeiti.ttc",
        ]
    else:  # Linux / 其他
        candidates = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        ]
    # 项目本地字体作为最后兜底
    candidates.append(os.path.join("fonts", "simhei.ttf"))
    for fp in candidates:
        if os.path.exists(fp):
            return fp
    return candidates[0]  # 返回第一个候选路径（供用户参考）


# ============================================================
# 0. 字体配置（在其余参数之前加载）
# ============================================================

FONT_PATH = os.environ.get("POLICE_FONT_PATH", _get_default_font_path())

# ============================================================
# 1. 状态机参数
# ============================================================

STATE_IDLE   = 0   # 空闲：等待动作开始
STATE_ACTIVE = 1   # 活跃：正在执行动作

# 动作最低持续帧数（少于此视为误触，不输出结果）
MIN_ACTION_FRAMES = 3

# 超时强制结束帧数（适应复杂动作的多个摆动周期）
MAX_FRAMES = 40

# 连续满足触发条件此帧数后进入 ACTIVE（防止瞬时抖动）
START_CONFIRM = 1           # 一帧触发，立即响应

# 动作结束后冷却期帧数（防止瞬间重复触发）
COOLDOWN_FRAMES = 2

# 人体位移阈值（世界坐标米制）：动作期间身体移动超过此值视为行走
BODY_MOVE_THRESHOLD = 0.30  # 30cm

# ----------------------------------------------------------
# 全身移动检测（基于下半身骨架关键点，非单纯距离）
# ----------------------------------------------------------
# 下半身关键点滑动窗口大小（帧）
BODY_WINDOW_SIZE = 10
# 下肢关键点（髋/膝/踝）滑动窗口内的平均位移阈值（米）
# 超过此值视为整个人在移动（行走）
LOWER_BODY_DISP_THRESHOLD = 0.15  # 15cm

# ----------------------------------------------------------
# 动作开始/停止检测
# ----------------------------------------------------------
# 动作开始：单手离开hip，单帧立即触发（不使用窗口）
# 动作停止：双手必须连续在hip达到此帧数才算真正停止
#           短暂回落（靠边停车的左右手切换、左待转的上下摆动）不会中断
HIP_CONSECUTIVE_STOP = 8  # 连续8帧双手在hip → 动作停止 (~267ms @30fps)

# 显示层锁定：DL 预测同一手势连续 N 帧才锁定显示
# 过滤动作初期的不稳定预测（如变道初帧被误判为右转弯）
LOCK_EARLY_SHOW = 5       # 连续5帧相同 → 早期结果 (~167ms @30fps)
LOCK_CONSECUTIVE_CONFIRM = 7  # 持续判断到7帧，5-7帧内可变

# 稳定识别间隔：每 N 帧做一次中间手势判定（用于取众数去噪）
STABLE_CLASSIFY_INTERVAL = 5

# 识别结果在画面上的持续显示帧数（超过后自动清除，回到"等待动作"）
RESULT_DISPLAY_FRAMES = 90   # 约 3 秒 @ 30fps

# ============================================================
# 2. 区域划分参数（基于手与肩的高度差，世界坐标米制）
# ============================================================
# dy = shoulder_y - hand_y（MediaPipe 世界坐标 y 轴向下，肩在上方数值小）
# 头部区域：dy > HEAD_THRESHOLD    （手明显高于肩，如举手过头）
# 肩部区域：SHOULDER_LOWER <= dy <= HEAD_THRESHOLD
# 腰部区域：WAIST_LOWER <= dy < SHOULDER_LOWER
# 胯部区域：dy < WAIST_LOWER        （手远低于肩，如自然下垂）
#
# 不依赖髋部坐标，仅使用肩作为锚点，避免 hip 关键点丢失导致划分失效。

HEAD_THRESHOLD = 0.05      # 手高于肩 5cm 以上 → head
SHOULDER_LOWER = -0.15     # 肩下15cm ~ 肩上5cm → shoulder  (20cm宽)
WAIST_LOWER = -0.47        # 肩下47cm ~ 肩下15cm → waist  (32cm宽)

# ============================================================
# 3. 手臂方位分类阈值（用于 classify_arm_orientation）
# ============================================================

ORI_FWD_STRONG  = 0.20   # fwd > 此值 → 明确前伸
ORI_FWD_WEAK    = 0.10   # fwd > 此值且 lat 小 → 弱前伸
ORI_LAT_STRONG  = 0.25   # |lat| > 此值 → 明确侧伸
ORI_LAT_TIGHT   = 0.25   # |lat| < 此值 → 非侧向（用于前向判定）
ORI_DY_STRONG   = 0.25   # |dy| > 此值 → 明确上举/下垂
ORI_DY_FWD_CHECK = 0.20  # 前向判定时区分 forward_up vs forward

# ============================================================
# 4. 摆动检测阈值（世界坐标，米制）
# ============================================================

# 位移阈值 = 肩宽 × 此因子（用于累计位移判定）
# 从 0.30 降至 0.15：水平合位移 = sqrt(fwd²+lat²) 方向分量合并后更容易达到阈值
SWING_DISP_THRESH_FACTOR = 0.15

# 轨迹方差阈值（画圈判定）
CIRCULAR_VAR_THRESHOLD = 0.005

# 最少峰值数量（判断往返摆动）
OSCILLATION_MIN_PEAKS = 2

# ============================================================
# 5. 手臂稳定性阈值
# ============================================================

# 手腕轨迹方差阈值（世界坐标，米制^2），小于此值视为稳定不动
# 放宽到 0.10，允许小幅自然颤动，避免检测噪声导致误判为不稳定
ARM_STABLE_THRESHOLD = 0.10

# ============================================================
# 6. 手掌朝向阈值
# ============================================================

PALM_FWD_NZ = -0.25       # nz < 此值 → forward（掌心向前/朝向相机）
PALM_BWD_NZ = 0.25        # nz > 此值 → backward（掌心向后/远离相机）
PALM_PROJ_THRESH = 0.40   # proj_up / proj_right 阈值

# ============================================================
# 7. 分类决策置信度
# ============================================================

CONFIDENCE_BASE  = 0.70   # 基础条件满足时的基础置信度
CONFIDENCE_EXTRA = 0.10   # 每个额外条件加分
CONFIDENCE_MAX   = 0.95   # 置信度上限

# ============================================================
# 8. 模型与摄像头参数
# ============================================================

# 前置摄像头镜像标志。
# ⚠️ 分类器规则基于人物解剖学左右（如左转弯=人物左手做动作），
#    features.py 中的交换会破坏这个一致性。
#    保持 False，让 L/R 始终=人物自身左右。
CAMERA_MIRRORED = False

# AI 推理跳帧数（每隔 N 帧推理一次，提升流畅度）
SKIP_FRAMES = 2

# 推理缩放比（原始帧缩放到此比例后送入模型，平衡速度与精度）
INFER_SCALE = 0.50

# --- 模型文件路径（相对于项目根目录） ---
# 可通过环境变量覆盖
POSE_MODEL_PATH = os.environ.get(
    "POLICE_POSE_MODEL",
    os.path.join("backend", "pose_landmarker_lite.task")
)
HAND_MODEL_PATH = os.environ.get(
    "POLICE_HAND_MODEL",
    os.path.join("backend", "hand_landmarker.task")
)

# --- 默认视频源 ---
DEFAULT_RTSP_URL = "rtsp://127.0.0.1:8554/test"

# --- 媒体管道参数 ---
POSE_DETECTION_CONFIDENCE  = 0.5
POSE_PRESENCE_CONFIDENCE   = 0.5
POSE_TRACKING_CONFIDENCE   = 0.5
HAND_DETECTION_CONFIDENCE  = 0.4    # 降低检测阈值，提高召回率（VIDEO模式下跟踪更稳定）
HAND_PRESENCE_CONFIDENCE   = 0.4
HAND_TRACKING_CONFIDENCE   = 0.3    # VIDEO模式跟踪阈值放低，减少跟丢
NUM_POSES = 1
NUM_HANDS = 2

# --- 手部检测专用参数 ---
# 手部检测缩放比（独立于 INFER_SCALE，手部需要更高分辨率才能准确定位）
HAND_INFER_SCALE = 1.0            # 全分辨率，手部特征不会被缩小抹平
# 手部数据持久化帧数（检测丢失时保留最近数据，防止闪烁）
# 降低到 2-3 帧，避免旧数据滞留导致识别延迟
HAND_PERSIST_FRAMES = 3
# EMA 平滑因子（0~1，越大越灵敏，0.65 兼顾响应速度与防抖）
HAND_EMA_ALPHA = 0.65
# associate_hands 最近邻距离阈值（归一化坐标，放宽以减少漏检）
HAND_ASSOCIATE_THRESH = 0.22
# 被遮挡手臂特征冻结的 visibility 阈值
HAND_VISIBILITY_FREEZE = 0.25
