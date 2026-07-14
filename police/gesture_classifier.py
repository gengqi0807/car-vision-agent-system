"""
gesture_classifier.py — 状态机 + 8 种交警手势分类决策树

核心组件：
  - GestureStateMachine : 状态机（IDLE → ACTIVE → IDLE）
  - reset_action_data   : 创建动作数据容器
  - classify_action     : 8 种手势的严格判定（含人体移动检测）
  - mode_str / mode_list : 众数工具

状态机规则：
  - 进入 ACTIVE : 至少一只手离开胯部，且至少一只手在肩/头/腰区域
  - 在 ACTIVE 中: 持续记录特征；双手都回到hip则立即结束
  - 稳定识别   : ACTIVE期间每N帧做中间判定，取众数作为最终结果
  - 人体移动   : 动作期间身体位移超阈值 → 判定为"无动作"
  - 结束后     : 进入"无动作"显示，冷却期不触发新动作
"""

import math
from collections import Counter

from . import config


# ============================================================
# 众数工具
# ============================================================

def mode_list(items: list) -> str:
    """返回列表中频率最高的元素，空列表返回 'unknown'。"""
    if not items:
        return 'unknown'
    return Counter(items).most_common(1)[0][0]


def mode_str(items: list) -> str:
    """返回列表中频率最高的元素，空列表返回 '?'。"""
    if not items:
        return '?'
    return Counter(items).most_common(1)[0][0]


# ============================================================
# 动作数据容器 - reset_action_data
# ============================================================

def reset_action_data(shoulder_width: float,
                      initial_feat: dict,
                      left_palm_ori: str,
                      right_palm_ori: str) -> dict:
    """
    创建并初始化动作数据容器。

    包含动作期间的极值、轨迹历史、区域历史、掌心朝向等累积字段。
    第一次调用时填入初始帧的特征值。

    Args:
        shoulder_width:  肩宽（米），用于自适应摆动阈值
        initial_feat:    第一帧的 extract_features 结果
        left_palm_ori:   左手掌朝向标签
        right_palm_ori:  右手掌朝向标签

    Returns:
        初始化的 action_data 字典
    """
    sw = shoulder_width
    f = initial_feat

    active_arm = "both"
    if f["left_region"] != "hip" and f["right_region"] == "hip":
        active_arm = "left"
    elif f["right_region"] != "hip" and f["left_region"] == "hip":
        active_arm = "right"

    data = {
        # --- 抬升值极值 ---
        "min_left_raise":  f["left_raise"],
        "min_right_raise": f["right_raise"],
        "max_left_raise":  f["left_raise"],
        "max_right_raise": f["right_raise"],
        # --- 伸展距离极值 ---
        "max_left_stretch":  f["left_stretch"],
        "max_right_stretch": f["right_stretch"],
        "min_left_stretch":  f["left_stretch"],
        "min_right_stretch": f["right_stretch"],
        # --- 肘角最小值 ---
        "min_left_angle":  f["left_arm_angle"],
        "min_right_angle": f["right_arm_angle"],
        # --- Z 轴深度极值 ---
        "min_left_z_diff":  f["left_z_diff"],
        "min_right_z_diff": f["right_z_diff"],
        "max_left_z_diff":  f["left_z_diff"],
        "max_right_z_diff": f["right_z_diff"],
        "left_z_diffs":  [f["left_z_diff"]],
        "right_z_diffs": [f["right_z_diff"]],
        # --- 世界坐标 X 极值 ---
        "max_left_wx":  f["left_wx"],
        "min_left_wx":  f["left_wx"],
        "max_right_wx": f["right_wx"],
        "min_right_wx": f["right_wx"],
        "left_sx":  f["left_sx"],
        "right_sx": f["right_sx"],
        "left_sy":  f["left_sy"],
        "right_sy": f["right_sy"],
        # --- 抬升历史 ---
        "raise_history": [
            (1, f["left_raise"], f["right_raise"])
        ],
        "frame_count": 1,
        # --- 手腕轨迹（(raise, horiz)，horiz = 水平合位移带方向） ---
        # horiz = sqrt(fwd² + lat²)，符号用 lat 方向（兼顾侧面站立时 fwd 方向的位移）
        "left_wrist_trail": [
            (f["left_raise"],
             math.copysign(math.sqrt(f["left_fwd"]**2 + f["left_lat"]**2),
                           f["left_lat"]))
        ],
        "right_wrist_trail": [
            (f["right_raise"],
             math.copysign(math.sqrt(f["right_fwd"]**2 + f["right_lat"]**2),
                           f["right_lat"]))
        ],
        # --- 手掌朝向历史 ---
        "left_palm_orientations":  [left_palm_ori],
        "right_palm_orientations": [right_palm_ori],
        # --- 手臂方位历史 ---
        "left_orientations":  [f["left_orient"]],
        "right_orientations": [f["right_orient"]],
        # --- 手腕空间区域历史 ---
        "left_regions":  [f["left_region"]],
        "right_regions": [f["right_region"]],
        # --- 原始方向向量 ---
        "left_dir_raws":  [f["left_dir_raw"]],
        "right_dir_raws": [f["right_dir_raw"]],
        # --- 人物坐标系投影分量 ---
        "left_fwds":  [f["left_fwd"]],
        "left_lats":  [f["left_lat"]],
        "right_fwds": [f["right_fwd"]],
        "right_lats": [f["right_lat"]],
        # --- 人物朝向向量 ---
        "body_forwards": [f["body_forward"]],
        # --- 手指数据可用性 ---
        "left_finger_available":  f.get("left_hand_y") is not None,
        "right_finger_available": f.get("right_hand_y") is not None,
        # --- 手臂详细姿态 ---
        "left_poses":  [f["left_pose"]],
        "right_poses": [f["right_pose"]],
        # --- 主动手标记 ---
        "active_arm": active_arm,
        # --- 摆动位移累积 ---
        "prev_left_stretch":  f["left_stretch"],
        "prev_left_raise":    f["left_raise"],
        "prev_right_stretch": f["right_stretch"],
        "prev_right_raise":   f["right_raise"],
        "sum_left_dx":  0.0,
        "sum_left_dy":  0.0,
        "sum_right_dx": 0.0,
        "sum_right_dy": 0.0,
        # --- 退出逻辑 ---
        "hip_hold_count": 1 if (
            f["left_region"] == "hip" and f["right_region"] == "hip"
        ) else 0,
        # --- 人体位移追踪（行走检测） ---
        "body_start_cx": f.get("body_cx", 0.0),
        "body_start_cz": f.get("body_cz", 0.0),
        "body_max_disp": 0.0,  # 动作期间最大位移（米）
        # --- 稳定识别 ---
        "stable_results": [],   # 中间判定的累积结果
        # --- 可见性标记 ---
        "left_visible": True,
        "right_visible": True,
        # --- 肩宽 ---
        "shoulder_width": sw,
    }

    if data.get("_verbose", True):
        print(f"  [active_arm] = {active_arm}")
    return data


def _update_action_data(action_data: dict, feat: dict,
                        left_palm_ori: str, right_palm_ori: str):
    """
    将新一帧的特征追加到 action_data（极值更新 + 历史追加）。

    此函数直接在 action_data 上修改，不返回值。
    """
    fc = action_data["frame_count"] + 1
    action_data["frame_count"] = fc

    # ---- 区域检测（用于退出判断）----
    left_region_cur  = feat["left_region"]
    right_region_cur = feat["right_region"]

    # ---- 双手回落检测（必须双手都在hip才计数） ----
    if left_region_cur == "hip" and right_region_cur == "hip":
        action_data["hip_hold_count"] += 1
    else:
        action_data["hip_hold_count"] = 0

    # ---- 人体位移追踪（行走检测） ----
    if "body_start_cx" in action_data:
        cx = feat.get("body_cx", action_data["body_start_cx"])
        cz = feat.get("body_cz", action_data["body_start_cz"])
        d = math.sqrt(
            (cx - action_data["body_start_cx"])**2 +
            (cz - action_data["body_start_cz"])**2
        )
        if d > action_data.get("body_max_disp", 0.0):
            action_data["body_max_disp"] = d

    # ---- 极值更新 ----
    # raise
    if feat["left_raise"] < action_data["min_left_raise"]:
        action_data["min_left_raise"] = feat["left_raise"]
    if feat["right_raise"] < action_data["min_right_raise"]:
        action_data["min_right_raise"] = feat["right_raise"]
    if feat["left_raise"] > action_data["max_left_raise"]:
        action_data["max_left_raise"] = feat["left_raise"]
    if feat["right_raise"] > action_data["max_right_raise"]:
        action_data["max_right_raise"] = feat["right_raise"]

    # stretch
    if feat["left_stretch"] > action_data["max_left_stretch"]:
        action_data["max_left_stretch"] = feat["left_stretch"]
    if feat["right_stretch"] > action_data["max_right_stretch"]:
        action_data["max_right_stretch"] = feat["right_stretch"]
    if feat["left_stretch"] < action_data["min_left_stretch"]:
        action_data["min_left_stretch"] = feat["left_stretch"]
    if feat["right_stretch"] < action_data["min_right_stretch"]:
        action_data["min_right_stretch"] = feat["right_stretch"]

    # elbow angle
    if feat["left_arm_angle"] < action_data["min_left_angle"]:
        action_data["min_left_angle"] = feat["left_arm_angle"]
    if feat["right_arm_angle"] < action_data["min_right_angle"]:
        action_data["min_right_angle"] = feat["right_arm_angle"]

    # z_diff
    if feat["left_z_diff"] < action_data["min_left_z_diff"]:
        action_data["min_left_z_diff"] = feat["left_z_diff"]
    if feat["right_z_diff"] < action_data["min_right_z_diff"]:
        action_data["min_right_z_diff"] = feat["right_z_diff"]
    if feat["left_z_diff"] > action_data["max_left_z_diff"]:
        action_data["max_left_z_diff"] = feat["left_z_diff"]
    if feat["right_z_diff"] > action_data["max_right_z_diff"]:
        action_data["max_right_z_diff"] = feat["right_z_diff"]
    action_data["left_z_diffs"].append(feat["left_z_diff"])
    action_data["right_z_diffs"].append(feat["right_z_diff"])

    # world X
    if feat["left_wx"] > action_data["max_left_wx"]:
        action_data["max_left_wx"] = feat["left_wx"]
    if feat["left_wx"] < action_data["min_left_wx"]:
        action_data["min_left_wx"] = feat["left_wx"]
    if feat["right_wx"] > action_data["max_right_wx"]:
        action_data["max_right_wx"] = feat["right_wx"]
    if feat["right_wx"] < action_data["min_right_wx"]:
        action_data["min_right_wx"] = feat["right_wx"]

    # raise 历史
    action_data["raise_history"].append(
        (fc, feat["left_raise"], feat["right_raise"]))

    # 腕轨迹（最多保留 30 帧）—— 非主动手冻结轨迹
    active_arm = action_data.get("active_arm", "both")

    # --- 左臂轨迹 ---
    if active_arm in ("left", "both"):
        left_horiz = math.copysign(math.sqrt(feat["left_fwd"]**2 + feat["left_lat"]**2),
                                   feat["left_lat"])
        action_data["left_wrist_trail"].append(
            (feat["left_raise"], left_horiz))
    else:
        # 非主动手：复制上一帧，轨迹保持平直
        if action_data["left_wrist_trail"]:
            action_data["left_wrist_trail"].append(
                action_data["left_wrist_trail"][-1])
        else:
            action_data["left_wrist_trail"].append((0.0, 0.0))

    # --- 右臂轨迹 ---
    if active_arm in ("right", "both"):
        right_horiz = math.copysign(math.sqrt(feat["right_fwd"]**2 + feat["right_lat"]**2),
                                    feat["right_lat"])
        action_data["right_wrist_trail"].append(
            (feat["right_raise"], right_horiz))
    else:
        # 非主动手：复制上一帧，轨迹保持平直
        if action_data["right_wrist_trail"]:
            action_data["right_wrist_trail"].append(
                action_data["right_wrist_trail"][-1])
        else:
            action_data["right_wrist_trail"].append((0.0, 0.0))

    if len(action_data["left_wrist_trail"]) > 30:
        action_data["left_wrist_trail"].pop(0)
        action_data["right_wrist_trail"].pop(0)

    # 掌朝向
    action_data["left_palm_orientations"].append(left_palm_ori)
    action_data["right_palm_orientations"].append(right_palm_ori)

    # 臂方位
    action_data["left_orientations"].append(feat["left_orient"])
    action_data["right_orientations"].append(feat["right_orient"])

    # 手腕区域
    action_data["left_regions"].append(feat["left_region"])
    action_data["right_regions"].append(feat["right_region"])

    # 手指可用性
    if feat.get("left_hand_y") is not None:
        action_data["left_finger_available"] = True
    if feat.get("right_hand_y") is not None:
        action_data["right_finger_available"] = True

    # 原始方向
    action_data["left_dir_raws"].append(feat["left_dir_raw"])
    action_data["right_dir_raws"].append(feat["right_dir_raw"])

    # 投影分量
    action_data["left_fwds"].append(feat["left_fwd"])
    action_data["left_lats"].append(feat["left_lat"])
    action_data["right_fwds"].append(feat["right_fwd"])
    action_data["right_lats"].append(feat["right_lat"])

    # 人物朝向
    action_data["body_forwards"].append(feat["body_forward"])

    # 详细臂姿态
    action_data["left_poses"].append(feat["left_pose"])
    action_data["right_poses"].append(feat["right_pose"])

    # 摆动累积
    pls = action_data["prev_left_stretch"]
    plr = action_data["prev_left_raise"]
    prs = action_data["prev_right_stretch"]
    prr = action_data["prev_right_raise"]
    if pls is not None:
        action_data["sum_left_dx"] += feat["left_stretch"] - pls
        action_data["sum_left_dy"] += feat["left_raise"] - plr
    if prs is not None:
        action_data["sum_right_dx"] += feat["right_stretch"] - prs
        action_data["sum_right_dy"] += feat["right_raise"] - prr
    action_data["prev_left_stretch"]  = feat["left_stretch"]
    action_data["prev_left_raise"]    = feat["left_raise"]
    action_data["prev_right_stretch"] = feat["right_stretch"]
    action_data["prev_right_raise"]   = feat["right_raise"]


# ============================================================
# 8 手势分类决策树
# ============================================================

def classify_action(action_data: dict, verbose: bool = True) -> tuple[str, float]:
    """
    基于双手区域的严格判定，不依赖摆动检测/手掌朝向/稳定度。

    判定核心：
      - 手部区域（head / shoulder / waist / hip）
      - 必须在该区域有 >=1 帧停留（而非瞬间经过）
      - "始终" 条件：必须全程在该区域

    Args:
        action_data: 动作期间的累积特征数据

    Returns:
        (gesture_name, confidence)
    """
    left_regions: list = action_data.get("left_regions", [])
    right_regions: list = action_data.get("right_regions", [])

    # ---- 提取区域特征 ----
    L_Region = mode_list(left_regions)         # 众数区域
    R_Region = mode_list(right_regions)

    # 是否"始终"在 hip（一帧都没离开过）
    L_always_hip = all(r == "hip" for r in left_regions) if left_regions else False
    R_always_hip = all(r == "hip" for r in right_regions) if right_regions else False

    # 是否"多数"在 hip（≥70%帧，容忍短暂离开，用于变道等放宽判定）
    L_mostly_hip = (left_regions.count("hip") >= max(1, len(left_regions) * 0.7)
                    if left_regions else False)

    # >=N 帧停留的快捷判断
    def stayed(regions, target, n=1):
        return regions.count(target) >= n

    fc = action_data.get("frame_count", 0)

    # 左臂侧展检测（基于横向投影 lat，不依赖高度）
    # 用于区分右转弯（左臂侧展） vs 变道（左臂无动作/仅下垂）
    left_poses = action_data.get("left_poses", [])
    left_side_count = sum(1 for p in left_poses if p.get("is_side", False))

    # 调试
    if verbose:
        region_str = lambda lst: ",".join(r[0].upper() for r in lst) if lst else "?"
        print(f"  [classify] frames={fc}  L={L_Region}({region_str(left_regions)})  R={R_Region}({region_str(right_regions)})")
        print(f"  [classify] L_always_hip={L_always_hip}  L_mostly_hip={L_mostly_hip}  R_always_hip={R_always_hip}")
        print(f"  [classify] body_max_disp={action_data.get('body_max_disp', 0):.3f}m")
        print(f"  [classify] left_side_count={left_side_count}/{fc}")

    # ================================================================
    # 0. 人体移动检测（行走/转身 → 无动作）
    #    动作期间髋部中心位移超过阈值，说明人在移动，手势不可靠
    # ================================================================
    if action_data.get("body_max_disp", 0.0) > config.BODY_MOVE_THRESHOLD:
        if verbose:
            print(f"  ⚠ 人体移动中 (位移={action_data['body_max_disp']:.3f}m > {config.BODY_MOVE_THRESHOLD}m) → 无动作")
        return ("无动作", 0.0)

    # ================================================================
    # 1. 停止信号
    #    左手在头部(≥1帧)  +  右手在胯部(始终)
    #    ★ 右手必须是hip，排除右手在头部的镜像误判
    # ================================================================
    if (L_Region == "head" and stayed(left_regions, "head") and R_always_hip):
        if verbose:
            print("  ✅ 命中 → 停止信号")
        return ("停止信号", 0.85)

    # ================================================================
    # 2. 直行信号
    #    一手在肩部(≥1帧) + 另一手在肩部或胯部
    #    侧身时一只手臂指向画面外，摄像头只看到 hip
    #    关键：肩部手不能访问过腰部（否则是转弯待转）
    #    双侧排查兼容镜像
    # ================================================================
    straight_left  = (L_Region == "shoulder" and stayed(left_regions, "shoulder")
                      and not stayed(left_regions, "waist")
                      and R_Region in ("shoulder", "hip"))
    straight_right = (R_Region == "shoulder" and stayed(right_regions, "shoulder")
                      and not stayed(right_regions, "waist")
                      and L_Region in ("shoulder", "hip"))
    if straight_left or straight_right:
        if verbose:
            print("  ✅ 命中 → 直行信号")
        return ("直行信号", 0.85)

    # ================================================================
    # 3. 左转弯信号
    #    L=腰部(≥1帧)  +  R=肩部(≥1帧, 前伸)
    # ================================================================
    if (L_Region == "waist" and stayed(left_regions, "waist")
            and R_Region == "shoulder" and stayed(right_regions, "shoulder")):
        if verbose:
            print("  ✅ 命中 → 左转弯信号")
        return ("左转弯信号", 0.85)

    # ================================================================
    # 4. 左转弯待转信号
    #    L=腰部或肩部(≥1帧waist)  +  R=胯部(始终)
    #    左手在 shoulder/waist 边界抖动时宽容处理
    # ================================================================
    if (L_Region in ("waist", "shoulder") and stayed(left_regions, "waist")
            and R_always_hip):
        if verbose:
            print("  ✅ 命中 → 左转弯待转信号")
        return ("左转弯待转信号", 0.85)

    # ================================================================
    # 5. 右转弯信号
    #    L=肩部(≥1帧)  +  R=腰部或肩部(≥1帧waist)
    #    ★★★ 左臂必须有侧展 (is_side, 基于横向投影 lat, 不依赖高度)
    #         变道左臂无动作 → left_side_count=0 → 不会被误判为右转弯 ★★★
    #    右手在 shoulder/waist 边界抖动时宽容处理
    # ================================================================
    if (L_Region == "shoulder" and stayed(left_regions, "shoulder")
            and R_Region in ("waist", "shoulder") and stayed(right_regions, "waist")
            and left_side_count >= 1):
        if verbose:
            print("  ✅ 命中 → 右转弯信号")
        return ("右转弯信号", 0.85)

    # ================================================================
    # 6. 变道信号 / 7. 减速慢行信号
    #    L=hip多数(≥70%帧, 容忍偶尔离开) + R=waist/shoulder(≥1帧waist)
    #    优先输出变道；如需区分可后续加入手臂方位判断
    # ================================================================
    if (L_mostly_hip
            and R_Region in ("waist", "shoulder") and stayed(right_regions, "waist")):
        if verbose:
            print("  ✅ 命中 → 变道信号/减速慢行 (区域条件一致)")
        return ("变道信号", 0.80)

    # ================================================================
    # 8. 示意车辆靠边停车信号
    #    L=头部(≥1帧)  +  R=腰部(≥1帧)
    # ================================================================
    if (L_Region == "head" and stayed(left_regions, "head")
            and stayed(right_regions, "waist")):
        if verbose:
            print("  ✅ 命中 → 示意车辆靠边停车信号")
        return ("示意车辆靠边停车信号", 0.85)

    # --- 无匹配 ---
    if verbose:
        print("  ⚠ 未命中任何手势")
    return ("其他手势", 0.0)


# ============================================================
# 状态机类
# ============================================================

class GestureStateMachine:
    """
    交警手势识别状态机（基于滑动窗口的动作开始/停止检测 + 全身移动过滤）。

    三态模型：
      MOVING  — 整个人在移动（下半身骨架关键点位移超阈值）→ "无动作"
      STOPPED — 站定且双手在hip（窗口内>50%帧）        → "动作停止"
      ACTIVE  — 站定且至少一只手离开hip                 → "动作中"

    动作区间 = STOPPED → ACTIVE → STOPPED。
    区间内每 N 帧做中间判定，结束后取众数 → 稳定去噪结果。

    Usage:
        sm = GestureStateMachine()
        for each frame:
            result = sm.update(feat, left_palm_ori, right_palm_ori, frame_num)
            if result:
                print(result)
            print(sm.display_text)  # 当前状态文字: "无动作" / "动作停止" / "动作中"
    """

    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: 是否输出调试信息
        """
        self._verbose = verbose

        # ---- 三态 ----
        # "stopped" / "active" / "moving"
        self._logical_state = "stopped"

        # ---- 下半身骨架关键点历史（用于全身移动检测） ----
        self._lower_body_history = []  # [ {kps}, {kps}, ... ]

        # ---- hip 连续帧计数器（仅用于动作停止判定） ----
        self._hip_consecutive = 0  # 连续双手在hip的帧数

        # ---- 动作区间追踪 ----
        self._action_data = None       # 当前动作累积数据
        self._last_result = None       # 最近一次识别结果
        self._last_confidence = 0.0
        self._display_text = "动作停止"  # UI 显示文本

        # ---- 结果持久化 ----
        self._result_display_timer = 0

    # ---- 属性 ----

    @property
    def state(self) -> int:
        """兼容旧接口：STATE_ACTIVE / STATE_IDLE。"""
        return config.STATE_ACTIVE if self._logical_state == "active" else config.STATE_IDLE

    @property
    def state_name(self) -> str:
        """人类可读的状态名。"""
        return self._logical_state.upper()

    @property
    def action_data(self) -> dict | None:
        return self._action_data

    @property
    def last_result(self) -> str | None:
        return self._last_result

    @property
    def last_confidence(self) -> float:
        return self._last_confidence

    @property
    def display_text(self) -> str:
        """当前 UI 显示文本。"""
        if self._logical_state == "moving":
            return "无动作"
        elif self._logical_state == "stopped":
            return "动作停止"
        elif self._logical_state == "active":
            return "动作中"
        return "—"

    # ---- 核心方法 ----

    def update(self,
               feat: dict,
               left_palm_ori: str,
               right_palm_ori: str,
               global_frame: int = 0,
               shoulder_width: float = 0.35) -> tuple[str, float] | None:
        """
        处理一帧特征，维护状态机。

        返回:
            (手势名, 置信度)  →  动作区间结束时
            None              →  状态仍在等待或动作进行中
        """
        left_region_cur  = feat["left_region"]
        right_region_cur = feat["right_region"]
        both_at_hip = (left_region_cur == "hip" and right_region_cur == "hip")

        # ---- 1) 更新下半身骨架历史 ----
        self._update_lower_body_history(feat)

        # ---- 2) 更新 hip 连续帧计数器 ----
        if both_at_hip:
            self._hip_consecutive += 1
        else:
            self._hip_consecutive = 0

        # ---- 3) 判定是否为全身移动 ----
        is_body_moving = self._check_whole_body_moving(shoulder_width)

        # ---- 4) 状态决策 ----
        new_state = self._logical_state

        if is_body_moving:
            new_state = "moving"
        elif self._logical_state == "stopped":
            # 停止 → 动作中：只要有一只手离开hip，单帧立即触发
            if not both_at_hip:
                new_state = "active"
        elif self._logical_state == "active":
            # 动作中 → 停止：双手必须连续在hip达到阈值（短暂回落不中断动作）
            if self._hip_consecutive >= config.HIP_CONSECUTIVE_STOP:
                new_state = "stopped"
        elif self._logical_state == "moving":
            # 不再移动 → 回到停止
            new_state = "stopped"

        # ---- 6) 状态迁移处理 ----
        result = None

        if self._logical_state == "stopped" and new_state == "active":
            # ★ 动作开始：创建 action_data
            self._hip_consecutive = 0  # 新动作开始，重置停止计数器
            if self._verbose:
                print(f"\n▶ [帧 {global_frame}] 动作开始！"
                      f"  L={left_region_cur}  R={right_region_cur}")
            self._action_data = reset_action_data(
                shoulder_width, feat, left_palm_ori, right_palm_ori
            )
            self._action_data["_verbose"] = self._verbose

        elif self._logical_state == "active" and new_state == "stopped" and self._action_data is not None:
            # ★ 动作结束：分类判定
            ad = self._action_data
            fc = ad["frame_count"]

            if self._verbose:
                print(f"\n◀ [帧 {global_frame}] 动作结束！"
                      f"  frames={fc}  hip_consecutive={self._hip_consecutive}")

            if fc < config.MIN_ACTION_FRAMES:
                if self._verbose:
                    print(f"   ⚠ 误触过滤（仅 {fc} 帧）\n")
                self._last_result = "误触过滤"
                self._last_confidence = 0.0
            else:
                # 稳定识别：中间判定取众数（去噪）
                stable_results = ad.get("stable_results", [])
                if stable_results:
                    gesture = mode_list(stable_results)
                    conf = 0.85
                    if self._verbose:
                        print(f"   稳定判断[众数 {len(stable_results)}次]: {gesture}\n")
                else:
                    gesture, conf = classify_action(ad, verbose=self._verbose)

                self._last_result = gesture
                self._last_confidence = conf
                result = (gesture, conf)

            self._action_data = None

        elif self._logical_state == "active" and new_state == "moving" and self._action_data is not None:
            # 动作中被全身移动打断
            if self._verbose:
                print(f"\n⚠ [帧 {global_frame}] 动作中断（人体移动）")
            self._last_result = "无动作"
            self._last_confidence = 0.0
            self._action_data = None
            result = ("无动作", 0.0)

        elif self._logical_state == "active" and new_state == "active" and self._action_data is not None:
            # 动作持续中：追加特征 + 中间判定
            _update_action_data(self._action_data, feat, left_palm_ori, right_palm_ori)
            ad = self._action_data
            fc = ad["frame_count"]

            if self._verbose:
                print(f"  [debug] fc={fc} L={feat['left_region']} R={feat['right_region']}"
                      f"  hip_cons={self._hip_consecutive}  moving={is_body_moving}",
                      end="\r")

            # 中间稳定判定（每 N 帧）
            if fc > config.MIN_ACTION_FRAMES and fc % config.STABLE_CLASSIFY_INTERVAL == 0:
                mid_result, _ = classify_action(ad, verbose=False)
                if mid_result not in ("其他手势", "无动作"):
                    ad.setdefault("stable_results", []).append(mid_result)
                if self._verbose:
                    sc = len(ad.get("stable_results", []))
                    print(f"\n  [mid #{sc}] fc={fc} → {mid_result}", end="")

            # 超时强制结束
            if fc >= config.MAX_FRAMES:
                if self._verbose:
                    print(f"\n⏰ [帧 {global_frame}] 超时强制结束")
                ad2 = self._action_data
                stable_results = ad2.get("stable_results", [])
                if stable_results:
                    gesture = mode_list(stable_results)
                    conf = 0.85
                else:
                    gesture, conf = classify_action(ad2, verbose=self._verbose)
                self._last_result = gesture
                self._last_confidence = conf
                self._action_data = None
                new_state = "stopped"
                result = (gesture, conf)

        # ---- 7) 提交状态变更 ----
        self._logical_state = new_state
        self._display_text = self.display_text  # 触发 property 刷新

        return result

    def cancel_action(self, global_frame: int = 0):
        """强制取消当前动作（人体丢失时调用）。"""
        if self._action_data is not None:
            if self._verbose:
                print(f"⚠️ [帧 {global_frame}] 动作中断（未检测到人体）")
            self._logical_state = "stopped"
            self._action_data = None
            self._lower_body_history.clear()
            self._hip_consecutive = 0

    def reset(self):
        """重置状态机到初始状态。"""
        self._logical_state = "stopped"
        self._action_data = None
        self._last_result = None
        self._last_confidence = 0.0
        self._display_text = "动作停止"
        self._lower_body_history.clear()
        self._hip_consecutive = 0

    # ================================================================
    # 内部工具方法
    # ================================================================

    def _update_lower_body_history(self, feat: dict):
        """更新下半身关键点历史窗口。"""
        entry = {
            "body_cx": feat.get("body_cx", 0.0),
            "body_cz": feat.get("body_cz", 0.0),
            "left_hip_xz":  feat.get("left_hip_xz", (0.0, 0.0)),
            "right_hip_xz": feat.get("right_hip_xz", (0.0, 0.0)),
            "left_knee_xz":  feat.get("left_knee_xz", (0.0, 0.0)),
            "right_knee_xz": feat.get("right_knee_xz", (0.0, 0.0)),
            "left_ankle_xz":  feat.get("left_ankle_xz", (0.0, 0.0)),
            "right_ankle_xz": feat.get("right_ankle_xz", (0.0, 0.0)),
        }
        self._lower_body_history.append(entry)
        if len(self._lower_body_history) > config.BODY_WINDOW_SIZE:
            self._lower_body_history.pop(0)

    def _check_whole_body_moving(self, shoulder_width: float) -> bool:
        """
        使用下半身骨架关键点检测全身移动。

        不只依赖单一距离，而是综合检查髋、膝、踝多个关键点
        在滑动窗口内的位移。如果下肢多个关键点都在移动 →
        说明整个人在走动（而非站着做手势）。

        Args:
            shoulder_width: 肩宽（米），用于自适应阈值

        Returns:
            True  → 全身在移动
            False → 站立不动
        """
        if len(self._lower_body_history) < config.BODY_WINDOW_SIZE:
            return False

        first = self._lower_body_history[0]
        last  = self._lower_body_history[-1]

        # 计算每个下肢关键点从窗口首帧到末帧的位移
        def disp_2d(p1, p2):
            return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

        displacements = []

        # 髋部中心
        displacements.append(
            math.sqrt((last["body_cx"] - first["body_cx"])**2
                    + (last["body_cz"] - first["body_cz"])**2)
        )

        # 左/右髋
        displacements.append(disp_2d(first["left_hip_xz"],  last["left_hip_xz"]))
        displacements.append(disp_2d(first["right_hip_xz"], last["right_hip_xz"]))

        # 左/右膝
        displacements.append(disp_2d(first["left_knee_xz"],  last["left_knee_xz"]))
        displacements.append(disp_2d(first["right_knee_xz"], last["right_knee_xz"]))

        # 左/右脚踝
        displacements.append(disp_2d(first["left_ankle_xz"],  last["left_ankle_xz"]))
        displacements.append(disp_2d(first["right_ankle_xz"], last["right_ankle_xz"]))

        # 阈值：基于肩宽自适应
        threshold = max(shoulder_width * 0.8, config.LOWER_BODY_DISP_THRESHOLD)

        # 如果有 3 个以上关键点位移超过阈值 → 全身移动
        moving_count = sum(1 for d in displacements if d > threshold)

        return moving_count >= 3
