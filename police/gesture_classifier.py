"""
gesture_classifier.py — 状态机 + 8 种交警手势分类决策树

核心组件：
  - GestureStateMachine : 状态机（IDLE → ACTIVE → IDLE）
  - reset_action_data   : 创建动作数据容器
  - classify_action     : 8 种手势的严格判定
  - mode_str / mode_list : 众数工具

状态机规则：
  - 进入 ACTIVE : 至少一只手离开胯部，且至少一只手在肩/头/腰区域，
                  连续 START_CONFIRM 帧满足
  - 在 ACTIVE 中: 持续记录特征；主动手回落胯部或超时则结束
  - 结束后冷却  : COOLDOWN_FRAMES 帧内不触发新动作
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

    # ---- 主动手回落检测 ----
    active_arm = action_data.get("active_arm", "both")
    if active_arm == "left":
        if left_region_cur == "hip" and right_region_cur != "head":
            action_data["hip_hold_count"] += 1
        else:
            action_data["hip_hold_count"] = 0
    elif active_arm == "right":
        if right_region_cur == "hip" and left_region_cur != "head":
            action_data["hip_hold_count"] += 1
        else:
            action_data["hip_hold_count"] = 0
    else:  # both
        if left_region_cur == "hip" and right_region_cur == "hip":
            action_data["hip_hold_count"] += 1
        else:
            action_data["hip_hold_count"] = 0

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

    # >=N 帧停留的快捷判断
    def stayed(regions, target, n=1):
        return regions.count(target) >= n

    fc = action_data.get("frame_count", 0)

    # 调试
    if verbose:
        region_str = lambda lst: ",".join(r[0].upper() for r in lst) if lst else "?"
        print(f"  [classify] frames={fc}  L={L_Region}({region_str(left_regions)})  R={R_Region}({region_str(right_regions)})")
        print(f"  [classify] L_always_hip={L_always_hip}  R_always_hip={R_always_hip}")

    # ================================================================
    # 1. 停止信号
    #    一手在头部(≥1帧)  +  另一手在胯部(始终)
    #    双侧排查以兼容摄像头镜像（交警左手=画面右手）
    # ================================================================
    if ((L_Region == "head" and stayed(left_regions, "head") and R_always_hip)
            or (R_Region == "head" and stayed(right_regions, "head") and L_always_hip)):
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
    #    L=肩部(≥1帧, 前伸)  +  R=腰部或肩部(≥1帧waist)
    #    右手在 shoulder/waist 边界抖动时宽容处理
    # ================================================================
    if (L_Region == "shoulder" and stayed(left_regions, "shoulder")
            and R_Region in ("waist", "shoulder") and stayed(right_regions, "waist")):
        if verbose:
            print("  ✅ 命中 → 右转弯信号")
        return ("右转弯信号", 0.85)

    # ================================================================
    # 6. 变道信号 / 7. 减速慢行信号
    #    两者按当前区域规则条件一致（L=hip始终 + R=waist≥1帧）,
    #    优先输出变道；如需区分可后续加入手臂方位判断
    # ================================================================
    if (L_always_hip
            and R_Region == "waist" and stayed(right_regions, "waist")):
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
    交警手势识别状态机。

    状态流转：
      IDLE ──(连续触发)──▶ ACTIVE ──(回落/超时)──▶ IDLE ──(冷却)──▶ IDLE

    在 ACTIVE 状态中累计动作数据，结束后调用 classify_action 判定手势。
    外部主循环每帧调用 update()，当手势完成时返回判定结果。

    Usage:
        sm = GestureStateMachine()
        for each frame:
            result = sm.update(feat, left_palm_ori, right_palm_ori, frame_num)
            if result:
                print(result)
    """

    def __init__(self, verbose: bool = True):
        """初始化状态机（IDLE 状态）。

        Args:
            verbose: 是否在终端输出调试信息（默认开启）
        """
        self._verbose = verbose
        self._state = config.STATE_IDLE
        self._action_data = None
        self._last_result = None
        self._last_confidence = 0.0
        self._idle_trigger_count = 0
        self._cooldown_counter = 0

    # ---- 属性（供外部 UI 读取） ----

    @property
    def state(self) -> int:
        """当前状态：STATE_IDLE 或 STATE_ACTIVE。"""
        return self._state

    @property
    def state_name(self) -> str:
        """状态的人类可读描述。"""
        if self._cooldown_counter > 0:
            return "COOLDOWN"
        return "ACTIVE" if self._state == config.STATE_ACTIVE else "IDLE"

    @property
    def action_data(self) -> dict | None:
        """当前动作的累积数据（ACTIVE 期间有效）。"""
        return self._action_data

    @property
    def last_result(self) -> str | None:
        """最近一次识别的手势名称，初始为 None。"""
        return self._last_result

    @property
    def last_confidence(self) -> float:
        """最近一次识别的置信度。"""
        return self._last_confidence

    @property
    def cooldown_counter(self) -> int:
        """冷却剩余帧数。"""
        return self._cooldown_counter

    @property
    def trigger_count(self) -> int:
        """IDLE 下触发计数器（用于 UI 显示进度）。"""
        return self._idle_trigger_count

    @property
    def trigger_target(self) -> int:
        """触发目标帧数。"""
        return config.START_CONFIRM

    # ---- 核心逻辑 ----

    def update(self,
               feat: dict,
               left_palm_ori: str,
               right_palm_ori: str,
               global_frame: int = 0,
               shoulder_width: float = 0.35) -> tuple[str, float] | None:
        """
        处理一帧特征，维护状态机。

        Args:
            feat:           extract_features 返回的特征字典
            left_palm_ori:  左手掌朝向标签
            right_palm_ori: 右手掌朝向标签
            global_frame:   全局帧号（仅用于日志）
            shoulder_width: 肩宽（用于自适应阈值）

        Returns:
            (gesture_name, confidence) 当手势判定完成时；
            None 当状态机仍在等待或动作进行中。
        """
        left_region_cur  = feat["left_region"]
        right_region_cur = feat["right_region"]

        # 当前帧的触发条件
        any_leave_hip = (left_region_cur != "hip" or right_region_cur != "hip")
        at_least_one_raised = (
            left_region_cur in ("shoulder", "head", "waist")
            or right_region_cur in ("shoulder", "head", "waist")
        )

        # ---- 冷却期倒计时 ----
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1

        # ================================================================
        # IDLE 状态
        # ================================================================
        if self._state == config.STATE_IDLE:
            if self._cooldown_counter > 0:
                self._idle_trigger_count = 0
            elif any_leave_hip and at_least_one_raised:
                self._idle_trigger_count += 1
            else:
                self._idle_trigger_count = 0

            if self._idle_trigger_count >= config.START_CONFIRM:
                self._idle_trigger_count = 0
                self._state = config.STATE_ACTIVE

                # ---- 创建并初始化 action_data ----
                self._action_data = reset_action_data(
                    shoulder_width, feat, left_palm_ori, right_palm_ori
                )
                self._action_data["_verbose"] = self._verbose

                if self._verbose:
                    print(f"\n▶ [帧 {global_frame}] 动作开始！"
                          f"  L_Raise={feat['left_raise']:.2f} R_Raise={feat['right_raise']:.2f}"
                          f"  L_Angle={feat['left_arm_angle']:.0f}° R_Angle={feat['right_arm_angle']:.0f}°"
                          f"  L_Region={left_region_cur} R_Region={right_region_cur}"
                          f"  L_Z={feat['left_z_diff']:.2f} R_Z={feat['right_z_diff']:.2f}")

                return None  # 动作刚开始，尚无判定结果

            return None

        # ================================================================
        # ACTIVE 状态
        # ================================================================
        if self._state == config.STATE_ACTIVE:
            # 追加特征到 action_data（reset_action_data 已处理入口帧）
            _update_action_data(self._action_data, feat,
                                left_palm_ori, right_palm_ori)

            ad = self._action_data
            fc = ad["frame_count"]

            # 调试：打印 hip_hold_count
            if self._verbose:
                print(f"  [debug] hip_hold_count={ad.get('hip_hold_count', 0)}/{config.HOLD_FRAMES}"
                      f"  active_arm={ad.get('active_arm', '?')}"
                      f"  L={feat['left_region']}  R={feat['right_region']}",
                      end="\r")

            # ---- 退出判定 ----
            end_reason = None
            if ad.get("hip_hold_count", 0) >= config.HOLD_FRAMES and fc >= config.MIN_ACTION_FRAMES:
                end_reason = "⬇ 双手持续回落胯部结束"
            elif fc > config.MAX_FRAMES:
                end_reason = "⏰ 超时强制结束"

            if end_reason:
                if self._verbose:
                    print(f"\n◀ [帧 {global_frame}] 动作结束！（{end_reason}）"
                          f"  frames={fc}"
                          f"  hip_hold_count={ad.get('hip_hold_count', 0)}"
                          f"  L_Z_min={ad['min_left_z_diff']:.2f}"
                          f"  R_Z_min={ad['min_right_z_diff']:.2f}")

                if fc < config.MIN_ACTION_FRAMES:
                    if self._verbose:
                        print(f"   ⚠ 误触过滤（仅 {fc} 帧，需 >= {config.MIN_ACTION_FRAMES}）\n")
                    result, confidence = ("误触过滤", 0.0)
                else:
                    result, confidence = classify_action(ad, verbose=self._verbose)
                    if self._verbose:
                        print(f"   判定结果: {result}  (置信度: {confidence:.0%})\n")

                self._last_result = result
                self._last_confidence = confidence
                self._state = config.STATE_IDLE
                self._action_data = None
                self._cooldown_counter = config.COOLDOWN_FRAMES
                self._idle_trigger_count = 0

                return (result, confidence)

            return None

        return None

    def cancel_action(self, global_frame: int = 0):
        """
        强制取消当前动作（如人体丢失时调用）。

        Args:
            global_frame: 全局帧号（仅用于日志）
        """
        if self._state == config.STATE_ACTIVE:
            if self._verbose:
                print(f"⚠️ [帧 {global_frame}] 动作中断（未检测到人体）")
            self._state = config.STATE_IDLE
            self._action_data = None
            self._cooldown_counter = config.COOLDOWN_FRAMES
            self._idle_trigger_count = 0

    def reset(self):
        """重置状态机到初始状态。"""
        self._state = config.STATE_IDLE
        self._action_data = None
        self._last_result = None
        self._last_confidence = 0.0
        self._idle_trigger_count = 0
        self._cooldown_counter = 0
