"""
CTPGREngine: 封装旧系统姿态估计 + 手势 RNN 模型，用于实时视频流推理。
纯内存推理，不落盘 .pkl 文件。

提供两种推理路径：
  1. predict_frame()       — 完整路径：VGG+PAFs 关键点检测 → 特征 → LSTM（慢，保留兼容）
  2. predict_from_keypoints() — 快速路径：直接输入已提取的关键点 → 特征 → LSTM（快，推荐）
"""

import numpy as np
import torch

from pred.human_keypoint_pred import HumanKeypointPredict
from models.gesture_recognition_model import GestureRecognitionModel
from constants.enum_keys import PG
from pgdataset.s3_handcraft import BoneLengthAngle

# ---------------------------------------------------------------------------
# MediaPipe 33 关键点 → AIChallenger 14 关键点 映射
# ---------------------------------------------------------------------------
# AIChallenger 索引（0-based）：
#   0:右肩  1:右肘  2:右腕  3:左肩  4:左肘  5:左腕
#   6:右髋  7:右膝  8:右踝  9:左髋  10:左膝  11:左踝
#   12:头顶  13:颈部
#
# MediaPipe Pose 关键点（索引 0-32）：
#   0:鼻子      1:左眼内角   2:左眼      3:左眼外角
#   4:右眼内角   5:右眼      6:右眼外角   7:左耳
#   8:右耳      9:嘴左角    10:嘴右角   11:左肩
#   12:右肩     13:左肘     14:右肘     15:左腕
#   16:右腕     17:左小指   18:右小指   19:左食指
#   20:右食指   21:左拇指   22:右拇指   23:左髋
#   24:右髋     25:左膝     26:右膝     27:左踝
#   28:右踝     29:左脚跟   30:右足尖   31:左足尖
#   32:右足尖
#
# MediaPipe 肩部关键点位于肩峰（肩膀最外侧），AIChallenger 的肩部标注可能
# 更偏近肩关节中心。因此需要微调肩点位置使其与训练数据分布更一致。
# ---------------------------------------------------------------------------

# MediaPipe → AIC 直接映射表（12 个躯干+四肢关键点）
_MP_TO_AIC = [
    (12, 0),   # 右肩
    (14, 1),   # 右肘
    (16, 2),   # 右腕
    (11, 3),   # 左肩
    (13, 4),   # 左肘
    (15, 5),   # 左腕
    (24, 6),   # 右髋
    (26, 7),   # 右膝
    (28, 8),   # 右踝
    (23, 9),   # 左髋
    (25, 10),  # 左膝
    (27, 11),  # 左踝
]

# 参考值：AIChallenger 训练数据中，躯干长度（颈→髋中点）的典型归一化值
_REF_TORSO_LEN = 0.22


def mediapipe_to_aic14(mp_landmarks) -> np.ndarray:
    """
    将 MediaPipe 33 关键点（归一化坐标 0~1）映射为 AIChallenger 14 关键点。

    改进点（相比初版）：
      1. 颈部：用嘴巴关键点推算下巴位置，再将颈部放在下巴→肩中点之间
         （约 38% 处），保证颈-肩-肘角度与真实解剖结构一致
      2. 头顶：用双眼间距 × 1.8 推算（标准面部比例：眼到头顶≈3.5-4倍眼宽）
      3. 肩部：从肩峰向内微调 6%（MediaPipe 肩在肩峰外侧，AIChallenger 在关节中心）
      4. 尺度归一化：当躯干长度偏离训练分布 >30% 时缩放坐标，使骨长特征跨
         摄像头距离一致

    Args:
        mp_landmarks: MediaPipe PoseLandmark 列表（33 个），每个有 .x .y .z .visibility

    Returns:
        np.ndarray, shape (2, 14), dtype float64
            coord[0, i] = x 归一化坐标
            coord[1, i] = y 归一化坐标
    """
    coord = np.zeros((2, 14), dtype=np.float64)

    # ---- 步骤 1: 提取所有需要的 MediaPipe 关键点 ----
    # 面部
    nose_x, nose_y = mp_landmarks[0].x, mp_landmarks[0].y
    leye_x, leye_y = mp_landmarks[2].x, mp_landmarks[2].y
    reye_x, reye_y = mp_landmarks[5].x, mp_landmarks[5].y
    eye_mid_x = (leye_x + reye_x) / 2.0
    eye_mid_y = (leye_y + reye_y) / 2.0
    eye_dist = np.sqrt((reye_x - leye_x) ** 2 + (reye_y - leye_y) ** 2)

    # 嘴巴（用于推算下巴）
    mouth_l_vis = mp_landmarks[9].visibility
    mouth_r_vis = mp_landmarks[10].visibility

    # 肩部
    ls_x, ls_y = mp_landmarks[11].x, mp_landmarks[11].y
    rs_x, rs_y = mp_landmarks[12].x, mp_landmarks[12].y
    shoulder_mid_x = (ls_x + rs_x) / 2.0
    shoulder_mid_y = (ls_y + rs_y) / 2.0
    shoulder_width = np.sqrt((rs_x - ls_x) ** 2 + (rs_y - ls_y) ** 2)

    # ---- 步骤 2: 估算头顶 (AIC 12) ----
    # 标准面部比例：双眼到头顶 ≈ 1.8× 眼间距（眼球中心到眼球中心）
    # 使用 max(0, ...) 防止极端角度下越界
    if eye_dist > 0.001:
        head_top_y = eye_mid_y - eye_dist * 1.8
    else:
        # 回退：眼睛上方约 0.12（占图像高度比例）
        head_top_y = eye_mid_y - 0.12
    coord[0, 12] = eye_mid_x
    coord[1, 12] = max(0.0, head_top_y)

    # ---- 步骤 3: 估算颈部 (AIC 13) —— 最关键的改进 ----
    # 3a. 推算下巴位置
    if mouth_l_vis > 0.5 or mouth_r_vis > 0.5:
        # 有嘴巴关键点 → 精算：下巴 ≈ 嘴巴下方 (嘴到鼻距离 × 1.1)
        mouth_x = (mp_landmarks[9].x + mp_landmarks[10].x) / 2.0
        mouth_y = (mp_landmarks[9].y + mp_landmarks[10].y) / 2.0
        mouth_to_nose = nose_y - mouth_y  # y轴向下，鼻子在上 → 负值
        chin_x = mouth_x
        chin_y = mouth_y + abs(mouth_to_nose) * 1.1
    else:
        # 无嘴巴 → 从鼻子和眼睛关系推算：下巴 ≈ 鼻子下方 (眼到鼻距离 × 0.7)
        eye_to_nose = nose_y - eye_mid_y  # 鼻子在眼下方 → 正值
        chin_x = nose_x
        chin_y = nose_y + eye_to_nose * 0.7

    # 3b. 颈部 ≈ 下巴和肩中点之间的 38% 处（解剖学上 C7 椎骨位于此区间）
    neck_x = chin_x + (shoulder_mid_x - chin_x) * 0.38
    neck_y = chin_y + (shoulder_mid_y - chin_y) * 0.38
    # 约束：颈部在垂直方向上不应低于肩中点、不应高于下巴
    neck_y = max(chin_y, min(shoulder_mid_y, neck_y))
    neck_x = max(0.0, min(1.0, neck_x))
    coord[0, 13] = neck_x
    coord[1, 13] = neck_y

    # ---- 步骤 4: 12 个直接映射 ----
    # 注意：不再对肩部做内移微调。MediaPipe 肩峰位置 vs AIChallenger 关节中心
    # 虽存在偏差，但任意的偏移量可能扭曲肩-肘-腕夹角，导致左转弯/左待转混淆。
    # 颈部已通过面部关键点精确估算，颈-肩-肘角度已得到改善。
    for mp_idx, aic_idx in _MP_TO_AIC:
        lm = mp_landmarks[mp_idx]
        coord[0, aic_idx] = lm.x
        coord[1, aic_idx] = lm.y

    # ---- 步骤 5: 尺度归一化 ----
    # 当人物距离摄像头很远或很近时，骨长特征会与训练分布严重偏离。
    # 以躯干长度（颈→髋中点）为基准，将坐标缩放到训练集典型尺度。
    mid_hip_x = (coord[0, 6] + coord[0, 9]) / 2.0  # 右髋 + 左髋 中点
    mid_hip_y = (coord[1, 6] + coord[1, 9]) / 2.0
    torso_len = np.sqrt((neck_x - mid_hip_x) ** 2 + (neck_y - mid_hip_y) ** 2)

    if torso_len > 0.005:
        scale = _REF_TORSO_LEN / torso_len
        # 仅当尺度偏差 >30% 时进行归一化（避免过度修正）
        if scale < 0.7 or scale > 1.3:
            scale = max(0.3, min(3.0, scale))  # 安全限幅
            coord *= scale
            coord = np.clip(coord, -0.2, 1.2)  # 允许轻微越界（归一化后）

    return coord


class CTPGREngine:
    # 手势 ID → 中文名称映射（参考 pred/play_gesture_results.py gesture_dict_c）
    GESTURE_MAP = {
        0: "无手势",
        1: "停止",
        2: "直行",
        3: "左转弯",
        4: "左待转",
        5: "右转弯",
        6: "变道",
        7: "减速",
        8: "靠边停车",
    }

    def __init__(self, load_pose_model: bool = True):
        """
        Args:
            load_pose_model: 是否加载 VGG+PAFs 姿态估计模型。
                             设为 False 时仅加载 LSTM + 特征提取，需配合
                             predict_from_keypoints() 使用。
        """
        # ---- 手势识别 LSTM 模型（总是加载） ----
        self.g_model = GestureRecognitionModel(1)
        self.g_model.load_ckpt()
        self.g_model.eval()

        # ---- 手工特征提取器（骨长 + 骨夹角） ----
        self.bla = BoneLengthAngle()

        # ---- LSTM 初始隐藏状态 ----
        self.h, self.c = self.g_model.h0(), self.g_model.c0()

        # ---- VGG+PAFs 姿态估计模型（仅 predict_frame() 需要） ----
        if load_pose_model:
            self.pose_predictor = HumanKeypointPredict()
        else:
            self.pose_predictor = None

        # ============================================================
        # 时序稳定：EMA 平滑 + 身体旋转检测
        # ============================================================

        # EMA 对数平滑（消除单帧噪声，防止持续手势中途跳动）
        self.logit_ema = None           # EMA 平滑后的 logits，shape (9,)
        self.ema_alpha = 0.35           # 正常帧 EMA 系数
        self.ema_alpha_rot = 0.06       # 旋转帧 EMA 系数（极低→锁定输出）

        # 身体旋转检测（转身时 2D 投影肩宽缩小）
        # 交警指挥动作都是面向相机做的，转身会导致肩-肘-腕角度失真
        self.shoulder_max = None         # 历史最大归一化肩宽
        self.shoulder_decay = 0.998      # max 每推理帧衰减因子
        self.rotation_cooldown = 0        # 旋转结束后冷却帧数
        self.rotation_threshold = 0.55    # 肩宽比率 < 此值 ⇒ 判定为旋转中

    def reset_state(self):
        """重置 RNN 隐藏状态 + 所有时序缓冲区，用于切换视频时清空记忆"""
        self.h = self.g_model.h0()
        self.c = self.g_model.c0()
        self.logit_ema = None
        self.shoulder_max = None
        self.rotation_cooldown = 0

    # ------------------------------------------------------------------
    # 快速推理路径（推荐）：直接输入已提取的 AIC 14 关键点
    # ------------------------------------------------------------------

    def predict_from_keypoints(self, coord_norm: np.ndarray) -> dict:
        """
        使用预提取的 AIChallenger 14 关键点进行手势分类（跳过 VGG+PAFs）。

        内置两项时序稳定机制：
          1. EMA 对数平滑：消除单帧噪声，防止同一手势中途跳动到相邻类
          2. 身体旋转门控：检测 2D 投影肩宽缩小 → 转身时抑制误识别

        Args:
            coord_norm: np.ndarray, shape (2, 14), dtype float
                归一化坐标，coord[0, i]=x, coord[1, i]=y，范围 0~1

        Returns:
            dict: 同 predict_frame() 格式
        """
        # ============================================================
        # 步骤 0: 身体旋转检测（肩宽 2D 投影比率）
        # ============================================================
        # 交警动作都是面向相机做的，转身会导致骨长/骨夹角特征剧烈变化，
        # 被 LSTM 误判为目标手势（如左转弯、变道等）。通过监测肩宽压缩比
        # 提前拦截旋转帧，防止隐藏状态被污染。
        sw = np.sqrt(
            (coord_norm[0, 0] - coord_norm[0, 3]) ** 2
            + (coord_norm[1, 0] - coord_norm[1, 3]) ** 2
        )  # 右肩(AIC 0) ↔ 左肩(AIC 3) 的欧氏距离

        # 维护历史最大肩宽（含缓慢衰减，适应人物移动）
        if self.shoulder_max is None or sw > self.shoulder_max:
            self.shoulder_max = sw
        else:
            self.shoulder_max = max(sw, self.shoulder_max * self.shoulder_decay)

        sw_ratio = sw / max(self.shoulder_max, 0.001)
        is_rotating = sw_ratio < self.rotation_threshold

        # 旋转冷却期：旋转结束后再保持若干帧抑制（防止 LSTM 状态残留）
        if is_rotating:
            self.rotation_cooldown = 6
        elif self.rotation_cooldown > 0:
            self.rotation_cooldown -= 1
            is_rotating = True

        # ============================================================
        # 步骤 1: 手工特征提取（骨长 + 骨夹角）
        # ============================================================
        coord_input = coord_norm[np.newaxis, :, :].astype(np.float64)  # (2,14) → (1,2,14)
        feature_dict = self.bla.handcrafted_features(coord_input)

        bone_length = feature_dict[PG.BONE_LENGTH]
        angle_cos = feature_dict[PG.BONE_ANGLE_COS]
        angle_sin = feature_dict[PG.BONE_ANGLE_SIN]

        # ============================================================
        # 步骤 2: 拼接特征 → LSTM 推理
        # ============================================================
        # 注意：即使是旋转帧也照常喂入 LSTM，保持隐藏状态持续更新；
        # 靠输出端的 EMA 极低 α + 旋转门控来屏蔽错误结果。
        features = np.concatenate((bone_length, angle_cos, angle_sin), axis=1)  # (1, C)
        features = features[np.newaxis, :, :]  # (1, 1, C)
        features = features.transpose((1, 0, 2))  # (1, 1, C)
        features_tensor = torch.from_numpy(features).to(
            self.g_model.device, dtype=torch.float32
        )

        with torch.no_grad():
            _, h_new, c_new, class_out = self.g_model(
                features_tensor, self.h, self.c
            )

        self.h, self.c = h_new, c_new

        # ============================================================
        # 步骤 3: 几何纠偏（在 EMA 之前，防止类别反复横跳）
        # ============================================================
        raw_logits_np = class_out[0].cpu().numpy()  # shape (9,)

        # ★ 关键：几何纠偏前置 — 根据当前帧的骨骼空间位置，
        #   直接 boost/suppress raw logits，让 EMA 自然收敛到正确类别。
        #   这解决了旧版"EMA 偏向 A 类但几何说应是 B 类→逐帧反复修正"的问题。
        geo_adjusted = self._geo_boost_logits(raw_logits_np.copy(), coord_norm)

        # ============================================================
        # 步骤 4: EMA 平滑（平滑的是几何纠偏后的 logits）
        # ============================================================
        if self.logit_ema is None:
            self.logit_ema = geo_adjusted.copy()
        else:
            alpha = self.ema_alpha_rot if is_rotating else self.ema_alpha
            self.logit_ema = (
                alpha * geo_adjusted + (1.0 - alpha) * self.logit_ema
            )

        # ============================================================
        # 步骤 5: 解码 EMA + 轻量安全网
        # ============================================================
        result = self._decode_output(self.logit_ema.copy(), coord_norm)

        # ============================================================
        # 步骤 6: 旋转门控（区分"真实转身动作"与"纯转身无手势"）
        # ============================================================
        if is_rotating:
            # 计算手臂活跃度：手腕是否离开了髋部
            mid_hip_raw = np.array([
                (coord_norm[0, 6] + coord_norm[0, 9]) / 2.0,
                (coord_norm[1, 6] + coord_norm[1, 9]) / 2.0,
            ])
            neck_raw = coord_norm[:, 13]
            torso_raw = np.linalg.norm(neck_raw - mid_hip_raw) + 1e-8
            lw_raw = coord_norm[:, 5]
            rw_raw = coord_norm[:, 2]

            # ★ 旋转时 torso_len 缩小会虚增距离比，单靠距离比不可靠
            #   改用"腕肘高度差"判断：自然下垂→腕低于肘；手势→腕高于或平齐于肘
            #   这是旋转不变的——因为肘-腕在2D中始终保持上-下关系
            le_raw = coord_norm[:, 4]   # 左肘 AIC=4
            re_raw = coord_norm[:, 1]   # 右肘 AIC=1
            left_wrist_raised  = lw_raw[1] <= le_raw[1]   # y小=高
            right_wrist_raised = rw_raw[1] <= re_raw[1]

            left_dist  = np.linalg.norm(lw_raw - mid_hip_raw) / torso_raw
            right_dist = np.linalg.norm(rw_raw - mid_hip_raw) / torso_raw
            left_arm_up  = left_dist > 0.60 and left_wrist_raised
            right_arm_up = right_dist > 0.60 and right_wrist_raised
            any_arm_raised = left_arm_up or right_arm_up

            ROT_NATURAL = {"左转弯", "左待转", "右转弯"}
            ema_is_rot_gesture = result["gesture"] in ROT_NATURAL and result["confidence"] > 0.55

            if ema_is_rot_gesture:
                pass  # EMA 已确认旋转手势，保留
            elif any_arm_raised:
                # 手臂真正抬起 → 解码当前帧原始 logits（含几何校验）
                raw_result = self._decode_output(
                    raw_logits_np.copy(), coord_norm, apply_lowconf_filter=False
                )
                if raw_result["gesture"] in ROT_NATURAL and raw_result["confidence"] > 0.50:
                    result = raw_result
            else:
                result["gesture"] = "无手势"
                result["confidence"] = 0.15

        return result

    # ------------------------------------------------------------------
    # 完整推理路径（保留兼容）：VGG+PAFs 关键点 → 特征 → LSTM
    # ------------------------------------------------------------------

    def predict_frame(self, frame_bgr: np.ndarray) -> dict:
        """
        对单帧 BGR 图像进行推理（完整路径，含 VGG+PAFs 关键点检测）。

        Args:
            frame_bgr: np.ndarray, uint8, shape (H, W, 3), BGR 格式

        Returns:
            dict: {
                "gesture": str,        # 中文手势名称
                "confidence": float,   # 置信度 (0~1)
                "keypoints": [         # 归一化关键点列表
                    {"x": float, "y": float},
                    ...
                ],
            }
        """
        if self.pose_predictor is None:
            raise RuntimeError(
                "CTPGREngine 初始化时未加载姿态估计模型 (load_pose_model=False)，"
                "请使用 predict_from_keypoints() 或重新初始化。"
            )

        # --- a. 姿态估计：获取归一化坐标 ---
        coord_result = self.pose_predictor.get_coordinates(frame_bgr)
        coord_norm = coord_result[PG.COORD_NORM]  # shape: (2, 14)

        # --- b~d. 特征提取 + RNN 推理 ---
        return self.predict_from_keypoints(coord_norm)

    # ------------------------------------------------------------------
    # 几何 logits 纠偏（EMA 之前调用，防止类别反复横跳）
    # ------------------------------------------------------------------

    def _geo_boost_logits(self, logits: np.ndarray, coord_norm: np.ndarray) -> np.ndarray:
        """基于关键点空间位置调整 raw logits，boost 几何合理的类、抑制不可能的类。

        关键设计：这是对 logits 的逐帧 boosting，结果送入 EMA 平滑后自然收敛。
        避免了旧版"EMA 偏 A、几何校验强制输出 B、下一帧 EMA 仍偏 A"的横跳问题。
        """
        adjusted = logits.copy()

        # ---- 提取关键点 & 通用特征 ----
        rs = coord_norm[:, 0]; re = coord_norm[:, 1]; rw = coord_norm[:, 2]
        ls = coord_norm[:, 3]; le = coord_norm[:, 4]; lw = coord_norm[:, 5]
        neck = coord_norm[:, 13]
        mid_hip = np.array([
            (coord_norm[0, 6] + coord_norm[0, 9]) / 2.0,
            (coord_norm[1, 6] + coord_norm[1, 9]) / 2.0,
        ])
        torso_len = np.linalg.norm(neck - mid_hip) + 1e-8
        left_active  = np.linalg.norm(lw - mid_hip) / torso_len > 0.55
        right_active = np.linalg.norm(rw - mid_hip) / torso_len > 0.55
        both_active  = left_active and right_active

        lw_outside   = float(ls[0] - lw[0])
        rw_outside   = float(rw[0] - rs[0])
        lw_above_ls  = float(ls[1] - lw[1])
        rw_above_rs  = float(rs[1] - rw[1])
        shoulder_w   = abs(float(ls[0] - rs[0]))
        sideways_body= shoulder_w < 0.12 or (shoulder_w / torso_len < 0.28)

        # ================================================================
        #  直行(2) / 右转弯(5) / 停止(1) 三向评分
        # ================================================================
        def _side_deg(v):  # 侧展程度
            if v > 0.030: return 4
            elif v > 0.015: return 3
            elif v > 0.005: return 1
            return 0

        def _fwd_deg(v):   # 前伸程度
            if v < -0.025: return 5
            elif v < -0.005: return 4
            elif v < 0.010: return 2
            return 0

        def _h_deg(v):     # 高度匹配程度
            if abs(v) < 0.015: return 3
            elif abs(v) < 0.035: return 2
            elif abs(v) < 0.050: return 1
            return 0

        # -- 直行得分 --
        lw_side = _side_deg(lw_outside); rw_side = _side_deg(rw_outside)
        lw_h = _h_deg(lw_above_ls);       rw_h = _h_deg(rw_above_rs)
        s_score = 0
        if both_active:                            s_score += 1
        s_score += lw_side + rw_side
        if lw_side >= 3: s_score += lw_h
        if rw_side >= 3: s_score += rw_h
        if both_active and lw_side >= 3 and rw_side >= 1 and lw_h >= 2 and rw_h >= 1:
            s_score += 2

        # -- 停止得分 --
        # ★ 修复：不再要求另一臂必须完全不活跃（not right_active），
        #   改为"另一臂没有明显抬起"。真实场景中闲置手臂会有自然晃动。
        stop_score = 0
        rw_not_raised = not right_active or rw_above_rs < 0.02
        lw_not_raised = not left_active  or lw_above_ls < 0.02
        if left_active and lw_above_ls > 0.12 and rw_not_raised:
            stop_score += 6
        elif right_active and rw_above_rs > 0.12 and lw_not_raised:
            stop_score += 6
        elif left_active and lw_above_ls > 0.08 and rw_not_raised:
            stop_score += 4
        elif right_active and rw_above_rs > 0.08 and lw_not_raised:
            stop_score += 4
        lw_above_head = float(coord_norm[1, 12] - lw[1])
        rw_above_head = float(coord_norm[1, 12] - rw[1])
        if lw_above_head > 0.02 and rw_not_raised:  stop_score += 2
        if rw_above_head > 0.02 and lw_not_raised:   stop_score += 2

        # -- 右转弯得分 --
        body_mid_x   = (float(ls[0]) + float(rs[0])) / 2.0
        lw_near_mid  = float(lw[0]) - body_mid_x > -0.05
        lw_fwd       = _fwd_deg(lw_outside)
        # ★ 右转弯右手是侧展划动，不是前伸——不纳入 rw_fwd
        r_score = 0
        if both_active:         r_score += 1
        r_score += lw_fwd                     # 只计左手前伸（核心标志）
        if rw_outside > 0.000:  r_score += 1  # 右手侧展
        if lw_near_mid:         r_score += 3

        # ★ 左臂前伸检测（右转弯核心标志）：腕在肩内侧 + 肘在内侧 + 肘伸直
        #   前伸不看高度——和左转弯 r_arm_forward 逻辑一致
        lw_outside_v = float(lw[0] - ls[0])    # 左腕 vs 左肩 x，<0=腕在肩内侧
        le_outside_v = float(le[0] - ls[0])     # 左肘 vs 左肩 x
        le_ls = np.linalg.norm(le - ls); le_lw = np.linalg.norm(le - lw)
        lw_ls = np.linalg.norm(lw - ls)
        le_straight = False
        if (le_ls + le_lw) > 0.02:
            le_straight = (lw_ls / (le_ls + le_lw)) > 0.78
        l_arm_forward = (left_active
                         and lw_outside_v < -0.015
                         and le_outside_v < -0.010
                         and le_straight)
        # ★ l_arm_forward 确认为 True 时直接为 r_score 大幅加分，
        #   确保右转弯在三向竞争中能压制变道。
        if l_arm_forward:
            r_score += 5

        # ★ 变道 vs 右转弯：
        #   右转弯 = 左臂前伸（l_arm_forward）
        #   变道   = 右臂侧展伸直齐肩 + 左臂无大动作
        #   左臂无大动作：不高举、不强烈侧展、不前伸
        re_outside_v = float(re[0] - rs[0])   # 右肘 x 侧展
        is_side_out  = rw_outside > 0.025      # 右腕明显在肩右侧
        is_re_side   = re_outside_v > 0.005    # 右肘也在右侧
        is_re_straight = False
        re_rs = np.linalg.norm(re - rs); re_rw = np.linalg.norm(re - rw)
        rw_rs = np.linalg.norm(rw - rs)
        if (re_rs + re_rw) > 0.02:
            is_re_straight = (rw_rs / (re_rs + re_rw)) > 0.78
        is_shoulder_h = abs(rw_above_rs) < 0.08   # 齐肩高度
        # 左臂无大动作：不高举、不强烈侧展、不前伸
        # ★ 若左臂活跃（远离髋部），腕必须明显低于肩才算"自然下垂"；
        #   否则臂处于前伸状态（右转弯标志），不能当作变道的左臂下垂条件。
        if left_active:
            lw_no_big_move = lw_above_ls < -0.02 and lw_outside < 0.10
        else:
            lw_no_big_move = lw_above_ls < 0.06 and lw_outside < 0.10
        lane_change_flag = (right_active and is_side_out and is_re_side
                            and is_re_straight and is_shoulder_h
                            and not l_arm_forward
                            and lw_no_big_move)
        if lane_change_flag:
            r_score = max(0, r_score - 4)
            adjusted[6] += 1.2   # 倾向变道
            adjusted[5] -= 1.5

        # ★ 右转弯强确认：左臂前伸 + 右手侧展 → 直接倾向右转弯、抑制变道
        #   lane_change_flag 已通过 not l_arm_forward 互斥，两者不会同时为 True。
        right_turn_strong = (l_arm_forward and right_active
                             and rw_outside > 0.000)
        if right_turn_strong:
            adjusted[5] += 2.0   # 倾向右转弯
            adjusted[6] -= 2.0   # 抑制变道

        # ★ 变道前摇：右臂齐肩但未侧展 + 左臂下垂 → 抑制右转弯
        #   变道手势：左臂自然下垂，右臂抬至肩高后左右滑动。
        #   右臂刚抬起尚未侧展时（rw_outside 不大），LSTM 容易误判为右转弯。
        right_raised_no_side = (right_active and not is_side_out
                                and abs(rw_above_rs) < 0.10
                                and not left_active)
        if right_raised_no_side:
            r_score = max(0, r_score - 3)
            adjusted[5] -= 1.2   # 抑制右转弯
            adjusted[6] += 0.5   # 倾向变道

        # ★ 停止信号互斥：单臂明显高举>0.12时，抑制右转弯/直行
        single_arm_high = ((left_active and lw_above_ls > 0.10 and rw_not_raised)
                           or (right_active and rw_above_rs > 0.10 and lw_not_raised))
        if single_arm_high:
            r_score = max(0, r_score - 3)

        # -- 应用 boosts：只在得分最高且领先对手 >2 时才调整 --
        max_125 = max(s_score, stop_score, r_score)
        if max_125 >= 5:
            if s_score == max_125 and s_score >= stop_score + 3 and s_score >= r_score + 3:
                adjusted[2] += 1.8
                adjusted[5] -= 1.5
                adjusted[1] -= 1.0
            elif r_score == max_125 and r_score >= stop_score + 3 and r_score >= s_score + 3:
                adjusted[5] += 1.8
                adjusted[2] -= 1.5
                adjusted[1] -= 1.0
                adjusted[6] -= 1.5   # 右转弯胜出时同步抑制变道
            elif stop_score == max_125 and stop_score >= 6 and stop_score >= s_score + 2 and stop_score >= r_score + 2:
                adjusted[1] += 1.8
                adjusted[2] -= 1.2
                adjusted[5] -= 1.2

        # ================================================================
        #  左转弯(3) vs 左待转(4) vs 面向左侧站立不动
        # ================================================================
        # ---- 肘伸直检测 ----
        re_rs = np.linalg.norm(re - rs)
        re_rw = np.linalg.norm(re - rw)
        rw_rs = np.linalg.norm(rw - rs)
        re_straight = False
        if (re_rs + re_rw) > 0.02:
            re_straight = (rw_rs / (re_rs + re_rw)) > 0.78

        # ★ 核心思想：左转弯手势的本质是右臂向身体前方平伸（不是向上抬）
        #   在2D图像中，"前伸"表现为手腕在肩内侧（x方向）且高度接近肩水平。
        #   左转弯 = 右臂前伸（rw_outside < 0）+ 左臂侧展划动（lw_outside > 0）
        #   左待转 = 左臂侧展划动 + 右臂无前伸/下垂

        # ★ 右臂前伸（左转弯核心标志）：腕在肩内侧且高度接近肩
        #   前伸臂应：腕在肩内侧(x) + 肘也在内侧(x) + 腕不高不低 + 肘伸直
        re_outside = float(re[0] - rs[0])       # 右肘 vs 右肩 x，<0=肘在肩内侧
        rw_outside_val = float(rw[0] - rs[0])   # 右腕 vs 右肩 x，<0=腕在肩内侧
        rw_above_rs_val = rw_above_rs           # 右腕 vs 右肩 y，>0=腕高于肩
        re_above_rs = float(rs[1] - re[1])      # 右肘 vs 右肩 y
        # 右臂前伸：手和肘都明显在肩内侧 + 高度不离谱 + 肘伸直
        r_arm_forward = (right_active
                         and rw_outside_val < -0.015   # 腕在肩内侧
                         and re_outside < -0.010       # 肘也在内侧（防甩手误判）
                         and abs(rw_above_rs_val) < 0.10  # 腕不高不低
                         and re_straight)              # 肘伸直

        # ★ 左臂侧展划动（左待转/左转弯的特征：腕明显在肩外侧 + 主动抬起）
        #   左待转 = 左臂在身体侧边上下摆动，必须是主动抬起的动作，
        #   自然下垂或轻微摆动不算。要求腕在肩外侧且不低于肩太多。
        lw_side_sweep = lw_outside > 0.025 and lw_above_ls > -0.03
        lw_elevated   = lw_above_ls > -0.07

        # ★ 右臂下垂（用于确定左待转）
        r_arm_down = (not right_active) or (rw_above_rs_val < -0.06
                      and not (rw_outside_val < -0.012))

        # ---- 检测"面朝左侧站立不动"：侧身 + 左臂不抬高、不外展 + 右臂下垂 ----
        #   手臂自然下垂/轻微摆动时腕可能略偏外或略高于肩，不应误判为手势。
        lw_not_sweeping = lw_outside < 0.08      # 容忍自然站立时腕的侧向偏移
        lw_not_raised   = lw_above_ls < 0.02     # 容忍轻微摆动时腕略超肩高
        # 右臂下垂：不活跃 或 腕明显低于肩（不再要求腕不在内侧，侧身时视觉误差大）
        r_arm_dropped = (not right_active) or (rw_above_rs_val < -0.06)
        stand_leftwards = (sideways_body and left_active
                           and lw_not_sweeping and lw_not_raised
                           and r_arm_dropped)

        if stand_leftwards:
            adjusted[4] -= 2.5
            adjusted[3] -= 2.5
            adjusted[0] += 2.0   # 强力推无手势
        else:
            # ---- 左转弯(3) vs 左待转(4)：核心看右臂是否前伸 ----
            if r_arm_forward:
                # 右臂前伸 = 左转弯（不管左臂状态，早期左臂未摆也应兜底）
                adjusted[3] += 3.5
                adjusted[4] -= 3.0
            elif left_active and lw_side_sweep and r_arm_down:
                # 右臂下垂 + 左臂侧展划动 → 左待转
                adjusted[4] += 2.0
                adjusted[3] -= 2.0
            elif left_active and lw_not_sweeping and not r_arm_forward:
                # 左臂在动但无侧展、右臂无前伸 → 弱左待转
                adjusted[4] += 0.3
                adjusted[3] -= 0.2
            # 其他情况（如左腕稍外展但不够 lw_side_sweep 阈值、或 r_arm_down=False）
            # 不做 boost，让 LSTM 自己判断

        return adjusted

    # ------------------------------------------------------------------
    # 内部方法：logits → 手势分类
    # ------------------------------------------------------------------

    def _decode_output(self, logits: np.ndarray, coord_norm: np.ndarray,
                        apply_lowconf_filter: bool = True) -> dict:
        """将 RNN 输出的 logits 解码为手势名称、置信度和关键点列表。

        Args:
            logits: 形状 (9,) 的 logits 数组
            coord_norm: 形状 (2, 14) 的归一化坐标
            apply_lowconf_filter: 是否应用 <0.70 低置信度兜底。
                旋转帧解码时设为 False，避免几何校验结果被阈值抹掉。
        """
        raw_logits = logits.copy()  # 保留原始 logits 供诊断

        # ---- class-0 偏差校准 ----
        # 训练数据不平衡导致 linear bias 对 class-0 产生 logit 偏差。
        # 当 class-0 logit 异常偏高时拉回，使其低于最高非0类 0.8 logit —
        # 这样 softmax 后 top 类能拿到 ~69%+ 的置信度，class-0 降至 ~30%。
        non0_max = np.max(logits[1:])  # 最高非0类 logit
        if logits[0] > non0_max + 1.0:
            logits[0] = non0_max - 0.8

        # ---- 温度缩放（T<1 锐化分布，增大类间区分度） ----
        # 0:无手势 1:停止 2:直行 3:左转弯 4:左待转 5:右转弯 6:变道 7:减速 8:靠边停车
        temperature = 0.70
        logits = logits / temperature

        # stable softmax
        logits_s = logits - np.max(logits)
        probs = np.exp(logits_s) / np.sum(np.exp(logits_s))
        gesture_id = int(np.argmax(probs))
        confidence = float(probs[gesture_id])

        # ---- 几何校验：纠正模型容易混淆的类对 ----
        gesture_id, confidence = self._geometric_verify(
            gesture_id, confidence, probs, coord_norm
        )

        # ---- 低置信度兜底 ----
        # 当 apply_lowconf_filter=True 时（正常 EMA 解码路径）：
        #   如果非0类信心不足（<0.70），回退到"无手势"
        # 当 apply_lowconf_filter=False 时（旋转帧原始解码）：
        #   跳过此过滤，让旋转门控自行判断
        if apply_lowconf_filter:
            if gesture_id != 0 and confidence < 0.70:
                gesture_id = 0
                confidence = float(probs[0])

        # 映射手势名称
        gesture_name = self.GESTURE_MAP.get(gesture_id, f"未知({gesture_id})")

        # 构建关键点列表
        keypoints = [
            {"x": float(coord_norm[0, i]), "y": float(coord_norm[1, i])}
            for i in range(coord_norm.shape[1])
        ]

        return {
            "gesture": gesture_name,
            "confidence": confidence,
            "keypoints": keypoints,
            "raw_logits": raw_logits.tolist(),
        }

    # ------------------------------------------------------------------
    # 几何安全网：仅纠正 EMA 输出中明显不合理的情况（轻量、高阈值）
    # ------------------------------------------------------------------
    # 主要纠偏工作已由 _geo_boost_logits 在 EMA 之前完成。
    # 此方法仅处理 EMA 滞后导致的"结果与当前几何明显矛盾"的极端情况。

    def _geometric_verify(
        self,
        gesture_id: int,
        confidence: float,
        probs: np.ndarray,
        coord_norm: np.ndarray,
    ) -> tuple:
        """轻量安全网：仅在几何证据极强时才纠正。"""
        rs = coord_norm[:, 0]; re_v = coord_norm[:, 1]; rw = coord_norm[:, 2]
        ls = coord_norm[:, 3]; lw = coord_norm[:, 5]
        neck = coord_norm[:, 13]
        mid_hip = np.array([
            (coord_norm[0, 6] + coord_norm[0, 9]) / 2.0,
            (coord_norm[1, 6] + coord_norm[1, 9]) / 2.0,
        ])
        torso_len = np.linalg.norm(neck - mid_hip) + 1e-8
        left_active  = np.linalg.norm(lw - mid_hip) / torso_len > 0.55
        right_active = np.linalg.norm(rw - mid_hip) / torso_len > 0.55
        both_active  = left_active and right_active

        lw_outside  = float(ls[0] - lw[0])
        lw_above_ls = float(ls[1] - lw[1])
        rw_above_rs = float(rs[1] - rw[1])

        # ---- 停止(1) → 直行(2)：双臂侧展不可能同时是停止 ----
        if gesture_id == 1 and both_active:
            gesture_id = 2
            confidence = max(float(probs[2]), 0.72)
            return gesture_id, confidence

        # ---- 左转弯(3) ↔ 左待转(4)：基于右臂前伸 ----
        if gesture_id in (3, 4):
            rw_outside_v = float(rw[0] - rs[0])
            re_outside_v = float(re_v[0] - rs[0])
            re_above_rs_v = float(rs[1] - re_v[1])
            rw_above_rs_v = rw_above_rs

            # 右臂前伸检测（同 _geo_boost_logits）
            re_rs_v = np.linalg.norm(re_v - rs); re_rw_v = np.linalg.norm(re_v - rw)
            rw_rs_v = np.linalg.norm(rw - rs)
            re_str = (rw_rs_v / (re_rs_v + re_rw_v)) > 0.78 if (re_rs_v + re_rw_v) > 0.02 else False

            r_arm_forward = (right_active
                             and rw_outside_v < -0.015
                             and re_outside_v < -0.010
                             and abs(rw_above_rs_v) < 0.10
                             and re_str)

            lw_side = lw_outside > -0.015
            lw_not_sweeping = lw_outside < 0.05
            lw_not_raised   = lw_above_ls < 0.0
            r_arm_dropped = (not right_active) or (rw_above_rs_v < -0.06)

            # 面朝左侧站立不动 → 无手势
            shoulder_w = abs(float(ls[0] - rs[0]))
            sideways_body = shoulder_w < 0.12 or (shoulder_w / torso_len < 0.28)
            stand_leftwards = (sideways_body and left_active
                               and lw_not_sweeping and lw_not_raised
                               and r_arm_dropped)
            if stand_leftwards:
                gesture_id = 0
                confidence = float(probs[0])
                return gesture_id, confidence

            # 右臂前伸 = 左转弯，EMA 说左待转 → 翻回左转弯
            if gesture_id == 4 and r_arm_forward:
                gesture_id = 3
                confidence = max(float(probs[3]), 0.72)
                return gesture_id, confidence

            if gesture_id == 3 and not both_active and lw_side and not r_arm_forward:
                # 模型说左转弯但只有左臂在动且右臂未前伸 → 左待转
                gesture_id = 4
                confidence = max(float(probs[4]), 0.72)

        # ---- 右转弯(5) ↔ 变道(6)：左臂前伸=右转弯，右臂侧展+左臂无大动作=变道 ----
        if gesture_id == 5:
            rw_outside_v = float(rw[0] - rs[0])
            re_outside_v = float(re_v[0] - rs[0])
            re_rs_v = np.linalg.norm(re_v - rs); re_rw_v = np.linalg.norm(re_v - rw)
            rw_rs_v = np.linalg.norm(rw - rs)
            re_str = (rw_rs_v / (re_rs_v + re_rw_v)) > 0.78 if (re_rs_v + re_rw_v) > 0.02 else False
            rw_above_rs_v = float(rs[1] - rw[1])
            # 左臂前伸检测（右转弯核心标志，不看高度）
            le_v = coord_norm[:, 4]
            lw_outside_v2 = float(lw[0] - ls[0])
            le_outside_v2 = float(le_v[0] - ls[0])
            le_ls_v = np.linalg.norm(le_v - ls); le_lw_v = np.linalg.norm(le_v - lw)
            lw_ls_v = np.linalg.norm(lw - ls)
            le_str = (lw_ls_v / (le_ls_v + le_lw_v)) > 0.78 if (le_ls_v + le_lw_v) > 0.02 else False
            l_arm_forward = (left_active
                             and lw_outside_v2 < -0.015
                             and le_outside_v2 < -0.010
                             and le_str)
            # 左臂无大动作：不高举、不强烈侧展、不前伸
            # ★ 与 _geo_boost_logits 逻辑一致：左臂活跃时必须明显低于肩
            if left_active:
                lw_no_big_move = lw_above_ls < -0.02 and lw_outside < 0.10
            else:
                lw_no_big_move = lw_above_ls < 0.06 and lw_outside < 0.10
            lane_change_flag = (right_active and rw_outside_v > 0.025
                                and re_outside_v > 0.005
                                and re_str
                                and abs(rw_above_rs_v) < 0.08
                                and not l_arm_forward
                                and lw_no_big_move)
            if lane_change_flag:
                gesture_id = 6
                confidence = max(float(probs[6]), 0.72)
                return gesture_id, confidence

        # ---- 变道(6) → 右转弯(5)：左臂前伸=右转弯，EMA若误判变道需纠正 ----
        if gesture_id == 6:
            le_v = coord_norm[:, 4]
            lw_outside_v2 = float(lw[0] - ls[0])
            le_outside_v2 = float(le_v[0] - ls[0])
            le_ls_v = np.linalg.norm(le_v - ls); le_lw_v = np.linalg.norm(le_v - lw)
            lw_ls_v = np.linalg.norm(lw - ls)
            le_str = (lw_ls_v / (le_ls_v + le_lw_v)) > 0.78 if (le_ls_v + le_lw_v) > 0.02 else False
            l_arm_forward = (left_active
                             and lw_outside_v2 < -0.015
                             and le_outside_v2 < -0.010
                             and le_str)
            if l_arm_forward:
                gesture_id = 5
                confidence = max(float(probs[5]), 0.72)
                return gesture_id, confidence

        # ---- 停止(1) ↔ 右转弯(5)：单臂高举不可能是右转弯 ----
        if gesture_id == 5:
            lw_not_raised = not left_active or lw_above_ls < 0.02
            rw_not_raised = not right_active or rw_above_rs < 0.02
            if left_active and lw_above_ls > 0.10 and rw_not_raised:
                gesture_id = 1
                confidence = max(float(probs[1]), 0.72)
            elif right_active and rw_above_rs > 0.10 and lw_not_raised:
                gesture_id = 1
                confidence = max(float(probs[1]), 0.72)

        return gesture_id, confidence
