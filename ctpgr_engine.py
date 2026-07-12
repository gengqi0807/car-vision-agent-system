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
        # 步骤 3: EMA 对数平滑（消除单帧预测噪声）
        # ============================================================
        raw_logits_np = class_out[0].cpu().numpy()  # shape (9,)

        if self.logit_ema is None:
            self.logit_ema = raw_logits_np.copy()
        else:
            # 旋转帧使用极低 α → EMA 几乎不变，输出被锁定
            alpha = self.ema_alpha_rot if is_rotating else self.ema_alpha
            self.logit_ema = (
                alpha * raw_logits_np + (1.0 - alpha) * self.logit_ema
            )

        # ============================================================
        # 步骤 4: 解码 + 旋转门控（区分"真实转身"与"手势侧身"）
        # ============================================================
        result = self._decode_output(self.logit_ema.copy(), coord_norm)

        if is_rotating:
            # ---- 4a: 从当前帧原始 logits 检测旋转型手势 ----
            # 左转弯/左待转/右转弯 本身涉及侧身 → 肩宽必然缩小。
            # 不能简单因为肩宽变小就压制 —— 要看"手是不是真的在做动作"。
            raw_logits_s = raw_logits_np - np.max(raw_logits_np)
            raw_probs = np.exp(raw_logits_s) / np.sum(np.exp(raw_logits_s))
            raw_id = int(np.argmax(raw_probs))
            raw_conf = float(raw_probs[raw_id])
            raw_gesture = self.GESTURE_MAP.get(raw_id, "")

            # 这些手势天然伴随身体转向，旋转期间应当允许通过
            ROT_NATURAL = {"左转弯", "左待转", "右转弯"}

            if raw_gesture in ROT_NATURAL and raw_conf > 0.50:
                # 当前帧强信号 → 旋转手势正在进行，覆盖 EMA 输出
                result["gesture"] = raw_gesture
                result["confidence"] = raw_conf
            elif result["confidence"] < 0.25:
                # 既非旋转手势，EMA 也低置信 → 纯粹转身（无动作），回退
                result["gesture"] = "无手势"
                ema_logits = self.logit_ema.copy()
                ema_s = ema_logits - np.max(ema_logits)
                ema_probs = np.exp(ema_s) / np.sum(np.exp(ema_s))
                result["confidence"] = float(ema_probs[0])

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
    # 内部方法：logits → 手势分类
    # ------------------------------------------------------------------

    def _decode_output(self, logits: np.ndarray, coord_norm: np.ndarray) -> dict:
        """将 RNN 输出的 logits 解码为手势名称、置信度和关键点列表。"""
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
        # 如果最高非0类的信心不足（<0.70），说明 LSTM 没有明确识别到任何手势，
        # 强制回退到"无手势"，避免站立/过渡动作被误判。
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
    # 几何校验：利用关键点空间位置纠正易混淆手势
    # ------------------------------------------------------------------
    # AIC 索引: 0=右肩 1=右肘 2=右腕  3=左肩 4=左肘 5=左腕
    #           6=右髋 7=右膝 8=右踝  9=左髋 10=左膝 11=左踝
    #           12=头顶 13=颈部
    #
    # 交警手势真值：
    #   左转弯 (3) = 双臂动作：右臂前伸 + 左臂左前方划动
    #   左待转 (4) = 单臂动作：右臂不动 + 左臂身体侧面划动
    #   直行   (2) = 双臂动作：两臂侧展摆动，基本在肩水平面
    #   右转弯 (5) = 双臂动作：左臂前伸 + 右臂右前方划动

    def _geometric_verify(
        self,
        gesture_id: int,
        confidence: float,
        probs: np.ndarray,
        coord_norm: np.ndarray,
    ) -> tuple:
        """基于关键点空间位置纠正模型容易混淆的类对。"""
        # ---- 提取关键点 ----
        rs = coord_norm[:, 0]   # 右肩
        re = coord_norm[:, 1]   # 右肘
        rw = coord_norm[:, 2]   # 右腕
        ls = coord_norm[:, 3]   # 左肩
        le = coord_norm[:, 4]   # 左肘
        lw = coord_norm[:, 5]   # 左腕
        neck = coord_norm[:, 13]
        mid_hip = np.array([
            (coord_norm[0, 6] + coord_norm[0, 9]) / 2.0,
            (coord_norm[1, 6] + coord_norm[1, 9]) / 2.0,
        ])

        # ---- 通用特征 ----
        # 单臂/双臂判断：手腕远离髋部 = 手臂抬起了
        torso_len = np.linalg.norm(neck - mid_hip) + 1e-8
        left_active  = np.linalg.norm(lw - mid_hip) / torso_len > 0.55
        right_active = np.linalg.norm(rw - mid_hip) / torso_len > 0.55
        both_active  = left_active and right_active

        # 腕 vs 肩 水平偏移（>0 = 腕在肩外侧/侧展；<0 = 腕在肩内侧/前伸）
        lw_outside  = float(ls[0] - lw[0])   # >0 左腕在左肩左侧（侧展）
        rw_outside  = float(rw[0] - rs[0])   # >0 右腕在右肩右侧（侧展）
        lw_inside   = float(lw[0] - ls[0])   # >0 左腕在左肩右侧（前伸向内）
        rw_inside   = float(rs[0] - rw[0])   # >0 右腕在右肩左侧（前伸向内）

        # 腕与肩同高（y 增大 = 向下，>0 表示腕高于肩）
        lw_above_ls = float(ls[1] - lw[1])
        rw_above_rs = float(rs[1] - rw[1])
        at_shoulder  = abs(lw_above_ls) < 0.035 and abs(rw_above_rs) < 0.035

        # ============================================================
        #  左转弯 (3) vs 左待转 (4)
        # ============================================================
        #   左转弯 = 双臂：右臂前伸 + 左臂左前方划动
        #   左待转 = 单臂：右臂贴近身体不动 + 左臂身体侧面划动
        #
        #   关键差异：右腕是否贴近身体（用腕-髋距离/躯干长来衡量）
        if gesture_id in (3, 4):
            rw_to_hip   = np.linalg.norm(rw - mid_hip) / torso_len
            rw_near_body = rw_to_hip < 0.45         # 右腕贴近身体 = 手臂休息位

            # 侧身检测：肩膀 x 间距过小 → 身体侧转，右臂可能被遮挡
            #   MP 骨架会把右腕投射到左腕附近，rw_near_body 失效
            shoulder_width = abs(float(ls[0] - rs[0]))
            sideways_body = shoulder_width < 0.10 or (shoulder_width / torso_len < 0.25)

            left_turn_pat = (
                both_active                         # 双臂都抬起
                and rw_inside > 0.005               # 右腕在肩内侧（前伸）
                and not rw_near_body                # 右腕远离身体
            )
            # 正脸左待转：右臂贴在身上不动 + 左臂侧前方划动
            left_wait_pat = (
                left_active                         # 左臂抬起
                and rw_near_body                    # ★ 右腕贴近身体
                and lw_outside > -0.005             # 左腕不越过身体中线（放宽）
            )
            # 侧身左待转：右臂被遮挡，MP 可能误判右臂随左臂前伸
            #   → 只要求左臂抬起，右腕位置极不可靠，大幅放宽
            left_wait_sideways_pat = (
                sideways_body                       # 身体侧转中
                and left_active                     # 左臂抬起
                and lw_outside > -0.015             # 左腕不跑到身体右侧
                and not (rw_inside > 0.030)         # 右腕没越过左腕位置（极宽）
                and rw_to_hip < 0.90                # 右腕别飞太远即可
            )

            if gesture_id == 3 and (left_wait_pat or left_wait_sideways_pat) and not left_turn_pat:
                # 模型说左转弯，但几何特征指向左待转 → 改左待转
                gesture_id = 4
                confidence = float(probs[4])
            elif gesture_id == 4 and left_turn_pat and not (left_wait_pat or left_wait_sideways_pat):
                # 模型说左待转，但右臂明显前伸 → 改左转弯
                gesture_id = 3
                confidence = float(probs[3])
            elif gesture_id in (3, 4) and not (left_turn_pat or left_wait_pat or left_wait_sideways_pat):
                gesture_id = 0
                confidence = float(probs[0])

        # ============================================================
        #  右转弯 (5) vs 直行 (2) vs 停止 (1)
        # ============================================================
        #   直行   = 两臂大幅侧展，腕远离身体、几乎在肩外侧最远点
        #   右转弯 = 左臂前伸（腕在身体前方），右臂在身体右前方划动
        #   停止   = 单臂（通常左臂）高举过头，另一臂自然下垂
        #
        #   用评分制：各自算匹配度，分高且差距 >2 才触发纠正
        if gesture_id in (1, 2, 5):
            # ---- 停止得分（单臂高举 + 另一臂下垂） ----
            # 当前帧左臂高举过头且右臂未激活 → 极可能是停止
            stop_score = 0
            if left_active and lw_above_ls > 0.06 and not right_active:
                stop_score += 6    # 左臂高举 + 右臂下垂（强证据）
            elif right_active and rw_above_rs > 0.06 and not left_active:
                stop_score += 6    # 右臂高举 + 左臂下垂（强证据）
            elif left_active and lw_above_ls > 0.04 and not right_active:
                stop_score += 4
            elif right_active and rw_above_rs > 0.04 and not left_active:
                stop_score += 4

            # 右臂高举且手掌向前 = 交警标准停止（额外加分）
            lw_above_head = float(coord_norm[1, 12] - lw[1])  # 左腕高于头顶
            rw_above_head = float(coord_norm[1, 12] - rw[1])  # 右腕高于头顶
            if left_active and lw_above_head > 0.02 and not right_active:
                stop_score += 2
            elif right_active and rw_above_head > 0.02 and not left_active:
                stop_score += 2

            # ---- 直行得分（两腕侧展程度） ----
            s_score = 0
            if both_active:           s_score += 1
            if lw_outside > 0.010:    s_score += 3    # 左腕侧展（关键）
            if rw_outside > 0.010:    s_score += 3    # 右腕侧展（关键）
            if at_shoulder:           s_score += 1

            # ---- 右转弯得分（左腕前伸 + 右腕右前方） ----
            r_score = 0
            if both_active:           r_score += 1
            if lw_outside < 0.010:    r_score += 4    # 左腕不侧展 = 前伸（核心）
            if rw_outside > 0.000:    r_score += 2    # 右腕在身体右侧
            if rw_inside > -0.010:    r_score += 1    # 右腕不在身体左侧

            # ---- 比较得分，差距 >2 才纠正（避免歧义帧反复跳动） ----
            if gesture_id == 5:
                # 右转弯必须是双臂动作；单臂高举更像停止
                if stop_score > r_score and stop_score >= 6:
                    gesture_id = 1
                    confidence = float(probs[1])
                elif s_score > r_score + 2:
                    gesture_id = 2
                    confidence = float(probs[2])
                elif r_score < 3:
                    gesture_id = 0
                    confidence = float(probs[0])
                elif r_score >= 5:
                    # 几何特征强烈确认右转弯 → 提升置信度，防止被 0.70 门槛误杀
                    confidence = max(confidence, 0.71)
            elif gesture_id == 2:
                if stop_score > s_score and stop_score >= 6:
                    gesture_id = 1
                    confidence = float(probs[1])
                elif r_score > s_score + 2:
                    gesture_id = 5
                    confidence = float(probs[5])
                elif s_score >= 5:
                    confidence = max(confidence, 0.71)
            elif gesture_id == 1:
                # 模型说停止，但双臂都在做动作 → 更可能是其他手势
                if both_active and stop_score < 4:
                    if r_score > s_score:
                        gesture_id = 5
                        confidence = float(probs[5])
                    elif s_score > r_score:
                        gesture_id = 2
                        confidence = float(probs[2])

        return gesture_id, confidence
