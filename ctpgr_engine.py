"""
CTPGREngine: 封装旧系统姿态估计 + 手势 RNN 模型，用于实时视频流推理。
纯内存推理，不落盘 .pkl 文件。
"""

import numpy as np
import torch

from pred.human_keypoint_pred import HumanKeypointPredict
from pred.gesture_pred import GesturePred
from constants.enum_keys import PG
from pgdataset.s3_handcraft import BoneLengthAngle


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

    def __init__(self):
        # 姿态估计模型（关键点检测）
        self.pose_predictor = HumanKeypointPredict()
        # 手势识别模型（RNN）
        self.gesture_predictor = GesturePred()
        # 手工特征提取器（骨长 + 骨夹角）
        self.bla = BoneLengthAngle()

        # 初始化 RNN 隐藏状态
        self.h, self.c = self.gesture_predictor.g_model.h0(), self.gesture_predictor.g_model.c0()

    def reset_state(self):
        """重置 RNN 隐藏状态，用于切换视频时清空时序记忆"""
        self.h = self.gesture_predictor.g_model.h0()
        self.c = self.gesture_predictor.g_model.c0()

    def predict_frame(self, frame_bgr: np.ndarray) -> dict:
        """
        对单帧 BGR 图像进行推理。

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
        # --- a. 姿态估计：获取归一化坐标 ---
        # get_coordinates 返回 {PG.COORD_NATIVE: (2,14), PG.COORD_NORM: (2,14)}
        coord_result = self.pose_predictor.get_coordinates(frame_bgr)
        coord_norm = coord_result[PG.COORD_NORM]  # shape: (2, 14)

        # --- b. 手工特征提取 ---
        # BoneLengthAngle.handcrafted_features 期望输入 (F, X, J)，即 (frames, 2, 14)
        coord_input = coord_norm[np.newaxis, :, :]  # (2, 14) → (1, 2, 14)
        feature_dict = self.bla.handcrafted_features(coord_input)
        # bone_len: (1, B), angle_cos: (1, P), angle_sin: (1, P)

        bone_length = feature_dict[PG.BONE_LENGTH]
        angle_cos = feature_dict[PG.BONE_ANGLE_COS]
        angle_sin = feature_dict[PG.BONE_ANGLE_SIN]

        # --- c. 拼接特征 → RNN 输入 tensor ---
        features = np.concatenate((bone_length, angle_cos, angle_sin), axis=1)  # (1, C)
        features = features[np.newaxis, :, :]  # (1, 1, C)
        features = features.transpose((1, 0, 2))  # (1, 1, C)，符合 RNN (seq, batch, input) 格式
        features_tensor = torch.from_numpy(features).to(
            self.gesture_predictor.g_model.device, dtype=torch.float32
        )

        # --- d. RNN 推理 ---
        with torch.no_grad():
            _, h_new, c_new, class_out = self.gesture_predictor.g_model(
                features_tensor, self.h, self.c
            )

        # 更新隐藏状态（保持时序记忆）
        self.h, self.c = h_new, c_new

        # class_out 形状: (1, num_classes=9)，取 softmax 得到概率
        logits = class_out[0].cpu().numpy()  # (9,)
        probs = np.exp(logits) / np.sum(np.exp(logits))  # softmax
        gesture_id = int(np.argmax(probs))
        confidence = float(probs[gesture_id])

        # --- e. 映射手势名称 ---
        gesture_name = self.GESTURE_MAP.get(gesture_id, f"未知({gesture_id})")

        # --- f. 构建关键点列表 ---
        # coord_norm shape: (2, 14)，第一行为 x，第二行为 y
        keypoints = [
            {"x": float(coord_norm[0, i]), "y": float(coord_norm[1, i])}
            for i in range(coord_norm.shape[1])
        ]

        return {
            "gesture": gesture_name,
            "confidence": confidence,
            "keypoints": keypoints,
        }
