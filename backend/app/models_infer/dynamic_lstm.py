"""
动态手势 BiLSTM 模型定义 + 推理分类器。

模型: 2 层双向 LSTM
  - input:  (T, 63)  归一化后的逐帧 21 关键点 xyz
  - hidden: 128 (双向 → 256)
  - output: num_classes (softmax)

推理类 DynamicLSTMClassifier:
  - 加载训练好的 .pt 模型
  - 支持滑动窗口实时打分 + 边界分段最终判定
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# 模型定义
# ----------------------------------------------------------------

class BiLSTMGesture(nn.Module):
    """双向 LSTM 动态手势分类器。"""

    def __init__(
        self,
        input_size: int = 63,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, T, 63) 或单条 (T, 63)
            lengths: (batch,) 每条序列的真实长度，用于 pack_padded_sequence

        Returns:
            logits: (batch, num_classes)
        """
        single = x.dim() == 2
        if single:
            x = x.unsqueeze(0)  # (1, T, 63)

        if lengths is not None:
            # 按长度降序排列
            lengths_sorted, sort_idx = lengths.sort(descending=True)
            x_sorted = x[sort_idx]
            packed = nn.utils.rnn.pack_padded_sequence(
                x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
            )
            lstm_out, (hidden, _) = self.lstm(packed)
            # 取最后时刻的双向隐藏状态拼接
            # hidden shape: (num_layers*2, batch, hidden_size)
            # 取最后一层: 正向 hidden[-2], 反向 hidden[-1]
            h_forward = hidden[-2, :, :]   # (batch, hidden_size)
            h_backward = hidden[-1, :, :]  # (batch, hidden_size)
            # 恢复原始顺序
            _, unsort_idx = sort_idx.sort()
            h_forward = h_forward[unsort_idx]
            h_backward = h_backward[unsort_idx]
        else:
            lstm_out, (hidden, _) = self.lstm(x)
            # lstm_out: (batch, T, hidden*2), take last timestep
            h_forward = lstm_out[:, -1, :self.hidden_size]
            h_backward = lstm_out[:, -1, self.hidden_size:]

        combined = torch.cat([h_forward, h_backward], dim=1)  # (batch, 256)
        combined = self.dropout(combined)
        logits = self.classifier(combined)  # (batch, num_classes)

        if single:
            logits = logits.squeeze(0)
        return logits

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """返回 softmax 概率 numpy (num_classes,) 或 (batch, num_classes)。"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            return probs.cpu().numpy()

    def predict(self, x: torch.Tensor) -> tuple[str, float]:
        """对单条序列预测，返回 (label, confidence)。"""
        probs = self.predict_proba(x)
        if probs.ndim == 2:
            probs = probs[0]
        idx = int(np.argmax(probs))
        label = self._label_names[idx] if hasattr(self, "_label_names") else str(idx)
        return label, float(probs[idx])


# ----------------------------------------------------------------
# 推理分类器
# ----------------------------------------------------------------

class DynamicLSTMClassifier:
    """动态手势 LSTM 推理分类器。

    两种工作模式:
      - classify_sequence(sequence): 对一段完整轨迹序列做一次分类（边界分段模式）
      - classify_window(buffer, n_frames): 取最近 n_frames 做滑动窗口打分

    配合:
      - 运动能量门控: 食指尖位移方差 > motion_threshold 才触发动态分支
      - EMA 平滑: 连续窗口的打分做指数移动平均去抖
      - 置信度阈值: 低于 threshold → unknown
    """

    # 每帧特征维度
    FEATURE_DIM: int = 63

    # 模型输入要求的最少帧数
    MIN_SEQUENCE_LENGTH: int = 10

    # 滑动窗口帧数（用于实时预览打分）
    WINDOW_FRAMES: int = 32

    # 运动能量阈值（食指尖 xy 归一化坐标方差）
    MOTION_THRESHOLD: float = 0.0004

    # 置信度阈值（低于此值判 unknown）
    # 临时放宽用于调试过拟合模型，后续样本增多后可调回 0.6
    CONFIDENCE_THRESHOLD: float = 0.4

    # EMA 平滑系数
    EMA_ALPHA: float = 0.3

    def __init__(self):
        self._model: BiLSTMGesture | None = None
        self._label_names: list[str] = []
        self._device: torch.device = torch.device("cpu")

        # 滑动窗口 EMA 状态
        self._ema_probs: np.ndarray | None = None
        self._last_prediction: tuple[str, float] = ("unknown", 0.0)

        # 运动缓冲: 最近 N 帧的食指尖 (x,y) 用于计算运动能量
        self._motion_buffer: list[tuple[float, float]] = []
        self._motion_buffer_max: int = 20

        # 轨迹缓冲区：累积归一化后的帧特征向量
        self._trajectory_buffer: list[np.ndarray] = []
        self._trajectory_max: int = 100  # 最多缓存帧数

        self._load_model()

    # ----------------------------------------------------------------
    # 模型加载
    # ----------------------------------------------------------------

    def _load_model(self) -> None:
        """加载训练好的 PyTorch 模型。"""
        try:
            from app.core.config import settings
            model_path = os.path.join(settings.models_dir, "gesture_dynamic_lstm.pt")
            if not os.path.exists(model_path):
                logger.info("动态 LSTM 模型不存在: %s，将使用启发式规则", model_path)
                return

            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            self._label_names = checkpoint.get("label_names", [])
            num_classes = len(self._label_names)
            if num_classes == 0:
                logger.warning("动态 LSTM 模型 label_names 为空")
                return

            self._model = BiLSTMGesture(
                input_size=self.FEATURE_DIM,
                hidden_size=checkpoint.get("hidden_size", 128),
                num_layers=checkpoint.get("num_layers", 2),
                num_classes=num_classes,
                dropout=checkpoint.get("dropout", 0.3),
            )
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.eval()
            self._model._label_names = self._label_names

            # 尝试使用 GPU
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            self._model.to(self._device)

            logger.info(
                "已加载动态 LSTM 模型，类别: %s，设备: %s",
                self._label_names, self._device
            )
        except Exception as e:
            logger.warning("加载动态 LSTM 模型失败: %s", e)
            self._model = None
            self._label_names = []

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and len(self._label_names) > 0

    # ----------------------------------------------------------------
    # 特征预处理
    # ----------------------------------------------------------------

    @staticmethod
    def _normalize_frame(keypoints: list[dict]) -> np.ndarray:
        """单帧归一化，返回 (63,) float32。与静态特征提取保持一致。"""
        from app.models_infer.hand_utils import normalize_hand_landmarks_array
        return normalize_hand_landmarks_array(keypoints)

    # ----------------------------------------------------------------
    # 运动能量门控
    # ----------------------------------------------------------------

    def update_motion(self, keypoints: list[dict]) -> float:
        """添加一帧食指尖位置，返回当前运动能量（方差）。

        Returns:
            motion_energy: 归一化坐标的方差，< MOTION_THRESHOLD 表示静止
        """
        if len(keypoints) != 21:
            return 0.0

        tip_x, tip_y = keypoints[8]["x"], keypoints[8]["y"]
        self._motion_buffer.append((tip_x, tip_y))
        if len(self._motion_buffer) > self._motion_buffer_max:
            self._motion_buffer = self._motion_buffer[-self._motion_buffer_max:]

        if len(self._motion_buffer) < 5:
            return 0.0

        xs = [p[0] for p in self._motion_buffer]
        ys = [p[1] for p in self._motion_buffer]
        var_x = float(np.var(xs))
        var_y = float(np.var(ys))
        return var_x + var_y

    def is_moving(self) -> bool:
        """当前是否处于运动中。"""
        return self.update_motion_direct() > self.MOTION_THRESHOLD

    def update_motion_direct(self) -> float:
        """直接计算当前运动缓冲区的方差。"""
        if len(self._motion_buffer) < 5:
            return 0.0
        xs = [p[0] for p in self._motion_buffer]
        ys = [p[1] for p in self._motion_buffer]
        return float(np.var(xs)) + float(np.var(ys))

    # ----------------------------------------------------------------
    # 轨迹缓冲区管理
    # ----------------------------------------------------------------

    def add_frame(self, keypoints: list[dict]) -> None:
        """向轨迹缓冲添加归一化后的帧特征。"""
        feat = self._normalize_frame(keypoints)
        self._trajectory_buffer.append(feat)
        if len(self._trajectory_buffer) > self._trajectory_max:
            self._trajectory_buffer = self._trajectory_buffer[-self._trajectory_max:]

    def get_trajectory(self) -> np.ndarray:
        """获取当前轨迹缓冲 (T, 63)。"""
        if not self._trajectory_buffer:
            return np.empty((0, self.FEATURE_DIM), dtype=np.float32)
        return np.stack(self._trajectory_buffer, axis=0)

    def trajectory_length(self) -> int:
        return len(self._trajectory_buffer)

    def reset_trajectory(self) -> None:
        self._trajectory_buffer.clear()
        self._motion_buffer.clear()
        self._ema_probs = None
        self._last_prediction = ("unknown", 0.0)

    # ----------------------------------------------------------------
    # 边界分段 — 最终判定（手消失/静止后触发）
    # ----------------------------------------------------------------

    def classify_sequence(self, sequence: np.ndarray) -> tuple[str, float]:
        """
        对一段完整轨迹做一次分类——边界分段模式的最终判定。

        Args:
            sequence: (T, 63) 轨迹特征序列

        Returns:
            (gesture_name, confidence)
        """
        if not self.is_loaded:
            return "unknown", 0.0
        if sequence.shape[0] < self.MIN_SEQUENCE_LENGTH:
            return "unknown", 0.0

        tensor = torch.from_numpy(sequence).float().to(self._device)  # (T, 63)
        probs = self._model.predict_proba(tensor)  # (num_classes,)

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        if confidence < self.CONFIDENCE_THRESHOLD:
            return "unknown", confidence
        return self._label_names[idx], confidence

    # ----------------------------------------------------------------
    # 滑动窗口打分 — 实时预览模式（取最近 K 帧）
    # ----------------------------------------------------------------

    def classify_window(self) -> tuple[str, float]:
        """
        取轨迹缓冲区中最近 WINDOW_FRAMES 帧做滑动窗口打分。
        结合 EMA 平滑输出稳定结果。

        Returns:
            (gesture_name, confidence)
        """
        if not self.is_loaded:
            return "unknown", 0.0

        buffer = self.get_trajectory()
        if buffer.shape[0] < self.MIN_SEQUENCE_LENGTH:
            return "unknown", 0.0

        # 取最近窗口
        window = buffer[-self.WINDOW_FRAMES:] if buffer.shape[0] > self.WINDOW_FRAMES else buffer
        tensor = torch.from_numpy(window).float().to(self._device)

        probs = self._model.predict_proba(tensor)  # (num_classes,)

        # EMA 平滑
        if self._ema_probs is None:
            self._ema_probs = probs
        else:
            self._ema_probs = (
                self.EMA_ALPHA * probs + (1.0 - self.EMA_ALPHA) * self._ema_probs
            )

        idx = int(np.argmax(self._ema_probs))
        confidence = float(self._ema_probs[idx])
        if confidence < self.CONFIDENCE_THRESHOLD:
            self._last_prediction = ("unknown", confidence)
        else:
            self._last_prediction = (self._label_names[idx], confidence)

        return self._last_prediction

    # ----------------------------------------------------------------
    # 统一入口：边界分段 + 滑动窗口结合
    # ----------------------------------------------------------------

    def classify(
        self,
        keypoints: list[dict] | None,
        is_boundary: bool = False,
    ) -> tuple[str, float]:
        """
        统一动态手势分类入口。

        Args:
            keypoints: 当前帧 21 关键点，None 表示无手
            is_boundary: 是否触发边界分段判定（手消失 / 静止时调用）

        Returns:
            (gesture_name, confidence)
        """
        # 无手 / 空关键点
        if keypoints is None or len(keypoints) != 21:
            return "unknown", 0.0

        # 运动能量检查
        motion = self.update_motion(keypoints)

        # 添加帧到缓冲区
        self.add_frame(keypoints)

        # 边界模式：对完整轨迹做最终判定
        if is_boundary:
            trajectory = self.get_trajectory()
            if trajectory.shape[0] >= self.MIN_SEQUENCE_LENGTH:
                result = self.classify_sequence(trajectory)
            else:
                result = ("unknown", 0.0)
            # 边界判定后重置（不清轨迹，留给外部显式 reset）
            return result

        # 非边界模式：运动量不够 → 静默
        if motion < self.MOTION_THRESHOLD:
            return "unknown", 0.0

        # 滑动窗口打分
        return self.classify_window()
