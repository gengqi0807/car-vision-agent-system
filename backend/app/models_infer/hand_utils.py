"""
手部关键点归一化工具 — 训练与推理共用。
特征顺序: [x0..x20, y0..y20, z0..z20]
"""

import numpy as np


def normalize_hand_landmarks(kp: list[dict]) -> list[float]:
    """
    以手腕(点0)为原点、手腕→中指MCP(点9)距离为尺度，做平移+缩放归一化。
    返回 63 维 float 列表。
    """
    wx, wy, wz = kp[0]["x"], kp[0]["y"], kp[0]["z"]
    mx, my, mz = kp[9]["x"], kp[9]["y"], kp[9]["z"]

    scale = float(np.sqrt((mx - wx) ** 2 + (my - wy) ** 2 + (mz - wz) ** 2))
    if scale < 1e-6:
        scale = 1.0

    flat = []
    for axis in ("x", "y", "z"):
        origin = kp[0][axis]
        for i in range(21):
            flat.append((kp[i][axis] - origin) / scale)
    return flat


def normalize_hand_landmarks_array(kp: list[dict]) -> np.ndarray:
    """同 normalize_hand_landmarks，但返回 numpy 数组 (63,) float32。"""
    return np.array(normalize_hand_landmarks(kp), dtype=np.float32)


# ----------------------------------------------------------------
# 模型包装器（joblib 序列化需要模块级别定义）
# ----------------------------------------------------------------
class SingleClassWrapper:
    """包装 OneClassSVM，对外暴露 predict / predict_proba 接口以兼容 SVC。"""

    def __init__(self, ocsvm, label_names, scaler):
        self.ocsvm = ocsvm
        self.classes_ = np.arange(len(label_names))
        self.label_names = label_names
        self.scaler = scaler
        self.n_classes_ = len(label_names)

    def predict(self, X_in):
        raw = self.ocsvm.predict(X_in)
        return np.where(raw == 1, 0, 0)

    def predict_proba(self, X_in):
        raw = self.ocsvm.decision_function(X_in)
        prob = 1.0 / (1.0 + np.exp(-raw))
        prob = np.clip(prob, 0.0, 1.0)
        return np.column_stack([1.0 - prob, prob])[:, :1]
