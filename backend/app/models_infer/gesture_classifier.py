"""
手势分类器 — 静态手势（基于 21 关键点手指伸展度） + 动态手势状态机。

静态手势:
  - palm      : 五指全伸展 → 唤醒
  - fist      : 五指全弯曲 → 确认
  - thumb_up  : 仅拇指伸展且朝上 → 接听
  - thumb_down: 仅拇指伸展且朝下 → 挂断
  - pointing  : 仅食指伸展 → 用于触发动态追踪

动态手势（通过 HandGestureTracker 时序状态机）:
  - circle_cw : 食指尖顺时针画圈 → 音量+
  - circle_ccw: 食指尖逆时针画圈 → 音量-
  - swipe_left : 手腕向左滑动 → 上一个功能
  - swipe_right: 手腕向右滑动 → 下一个功能
  - wave       : 手腕往复摆动 → 返回主页

接口预留:
  - GestureClassifier(domain="police") → 交警手势复用同一分类器框架
"""

import math
import os
import logging

import numpy as np

logger = logging.getLogger(__name__)


class GestureClassifier:
    """手势分类器，支持 domain="owner" / "police" 两种模式。

    owner 模式下优先使用训练的 SVM 模型，模型不存在时回退到启发式规则。
    """

    # ----------------------------------------------------------------
    # 时序保护常量（classify_frame 使用）
    # ----------------------------------------------------------------
    DYNAMIC_COOLDOWN_FRAMES: int = 18  # 动态触发后抑制静态的帧数 (~0.6s @30fps)
    NO_HAND_REARM_FRAMES: int = 3      # 手消失多少帧算「手势边界」，重新武装
    STATIC_STABLE_FRAMES: int = 3      # 静态手势需连续稳定几帧才输出

    def __init__(self, domain: str = "owner"):
        self.domain = domain
        self.tracker: HandGestureTracker | None = (
            HandGestureTracker() if domain == "owner" else None
        )
        self._ml_model = None
        self._ml_scaler = None
        self._ml_labels = None

        # 时序保护状态
        self._cooldown: int = 0
        self._no_hand: int = 0
        self._stable_gesture: str | None = None
        self._stable_count: int = 0

        # 尝试加载训练好的 SVM 模型
        if domain == "owner":
            self._load_ml_model()

    # ----------------------------------------------------------------
    # ML 模型加载与推理
    # ----------------------------------------------------------------

    def _load_ml_model(self) -> None:
        """尝试从 models/ 目录加载训练好的 SVM 模型。"""
        try:
            from app.core.config import settings
            model_path = os.path.join(settings.models_dir, "gesture_classifier_svm.joblib")
            if not os.path.exists(model_path):
                logger.info("SVM 模型文件不存在，将使用启发式规则: %s", model_path)
                return

            import joblib
            bundle = joblib.load(model_path)
            self._ml_model = bundle["model"]
            self._ml_scaler = bundle["scaler"]
            self._ml_labels = bundle["label_names"]
            logger.info("已加载 SVM 手势模型，类别: %s", self._ml_labels.tolist())
        except Exception as e:
            logger.warning("加载 SVM 模型失败，回退到启发式规则: %s", e)
            self._ml_model = None

    def _classify_ml(self, keypoints: list[dict]) -> tuple[str, float] | None:
        """使用 SVM 模型预测手势。失败返回 None，由调用方回退到启发式规则。"""
        if self._ml_model is None or self._ml_scaler is None:
            return None
        if len(keypoints) != 21:
            return None

        try:
            from app.models_infer.hand_utils import normalize_hand_landmarks_array, SingleClassWrapper  # noqa: F401
            feat = normalize_hand_landmarks_array(keypoints).reshape(1, -1)  # (1, 63)
            feat_scaled = self._ml_scaler.transform(feat)

            # 尝试获取概率
            if hasattr(self._ml_model, "predict_proba"):
                proba = self._ml_model.predict_proba(feat_scaled)[0]
                idx = int(np.argmax(proba))
                gesture = str(self._ml_labels[idx])
                confidence = float(proba[idx])
                return gesture, confidence
            else:
                idx = int(self._ml_model.predict(feat_scaled)[0])
                gesture = str(self._ml_labels[idx])
                return gesture, 0.75
        except Exception as e:
            logger.debug("ML 推理失败: %s", e)
            return None

    # ----------------------------------------------------------------
    # 静态手势分类
    # ----------------------------------------------------------------

    def classify_static(self, keypoints: list[dict]) -> tuple[str, float]:
        """
        根据 21 个手部关键点判定静态手势。
        优先使用 ML 模型，不存在时回退到启发式规则。

        Returns:
            (gesture_name, confidence)
        """
        if len(keypoints) != 21:
            return "unknown", 0.0

        # 优先: 训练好的 SVM/ML 模型
        ml_result = self._classify_ml(keypoints)
        if ml_result is not None:
            return ml_result

        # 回退: 启发式规则
        return self._classify_heuristic(keypoints)

    # ----------------------------------------------------------------
    # 统一入口：静态 + 动态 + 时序去抖
    # ----------------------------------------------------------------

    def classify_frame(self, keypoints: list[dict] | None) -> tuple[str, float]:
        """
        统一手势分类入口，融合静态识别、动态追踪、时序去抖。

        传入 None 表示当前帧无手。
        - 动态手势触发后进入冷却期，抑制静态输出返回 idle
        - 手消失 ≥ NO_HAND_REARM_FRAMES 帧 → 解除冷却（手势边界）
        - 静态手势需连续稳定 STATIC_STABLE_FRAMES 帧才输出

        Returns:
            (gesture_name, confidence)
        """
        # ---- 无手：重置 tracker + 计数 ----
        if keypoints is None:
            if self.tracker:
                self.tracker.reset()
            self._no_hand += 1
            self._stable_gesture = None
            self._stable_count = 0
            if self._no_hand >= self.NO_HAND_REARM_FRAMES:
                self._cooldown = 0  # 手势边界 → 解除冷却
            return "unknown", 0.0

        if len(keypoints) != 21:
            return "unknown", 0.0

        self._no_hand = 0

        # ---- 静态 + 动态推理 ----
        static_gesture, static_conf = self.classify_static(keypoints)
        dynamic = self.tracker.update(keypoints) if self.tracker else None

        # 1) 动态触发：立即输出 + 进入冷却
        if dynamic:
            self._cooldown = self.DYNAMIC_COOLDOWN_FRAMES
            self._stable_gesture = None
            self._stable_count = 0
            return dynamic, 0.85

        # 2) 冷却期：抑制所有静态输出
        if self._cooldown > 0:
            self._cooldown -= 1
            self._stable_gesture = None
            self._stable_count = 0
            return "idle", 0.0

        # 3) 冷却结束：静态需连续稳定 K 帧才输出
        if static_gesture == self._stable_gesture:
            self._stable_count += 1
        else:
            self._stable_gesture = static_gesture
            self._stable_count = 1

        if self._stable_count >= self.STATIC_STABLE_FRAMES:
            return static_gesture, static_conf

        return "idle", 0.0

    def _classify_heuristic(self, keypoints: list[dict]) -> tuple[str, float]:

        fingers = self._finger_states(keypoints)
        extended_count = sum(fingers)

        # 全部弯曲 → 握拳
        if extended_count == 0:
            return "fist", 0.95

        # 全部伸展 → 手掌张开
        if extended_count >= 4:  # 容忍 1 指误判
            return "palm", 0.92

        # 仅拇指伸展 → thumb_up / thumb_down
        if extended_count == 1 and fingers[0]:
            thumb_tip = keypoints[4]
            thumb_mcp = keypoints[2]
            if thumb_tip["y"] < thumb_mcp["y"] - 0.03:
                return "thumb_up", 0.88
            elif thumb_tip["y"] > thumb_mcp["y"] + 0.03:
                return "thumb_down", 0.88
            else:
                return "thumb_up", 0.70  # 默认向上

        # 仅食指伸展 → pointing（用于触发画圈/滑动追踪）
        if extended_count == 1 and fingers[1]:
            return "pointing", 0.90

        # 仅食指+中指伸展 → 可能是 V 手势，暂归为 pointing
        if extended_count == 2 and fingers[1] and fingers[2]:
            return "pointing", 0.78

        return "unknown", 0.40

    # ----------------------------------------------------------------
    # 手指伸展判定
    # ----------------------------------------------------------------

    def _finger_states(self, kp: list[dict]) -> list[bool]:
        """
        返回 5 个布尔值: [thumb, index, middle, ring, pinky]

        判定逻辑: 指尖到腕距 > 第二关节到腕距 × 1.2 表示伸展。
        拇指特殊处理: 指尖-IP距 > IP-MCP距 × 1.2。
        """
        wrist = kp[0]
        # 每根手指: (tip, pip/ip, mcp)
        fingers_def = [
            (4, 3, 2),   # thumb:  tip=4,  ip=3,  mcp=2
            (8, 6, 5),   # index:  tip=8,  pip=6,  mcp=5
            (12, 10, 9), # middle: tip=12, pip=10, mcp=9
            (16, 14, 13),# ring:   tip=16, pip=14, mcp=13
            (20, 18, 17),# pinky:  tip=20, pip=18, mcp=17
        ]

        results: list[bool] = []
        for tip_i, pip_i, mcp_i in fingers_def:
            if tip_i == 4:  # 拇指: 用 MCP→IP 与 IP→TIP 夹角判断（比距离比更鲁棒）
                v1 = (kp[pip_i]["x"] - kp[mcp_i]["x"],
                      kp[pip_i]["y"] - kp[mcp_i]["y"])
                v2 = (kp[tip_i]["x"] - kp[pip_i]["x"],
                      kp[tip_i]["y"] - kp[pip_i]["y"])
                mag1 = math.hypot(v1[0], v1[1])
                mag2 = math.hypot(v2[0], v2[1])
                if mag1 < 1e-8 or mag2 < 1e-8:
                    results.append(False)
                else:
                    cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)
                    # cos > 0.35 → 夹角 < ~70° → 拇指伸直
                    results.append(cos_angle > 0.35)
            else:
                tip_wrist = GestureClassifier._dist(kp[tip_i], wrist)
                pip_wrist = GestureClassifier._dist(kp[pip_i], wrist)
                results.append(tip_wrist > pip_wrist * 1.2 if pip_wrist > 1e-8 else False)

        return results

    @staticmethod
    def _dist(a: dict, b: dict) -> float:
        return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


# ================================================================
# HandGestureTracker — 动态手势时序状态机
# ================================================================

class HandGestureTracker:
    """
    追踪手部运动轨迹，识别动态手势。

    三种动态手势:
      - 画圈 (circle_cw / circle_ccw): 食指尖累积转角 ≥ 300° 且轨道半径稳定
      - 滑动 (swipe_left / swipe_right): 手腕 x 方向位移超阈值
      - 挥手 (wave): 手腕 x 方向往复反转 ≥ 2 次

    每帧调用 update()，识别成功后自动 reset。
    """

    def __init__(self):
        self.history: list[tuple[int, float, float, float, float]] = []
        # (frame_idx, wrist_x, wrist_y, index_tip_x, index_tip_y)
        self.frame_count: int = 0
        self._max_history: int = 60

    def reset(self) -> None:
        self.history.clear()
        self.frame_count = 0

    def update(self, keypoints: list[dict]) -> str | None:
        """
        输入 21 关键点，返回识别的动态手势名称或 None。

        手势优先级: 挥手 > 画圈 > 滑动
        """
        if len(keypoints) != 21:
            return None

        self.frame_count += 1

        wrist = keypoints[0]
        index_tip = keypoints[8]
        entry = (
            self.frame_count,
            wrist["x"], wrist["y"],
            index_tip["x"], index_tip["y"],
        )
        self.history.append(entry)
        if len(self.history) > self._max_history:
            self.history = self.history[-self._max_history:]

        if len(self.history) < 8:
            return None

        # 挥手优先级最高
        wave = self._detect_wave()
        if wave:
            self.reset()
            return wave

        # 画圈
        circle = self._detect_circle()
        if circle:
            self.reset()
            return circle

        # 滑动
        swipe = self._detect_swipe()
        if swipe:
            self.reset()
            return swipe

        return None

    # ------------------------------------------------------------
    # 画圈检测
    # ------------------------------------------------------------

    def _detect_circle(self) -> str | None:
        if len(self.history) < 15:
            return None

        points = [(h[3], h[4]) for h in self.history[-30:]]  # (tip_x, tip_y)

        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)

        # 累积转角
        total_angle = 0.0
        for i in range(1, len(points)):
            a1 = math.atan2(points[i - 1][1] - cy, points[i - 1][0] - cx)
            a2 = math.atan2(points[i][1] - cy, points[i][0] - cx)
            d = a2 - a1
            while d > math.pi:
                d -= 2 * math.pi
            while d < -math.pi:
                d += 2 * math.pi
            total_angle += d

        # 半径稳定性
        radii = [math.hypot(p[0] - cx, p[1] - cy) for p in points]
        mean_r = sum(radii) / len(radii)
        if mean_r < 0.015:  # 轨道太小不可靠
            return None
        max_dev = max(abs(r - mean_r) for r in radii) / mean_r

        if abs(total_angle) >= math.radians(300) and max_dev < 0.55:
            return "circle_cw" if total_angle > 0 else "circle_ccw"

        return None

    # ------------------------------------------------------------
    # 滑动检测
    # ------------------------------------------------------------

    def _detect_swipe(self) -> str | None:
        if len(self.history) < 8:
            return None

        start_x = self.history[0][1]   # wrist_x
        end_x = self.history[-1][1]
        displacement = end_x - start_x

        threshold = 0.06  # 归一化坐标阈值

        if displacement > threshold:
            return "swipe_right"
        elif displacement < -threshold:
            return "swipe_left"

        return None

    # ------------------------------------------------------------
    # 挥手检测
    # ------------------------------------------------------------

    def _detect_wave(self) -> str | None:
        if len(self.history) < 15:
            return None

        wrist_x = [h[1] for h in self.history]

        # 每隔 5 帧采样方向，统计反转次数
        reversals = 0
        prev_dir = 0
        for i in range(5, len(wrist_x)):
            diff = wrist_x[i] - wrist_x[i - 5]
            if abs(diff) < 0.012:
                continue
            curr_dir = 1 if diff > 0 else -1
            if prev_dir != 0 and curr_dir != prev_dir:
                reversals += 1
                if reversals >= 2:
                    return "wave"
            prev_dir = curr_dir

        return None
