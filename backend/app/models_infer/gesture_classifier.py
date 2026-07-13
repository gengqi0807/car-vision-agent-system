"""
手势分类器 — 静态手势（基于 21 关键点手指伸展度） + 动态手势状态机。

静态手势:
  - palm      : 五指全伸展 → 唤醒
  - fist      : 五指全弯曲 → 确认
  - thumb_up  : 仅拇指伸展且朝上 → 接听
  - thumb_down: 仅拇指伸展且朝下 → 挂断
  - pointing  : 仅食指伸展 → 用于触发动态追踪

动态手势（通过 HandGestureTracker 时序状态机）:
  - circle_ccw : 逆时针画圈 → 音量-
  - circle_cw  : 顺时针画圈 → 音量+
  - swipe_left : fist→palm 1秒内切换 → 上一个功能（单手时序）
  - swipe_right: palm→fist 1秒内切换 → 下一个功能（单手时序）
  - wave       : 手腕往复摆动 → 返回主页

接口预留:
  - GestureClassifier(domain="police") → 交警手势复用同一分类器框架
"""

import math
import os
import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


class GestureClassifier:
    """手势分类器，支持 domain="owner" / "police" 两种模式。

    owner 模式下：
      - 静态手势: 训练好的 SVM 模型，不存在时回退到启发式规则
      - 动态手势: 训练好的 BiLSTM 模型，不存在时回退到 HandGestureTracker 启发式
      - meta-router: 融合静态/动态分支，选择置信度更高的输出
    """

    # ----------------------------------------------------------------
    # 时序保护常量（classify_frame 使用）
    # ----------------------------------------------------------------
    DYNAMIC_COOLDOWN_FRAMES: int = 18  # 动态触发后抑制静态的帧数 (~0.6s @30fps)
    NO_HAND_REARM_FRAMES: int = 3      # 手消失多少帧算「手势边界」，重新武装
    STATIC_STABLE_FRAMES: int = 3      # 静态手势需连续稳定几帧才输出（已废弃，改用 SETTLE 时间）
    STATIC_SETTLE_SECONDS: float = 1.0 # 静态手势需持续同一手势达 N 秒才输出

    # Meta-router: 动态 LSTM 置信度需比静态高多少才切换为动态输出
    DYNAMIC_CONF_MARGIN: float = 0.05

    # 运动能量阈值：超过此值认为手在显著运动，倾向动态手势
    MOTION_FIRE_THRESHOLD: float = 0.0004

    # 持续运动帧数：超过此值强制让动态手势胜出（抑制静态"确认/唤醒"误判）
    SUSTAINED_MOTION_FRAMES: int = 4

    # wave（主页）需连续确认的帧数，防误触发
    WAVE_STABLE_FRAMES: int = 5

    # 动态手势稳定延迟：动作开始 N 秒后才输出结果，避免初期跳变
    DYNAMIC_SETTLE_SECONDS: float = 1.0

    def __init__(self, domain: str = "owner"):
        self.domain = domain
        self.tracker: HandGestureTracker | None = (
            HandGestureTracker() if domain == "owner" else None
        )
        self._ml_model = None
        self._ml_scaler = None
        self._ml_labels = None

        # 动态 LSTM 分类器（仅在 owner 模式下加载）
        self._dynamic_lstm = None

        # 时序保护状态
        self._cooldown: int = 0
        self._no_hand: int = 0
        self._stable_gesture: str | None = None
        self._stable_start_time: float = 0.0  # 静态手势首次出现的时刻（time.time()）
        self._wave_count: int = 0            # wave 连续确认计数

        # 动态手势稳定延迟计时
        self._dyn_active: bool = False    # 动态动作是否已激活（开始计时）
        self._dyn_start_time: float = 0.0 # 动作开始时刻（time.time()）

        # 单手时序切换状态（fist↔palm 在 1s 内完成）
        self._swipe_from: str | None = None
        self._swipe_from_time: float = 0.0

        # 运动追踪（独立于 LSTM，用于元路由器判断"手是否在运动"）
        self._motion_history: list[tuple[float, float]] = []  # (wrist_x, wrist_y)
        self._motion_history_max: int = 15
        self._motion_frames: int = 0          # 连续高运动帧计数

        # 尝试加载训练好的 SVM 模型 + 动态 LSTM 模型
        if domain == "owner":
            self._load_ml_model()
            self._load_dynamic_lstm()

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

    def _load_dynamic_lstm(self) -> None:
        """尝试加载动态手势 LSTM 模型。加载失败不影响静态分类。"""
        try:
            from app.models_infer.dynamic_lstm import DynamicLSTMClassifier
            self._dynamic_lstm = DynamicLSTMClassifier()
            if not self._dynamic_lstm.is_loaded:
                logger.info("动态 LSTM 模型未加载，动态手势回退到启发式规则")
        except Exception as e:
            logger.warning("初始化动态 LSTM 失败: %s", e)
            self._dynamic_lstm = None

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
    # 统一入口：静态 SVM + 动态 LSTM/Tracker + meta-router + 时序去抖
    # ----------------------------------------------------------------

    def classify_frame(self, keypoints: list[dict] | None) -> tuple[str, float]:
        """
        统一手势分类入口，融合静态识别、动态追踪、meta-router、时序去抖。

        路由逻辑:
          1. 无手 → 触发 LSTM 边界分段判定（对完整轨迹一次分类）
                 → 重置 tracker/LSTM 缓冲
                 → 手消失 ≥ REARM 帧 → 解除冷却
          2. 有手 → 静态 SVM + 单手时序切换判定（fist↔palm 1s 内 = swipe）
                 → 动态 LSTM 滑动窗口打分
                 → meta-router: 动态置信度 > 静态 + margin → 输出动态
                 → 动态触发后冷却期内抑制静态
                 → 静态需连续稳定 K 帧才输出

        Returns:
            (gesture_name, confidence)
        """
        # ---- 无手：边界分段 + 重置 ----
        if keypoints is None:
            boundary_result = self._handle_boundary()
            if self.tracker:
                self.tracker.reset()
            self._dyn_active = False  # 手消失，重置稳定计时
            self._no_hand += 1
            self._stable_gesture = None
            self._stable_start_time = 0.0
            self._motion_history.clear()
            self._motion_frames = 0
            self._swipe_from = None
            self._swipe_from_time = 0.0
            if self._no_hand >= self.NO_HAND_REARM_FRAMES:
                self._cooldown = 0  # 手势边界 → 解除冷却

            # 边界分段有有效动态结果 → 立即输出
            if boundary_result[0] != "unknown":
                self._cooldown = self.DYNAMIC_COOLDOWN_FRAMES
                return boundary_result
            return "unknown", 0.0

        if len(keypoints) != 21:
            return "unknown", 0.0

        self._no_hand = 0

        # ---- 静态 SVM ----
        static_gesture, static_conf = self.classify_static(keypoints)

        # ---- 单手时序切换判定（1s 内 fist↔palm）----
        now = time.time()
        swipe = self._detect_swipe_transition(static_gesture, now)
        if swipe:
            self._cooldown = self.DYNAMIC_COOLDOWN_FRAMES
            self._stable_gesture = None
            self._stable_start_time = 0.0
            self._swipe_from = None
            self._swipe_from_time = 0.0
            if self.tracker:
                self.tracker.reset()
            return swipe, 0.9

        # ---- 动态分支 ----
        dynamic_gesture: str | None = None
        dynamic_conf: float = 0.0

        # 优先 LSTM（如果已加载）
        lstm_result = self._classify_dynamic_lstm(keypoints)
        if lstm_result is not None:
            dynamic_gesture, dynamic_conf = lstm_result
        elif self.tracker:
            # 回退到启发式状态机
            heuristic = self.tracker.update(keypoints)
            if heuristic:
                dynamic_gesture, dynamic_conf = heuristic, 0.85

        # 屏蔽 LSTM/启发式 的 swipe 输出（swipe 已改为单手时序切换判定）
        if dynamic_gesture in ("swipe_left", "swipe_right"):
            dynamic_gesture, dynamic_conf = None, 0.0

        # ---- meta-router ----
        wave_confirmed_this_frame = False

        if dynamic_gesture and dynamic_gesture != "unknown":
            # ---- 2 秒稳定延迟：动作刚开始轨迹不完整、结果跳变，先静默 ----
            now = time.time()
            if not self._dyn_active:
                self._dyn_active = True
                self._dyn_start_time = now
            if now - self._dyn_start_time < self.DYNAMIC_SETTLE_SECONDS:
                # 未满 2 秒 → 抑制跳变结果，返回 idle
                self._stable_gesture = None
                self._stable_start_time = 0.0
                self._wave_count = 0
                return "idle", 0.0

            # 运动能量（独立于 LSTM，基于手腕轨迹方差）
            motion_energy = self._update_motion(keypoints)

            should_fire_dynamic = False

            if static_gesture == "unknown" or static_gesture == "pointing":
                # 静态不可靠时，动态直接触发
                should_fire_dynamic = True
                self._motion_frames = min(self._motion_frames + 1, 10)
            elif motion_energy > self.MOTION_FIRE_THRESHOLD:
                # 手在显著运动 → 大幅惩罚静止手势（fist/palm 不应该运动中触发）
                self._motion_frames += 1
                if static_gesture in ("fist", "palm"):
                    # 运动中出现的 "确认/唤醒" 几乎一定是误判 → 惩罚 65%
                    adjusted_static_conf = static_conf * 0.35
                else:
                    adjusted_static_conf = static_conf * 0.65

                if self._motion_frames >= self.SUSTAINED_MOTION_FRAMES:
                    # 持续运动 N 帧 → 确定是动态手势，低置信也触发
                    should_fire_dynamic = dynamic_conf >= 0.20
                else:
                    # 刚开始运动 → 动态只需打赢被惩罚后的静态
                    should_fire_dynamic = dynamic_conf > adjusted_static_conf + 0.02
            elif dynamic_conf > static_conf + self.DYNAMIC_CONF_MARGIN:
                # 低运动但动态置信度显著高于静态 → 动态优先
                should_fire_dynamic = True
                self._motion_frames = max(0, self._motion_frames - 1)
            else:
                self._motion_frames = max(0, self._motion_frames - 1)

            if should_fire_dynamic:
                # wave（主页）需要连续5帧确认才输出，防误触发
                if dynamic_gesture == "wave":
                    self._wave_count += 1
                    wave_confirmed_this_frame = True
                    if self._wave_count >= self.WAVE_STABLE_FRAMES:
                        self._cooldown = self.DYNAMIC_COOLDOWN_FRAMES
                        self._stable_gesture = None
                        self._stable_start_time = 0.0
                        self._wave_count = 0
                        return dynamic_gesture, round(dynamic_conf, 4)
                    return "idle", 0.0

                # 其他动态手势直接触发
                self._cooldown = self.DYNAMIC_COOLDOWN_FRAMES
                self._stable_gesture = None
                self._stable_start_time = 0.0
                self._wave_count = 0
                return dynamic_gesture, round(dynamic_conf, 4)

            # 动态手势激活中但未触发 → 抑制静态，返回 idle
            self._stable_gesture = None
            self._stable_start_time = 0.0
            return "idle", 0.0

        # 无有效动态手势 → 重置稳定计时，下次运动重新计 2 秒
        if dynamic_gesture is None or dynamic_gesture == "unknown":
            self._dyn_active = False

        # 本帧未确认 wave → 重置连续计数（中断则从头来）
        if not wave_confirmed_this_frame:
            self._wave_count = 0

        # ---- 冷却期 ----
        if self._cooldown > 0:
            self._cooldown -= 1
            self._stable_gesture = None
            self._stable_start_time = 0.0
            return "idle", 0.0

        # ---- 静态稳定投票（时间基准：持续 N 秒同一手势才输出） ----
        now = time.time()
        if static_gesture == self._stable_gesture and self._stable_start_time > 0:
            pass  # 手势不变，计时继续
        else:
            self._stable_gesture = static_gesture
            self._stable_start_time = now

        if self._stable_start_time > 0 and now - self._stable_start_time >= self.STATIC_SETTLE_SECONDS:
            return static_gesture, round(static_conf, 4)

        return "idle", 0.0

    # ----------------------------------------------------------------
    # 动态 LSTM 推理 + 边界分段
    # ----------------------------------------------------------------

    def _classify_dynamic_lstm(self, keypoints: list[dict]) -> tuple[str, float] | None:
        """使用 LSTM 滑动窗口对当前帧打分。返回 (gesture, conf) 或 None。"""
        if self._dynamic_lstm is None or not self._dynamic_lstm.is_loaded:
            return None
        try:
            gesture, conf = self._dynamic_lstm.classify(keypoints, is_boundary=False)
            if gesture == "unknown":
                return None
            return gesture, conf
        except Exception as e:
            logger.debug("动态 LSTM 推理异常: %s", e)
            return None

    def _handle_boundary(self) -> tuple[str, float]:
        """
        手消失时触发边界分段：对累积的完整轨迹做一次 LSTM 最终判定。

        Returns:
            (gesture, confidence) — 有效动态手势或 unknown
        """
        if self._dynamic_lstm is None or not self._dynamic_lstm.is_loaded:
            return "unknown", 0.0
        try:
            trajectory = self._dynamic_lstm.get_trajectory()
            min_len = self._dynamic_lstm.MIN_SEQUENCE_LENGTH
            if trajectory.shape[0] < min_len:
                self._dynamic_lstm.reset_trajectory()
                return "unknown", 0.0

            result = self._dynamic_lstm.classify_sequence(trajectory)
            self._dynamic_lstm.reset_trajectory()
            return result
        except Exception as e:
            logger.debug("边界分段判定异常: %s", e)
            if self._dynamic_lstm:
                self._dynamic_lstm.reset_trajectory()
            return "unknown", 0.0

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
    # 单手时序切换判定（fist↔palm 在 1s 内完成）
    # ----------------------------------------------------------------

    def _detect_swipe_transition(self, gesture: str, now: float) -> str | None:
        """单手时序切换: fist→palm=swipe_left, palm→fist=swipe_right。"""
        WINDOW = 1.8       # 给实际推理帧率和手型过渡留出余量
        MIN_HOLD = 0.10    # 起点手势至少保持 0.1s，过滤抖动误触发
        gesture = "palm" if gesture == "open_palm" else gesture
        if gesture not in ("fist", "palm"):
            # 张合过程中常有几帧 unknown/pointing，短暂中间态不应清空起点。
            if self._swipe_from_time > 0 and now - self._swipe_from_time > WINDOW:
                self._swipe_from = None
                self._swipe_from_time = 0.0
            return None
        if self._swipe_from is None:
            self._swipe_from = gesture
            self._swipe_from_time = now
            return None
        if gesture == self._swipe_from:
            return None
        dt = now - self._swipe_from_time
        result = None
        if MIN_HOLD <= dt <= WINDOW:
            if self._swipe_from == "fist" and gesture == "palm":
                result = "swipe_left"
            elif self._swipe_from == "palm" and gesture == "fist":
                result = "swipe_right"
        self._swipe_from = gesture
        self._swipe_from_time = now
        return result

    # ----------------------------------------------------------------
    # 运动追踪（供元路由器使用）
    # ----------------------------------------------------------------

    def _update_motion(self, keypoints: list[dict]) -> float:
        """更新手腕运动缓冲区，返回当前运动能量（归一化坐标方差）。

        独立于 LSTM，确保启发式 tracker 回退时也能受益。
        """
        if len(keypoints) != 21:
            return 0.0

        wrist_x, wrist_y = keypoints[0]["x"], keypoints[0]["y"]
        self._motion_history.append((wrist_x, wrist_y))
        if len(self._motion_history) > self._motion_history_max:
            self._motion_history = self._motion_history[-self._motion_history_max:]

        if len(self._motion_history) < 5:
            return 0.0

        xs = [p[0] for p in self._motion_history]
        ys = [p[1] for p in self._motion_history]
        return float(np.var(xs)) + float(np.var(ys))

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

    两种动态手势:
      - 画圈 (circle_cw / circle_ccw): 手腕轨迹累计转角 ≈ 2π 且闭环
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

        手势优先级: 挥手 > 画圈
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

        # 画圈 (circle_cw / circle_ccw)
        circle = self._detect_circle()
        if circle:
            self.reset()
            return circle

        return None

    # ------------------------------------------------------------
    # 画圈检测 (circle_cw / circle_ccw)
    # ------------------------------------------------------------

    def _detect_circle(self) -> str | None:
        """检测画圈手势，基于累计转向角 + 闭环判定。"""
        if len(self.history) < 12:
            return None

        xs = np.array([h[1] for h in self.history], dtype=np.float64)
        ys = np.array([h[2] for h in self.history], dtype=np.float64)

        # 平滑降噪（3点均值滤波）
        if len(xs) >= 3:
            kernel = np.ones(3) / 3.0
            xs_smooth = np.convolve(xs, kernel, mode='valid')
            ys_smooth = np.convolve(ys, kernel, mode='valid')
        else:
            xs_smooth, ys_smooth = xs, ys

        total_angle = 0.0
        valid_steps = 0
        n = len(xs_smooth)

        for i in range(1, n - 1):
            v1_x = xs_smooth[i] - xs_smooth[i - 1]
            v1_y = ys_smooth[i] - ys_smooth[i - 1]
            v2_x = xs_smooth[i + 1] - xs_smooth[i]
            v2_y = ys_smooth[i + 1] - ys_smooth[i]

            mag1 = math.hypot(v1_x, v1_y)
            mag2 = math.hypot(v2_x, v2_y)
            if mag1 < 0.0015 or mag2 < 0.0015:
                continue

            cos_angle = (v1_x * v2_x + v1_y * v2_y) / (mag1 * mag2)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle = math.acos(cos_angle)

            # 叉积判断顺时针/逆时针 (屏幕坐标系 y 向下，与数学坐标系相反)
            cross = v1_x * v2_y - v1_y * v2_x
            if cross > 0:
                angle = -angle

            total_angle += angle
            valid_steps += 1

        if valid_steps < 6:
            return None

        # 累计转角 > 5.0 rad (~286°) 认为画了圈
        if abs(total_angle) > 5.0:
            # 闭环：起点和终点接近
            net_x = xs[-1] - xs[0]
            net_y = ys[-1] - ys[0]
            net_dist = math.hypot(net_x, net_y)

            # 轨迹范围（过滤原地抖动）
            x_range = float(np.max(xs) - np.min(xs))
            y_range = float(np.max(ys) - np.min(ys))
            span = max(x_range, y_range)

            if net_dist < 0.05 and span > 0.03:
                # total_angle > 0 → 逆时针 (ccw), total_angle < 0 → 顺时针 (cw)
                return "circle_ccw" if total_angle > 0 else "circle_cw"

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
