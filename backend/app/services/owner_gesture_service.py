"""
车主手势识别服务 — 后端拉流模式。

模式:
  - POST /stream/start   → 启动后台线程，从 RTSP/摄像头拉流 + MediaPipe Hands 推理
  - POST /stream/stop    → 停止拉流
  - GET  /current        → 获取最近一次识别结果
  - WS   /ws             → 实时推送识别结果

手势 → 车机动作映射:
  palm        → wake      (唤醒)
  fist        → confirm   (确认)
  circle_ccw  → volume_down(音量-)
  circle_cw   → volume_up (音量+)
  swipe_left  → prev_func (上一个功能)
  swipe_right → next_func (下一个功能)
  thumb_up    → call_answer (接听)
  thumb_down  → call_hangup(挂断)
  wave        → home      (主页)

接口预留:
  - _notify_control_panel(action) → 车机控制面板
  - _notify_alert(gesture, confidence) → 告警智能体
  - result_history 列表 → 历史持久化模块
"""

import asyncio
import os
import threading
import time
from datetime import datetime, timezone
from typing import ClassVar

import cv2
import numpy as np

from app.core.config import settings
from app.models_infer.gesture_classifier import GestureClassifier
from app.models_infer.mediapipe_hands import MediaPipeHands
from app.schemas.gesture import Keypoint, OwnerGestureResult, StreamState

# ----------------------------------------------------------------
# 手势 → 车机动作映射
# ----------------------------------------------------------------

GESTURE_ACTION_MAP: dict[str, str] = {
    "palm":        "wake",
    "fist":        "confirm",
    "circle_ccw":  "volume_down",
    "circle_cw":   "volume_up",
    "swipe_left":  "prev_func",
    "swipe_right": "next_func",
    "thumb_up":    "call_answer",
    "thumb_down":  "call_hangup",
    "wave":        "home",
    "pointing":    "idle",      # 食指单指 — 追踪中，不触发动作
    "unknown":     "idle",
}


def gesture_to_action(gesture: str) -> str:
    return GESTURE_ACTION_MAP.get(gesture, "idle")


# ----------------------------------------------------------------
# 服务主体
# ----------------------------------------------------------------

class OwnerGestureService:
    """
    单例服务：管理拉流线程 + 推理 + 结果存储。

    线程安全:
      - _lock 保护 latest_result / stream_state / running
      - _running 作为停止信号
    """

    _instance: ClassVar["OwnerGestureService | None"] = None
    _initialized: ClassVar[bool] = False

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

        # 最新识别结果
        self._latest_result: OwnerGestureResult | None = None

        # 流状态
        self._stream_state = StreamState(running=False)
        self._source: str = ""

        # 推理组件（首次使用时懒加载）
        self._classifier: GestureClassifier | None = None
        self._model_configured = False

        # 历史结果（预留：可接入数据库持久化）
        self.result_history: list[OwnerGestureResult] = []

        # WebSocket 回调（由 API 层注入）
        self._ws_callbacks: list = []

        # 车机控制回调（预留）
        self._control_callbacks: list = []

        # 告警回调（预留）
        self._alert_callbacks: list = []

    # ----------------------------------------------------------------
    # 单例获取
    # ----------------------------------------------------------------

    @classmethod
    def instance(cls) -> "OwnerGestureService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ----------------------------------------------------------------
    # 模型配置（lazy-load，仅首次调用时初始化）
    # ----------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model_configured:
            return

        model_path = os.path.join(settings.models_dir, settings.hand_landmarker_model)
        MediaPipeHands.configure(
            model_path=model_path,
            num_hands=settings.num_hands,
            min_detection_confidence=settings.min_hand_detection_confidence,
            min_presence_confidence=settings.min_hand_presence_confidence,
            min_tracking_confidence=settings.min_hand_tracking_confidence,
        )
        self._classifier = GestureClassifier(domain="owner")
        self._model_configured = True

    # ----------------------------------------------------------------
    # 注册回调
    # ----------------------------------------------------------------

    def register_ws_callback(self, cb) -> None:
        """注册 WebSocket 推送回调 (async callable)。"""
        self._ws_callbacks.append(cb)

    def unregister_ws_callback(self, cb) -> None:
        if cb in self._ws_callbacks:
            self._ws_callbacks.remove(cb)

    def register_control_callback(self, cb) -> None:
        """预留：车机控制回调 (action: str) → None。"""
        self._control_callbacks.append(cb)

    def register_alert_callback(self, cb) -> None:
        """预留：告警回调 (gesture: str, confidence: float) → None。"""
        self._alert_callbacks.append(cb)

    # ----------------------------------------------------------------
    # 公开属性
    # ----------------------------------------------------------------

    @property
    def latest_result(self) -> OwnerGestureResult | None:
        with self._lock:
            return self._latest_result

    @property
    def stream_state(self) -> StreamState:
        with self._lock:
            return self._stream_state

    # ----------------------------------------------------------------
    # 流控制
    # ----------------------------------------------------------------

    def start(self, source: str, fps: int = 15) -> StreamState:
        """启动后端拉流 + 推理线程。"""
        with self._lock:
            if self._running:
                return self._stream_state

            self._ensure_model()
            self._source = source
            self._running = True

        self._thread = threading.Thread(
            target=self._stream_loop,
            args=(fps,),
            daemon=True,
            name="owner-gesture-stream",
        )
        self._thread.start()

        with self._lock:
            self._stream_state = StreamState(
                running=True,
                source=source,
                fps=fps,
                started_at=datetime.now(timezone.utc),
            )
            return self._stream_state

    def stop(self) -> StreamState:
        """停止拉流线程。"""
        with self._lock:
            self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        with self._lock:
            self._stream_state = StreamState(running=False)
            return self._stream_state

    # ----------------------------------------------------------------
    # 历史接口（兼容旧 GET /current）
    # ----------------------------------------------------------------

    def current_result(self) -> OwnerGestureResult:
        """返回最近一次识别结果（GET /current 使用）。"""
        result = self.latest_result
        if result:
            return result
        return OwnerGestureResult(
            gesture="unknown",
            action="idle",
            confidence=0.0,
            keypoints=[],
            hand_count=0,
            updated_at=datetime.now(timezone.utc),
        )

    # ----------------------------------------------------------------
    # 推理主循环（在后台线程中运行）
    # ----------------------------------------------------------------

    def _stream_loop(self, fps: int) -> None:
        """
        后端拉流 → 逐帧推理 → 更新 latest_result → 触发回调。
        参考 police_local.py 的 RTSP 拉流模式。
        """
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            with self._lock:
                self._running = False
            return

        classifier = self._classifier
        frame_interval = 1.0 / max(fps, 1)

        while self._running:
            start_tick = time.time()

            ret, frame_bgr = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            try:
                # MediaPipe Hands 推理
                hands = MediaPipeHands.infer(frame_bgr)
                hand_kp = hands[0] if hands else None

                # 统一手势分类（静态 + 动态 + 时序去抖）
                gesture, confidence = classifier.classify_frame(hand_kp, all_hands=hands)

                # 转换为 keypoints
                if hands:
                    keypoints = [
                        Keypoint(x=k["x"], y=k["y"], score=1.0) for k in hand_kp
                    ]
                else:
                    keypoints = []

                result = OwnerGestureResult(
                    gesture=gesture,
                    action=gesture_to_action(gesture),
                    confidence=round(confidence, 4),
                    keypoints=keypoints,
                    hand_count=len(hands),
                    updated_at=datetime.now(timezone.utc),
                )

                # 更新最新结果
                with self._lock:
                    self._latest_result = result

                self.result_history.append(result)

                # 触发控制回调（预留）
                if result.action != "idle":
                    for cb in self._control_callbacks:
                        try:
                            cb(result.action)
                        except Exception:
                            pass
                    for cb in self._alert_callbacks:
                        try:
                            cb(result.gesture, result.confidence)
                        except Exception:
                            pass

                # WebSocket 推送
                if self._ws_callbacks:
                    payload = result.model_dump(mode="json")
                    for cb in self._ws_callbacks:
                        try:
                            asyncio.run_coroutine_threadsafe(
                                cb(payload), asyncio.get_event_loop()
                            )
                        except Exception:
                            pass

            except Exception:
                pass  # 单帧出错不影响整体流程

            # 帧率控制
            elapsed = time.time() - start_tick
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()
        MediaPipeHands.reset()
