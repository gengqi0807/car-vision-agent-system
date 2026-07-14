from __future__ import annotations

import asyncio
import base64
from collections import deque
from dataclasses import dataclass, field
import logging
import math
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from time import perf_counter
from typing import TYPE_CHECKING, Any, ClassVar, Optional
from uuid import uuid4

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.owner_gesture_record import OwnerGestureRecord
from app.models.user_operation_log import UserOperationLog
from app.schemas.gesture import (
    ControlPanelState,
    GestureFrameResult,
    Keypoint,
    OwnerGestureResult,
    StreamState,
)
from app.services.alert_service import AlertService
from app.services.monitor_service import MonitorService

try:
    from police.visualization import draw_chinese_text_lines
except ModuleNotFoundError:
    import sys

    project_root = Path(__file__).resolve().parents[3]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    from police.visualization import draw_chinese_text_lines

if TYPE_CHECKING:
    from app.models_infer.gesture_classifier import GestureClassifier
    from app.models_infer.mediapipe_hands import MediaPipeHands

logger = logging.getLogger(__name__)

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)

GESTURE_ACTION_MAP: dict[str, str] = {
    "open_palm": "wake",
    "palm": "wake",
    "fist": "confirm",
    "index_circle": "idle",
    "circle_cw": "volume_up",
    "circle_ccw": "volume_down",
    "swipe_left": "prev_func",
    "swipe_right": "next_func",
    "thumbs_up": "call_answer",
    "thumb_up": "call_answer",
    "thumbs_down": "call_hangup",
    "thumb_down": "call_hangup",
    "thunb_index": "home",
    "thumb_index": "home",
    "wave": "idle",
    "point": "idle",
    "pointing": "idle",
    "idle": "idle",
    "unknown": "idle",
    "未检测到手部": "idle",
}

GESTURE_DISPLAY_MAP: dict[str, str] = {
    "open_palm": "张开手掌",
    "palm": "张开手掌",
    "fist": "握拳",
    "point": "待机",
    "pointing": "待机",
    "index_circle": "单指画圈",
    "circle_cw": "顺时针画圈",
    "circle_ccw": "逆时针画圈",
    "swipe_left": "张开拳头",
    "swipe_right": "收回拳头",
    "thumbs_up": "拇指向上",
    "thumb_up": "拇指向上",
    "thumbs_down": "拇指向下",
    "thumb_down": "拇指向下",
    "thunb_index": "捏指",
    "thumb_index": "捏指",
    "wave": "挥手",
    "idle": "待机",
    "unknown": "未识别",
    "未检测到手部": "未检测到手部",
}

COMMAND_DISPLAY_MAP: dict[str, str] = {
    "WakeSystem": "系统唤醒",
    "ConfirmAction": "确认执行",
    "AdjustVolume": "音量调节",
    "AdjustVolumeUp": "音量升高",
    "AdjustVolumeDown": "音量降低",
    "SwitchPrevFeature": "切换上一个功能",
    "SwitchNextFeature": "切换下一个功能",
    "AnswerCall": "接听电话",
    "HangUpCall": "挂断电话",
    "ReturnHome": "返回主页",
}

GESTURE_ACTION_DISPLAY_MAP: dict[str, str] = {
    "wake": "唤醒",
    "confirm": "确认",
    "volume_adjust": "音量调节",
    "volume_up": "音量增加",
    "volume_down": "音量降低",
    "prev_func": "切换上一个功能",
    "next_func": "切换下一个功能",
    "call_answer": "接听",
    "call_hangup": "挂断",
    "home": "返回主页",
    "idle": "等待动作",
}

GESTURE_LOG_LABEL_MAP: dict[str, str] = {
    "open_palm": "张开手掌",
    "palm": "张开手掌",
    "fist": "握拳",
    "point": "待机",
    "pointing": "待机",
    "index_circle": "画圈",
    "circle_cw": "顺时针画圈",
    "circle_ccw": "逆时针画圈",
    "swipe_left": "张开拳头",
    "swipe_right": "收回拳头",
    "thumbs_up": "竖起大拇指",
    "thumb_up": "竖起大拇指",
    "thumbs_down": "倒拇指",
    "thumb_down": "倒拇指",
    "thunb_index": "捏指",
    "thumb_index": "捏指",
    "wave": "挥手",
    "idle": "待机",
    "unknown": "未知",
    "未检测到手部": "未检测到手部",
}

COMMAND_LOG_LABEL_MAP: dict[str, str] = {
    "WakeSystem": "唤醒系统",
    "ConfirmAction": "确认操作",
    "AdjustVolume": "调节音量",
    "AdjustVolumeUp": "提高音量",
    "AdjustVolumeDown": "降低音量",
    "SwitchPrevFeature": "切换上一个功能",
    "SwitchNextFeature": "切换下一个功能",
    "AnswerCall": "接听电话",
    "HangUpCall": "挂断电话",
    "ReturnHome": "返回主页",
    "None": "无",
}

VOLUME_GESTURES = {"circle_cw", "circle_ccw"}
VOLUME_ACTIONS = {"volume_up", "volume_down"}
SWITCH_GESTURES = {"swipe_left", "swipe_right"}
SWITCH_ENDPOINT_GESTURES = {"fist", "palm", "open_palm"}


@dataclass
class RuntimeGestureRecord:
    gesture: str
    control_action: str
    is_triggered: bool
    created_at: datetime
    hand_landmarks: list[dict] = field(default_factory=list)


@dataclass
class RuntimeGestureSession:
    panel_state: ControlPanelState
    recent_records: list[RuntimeGestureRecord] = field(default_factory=list)
    inactivity_started_at: datetime | None = None
    no_hand_count: int = 0
    hand_gone_confirmed: bool = False
    result_locked: bool = False
    locked_gesture: str | None = None
    locked_confidence: float = 0.0
    last_fired_action: str | None = None
    last_fired_at_monotonic: float = 0.0
    hand_present_prev: bool = False
    grace_count: int = 0


class LatestFrameCapture:
    """Continuously drain a capture and expose only its newest frame."""

    def __init__(self, capture: cv2.VideoCapture) -> None:
        self._capture = capture
        self._condition = threading.Condition()
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="owner-frame-reader")

    def start(self) -> None:
        self._thread.start()

    def read_latest(
        self,
        last_sequence: int,
        *,
        timeout: float = 1.0,
    ) -> tuple[int, np.ndarray | None]:
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > last_sequence or self._stopped,
                timeout=timeout,
            )
            if self._sequence <= last_sequence:
                return last_sequence, None
            return self._sequence, self._frame

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._capture.release()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    if self._stopped:
                        return
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue
                with self._condition:
                    self._frame = frame
                    self._sequence += 1
                    self._condition.notify_all()
        finally:
            with self._condition:
                self._stopped = True
                self._condition.notify_all()


class OwnerGestureService:
    _instance: ClassVar["OwnerGestureService | None"] = None
    _feature_modes = ("home", "media", "comfort", "vehicle")
    _hands: Optional["MediaPipeHands"] = None
    _classifier: Optional["GestureClassifier"] = None
    _stream_classifier: Optional["GestureClassifier"] = None
    _max_inference_edge = 640
    _max_annotated_edge = 1024
    _hold_frame_count = 2
    _trigger_cooldown = timedelta(seconds=2)
    _idle_behavior_interval = timedelta(seconds=30)
    _camera_grace_frames = 8
    _continuous_volume_interval_seconds = 0.55
    _stream_no_hand_warning_delay_seconds = 60
    _stream_low_confidence_window_seconds = 30
    _stream_high_confidence_log_gap_seconds = 1.5
    _default_panel_state = {
        "system_awake": False,
        "volume": 32,
        "climate_temperature": 24,
        "phone_call_active": False,
        "current_mode": "home",
        "media_playing": True,
        "comfort_scene": "标准",
        "vehicle_status": "就绪",
        "focus_tile": "home",
        "last_feedback": None,
    }

    def __init__(self) -> None:
        self._stream_lock = threading.Lock()
        self._frame_condition = threading.Condition()
        self._stream_thread: threading.Thread | None = None
        self._mediamtx_process: subprocess.Popen[bytes] | None = None
        self._owner_h264_encoder: str | None = None
        self._stream_running = False
        self._stream_source = ""
        self._stream_loop: asyncio.AbstractEventLoop | None = None
        self._ws_callbacks: list[tuple[Any, asyncio.AbstractEventLoop | None]] = []
        self._control_callbacks: list[Any] = []
        self._alert_callbacks: list[Any] = []
        self._runtime_sessions: dict[tuple[int, str], RuntimeGestureSession] = {}
        self._latest_stream_result: OwnerGestureResult | None = None
        self._latest_stream_jpeg: bytes | None = None
        self._stream_runtime_session: RuntimeGestureSession | None = None
        self._stream_state = StreamState(running=False)
        self._live_panel_state = ControlPanelState(
            **self._default_panel_state,
            last_gesture=None,
            last_command=None,
            last_command_at=None,
            updated_at=None,
        )
        self._camera_runtime_ready = False

    @classmethod
    def instance(cls) -> "OwnerGestureService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def hands(self) -> "MediaPipeHands":
        if self._hands is None:
            from app.models_infer.mediapipe_hands import MediaPipeHands

            self._hands = MediaPipeHands(
                num_hands=settings.num_hands,
                min_detection_confidence=settings.min_hand_detection_confidence,
                min_presence_confidence=settings.min_hand_presence_confidence,
                min_tracking_confidence=settings.min_hand_tracking_confidence,
            )
            logger.info("OwnerGestureService MediaPipeHands loaded for image inference")
        return self._hands

    @property
    def classifier(self) -> "GestureClassifier":
        if self._classifier is None:
            from app.models_infer.gesture_classifier import GestureClassifier

            self._classifier = GestureClassifier(domain="owner")
        return self._classifier

    @property
    def stream_classifier(self) -> "GestureClassifier":
        if self._stream_classifier is None:
            from app.models_infer.gesture_classifier import GestureClassifier

            self._stream_classifier = GestureClassifier(domain="owner")
        return self._stream_classifier

    @property
    def stream_state(self) -> StreamState:
        with self._stream_lock:
            return self._stream_state

    def register_ws_callback(
        self,
        callback: Any,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        callback_loop = loop
        if callback_loop is None:
            try:
                callback_loop = asyncio.get_running_loop()
            except RuntimeError:
                callback_loop = None
        self._ws_callbacks.append((callback, callback_loop))

    def unregister_ws_callback(self, callback: Any) -> None:
        self._ws_callbacks = [(cb, loop) for cb, loop in self._ws_callbacks if cb is not callback]

    def register_control_callback(self, callback: Any) -> None:
        self._control_callbacks.append(callback)

    def register_alert_callback(self, callback: Any) -> None:
        self._alert_callbacks.append(callback)

    async def process_frame(
        self,
        image_bytes: bytes,
        filename: str,
        *,
        db: Session,
        user_id: int,
        session_id: str | None = None,
        input_mode: str = "camera",
    ) -> GestureFrameResult:
        started_at = perf_counter()

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            await self._capture_error(filename, "owner_gesture_decode_error", "无法解析图像字节数据。")
            raise ValueError(f"无法解析图像文件：{filename}")

        original_frame = frame
        if input_mode == "camera":
            frame = self._prepare_frame_for_inference(frame)
        logger.info("Processing owner gesture frame '%s' (%dx%d)", filename, frame.shape[1], frame.shape[0])

        verify_draw_result = None
        verify_gesture_label = None
        if input_mode == "camera":
            verify_process_frame, verify_draw_result, verify_gesture_label, _ = self._verify_owner_runtime()
            raw_gesture_label, _, cls_conf, num_hands, detected_hand = verify_process_frame(
                frame,
                self.stream_classifier,
            )
            observed_gesture_label = raw_gesture_label
            primary_hand = detected_hand or []
            hands = [primary_hand] if primary_hand else []
            raw_kps = list(primary_hand)
        else:
            infer_result = self.hands.infer(original_frame)
            raw_kps = infer_result["keypoints"]
            num_hands = infer_result.get("num_hands_detected", 0)
            hands = self._group_hands(raw_kps)

        active_session_id = session_id or uuid4().hex[:16]
        runtime_session: RuntimeGestureSession | None = None
        if input_mode == "camera":
            runtime_session = self._runtime_session(user_id=user_id, session_id=active_session_id)
            recent_records = list(reversed(runtime_session.recent_records))
        else:
            recent_records = self._recent_session_records(db, user_id=user_id, session_id=active_session_id)
        previous_record = recent_records[0] if recent_records else None
        if input_mode == "image":
            primary_hand = hands[0] if hands else []
            image_gesture = None
            image_confidence = None
        elif input_mode != "camera":
            primary_hand, image_gesture, image_confidence = self._select_primary_hand(
                hands,
                input_mode=input_mode,
            )

        if input_mode == "camera":
            raw_gesture_label, cls_conf = self._lock_camera_result(
                runtime_session,
                gesture=raw_gesture_label,
                confidence=cls_conf,
                hand_present=bool(primary_hand),
            )
        elif image_gesture is not None and image_confidence is not None:
            raw_gesture_label, cls_conf = image_gesture, image_confidence
        else:
            raw_gesture_label, cls_conf = self._classify_upload_gesture(
                raw_keypoints=primary_hand,
                num_hands=num_hands,
                recent_records=recent_records,
                input_mode=input_mode,
            )

        control_command, _ = self._map_gesture_to_command(raw_gesture_label)
        action = GESTURE_ACTION_MAP.get(raw_gesture_label, "idle")
        gesture_label = self._normalize_public_gesture(raw_gesture_label)
        if input_mode == "camera" and runtime_session is not None:
            triggered = self._should_fire_verified_camera_action(
                runtime_session,
                action=action,
                control_command=control_command,
                observed_gesture=observed_gesture_label,
            )
        else:
            triggered = self._should_trigger(
                gesture=gesture_label,
                control_command=control_command,
                recent_records=recent_records,
                input_mode=input_mode,
            )
        processing_time_ms = int((perf_counter() - started_at) * 1000)
        if input_mode == "camera" and verify_draw_result is not None and verify_gesture_label is not None:
            annotated_frame = verify_draw_result(
                original_frame.copy(),
                raw_gesture_label,
                verify_gesture_label(raw_gesture_label),
                cls_conf,
                num_hands,
                primary_hand,
                locked=bool(runtime_session and runtime_session.result_locked),
            )
            annotated_image = self._encode_frame_to_data_url(annotated_frame)
        else:
            display_hands = [primary_hand] if primary_hand else hands
            annotated_image = self._build_annotated_image(
                original_frame,
                hands=display_hands,
                gesture=gesture_label,
                confidence=cls_conf,
                action=action,
            )

        response_keypoints_source = primary_hand if primary_hand else raw_kps
        keypoints = [
            Keypoint(x=kp["x"], y=kp["y"], score=kp.get("z", 0.0))
            for kp in response_keypoints_source
        ]
        updated_at = datetime.utcnow()
        if input_mode == "camera" and runtime_session is not None:
            panel_state = self._update_runtime_session(
                runtime_session,
                gesture=gesture_label,
                control_command=control_command if triggered else None,
                updated_at=updated_at,
            )
            idle_behavior_due = self._consume_idle_behavior_interval(
                runtime_session,
                gesture=gesture_label,
                control_command=control_command if triggered else None,
                observed_at=updated_at,
            )
            runtime_session.recent_records.append(
                RuntimeGestureRecord(
                    gesture=gesture_label,
                    control_action=control_command or "None",
                    is_triggered=triggered,
                    created_at=updated_at,
                    hand_landmarks=raw_kps,
                )
            )
            if len(runtime_session.recent_records) > 12:
                runtime_session.recent_records = runtime_session.recent_records[-12:]
            if (
                triggered
                or idle_behavior_due
                or self._should_persist_camera_record(previous_record=previous_record, gesture=gesture_label)
            ):
                self._persist_gesture_record(
                    db,
                    user_id=user_id,
                    session_id=active_session_id,
                    gesture=gesture_label,
                    confidence=cls_conf,
                    control_command=control_command,
                    triggered=triggered,
                    processing_time_ms=processing_time_ms,
                    raw_kps=raw_kps,
                    previous_record=previous_record,
                    num_hands=num_hands,
                    idle_behavior_due=idle_behavior_due,
                )
        else:
            self._persist_gesture_record(
                db,
                user_id=user_id,
                session_id=active_session_id,
                gesture=gesture_label,
                confidence=cls_conf,
                control_command=control_command,
                triggered=triggered,
                processing_time_ms=processing_time_ms,
                raw_kps=raw_kps,
                previous_record=previous_record,
                num_hands=num_hands,
                idle_behavior_due=False,
            )
            panel_state = self._build_panel_state(db, user_id=user_id, session_id=active_session_id)
            await self._capture_monitor_log(
                event_type=(
                    "owner_gesture_success"
                    if cls_conf >= settings.alert_low_confidence_threshold
                    else "owner_gesture_low_confidence"
                ),
                title="车主手势帧处理完成",
                summary=f"{filename} 已处理完成，置信度为 {cls_conf:.2f}，检测到 {num_hands} 只手。",
                confidence=cls_conf,
                details={
                    "filename": filename,
                    "num_hands_detected": num_hands,
                    "frame_width": int(frame.shape[1]),
                    "frame_height": int(frame.shape[0]),
                    "triggered": triggered,
                    "control_command": control_command,
                },
                trigger_alert=cls_conf < settings.alert_low_confidence_threshold,
                level="info" if cls_conf >= settings.alert_low_confidence_threshold else "warning",
            )

        return GestureFrameResult(
            gesture=gesture_label,
            action=action,
            confidence=round(cls_conf, 4),
            keypoints=keypoints,
            annotated_image=annotated_image,
            control_command=control_command,
            triggered=triggered,
            panel_state=panel_state,
            updated_at=updated_at,
        )

    def start(self, source: str, fps: int = 15) -> StreamState:
        with self._stream_lock:
            if self._stream_running:
                return self._stream_state

            self._configure_stream_runtime()
            self._stream_running = True
            self._stream_source = source
            self._latest_stream_result = None
            self._latest_stream_jpeg = None
            self._stream_runtime_session = RuntimeGestureSession(panel_state=self._live_panel_state)
            self._live_panel_state = ControlPanelState(
                **self._default_panel_state,
                last_gesture=None,
                last_command=None,
                last_command_at=None,
                updated_at=None,
            )

            self._stream_thread = threading.Thread(
                target=self._stream_loop_worker,
                args=(source, fps),
                daemon=True,
                name="owner-gesture-stream",
            )
            self._stream_thread.start()
            self._stream_state = StreamState(
                running=True,
                source=source,
                fps=fps,
                published=False,
                publish_rtsp_url=self._owner_stream_publish_url(),
                playback_url=self._owner_stream_playback_url(),
                started_at=datetime.now(timezone.utc),
            )
            return self._stream_state

    def stop(self) -> StreamState:
        with self._stream_lock:
            self._stream_running = False

        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=3.0)

        with self._stream_lock:
            self._stream_state = StreamState(running=False)
            with self._frame_condition:
                self._frame_condition.notify_all()
            return self._stream_state

    def mjpeg_frames(self):
        last_frame: bytes | None = None
        while True:
            with self._frame_condition:
                self._frame_condition.wait_for(
                    lambda: self._latest_stream_jpeg is not None
                    and self._latest_stream_jpeg is not last_frame
                    or not self._stream_running,
                    timeout=1.0,
                )
                frame = self._latest_stream_jpeg
                running = self._stream_running
            if frame is not None and frame is not last_frame:
                last_frame = frame
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            if not running and frame is last_frame:
                break

    def current_stream_result(self) -> OwnerGestureResult:
        with self._stream_lock:
            if self._latest_stream_result is not None:
                return self._latest_stream_result
            return OwnerGestureResult(
                gesture="unknown",
                action="idle",
                confidence=0.0,
                keypoints=[],
                annotated_image=None,
                hand_count=0,
                control_command=None,
                triggered=False,
                panel_state=self._live_panel_state,
                updated_at=datetime.now(timezone.utc),
            )

    def control_panel(
        self,
        db: Session,
        user_id: int,
        *,
        session_id: str | None = None,
    ) -> ControlPanelState:
        runtime_panel = self._latest_runtime_panel_state(user_id=user_id, session_id=session_id)
        if runtime_panel is not None:
            return runtime_panel
        return self._build_panel_state(db, user_id=user_id, session_id=session_id)

    def _configure_stream_runtime(self) -> None:
        from app.models_infer.gesture_classifier import GestureClassifier
        from app.models_infer.mediapipe_hands import MediaPipeHands

        MediaPipeHands.reset()
        MediaPipeHands.configure(
            settings.resolved_hand_model_path,
            num_hands=settings.num_hands,
            min_detection_confidence=settings.min_hand_detection_confidence,
            min_presence_confidence=settings.min_hand_presence_confidence,
            min_tracking_confidence=settings.min_hand_tracking_confidence,
        )
        self._stream_classifier = GestureClassifier(domain="owner")

    def _infer_camera_hands(self, frame: np.ndarray) -> list[list[dict]]:
        if not self._camera_runtime_ready:
            self._camera_runtime_ready = True

        infer_result = self.hands.infer(frame)
        return self._group_hands(infer_result["keypoints"])

    def _stream_loop_worker(self, source: str, fps: int) -> None:
        from app.models_infer.mediapipe_hands import MediaPipeHands
        verify_process_frame, verify_draw_result, verify_gesture_label, _ = self._verify_owner_runtime()

        capture = cv2.VideoCapture(self._resolve_stream_source(source))
        if not capture.isOpened():
            logger.warning("Failed to open owner-gesture stream source: %s", source)
            with self._stream_lock:
                self._stream_running = False
                self._stream_state = StreamState(
                    running=False,
                    source=source,
                    fps=fps,
                    published=False,
                    last_error=f"无法打开后端摄像头 {source}，请确认摄像头未被其他软件占用。",
                )
            return

        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        capture.set(cv2.CAP_PROP_FPS, fps)
        frame_reader = LatestFrameCapture(capture)
        frame_reader.start()
        last_capture_sequence = 0

        frame_interval = 1.0 / max(fps, 1)
        no_hand_started_at: float | None = None
        low_confidence_hits: deque[float] = deque()
        last_high_confidence_log_at = 0.0
        last_high_confidence_gesture: str | None = None

        try:
            while True:
                with self._stream_lock:
                    if not self._stream_running:
                        break

                tick_started = time.time()
                last_capture_sequence, frame_bgr = frame_reader.read_latest(
                    last_capture_sequence,
                    timeout=0.5,
                )
                if frame_bgr is None:
                    continue

                try:
                    inference_frame = self._prepare_frame_for_inference(frame_bgr)
                    raw_gesture, _, confidence, hand_count, primary_hand = verify_process_frame(
                        inference_frame,
                        self.stream_classifier,
                    )
                    observed_gesture = raw_gesture
                    raw_gesture, confidence = self._lock_camera_result(
                        self._stream_runtime_session,
                        gesture=raw_gesture,
                        confidence=confidence,
                        hand_present=bool(primary_hand),
                    )
                    now_monotonic = time.monotonic()
                    no_hand_started_at = self._handle_owner_stream_monitor_event(
                        no_hand_started_at=no_hand_started_at,
                        low_confidence_hits=low_confidence_hits,
                        source=source,
                        gesture=raw_gesture,
                        confidence=confidence,
                        hand_count=hand_count,
                        now=now_monotonic,
                    )
                    if self._should_log_owner_high_confidence_stream_gesture(
                        gesture=raw_gesture,
                        confidence=confidence,
                        now=now_monotonic,
                        last_gesture=last_high_confidence_gesture,
                        last_logged_at=last_high_confidence_log_at,
                    ):
                        self._capture_monitor_log_sync(
                            event_type="owner_gesture_success",
                            title="手势控车识别成功",
                            summary=(
                                f"实时视频识别到高置信率手势："
                                f"{GESTURE_LOG_LABEL_MAP.get(raw_gesture, raw_gesture)}，"
                                f"置信度 {confidence:.2f}。"
                            ),
                            confidence=confidence,
                            details={
                                "source": source,
                                "gesture": raw_gesture,
                                "hand_count": hand_count,
                                "input_mode": "stream",
                            },
                            trigger_alert=False,
                            level="info",
                        )
                        last_high_confidence_log_at = now_monotonic
                        last_high_confidence_gesture = raw_gesture
                    control_command, _ = self._map_gesture_to_command(raw_gesture)
                    action = GESTURE_ACTION_MAP.get(raw_gesture, "idle")
                    gesture = self._normalize_public_gesture(raw_gesture)
                    keypoints = (
                        [Keypoint(x=point["x"], y=point["y"], score=point.get("z", 0.0)) for point in primary_hand]
                        if primary_hand
                        else []
                    )

                    should_fire = self._should_fire_verified_camera_action(
                        self._stream_runtime_session,
                        action=action,
                        control_command=control_command,
                        observed_gesture=observed_gesture,
                    )

                    panel_state = self._update_live_panel_state(
                        gesture=gesture,
                        control_command=control_command if should_fire else None,
                        updated_at=datetime.now(timezone.utc),
                    )
                    annotated_frame = verify_draw_result(
                        inference_frame.copy(),
                        raw_gesture,
                        verify_gesture_label(raw_gesture),
                        confidence,
                        hand_count,
                        primary_hand,
                        locked=bool(
                            self._stream_runtime_session
                            and self._stream_runtime_session.result_locked
                        ),
                    )
                    annotated_frame = self._resize_stream_frame(annotated_frame)
                    jpeg_frame = self._encode_frame_to_jpeg(annotated_frame)
                    if jpeg_frame is not None:
                        with self._frame_condition:
                            self._latest_stream_jpeg = jpeg_frame
                            self._frame_condition.notify_all()
                        with self._stream_lock:
                            if not self._stream_state.published:
                                self._stream_state = self._stream_state.model_copy(update={"published": True})
                    result = OwnerGestureResult(
                        gesture=gesture,
                        action=action,
                        confidence=round(confidence, 4),
                        keypoints=keypoints,
                        annotated_image=None,
                        hand_count=hand_count,
                        control_command=control_command if should_fire else None,
                        triggered=should_fire,
                        panel_state=panel_state,
                        updated_at=datetime.now(timezone.utc),
                    )

                    with self._stream_lock:
                        self._latest_stream_result = result
                    if should_fire:
                        for callback in self._control_callbacks:
                            try:
                                callback(action)
                            except Exception:
                                logger.debug("Owner gesture control callback failed", exc_info=True)
                        for callback in self._alert_callbacks:
                            try:
                                callback(gesture, confidence)
                            except Exception:
                                logger.debug("Owner gesture alert callback failed", exc_info=True)

                    self._emit_ws_payload(result.model_dump(mode="json"))
                except Exception as exc:
                    logger.debug("Owner gesture stream frame processing failed", exc_info=True)
                    with self._stream_lock:
                        self._stream_state = self._stream_state.model_copy(update={"last_error": str(exc)})

                elapsed = time.time() - tick_started
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            frame_reader.stop()
            MediaPipeHands.reset()
            with self._stream_lock:
                self._stream_running = False
                self._stream_state = self._stream_state.model_copy(update={"running": False, "published": False})
            with self._frame_condition:
                self._frame_condition.notify_all()

    def _emit_ws_payload(self, payload: dict[str, Any]) -> None:
        for callback, loop in list(self._ws_callbacks):
            if loop is None:
                continue
            try:
                asyncio.run_coroutine_threadsafe(callback(payload), loop)
            except Exception:
                logger.debug("Owner gesture websocket callback failed", exc_info=True)

    def _resolve_stream_source(self, source: str) -> str | int:
        stripped = source.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped

    def _is_meaningful_owner_gesture(self, gesture: str | None) -> bool:
        return str(gesture or "").strip() not in {
            "",
            "unknown",
            "idle",
            "未检测到手部",
            "鏈娴嬪埌鎵嬮儴",
        }

    def _prune_owner_low_confidence_hits(self, hits: deque[float], now: float) -> None:
        while hits and now - hits[0] > self._stream_low_confidence_window_seconds:
            hits.popleft()

    def _handle_owner_stream_monitor_event(
        self,
        *,
        no_hand_started_at: float | None,
        low_confidence_hits: deque[float],
        source: str,
        gesture: str,
        confidence: float,
        hand_count: int,
        now: float,
    ) -> float | None:
        if hand_count <= 0:
            low_confidence_hits.clear()
            if no_hand_started_at is None:
                no_hand_started_at = now
            no_hand_seconds = now - no_hand_started_at
            if no_hand_seconds >= self._stream_no_hand_warning_delay_seconds:
                self._capture_monitor_log_sync(
                    event_type="owner_gesture_no_hand",
                    title="手势控车长时间未识别到手",
                    summary=f"实时视频已连续 {int(no_hand_seconds)} 秒未识别到手部。",
                    confidence=0.0,
                    details={
                        "source": source,
                        "input_mode": "stream",
                        "duration_seconds": round(no_hand_seconds, 2),
                        "hand_count": hand_count,
                        "gesture": gesture,
                    },
                    trigger_alert=True,
                    level="warning",
                )
                return now
            return no_hand_started_at

        no_hand_started_at = None
        if not self._is_meaningful_owner_gesture(gesture):
            low_confidence_hits.clear()
            return no_hand_started_at

        if confidence >= settings.alert_low_confidence_threshold:
            low_confidence_hits.clear()
            return no_hand_started_at

        self._prune_owner_low_confidence_hits(low_confidence_hits, now)
        low_confidence_hits.append(now)
        threshold = max(int(settings.alert_low_confidence_window_size), 1)
        if len(low_confidence_hits) >= threshold:
            count = len(low_confidence_hits)
            low_confidence_hits.clear()
            self._capture_monitor_log_sync(
                event_type="owner_gesture_low_confidence",
                title="手势控车低置信度连续出现",
                summary=(
                    f"实时视频在 {self._stream_low_confidence_window_seconds} 秒内"
                    f"连续 {count} 次识别到低置信度手势："
                    f"{GESTURE_LOG_LABEL_MAP.get(gesture, gesture)}，当前置信度 {confidence:.2f}。"
                ),
                confidence=confidence,
                details={
                    "source": source,
                    "input_mode": "stream",
                    "gesture": gesture,
                    "hand_count": hand_count,
                    "count": count,
                    "window_seconds": self._stream_low_confidence_window_seconds,
                    "confidence_threshold": settings.alert_low_confidence_threshold,
                },
                trigger_alert=True,
                level="warning",
            )
        return no_hand_started_at

    def _should_log_owner_high_confidence_stream_gesture(
        self,
        *,
        gesture: str,
        confidence: float,
        now: float,
        last_gesture: str | None,
        last_logged_at: float,
    ) -> bool:
        if not self._is_meaningful_owner_gesture(gesture):
            return False
        if confidence < settings.alert_low_confidence_threshold:
            return False
        return gesture != last_gesture or now - last_logged_at >= self._stream_high_confidence_log_gap_seconds

    def _ensure_mediamtx_running(self) -> None:
        if self._is_tcp_port_open("127.0.0.1", 8554):
            return

        executable = Path(settings.owner_gesture_mediamtx_bin)
        if not executable.is_file():
            raise RuntimeError(f"找不到 MediaMTX：{executable}")

        config_path = executable.with_name("mediamtx.yml")
        command = [str(executable)]
        if config_path.is_file():
            command.append(str(config_path))

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._mediamtx_process = subprocess.Popen(
            command,
            cwd=str(executable.parent),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self._is_tcp_port_open("127.0.0.1", 8554):
                logger.info("MediaMTX started from %s", executable)
                return
            if self._mediamtx_process.poll() is not None:
                break
            time.sleep(0.1)
        raise RuntimeError("MediaMTX 启动失败，RTSP 端口 8554 未就绪。")

    def _is_tcp_port_open(self, host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return True
        except OSError:
            return False

    def _start_owner_stream_publisher(
        self,
        *,
        width: int,
        height: int,
        fps: int,
    ) -> subprocess.Popen[bytes]:
        encoder = self._resolve_owner_h264_encoder()
        command = [
            settings.owner_gesture_ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-use_wallclock_as_timestamps",
            "1",
            "-fflags",
            "+nobuffer+discardcorrupt",
            "-i",
            "-",
            "-flags",
            "low_delay",
            "-c:v",
            encoder,
            "-pix_fmt",
            "yuv420p",
        ]
        if encoder == "h264_qsv":
            command.extend(
                [
                    "-preset", "veryfast",
                    "-look_ahead", "0",
                    "-async_depth", "1",
                    "-global_quality", "24",
                ]
            )
        else:
            command.extend(
                [
                    "-preset", "ultrafast",
                    "-tune", "zerolatency",
                    "-x264-params",
                    "bframes=0:sync-lookahead=0:rc-lookahead=0:sliced-threads=1:force-cfr=0",
                    "-threads", "1",
                ]
            )
        command.extend(
            [
                "-g", str(fps),
                "-keyint_min", str(fps),
                "-sc_threshold", "0",
                "-bf", "0",
                "-b:v", settings.owner_gesture_push_bitrate,
                "-fps_mode", "passthrough",
                "-flush_packets", "1",
                "-muxdelay", "0",
                "-muxpreload", "0",
                "-rtsp_transport", "udp",
                "-f", "rtsp",
                self._owner_stream_publish_url(),
            ]
        )
        try:
            return subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"找不到 FFmpeg：{settings.owner_gesture_ffmpeg_bin}") from exc

    def _resolve_owner_h264_encoder(self) -> str:
        if self._owner_h264_encoder is not None:
            return self._owner_h264_encoder

        probe = [
            settings.owner_gesture_ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=size=64x64:rate=1",
            "-frames:v",
            "1",
            "-c:v",
            "h264_qsv",
            "-f",
            "null",
            "NUL" if os.name == "nt" else "/dev/null",
        ]
        try:
            completed = subprocess.run(
                probe,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
            self._owner_h264_encoder = "h264_qsv" if completed.returncode == 0 else "libx264"
        except (FileNotFoundError, subprocess.SubprocessError):
            self._owner_h264_encoder = "libx264"
        logger.info("Owner gesture stream encoder selected: %s", self._owner_h264_encoder)
        return self._owner_h264_encoder

    def _resize_stream_frame(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        longest_edge = max(width, height)
        if longest_edge <= self._max_annotated_edge:
            return frame
        scale = self._max_annotated_edge / float(longest_edge)
        return cv2.resize(
            frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def _probe_owner_stream_ready(self) -> bool:
        publish_url = self._owner_stream_publish_url()
        request = (
            f"DESCRIBE {publish_url} RTSP/1.0\r\n"
            "CSeq: 1\r\n"
            "Accept: application/sdp\r\n"
            "User-Agent: owner-gesture-readiness-probe\r\n\r\n"
        ).encode("ascii")
        try:
            with socket.create_connection(("127.0.0.1", 8554), timeout=0.4) as connection:
                connection.settimeout(0.4)
                connection.sendall(request)
                response = connection.recv(512)
            return response.startswith(b"RTSP/1.0 200")
        except OSError:
            return False

    def _shutdown_owner_stream_publisher(self, process: subprocess.Popen[bytes] | None) -> None:
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=3)
        except Exception:
            process.kill()

    def _owner_stream_publish_url(self) -> str:
        return (
            f"{settings.owner_gesture_rtsp_base_url.rstrip('/')}"
            f"/{settings.owner_gesture_stream_name.strip('/')}"
        )

    def _owner_stream_playback_url(self) -> str:
        return (
            f"{settings.owner_gesture_playback_base_url.rstrip('/')}"
            f"/{settings.owner_gesture_stream_name.strip('/')}"
        )

    def _update_live_panel_state(
        self,
        *,
        gesture: str,
        control_command: str | None,
        updated_at: datetime,
    ) -> ControlPanelState:
        state = self._live_panel_state.model_dump()
        state["last_gesture"] = gesture
        state["updated_at"] = updated_at
        if control_command:
            self._apply_command_to_state(state, control_command)
            state["last_command"] = control_command
            state["last_command_at"] = updated_at
        self._live_panel_state = ControlPanelState(**state)
        return self._live_panel_state

    def _log_operation(self, db: Session, *, user_id: int, response_status: str) -> None:
        db.add(
            UserOperationLog(
                user_id=user_id,
                operation_type="owner_gesture_recognition",
                response_status=response_status,
            )
        )

    def _map_gesture_to_command(self, gesture: str) -> tuple[str | None, bool]:
        command_map = {
            "open_palm": ("WakeSystem", True),
            "palm": ("WakeSystem", True),
            "fist": ("ConfirmAction", True),
            "circle_cw": ("AdjustVolumeUp", True),
            "circle_ccw": ("AdjustVolumeDown", True),
            "swipe_left": ("SwitchPrevFeature", True),
            "swipe_right": ("SwitchNextFeature", True),
            "thumbs_up": ("AnswerCall", True),
            "thumb_up": ("AnswerCall", True),
            "thumbs_down": ("HangUpCall", True),
            "thumb_down": ("HangUpCall", True),
            "thunb_index": ("ReturnHome", True),
            "thumb_index": ("ReturnHome", True),
        }
        return command_map.get(gesture, (None, False))

    def _recent_session_records(
        self,
        db: Session,
        *,
        user_id: int,
        session_id: str,
        limit: int = 6,
    ) -> list[OwnerGestureRecord]:
        return db.scalars(
            select(OwnerGestureRecord)
            .where(
                OwnerGestureRecord.user_id == user_id,
                OwnerGestureRecord.session_id == session_id,
            )
            .order_by(OwnerGestureRecord.created_at.desc())
            .limit(limit)
        ).all()

    def _runtime_session(self, *, user_id: int, session_id: str) -> RuntimeGestureSession:
        key = (user_id, session_id)
        session = self._runtime_sessions.get(key)
        if session is None:
            session = RuntimeGestureSession(
                panel_state=ControlPanelState(
                    **self._default_panel_state,
                    last_gesture=None,
                    last_command=None,
                    last_command_at=None,
                    updated_at=None,
                )
            )
            self._runtime_sessions[key] = session
        return session

    def _latest_runtime_panel_state(
        self,
        *,
        user_id: int,
        session_id: str | None = None,
    ) -> ControlPanelState | None:
        if session_id is not None:
            session = self._runtime_sessions.get((user_id, session_id))
            return session.panel_state if session is not None else None

        latest_session: ControlPanelState | None = None
        latest_at: datetime | None = None
        for (runtime_user_id, _), session in self._runtime_sessions.items():
            if runtime_user_id != user_id:
                continue
            updated_at = session.panel_state.updated_at
            if updated_at is None:
                continue
            if latest_at is None or updated_at > latest_at:
                latest_session = session.panel_state
                latest_at = updated_at
        return latest_session

    def _update_runtime_session(
        self,
        session: RuntimeGestureSession,
        *,
        gesture: str,
        control_command: str | None,
        updated_at: datetime,
    ) -> ControlPanelState:
        state = session.panel_state.model_dump()
        state["last_gesture"] = gesture
        state["updated_at"] = updated_at
        if control_command:
            self._apply_command_to_state(state, control_command)
            state["last_command"] = control_command
            state["last_command_at"] = updated_at
        session.panel_state = ControlPanelState(**state)
        return session.panel_state

    def _should_trigger(
        self,
        *,
        gesture: str,
        control_command: str | None,
        recent_records: list[OwnerGestureRecord],
        input_mode: str = "camera",
    ) -> bool:
        if not control_command:
            return False

        if input_mode == "image":
            return True

        required_hold_count = 1 if control_command in {
            "AdjustVolume",
            "AdjustVolumeUp",
            "AdjustVolumeDown",
            "SwitchPrevFeature",
            "SwitchNextFeature",
            "ReturnHome",
        } else self._hold_frame_count

        consecutive_same = 0
        for record in recent_records:
            if record.gesture != gesture:
                break
            consecutive_same += 1

        if consecutive_same < required_hold_count - 1:
            return False

        now = datetime.utcnow()
        for record in recent_records:
            if (
                record.is_triggered
                and record.gesture == gesture
                and record.control_action == control_command
                and now - record.created_at < self._trigger_cooldown
            ):
                return False
        return True

    def _should_log_operation(
        self,
        *,
        previous_record: OwnerGestureRecord | None,
        gesture: str,
        triggered: bool,
    ) -> bool:
        if triggered:
            return True
        if previous_record is None:
            return True
        return previous_record.gesture != gesture

    def _should_record_behavior(
        self,
        *,
        previous_record: OwnerGestureRecord | None,
        gesture: str,
        triggered: bool,
    ) -> bool:
        if triggered:
            return True
        if previous_record is None:
            return True
        return previous_record.gesture != gesture

    def _should_persist_camera_record(
        self,
        *,
        previous_record: RuntimeGestureRecord | OwnerGestureRecord | None,
        gesture: str,
    ) -> bool:
        if previous_record is None:
            return True
        return previous_record.gesture != gesture

    def _consume_idle_behavior_interval(
        self,
        session: RuntimeGestureSession,
        *,
        gesture: str,
        control_command: str | None,
        observed_at: datetime,
    ) -> bool:
        if control_command or not self._is_idle_behavior_gesture(gesture):
            session.inactivity_started_at = None
            return False

        if session.inactivity_started_at is None:
            session.inactivity_started_at = observed_at
            return False

        if observed_at - session.inactivity_started_at < self._idle_behavior_interval:
            return False

        session.inactivity_started_at = observed_at
        return True

    def _is_idle_behavior_gesture(self, gesture: str) -> bool:
        return gesture in {"idle", "unknown", "未检测到手部", "point"}

    def _persist_gesture_record(
        self,
        db: Session,
        *,
        user_id: int,
        session_id: str,
        gesture: str,
        confidence: float,
        control_command: str | None,
        triggered: bool,
        processing_time_ms: int,
        raw_kps: list[dict],
        previous_record: RuntimeGestureRecord | OwnerGestureRecord | None,
        num_hands: int,
        idle_behavior_due: bool,
    ) -> None:
        record_keypoints = [Keypoint(x=kp["x"], y=kp["y"], score=kp.get("z", 0.0)) for kp in raw_kps]
        record = OwnerGestureRecord(
            user_id=user_id,
            session_id=session_id,
            gesture=gesture,
            confidence=round(confidence, 4),
            control_action=control_command or "None",
            hand_landmarks=[kp.model_dump() for kp in record_keypoints],
            is_triggered=triggered,
            processing_time_ms=processing_time_ms,
        )
        db.add(record)
        if self._should_log_operation(
            previous_record=previous_record,
            gesture=gesture,
            triggered=triggered,
        ):
            self._log_operation(
                db,
                user_id=user_id,
                response_status="Success" if num_hands > 0 else "NoHandDetected",
            )
        db.commit()
        db.refresh(record)
        if idle_behavior_due or self._should_record_behavior(
            previous_record=previous_record,
            gesture=gesture,
            triggered=triggered,
        ):
            AlertService(db).record_behavior(
                source="owner-gesture",
                title=(
                    "手势控车识别完成"
                    if triggered
                    else "手势控车识别长时无动作" if idle_behavior_due else "手势控车识别更新"
                ),
                summary=self._build_behavior_summary(
                    gesture=gesture,
                    control_command=control_command,
                    triggered=triggered,
                    processing_time_ms=processing_time_ms,
                    idle_behavior_due=idle_behavior_due,
                ),
            )

    def _prepare_frame_for_inference(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        longest_edge = max(width, height)
        if longest_edge <= self._max_inference_edge:
            return frame

        scale = self._max_inference_edge / float(longest_edge)
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        return cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    def _group_hands(self, keypoints: list[dict]) -> list[list[dict]]:
        if not keypoints:
            return []
        grouped: list[list[dict]] = []
        for start in range(0, len(keypoints), 21):
            hand = keypoints[start:start + 21]
            if len(hand) == 21:
                grouped.append(hand)
        return grouped

    def _select_primary_hand(
        self,
        hands: list[list[dict]],
        *,
        input_mode: str,
    ) -> tuple[list[dict], str | None, float | None]:
        if not hands:
            return [], None, None
        if len(hands) == 1:
            return hands[0], None, None
        if input_mode == "camera":
            return hands[0], None, None

        best_hand = hands[0]
        best_gesture = "unknown"
        best_confidence = 0.0
        best_rank = (-1, -1.0, -1.0)

        for hand in hands:
            gesture, confidence = self.classifier.classify_static(hand[:21])
            rank = self._rank_image_hand_candidate(hand, gesture=gesture, confidence=confidence)
            if rank > best_rank:
                best_hand = hand
                best_gesture = gesture
                best_confidence = confidence
                best_rank = rank

        if input_mode == "image":
            return best_hand, best_gesture, best_confidence
        return best_hand, None, None

    def _rank_image_hand_candidate(
        self,
        hand: list[dict],
        *,
        gesture: str,
        confidence: float,
    ) -> tuple[int, float, float]:
        control_command, _ = self._map_gesture_to_command(gesture)
        if control_command:
            gesture_priority = 3
        elif gesture not in {"unknown", "idle", "未检测到手部", "point"}:
            gesture_priority = 2
        elif gesture == "point":
            gesture_priority = 1
        else:
            gesture_priority = 0
        return gesture_priority, confidence, self._hand_bbox_area(hand)

    def _hand_bbox_area(self, hand: list[dict]) -> float:
        if not hand:
            return 0.0
        xs = [float(point["x"]) for point in hand]
        ys = [float(point["y"]) for point in hand]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    def _classify_upload_gesture(
        self,
        *,
        raw_keypoints: list[dict],
        num_hands: int,
        recent_records: list[OwnerGestureRecord],
        input_mode: str,
    ) -> tuple[str, float]:
        if num_hands == 0 or len(raw_keypoints) < 21:
            return "未检测到手部", 0.0

        if input_mode == "image":
            return self.classifier.classify_static(raw_keypoints[:21])

        cls_result = self.classifier.classify(raw_keypoints[:21], domain="owner")
        gesture_label = self._refine_motion_gesture(
            gesture=cls_result["gesture"],
            raw_keypoints=raw_keypoints[:21],
            recent_records=recent_records,
        )
        return gesture_label, cls_result["confidence"]

    def _normalize_public_gesture(self, gesture: str) -> str:
        if gesture in {"point", "pointing", "index_circle"}:
            return "idle"
        return gesture

    def _lock_camera_result(
        self,
        session: RuntimeGestureSession | None,
        *,
        gesture: str,
        confidence: float,
        hand_present: bool,
    ) -> tuple[str, float]:
        if session is None:
            return gesture, confidence

        if hand_present != session.hand_present_prev:
            session.grace_count = self._camera_grace_frames
            session.result_locked = False
            session.locked_gesture = None
            session.locked_confidence = 0.0
            if hand_present:
                session.last_fired_action = None
                session.last_fired_at_monotonic = 0.0
        session.hand_present_prev = hand_present

        if session.grace_count > 0:
            session.grace_count -= 1
            return "idle", 0.0

        if not hand_present:
            return "idle", 0.0

        if session.result_locked:
            if (
                session.locked_gesture in SWITCH_ENDPOINT_GESTURES
                and gesture in SWITCH_GESTURES
            ):
                session.locked_gesture = gesture
                session.locked_confidence = confidence
            if (
                session.locked_gesture in VOLUME_GESTURES
                and gesture in VOLUME_GESTURES
                and gesture != session.locked_gesture
            ):
                session.locked_gesture = gesture
                session.locked_confidence = confidence
            return session.locked_gesture or gesture, session.locked_confidence

        if not session.result_locked:
            if gesture not in {"unknown", "idle"} and confidence > 0.0:
                session.result_locked = True
                session.locked_gesture = gesture
                session.locked_confidence = confidence

        if session.result_locked:
            return session.locked_gesture or gesture, session.locked_confidence
        return gesture, confidence

    def _verify_owner_runtime(self) -> tuple[Any, Any, Any, Any]:
        import sys

        project_root = Path(__file__).resolve().parents[3]
        project_root_str = str(project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)

        from verify_owner_gesture import draw_locked_marker, draw_result, gesture_label, process_frame

        return process_frame, draw_result, gesture_label, draw_locked_marker

    def _should_fire_verified_camera_action(
        self,
        session: RuntimeGestureSession | None,
        *,
        action: str,
        control_command: str | None,
        observed_gesture: str | None = None,
    ) -> bool:
        if session is None or action == "idle" or not control_command:
            return False
        now = time.monotonic()
        if action == session.last_fired_action:
            is_fresh_volume_observation = (
                action in VOLUME_ACTIONS
                and observed_gesture in VOLUME_GESTURES
            )
            if not is_fresh_volume_observation:
                return False
            if now - session.last_fired_at_monotonic < self._continuous_volume_interval_seconds:
                return False
        session.last_fired_action = action
        session.last_fired_at_monotonic = now
        return True

    def _draw_verified_locked_marker(
        self,
        frame: np.ndarray,
        session: RuntimeGestureSession | None,
    ) -> None:
        if session is None or not session.result_locked:
            return
        _, _, _, draw_locked_marker = self._verify_owner_runtime()
        locked_frame = draw_locked_marker(frame)
        if locked_frame is not frame:
            frame[:] = locked_frame

    def _draw_annotated_frame(
        self,
        frame: np.ndarray,
        *,
        hands: list[list[dict]],
        gesture: str,
        confidence: float,
        action: str | None = None,
        control_command: str | None = None,
    ) -> np.ndarray:
        annotated = frame.copy()
        height, width = annotated.shape[:2]
        longest_edge = max(width, height)
        header_height = max(118, min(176, int(height * 0.2)))
        title_font_size = 26 if longest_edge < 900 else 32 if longest_edge < 1400 else 40
        meta_font_size = 20 if longest_edge < 900 else 24 if longest_edge < 1400 else 30
        point_radius = 4 if longest_edge < 900 else 5 if longest_edge < 1400 else 7
        line_thickness = 1 if longest_edge < 900 else 2

        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (width, header_height), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0)

        for hand in hands:
            points: list[tuple[int, int]] = []
            for landmark in hand:
                x = int(min(max(float(landmark["x"]) * width, 0), width - 1))
                y = int(min(max(float(landmark["y"]) * height, 0), height - 1))
                points.append((x, y))

            for start_index, end_index in HAND_CONNECTIONS:
                cv2.line(
                    annotated,
                    points[start_index],
                    points[end_index],
                    (255, 255, 255),
                    line_thickness,
                    cv2.LINE_AA,
                )

            for index, point in enumerate(points):
                color = (0, 255, 255) if index in {4, 8, 12, 16, 20} else (255, 255, 255)
                cv2.circle(annotated, point, point_radius, color, -1, cv2.LINE_AA)

        gesture_text = GESTURE_DISPLAY_MAP.get(gesture, gesture)
        if action is None:
            action = GESTURE_ACTION_MAP.get(gesture, "idle")
        action_text = GESTURE_ACTION_DISPLAY_MAP.get(action, action)
        return draw_chinese_text_lines(
            annotated,
            [
                (f"手势: {gesture_text}", (10, 10), (0, 255, 0), title_font_size),
                (
                    f"功能: {action_text}",
                    (10, max(42, int(header_height * 0.46))),
                    (0, 215, 255),
                    meta_font_size,
                ),
                (
                    f"置信度: {confidence:.3f}  |  手部: {len(hands)}",
                    (10, max(76, int(header_height * 0.7))),
                    (200, 200, 200),
                    meta_font_size,
                ),
            ],
        )

    def _build_annotated_image(self, frame: np.ndarray, **kwargs) -> str | None:
        return self._encode_frame_to_data_url(self._draw_annotated_frame(frame, **kwargs))

    def _encode_frame_to_jpeg(self, frame: np.ndarray) -> bytes | None:
        preview = frame
        height, width = frame.shape[:2]
        longest_edge = max(width, height)
        if longest_edge > self._max_annotated_edge:
            scale = self._max_annotated_edge / float(longest_edge)
            preview = cv2.resize(
                frame,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buffer = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        return buffer.tobytes() if ok else None

    def _encode_frame_to_data_url(self, frame: np.ndarray) -> str | None:
        jpeg = self._encode_frame_to_jpeg(frame)
        if jpeg is None:
            return None
        encoded = base64.b64encode(jpeg).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _build_panel_state(
        self,
        db: Session,
        *,
        user_id: int,
        session_id: str | None = None,
    ) -> ControlPanelState:
        state = dict(self._default_panel_state)
        last_gesture: str | None = None
        last_command: str | None = None
        last_command_at: datetime | None = None
        updated_at: datetime | None = None

        record_query = select(OwnerGestureRecord).where(OwnerGestureRecord.user_id == user_id)
        active_session_id = session_id or db.scalar(
            select(OwnerGestureRecord.session_id)
            .where(
                OwnerGestureRecord.user_id == user_id,
                OwnerGestureRecord.session_id.is_not(None),
            )
            .order_by(OwnerGestureRecord.created_at.desc())
            .limit(1)
        )
        if active_session_id:
            record_query = record_query.where(OwnerGestureRecord.session_id == active_session_id)

        records = db.scalars(record_query.order_by(OwnerGestureRecord.created_at.asc())).all()
        for record in records:
            last_gesture = record.gesture
            updated_at = record.created_at
            if record.is_triggered and record.control_action != "None":
                last_command = record.control_action
                last_command_at = record.created_at
            self._apply_record_to_state(state, record)

        return ControlPanelState(
            system_awake=state["system_awake"],
            volume=state["volume"],
            climate_temperature=state["climate_temperature"],
            phone_call_active=state["phone_call_active"],
            current_mode=state["current_mode"],
            media_playing=state["media_playing"],
            comfort_scene=state["comfort_scene"],
            vehicle_status=state["vehicle_status"],
            focus_tile=state["focus_tile"],
            last_gesture=last_gesture,
            last_command=last_command,
            last_command_at=last_command_at,
            last_feedback=state["last_feedback"],
            updated_at=updated_at,
        )

    def _apply_record_to_state(self, state: dict[str, object], record: OwnerGestureRecord) -> None:
        if not record.is_triggered or record.control_action == "None":
            return
        self._apply_command_to_state(state, record.control_action)

    def _apply_command_to_state(self, state: dict[str, object], command: str) -> None:
        if command == "WakeSystem":
            if state["system_awake"]:
                state["last_feedback"] = "系统已唤醒。"
                return
            state["system_awake"] = True
            state["phone_call_active"] = False
            state["current_mode"] = "home"
            state["focus_tile"] = "home"
            state["last_feedback"] = "CMC 已唤醒，主页信息恢复显示。"
            return

        if not state["system_awake"]:
            return

        if command == "ConfirmAction":
            self._apply_confirm_action(state)
        elif command in {"AdjustVolume", "AdjustVolumeUp", "AdjustVolumeDown"}:
            state["current_mode"] = "media"
            state["focus_tile"] = "media"
            state["media_playing"] = True
            if command == "AdjustVolumeDown":
                state["volume"] = max(0, int(state["volume"]) - 6)
                state["last_feedback"] = f"媒体音量已下调至 {state['volume']}%。"
            else:
                state["volume"] = min(100, int(state["volume"]) + 6)
                state["last_feedback"] = f"媒体音量已调至 {state['volume']}%。"
        elif command == "SwitchPrevFeature":
            if state["phone_call_active"]:
                return
            next_mode = self._shift_mode(str(state["current_mode"]), direction=-1)
            state["current_mode"] = next_mode
            state["focus_tile"] = next_mode
            state["last_feedback"] = f"已切换至{self._mode_label(next_mode)}界面。"
        elif command == "SwitchNextFeature":
            if state["phone_call_active"]:
                return
            next_mode = self._shift_mode(str(state["current_mode"]), direction=1)
            state["current_mode"] = next_mode
            state["focus_tile"] = next_mode
            state["last_feedback"] = f"已切换至{self._mode_label(next_mode)}界面。"
        elif command == "AnswerCall":
            state["phone_call_active"] = True
            state["current_mode"] = "call"
            state["focus_tile"] = "call"
            state["last_feedback"] = "蓝牙电话已接通，通话界面接管前台。"
        elif command == "HangUpCall":
            state["phone_call_active"] = False
            state["current_mode"] = "home"
            state["focus_tile"] = "home"
            state["last_feedback"] = "通话已挂断，系统已回到主页。"
        elif command == "ReturnHome":
            if state["phone_call_active"]:
                return
            state["current_mode"] = "home"
            state["focus_tile"] = "home"
            state["last_feedback"] = "已捏指返回主页。"

    def _apply_confirm_action(self, state: dict[str, object]) -> None:
        current_mode = str(state["current_mode"])
        if current_mode == "call" and state["phone_call_active"]:
            state["focus_tile"] = "call"
            state["last_feedback"] = "当前正在通话，确认动作已转为通话内操作。"
            return
        if current_mode == "media":
            state["media_playing"] = not bool(state["media_playing"])
            state["focus_tile"] = "media"
            state["last_feedback"] = "媒体播放已确认继续。" if state["media_playing"] else "媒体播放已确认暂停。"
            return
        if current_mode == "comfort":
            state["comfort_scene"] = "舒享"
            state["climate_temperature"] = 22
            state["focus_tile"] = "comfort"
            state["last_feedback"] = "舒适模式已执行，空调与座椅联动完成。"
            return
        if current_mode == "vehicle":
            state["vehicle_status"] = "已完成整车检查"
            state["focus_tile"] = "vehicle"
            state["last_feedback"] = "车辆检查已执行，车况状态正常。"
            return
        state["current_mode"] = "home"
        state["focus_tile"] = "home"
        state["last_feedback"] = "主页快捷操作已确认执行。"

    def _shift_mode(self, current_mode: str, *, direction: int) -> str:
        modes = list(self._feature_modes)
        try:
            current_index = modes.index(current_mode)
        except ValueError:
            current_index = 0
        return modes[(current_index + direction) % len(modes)]

    def _mode_label(self, mode: str) -> str:
        labels = {
            "home": "主页",
            "media": "媒体",
            "comfort": "舒适",
            "vehicle": "车辆",
            "call": "通话",
        }
        return labels.get(mode, "主页")

    def _refine_motion_gesture(
        self,
        *,
        gesture: str,
        raw_keypoints: list[dict],
        recent_records: list[OwnerGestureRecord],
    ) -> str:
        if gesture == "未检测到手部" or len(raw_keypoints) < 21:
            return gesture

        if gesture == "open_palm":
            wrist_path = self._landmark_path(recent_records, raw_keypoints, landmark_index=0)
            if self._is_wave_motion(wrist_path):
                return "wave"
            swipe_gesture = self._classify_swipe(wrist_path)
            return swipe_gesture or gesture

        if gesture == "point":
            index_path = self._landmark_path(recent_records, raw_keypoints, landmark_index=8)
            if self._is_circle_motion(index_path):
                return "index_circle"
            swipe_gesture = self._classify_swipe(index_path)
            return swipe_gesture or gesture

        return gesture

    def _landmark_path(
        self,
        recent_records: list[OwnerGestureRecord],
        raw_keypoints: list[dict],
        *,
        landmark_index: int,
    ) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for record in reversed(recent_records[:5]):
            point = self._extract_landmark_point(record.hand_landmarks, landmark_index)
            if point is not None:
                points.append(point)
        current_point = self._extract_landmark_point(raw_keypoints, landmark_index)
        if current_point is not None:
            points.append(current_point)
        return points

    def _extract_landmark_point(
        self,
        hand_landmarks: list[dict] | None,
        landmark_index: int,
    ) -> tuple[float, float] | None:
        if not hand_landmarks or len(hand_landmarks) <= landmark_index:
            return None
        point = hand_landmarks[landmark_index]
        x = point.get("x")
        y = point.get("y")
        if x is None or y is None:
            return None
        return float(x), float(y)

    def _classify_swipe(self, points: list[tuple[float, float]]) -> str | None:
        if len(points) < 4:
            return None
        net_dx = points[-1][0] - points[0][0]
        net_dy = points[-1][1] - points[0][1]
        if abs(net_dx) < 0.16 or abs(net_dy) > 0.12:
            return None
        x_deltas = [right[0] - left[0] for left, right in zip(points, points[1:])]
        meaningful_deltas = [delta for delta in x_deltas if abs(delta) > 0.025]
        if len(meaningful_deltas) < 2:
            return None
        if max(abs(delta) for delta in meaningful_deltas) < 0.06:
            return None
        consistent_steps = sum(1 for delta in meaningful_deltas if delta * net_dx > 0)
        if consistent_steps < max(2, len(meaningful_deltas) - 1):
            return None
        return "swipe_right" if net_dx > 0 else "swipe_left"

    def _is_wave_motion(self, points: list[tuple[float, float]]) -> bool:
        if len(points) < 5:
            return False
        x_deltas = [right[0] - left[0] for left, right in zip(points, points[1:])]
        signs = [1 if delta > 0.03 else -1 if delta < -0.03 else 0 for delta in x_deltas]
        filtered_signs = [sign for sign in signs if sign != 0]
        if len(filtered_signs) < 4:
            return False
        direction_changes = sum(
            1 for previous, current in zip(filtered_signs, filtered_signs[1:]) if previous != current
        )
        span_x = max(point[0] for point in points) - min(point[0] for point in points)
        net_dx = abs(points[-1][0] - points[0][0])
        max_amplitude = max(abs(point[0] - points[0][0]) for point in points)
        return direction_changes >= 2 and span_x >= 0.16 and net_dx <= 0.12 and max_amplitude >= 0.06

    def _is_circle_motion(self, points: list[tuple[float, float]]) -> bool:
        if len(points) < 5:
            return False
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        if width < 0.10 or height < 0.10:
            return False
        aspect_ratio = width / height if height else 0.0
        if not 0.6 <= aspect_ratio <= 1.6:
            return False
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)
        radii = [math.hypot(x - center_x, y - center_y) for x, y in points]
        mean_radius = sum(radii) / len(radii)
        if mean_radius < 0.05:
            return False
        average_deviation = sum(abs(radius - mean_radius) for radius in radii) / len(radii)
        path_length = sum(
            math.hypot(right[0] - left[0], right[1] - left[1])
            for left, right in zip(points, points[1:])
        )
        closure_distance = math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])
        return (
            average_deviation / mean_radius <= 0.45
            and path_length >= mean_radius * 4.6
            and closure_distance <= max(width, height) * 0.72
        )

    def _build_behavior_summary(
        self,
        *,
        gesture: str,
        control_command: str | None,
        triggered: bool,
        processing_time_ms: int,
    ) -> str:
        gesture_label = GESTURE_LOG_LABEL_MAP.get(gesture, gesture)
        command_label = COMMAND_LOG_LABEL_MAP.get(control_command or "None", control_command or "无")
        if triggered and control_command:
            return (
                f"识别到手势 {gesture_label}，已触发控车指令 {command_label}，"
                f"处理耗时 {processing_time_ms} ms。"
            )
        return f"识别到手势 {gesture_label}，未触发控车指令，处理耗时 {processing_time_ms} ms。"

    def _build_behavior_summary(
        self,
        *,
        gesture: str,
        control_command: str | None,
        triggered: bool,
        processing_time_ms: int,
    ) -> str:
        gesture_label = GESTURE_LOG_LABEL_MAP.get(gesture, gesture)
        command_label = COMMAND_LOG_LABEL_MAP.get(control_command or "None", control_command or "无")
        if triggered and control_command:
            return (
                f"识别到手势 {gesture_label}，已触发控车指令 {command_label}，"
                f"处理耗时 {processing_time_ms} ms。"
            )
        return (
            f"长时间未识别到有效控车动作，当前手势结果为 {gesture_label}，"
            f"未触发控车指令，处理耗时 {processing_time_ms} ms。"
        )

    def _build_behavior_summary(
        self,
        *,
        gesture: str,
        control_command: str | None,
        triggered: bool,
        processing_time_ms: int,
        idle_behavior_due: bool,
    ) -> str:
        gesture_label = GESTURE_LOG_LABEL_MAP.get(gesture, gesture)
        command_label = COMMAND_LOG_LABEL_MAP.get(control_command or "None", control_command or "无")
        if triggered and control_command:
            return (
                f"识别到手势 {gesture_label}，已触发控车指令 {command_label}，"
                f"处理耗时 {processing_time_ms} ms。"
            )
        if idle_behavior_due:
            return (
                f"长时间未识别到有效控车动作，当前手势结果为 {gesture_label}，"
                f"未触发控车指令，处理耗时 {processing_time_ms} ms。"
            )
        return f"识别到手势 {gesture_label}，暂未触发控车指令，处理耗时 {processing_time_ms} ms。"

    async def _capture_monitor_log(
        self,
        *,
        event_type: str,
        title: str,
        summary: str,
        confidence: float | None = None,
        details: dict | None = None,
        trigger_alert: bool = False,
        level: str = "info",
    ) -> None:
        with SessionLocal() as session:
            await MonitorService(session).capture_event(
                category="owner_gesture",
                source="owner-gesture",
                event_type=event_type,
                title=title,
                summary=summary,
                level=level,
                status="processed" if confidence and confidence > 0 else "empty",
                confidence=confidence,
                details=details,
                trigger_alert=trigger_alert,
            )

    def _capture_monitor_log_sync(
        self,
        *,
        event_type: str,
        title: str,
        summary: str,
        confidence: float | None = None,
        details: dict | None = None,
        trigger_alert: bool = False,
        level: str = "info",
    ) -> None:
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(
                    self._capture_monitor_log(
                        event_type=event_type,
                        title=title,
                        summary=summary,
                        confidence=confidence,
                        details=details,
                        trigger_alert=trigger_alert,
                        level=level,
                    )
                )
            else:
                loop.create_task(
                    self._capture_monitor_log(
                        event_type=event_type,
                        title=title,
                        summary=summary,
                        confidence=confidence,
                        details=details,
                        trigger_alert=trigger_alert,
                        level=level,
                    )
                )
        except Exception:
            logger.warning("Failed to persist owner gesture monitor log: %s", title, exc_info=True)

    async def _capture_error(self, filename: str, event_type: str, summary: str) -> None:
        with SessionLocal() as session:
            await MonitorService(session).capture_event(
                category="owner_gesture",
                source="owner-gesture",
                event_type=event_type,
                title="车主手势帧处理失败",
                summary=f"{filename}: {summary}",
                level="warning",
                status="failed",
                details={"filename": filename},
            )
