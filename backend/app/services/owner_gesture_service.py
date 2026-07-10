from __future__ import annotations

import asyncio
import logging
import math
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

if TYPE_CHECKING:
    from app.models_infer.gesture_classifier import GestureClassifier
    from app.models_infer.mediapipe_hands import MediaPipeHands

logger = logging.getLogger(__name__)

GESTURE_ACTION_MAP: dict[str, str] = {
    "open_palm": "wake",
    "fist": "confirm",
    "index_circle": "volume_adjust",
    "swipe_left": "prev_func",
    "swipe_right": "next_func",
    "thumbs_up": "call_answer",
    "thumbs_down": "call_hangup",
    "wave": "home",
    "point": "idle",
    "idle": "idle",
    "unknown": "idle",
    "未检测到手部": "idle",
}


class OwnerGestureService:
    _instance: ClassVar["OwnerGestureService | None"] = None
    _feature_modes = ("home", "media", "comfort", "vehicle")
    _hands: Optional["MediaPipeHands"] = None
    _classifier: Optional["GestureClassifier"] = None
    _stream_classifier: Optional["GestureClassifier"] = None
    _max_inference_edge = 640
    _hold_frame_count = 2
    _trigger_cooldown = timedelta(seconds=2)
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
        self._stream_thread: threading.Thread | None = None
        self._stream_running = False
        self._stream_source = ""
        self._stream_loop: asyncio.AbstractEventLoop | None = None
        self._ws_callbacks: list[tuple[Any, asyncio.AbstractEventLoop | None]] = []
        self._control_callbacks: list[Any] = []
        self._alert_callbacks: list[Any] = []
        self._latest_stream_result: OwnerGestureResult | None = None
        self._stream_state = StreamState(running=False)
        self._live_panel_state = ControlPanelState(
            **self._default_panel_state,
            last_gesture=None,
            last_command=None,
            last_command_at=None,
            updated_at=None,
        )

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
    ) -> GestureFrameResult:
        started_at = perf_counter()

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            await self._capture_error(filename, "owner_gesture_decode_error", "无法解析图像字节数据。")
            raise ValueError(f"无法解析图像文件：{filename}")

        frame = self._prepare_frame_for_inference(frame)
        logger.info("Processing owner gesture frame '%s' (%dx%d)", filename, frame.shape[1], frame.shape[0])
        infer_result = self.hands.infer(frame)

        raw_kps = infer_result["keypoints"]
        num_hands = infer_result.get("num_hands_detected", 0)
        cls_result = self.classifier.classify(raw_kps, domain="owner")
        gesture_label = cls_result["gesture"]
        cls_conf = cls_result["confidence"]
        if num_hands == 0:
            gesture_label = "未检测到手部"
            cls_conf = 0.0

        active_session_id = session_id or uuid4().hex[:16]
        recent_records = self._recent_session_records(db, user_id=user_id, session_id=active_session_id)
        previous_record = recent_records[0] if recent_records else None
        gesture_label = self._refine_motion_gesture(
            gesture=gesture_label,
            raw_keypoints=raw_kps,
            recent_records=recent_records,
        )

        control_command, _ = self._map_gesture_to_command(gesture_label)
        triggered = self._should_trigger(
            gesture=gesture_label,
            control_command=control_command,
            recent_records=recent_records,
        )
        processing_time_ms = int((perf_counter() - started_at) * 1000)

        keypoints = [Keypoint(x=kp["x"], y=kp["y"], score=kp.get("z", 0.0)) for kp in raw_kps]
        record = OwnerGestureRecord(
            user_id=user_id,
            session_id=active_session_id,
            gesture=gesture_label,
            confidence=round(cls_conf, 4),
            control_action=control_command or "None",
            hand_landmarks=[kp.model_dump() for kp in keypoints],
            is_triggered=triggered,
            processing_time_ms=processing_time_ms,
        )
        db.add(record)
        if self._should_log_operation(
            previous_record=previous_record,
            gesture=gesture_label,
            triggered=triggered,
        ):
            self._log_operation(
                db,
                user_id=user_id,
                response_status="Success" if num_hands > 0 else "NoHandDetected",
            )
        db.commit()
        db.refresh(record)

        panel_state = self._build_panel_state(db, user_id=user_id, session_id=active_session_id)
        if self._should_record_behavior(
            previous_record=previous_record,
            gesture=gesture_label,
            triggered=triggered,
        ):
            AlertService(db).record_behavior(
                source="owner-gesture",
                title="手势控车识别完成" if triggered else "手势控车识别更新",
                summary=self._build_behavior_summary(
                    gesture=gesture_label,
                    control_command=control_command,
                    triggered=triggered,
                    processing_time_ms=processing_time_ms,
                ),
            )

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
            confidence=round(cls_conf, 4),
            keypoints=keypoints,
            control_command=control_command,
            triggered=triggered,
            panel_state=panel_state,
            updated_at=datetime.utcnow(),
        )

    def start(self, source: str, fps: int = 15) -> StreamState:
        with self._stream_lock:
            if self._stream_running:
                return self._stream_state

            self._configure_stream_runtime()
            self._stream_running = True
            self._stream_source = source
            self._latest_stream_result = None
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
            return self._stream_state

    def current_stream_result(self) -> OwnerGestureResult:
        with self._stream_lock:
            if self._latest_stream_result is not None:
                return self._latest_stream_result
            return OwnerGestureResult(
                gesture="unknown",
                action="idle",
                confidence=0.0,
                keypoints=[],
                hand_count=0,
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
        return self._build_panel_state(db, user_id=user_id, session_id=session_id)

    def _configure_stream_runtime(self) -> None:
        from app.models_infer.mediapipe_hands import MediaPipeHands

        MediaPipeHands.configure(
            settings.resolved_hand_model_path,
            num_hands=settings.num_hands,
            min_detection_confidence=settings.min_hand_detection_confidence,
            min_presence_confidence=settings.min_hand_presence_confidence,
            min_tracking_confidence=settings.min_hand_tracking_confidence,
        )
        _ = self.stream_classifier

    def _stream_loop_worker(self, source: str, fps: int) -> None:
        from app.models_infer.mediapipe_hands import MediaPipeHands

        capture = cv2.VideoCapture(self._resolve_stream_source(source))
        if not capture.isOpened():
            logger.warning("Failed to open owner-gesture stream source: %s", source)
            with self._stream_lock:
                self._stream_running = False
                self._stream_state = StreamState(running=False)
            return

        frame_interval = 1.0 / max(fps, 1)

        try:
            while True:
                with self._stream_lock:
                    if not self._stream_running:
                        break

                tick_started = time.time()
                ok, frame_bgr = capture.read()
                if not ok:
                    time.sleep(0.1)
                    continue

                try:
                    prepared = self._prepare_frame_for_inference(frame_bgr)
                    hands = MediaPipeHands.infer_video(prepared)
                    primary_hand = hands[0] if hands else None
                    gesture, confidence = self.stream_classifier.classify_frame(primary_hand)

                    control_command, _ = self._map_gesture_to_command(gesture)
                    action = GESTURE_ACTION_MAP.get(gesture, "idle")
                    keypoints = (
                        [Keypoint(x=point["x"], y=point["y"], score=point.get("z", 0.0)) for point in primary_hand]
                        if primary_hand
                        else []
                    )

                    panel_state = self._update_live_panel_state(
                        gesture=gesture,
                        control_command=control_command,
                        updated_at=datetime.now(timezone.utc),
                    )
                    result = OwnerGestureResult(
                        gesture=gesture,
                        action=action,
                        confidence=round(confidence, 4),
                        keypoints=keypoints,
                        hand_count=len(hands),
                        panel_state=panel_state,
                        updated_at=datetime.now(timezone.utc),
                    )

                    with self._stream_lock:
                        self._latest_stream_result = result

                    if action != "idle":
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
                except Exception:
                    logger.debug("Owner gesture stream frame processing failed", exc_info=True)

                elapsed = time.time() - tick_started
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            capture.release()
            MediaPipeHands.reset()
            with self._stream_lock:
                self._stream_running = False
                self._stream_state = StreamState(running=False)

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
            "index_circle": ("AdjustVolume", True),
            "circle_cw": ("AdjustVolume", True),
            "circle_ccw": ("AdjustVolume", True),
            "swipe_left": ("SwitchPrevFeature", True),
            "swipe_right": ("SwitchNextFeature", True),
            "thumbs_up": ("AnswerCall", True),
            "thumb_up": ("AnswerCall", True),
            "thumbs_down": ("HangUpCall", True),
            "thumb_down": ("HangUpCall", True),
            "wave": ("ReturnHome", True),
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

    def _should_trigger(
        self,
        *,
        gesture: str,
        control_command: str | None,
        recent_records: list[OwnerGestureRecord],
    ) -> bool:
        if not control_command:
            return False

        required_hold_count = 1 if control_command in {
            "AdjustVolume",
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

    def _prepare_frame_for_inference(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        longest_edge = max(width, height)
        if longest_edge <= self._max_inference_edge:
            return frame

        scale = self._max_inference_edge / float(longest_edge)
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        return cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

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
        elif command == "AdjustVolume":
            state["current_mode"] = "media"
            state["focus_tile"] = "media"
            state["media_playing"] = True
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
            state["last_feedback"] = "已挥手返回主页。"

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
        if len(points) < 3:
            return None
        net_dx = points[-1][0] - points[0][0]
        net_dy = points[-1][1] - points[0][1]
        if abs(net_dx) < 0.16 or abs(net_dy) > 0.12:
            return None
        x_deltas = [right[0] - left[0] for left, right in zip(points, points[1:])]
        meaningful_deltas = [delta for delta in x_deltas if abs(delta) > 0.015]
        if len(meaningful_deltas) < 2:
            return None
        consistent_steps = sum(1 for delta in meaningful_deltas if delta * net_dx > 0)
        if consistent_steps < max(2, len(meaningful_deltas) - 1):
            return None
        return "swipe_right" if net_dx > 0 else "swipe_left"

    def _is_wave_motion(self, points: list[tuple[float, float]]) -> bool:
        if len(points) < 5:
            return False
        x_deltas = [right[0] - left[0] for left, right in zip(points, points[1:])]
        signs = [1 if delta > 0.02 else -1 if delta < -0.02 else 0 for delta in x_deltas]
        filtered_signs = [sign for sign in signs if sign != 0]
        if len(filtered_signs) < 3:
            return False
        direction_changes = sum(
            1 for previous, current in zip(filtered_signs, filtered_signs[1:]) if previous != current
        )
        span_x = max(point[0] for point in points) - min(point[0] for point in points)
        net_dx = abs(points[-1][0] - points[0][0])
        return direction_changes >= 2 and span_x >= 0.16 and net_dx <= 0.12

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
        if triggered and control_command:
            return (
                f"识别到手势 {gesture}，已触发控车指令 {control_command}，"
                f"处理耗时 {processing_time_ms} ms。"
            )
        return f"识别到手势 {gesture}，未触发控车指令，处理耗时 {processing_time_ms} ms。"

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
