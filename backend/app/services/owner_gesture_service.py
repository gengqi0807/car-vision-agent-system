from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timedelta
from time import perf_counter
from typing import Optional
from uuid import uuid4

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.owner_gesture_record import OwnerGestureRecord
from app.models.user_operation_log import UserOperationLog
from app.models_infer import GestureClassifier, MediaPipeHands
from app.schemas.gesture import ControlPanelState, GestureFrameResult, Keypoint
from app.services.alert_service import AlertService
from app.services.monitor_service import MonitorService

logger = logging.getLogger(__name__)

NO_HAND_GESTURE = "\u672a\u68c0\u6d4b\u5230\u624b\u90e8"


class OwnerGestureService:
    """Owner (in-cabin) gesture service backed by MediaPipe Hands."""

    _hands: Optional[MediaPipeHands] = None
    _classifier: Optional[GestureClassifier] = None
    _max_inference_edge = 640
    _hold_frame_count = 2
    _trigger_cooldown = timedelta(seconds=2)
    _unrecognized_behavior_window_seconds = 30
    _default_panel_state = {
        "system_awake": False,
        "volume": 32,
        "climate_temperature": 24,
        "phone_call_active": False,
        "current_mode": "home",
    }

    @property
    def hands(self) -> MediaPipeHands:
        if self._hands is None:
            self._hands = MediaPipeHands()
            logger.info("OwnerGestureService loaded MediaPipeHands")
        return self._hands

    @property
    def classifier(self) -> GestureClassifier:
        if self._classifier is None:
            self._classifier = GestureClassifier()
        return self._classifier

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
            await self._capture_error(
                db,
                user_id=user_id,
                filename=filename,
                event_type="owner_gesture_decode_error",
                summary="Cannot decode image bytes.",
            )
            raise ValueError(f"Cannot decode image '{filename}'")

        frame = self._prepare_frame_for_inference(frame)
        logger.info("Processing hand-gesture frame '%s' (%dx%d)", filename, frame.shape[1], frame.shape[0])
        infer_result = self.hands.infer(frame)

        raw_kps = infer_result["keypoints"]
        num_hands = infer_result.get("num_hands_detected", 0)

        cls_result = self.classifier.classify(raw_kps, domain="owner")
        gesture_label = cls_result["gesture"]
        cls_conf = cls_result["confidence"]
        if num_hands == 0:
            gesture_label = NO_HAND_GESTURE
            cls_conf = 0.0

        active_session_id = session_id or uuid4().hex[:16]
        recent_records = self._recent_session_records(
            db,
            user_id=user_id,
            session_id=active_session_id,
        )
        previous_record = recent_records[0] if recent_records else None

        control_command = self._map_gesture_to_command(gesture_label)
        triggered = self._should_trigger(
            gesture=gesture_label,
            control_command=control_command,
            recent_records=recent_records,
        )
        processing_time_ms = int((perf_counter() - started_at) * 1000)
        is_unrecognized = self._is_unrecognized_result(gesture=gesture_label, num_hands=num_hands)

        keypoints = [
            Keypoint(x=kp["x"], y=kp["y"], score=kp.get("z", 0.0))
            for kp in raw_kps
        ]

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

        if is_unrecognized:
            AlertService(db).record_behavior_once(
                source="owner-gesture",
                title="Owner gesture not recognized",
                summary=self._build_unrecognized_behavior_summary(
                    filename=filename,
                    gesture=gesture_label,
                    num_detections=num_hands,
                    processing_time_ms=processing_time_ms,
                ),
                window_seconds=self._unrecognized_behavior_window_seconds,
            )
        else:
            await self._capture_monitor_log(
                db,
                user_id=user_id,
                event_type=(
                    "owner_gesture_success"
                    if cls_conf >= settings.alert_low_confidence_threshold
                    else "owner_gesture_low_confidence"
                ),
                title="Owner gesture frame processed",
                summary=(
                    f"{filename} processed: gesture={gesture_label}, "
                    f"confidence={cls_conf:.2f}, hands={num_hands}."
                ),
                confidence=cls_conf,
                details={
                    "filename": filename,
                    "session_id": active_session_id,
                    "num_hands_detected": num_hands,
                    "frame_width": int(frame.shape[1]),
                    "frame_height": int(frame.shape[0]),
                    "gesture": gesture_label,
                    "control_command": control_command,
                    "triggered": triggered,
                    "processing_time_ms": processing_time_ms,
                },
                trigger_alert=cls_conf < settings.alert_low_confidence_threshold,
                level="info" if cls_conf >= settings.alert_low_confidence_threshold else "warning",
            )

        panel_state = self._build_panel_state(db, user_id=user_id)
        if (not is_unrecognized) and self._should_record_behavior(
            previous_record=previous_record,
            gesture=gesture_label,
            triggered=triggered,
        ):
            AlertService(db).record_behavior(
                source="owner-gesture",
                title="Owner gesture control updated" if triggered else "Owner gesture observed",
                summary=self._build_behavior_summary(
                    gesture=gesture_label,
                    control_command=control_command,
                    triggered=triggered,
                    processing_time_ms=processing_time_ms,
                ),
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

    def current_result(self) -> GestureFrameResult:
        return GestureFrameResult(
            gesture="open_palm",
            confidence=0.92,
            keypoints=[
                Keypoint(x=0.42, y=0.18, score=0.99),
                Keypoint(x=0.48, y=0.26, score=0.98),
            ],
            updated_at=datetime.utcnow(),
        )

    def control_panel(self, db: Session, user_id: int) -> ControlPanelState:
        return self._build_panel_state(db, user_id=user_id)

    def _log_operation(self, db: Session, *, user_id: int, response_status: str) -> None:
        db.add(
            UserOperationLog(
                user_id=user_id,
                operation_type="owner_gesture_recognition",
                response_status=response_status,
            )
        )

    def _map_gesture_to_command(self, gesture: str) -> str | None:
        command_map = {
            "open_palm": "WakeSystem",
            "fist": "ConfirmAction",
            "thumbs_up": "AnswerCall",
            "thumbs_down": "HangUpCall",
        }
        return command_map.get(gesture)

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

        consecutive_same = 0
        for record in recent_records:
            if record.gesture != gesture:
                break
            consecutive_same += 1

        if consecutive_same < self._hold_frame_count - 1:
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
        if triggered or previous_record is None:
            return True
        return previous_record.gesture != gesture

    def _should_record_behavior(
        self,
        *,
        previous_record: OwnerGestureRecord | None,
        gesture: str,
        triggered: bool,
    ) -> bool:
        if triggered or previous_record is None:
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

    def _latest_user_record(
        self,
        db: Session,
        *,
        user_id: int,
        triggered_only: bool = False,
        control_actions: Sequence[str] | None = None,
        since: datetime | None = None,
    ) -> OwnerGestureRecord | None:
        statement = select(OwnerGestureRecord).where(OwnerGestureRecord.user_id == user_id)

        if triggered_only:
            statement = statement.where(OwnerGestureRecord.is_triggered.is_(True))
        if control_actions:
            statement = statement.where(OwnerGestureRecord.control_action.in_(tuple(control_actions)))
        if since is not None:
            statement = statement.where(OwnerGestureRecord.created_at >= since)

        return db.scalars(
            statement.order_by(OwnerGestureRecord.created_at.desc()).limit(1)
        ).first()

    def _build_panel_state(self, db: Session, *, user_id: int) -> ControlPanelState:
        state = dict(self._default_panel_state)
        last_gesture: str | None = None
        last_command: str | None = None
        updated_at: datetime | None = None

        last_record = self._latest_user_record(db, user_id=user_id)
        if last_record is not None:
            last_gesture = last_record.gesture
            last_command = last_record.control_action if last_record.control_action != "None" else None
            updated_at = last_record.created_at

        last_wake_record = self._latest_user_record(
            db,
            user_id=user_id,
            triggered_only=True,
            control_actions=("WakeSystem",),
        )
        if last_wake_record is not None:
            wake_started_at = last_wake_record.created_at
            state["system_awake"] = True

            last_mode_record = self._latest_user_record(
                db,
                user_id=user_id,
                triggered_only=True,
                control_actions=("WakeSystem", "ConfirmAction", "HangUpCall"),
                since=wake_started_at,
            )
            if last_mode_record is not None and last_mode_record.control_action == "ConfirmAction":
                state["current_mode"] = "control"

            last_call_record = self._latest_user_record(
                db,
                user_id=user_id,
                triggered_only=True,
                control_actions=("AnswerCall", "HangUpCall"),
                since=wake_started_at,
            )
            if last_call_record is not None and last_call_record.control_action == "AnswerCall":
                state["phone_call_active"] = True

        return ControlPanelState(
            system_awake=state["system_awake"],
            volume=state["volume"],
            climate_temperature=state["climate_temperature"],
            phone_call_active=state["phone_call_active"],
            current_mode=state["current_mode"],
            last_gesture=last_gesture,
            last_command=last_command,
            updated_at=updated_at,
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
                f"Gesture {gesture} triggered command {control_command}. "
                f"Processing time: {processing_time_ms} ms."
            )
        return f"Gesture {gesture} observed without control command. Processing time: {processing_time_ms} ms."

    def _is_unrecognized_result(self, *, gesture: str, num_hands: int) -> bool:
        return num_hands == 0 or gesture in {"unknown", NO_HAND_GESTURE}

    def _build_unrecognized_behavior_summary(
        self,
        *,
        filename: str,
        gesture: str,
        num_detections: int,
        processing_time_ms: int,
    ) -> str:
        return (
            f"{filename} did not produce a recognized owner gesture. "
            f"gesture={gesture}, hands={num_detections}, processing_time_ms={processing_time_ms}."
        )

    async def _capture_monitor_log(
        self,
        db: Session,
        *,
        user_id: int,
        event_type: str,
        title: str,
        summary: str,
        confidence: float | None = None,
        details: dict | None = None,
        trigger_alert: bool = False,
        level: str = "info",
    ) -> None:
        await MonitorService(db).capture_event(
            category="owner_gesture",
            source="owner-gesture",
            event_type=event_type,
            title=title,
            summary=summary,
            level=level,
            status="processed" if confidence and confidence > 0 else "empty",
            user_id=user_id,
            confidence=confidence,
            details=details,
            trigger_alert=trigger_alert,
        )

    async def _capture_error(
        self,
        db: Session,
        *,
        user_id: int,
        filename: str,
        event_type: str,
        summary: str,
    ) -> None:
        await MonitorService(db).capture_event(
            category="owner_gesture",
            source="owner-gesture",
            event_type=event_type,
            title="Owner gesture frame processing failed",
            summary=f"{filename}: {summary}",
            level="warning",
            status="failed",
            user_id=user_id,
            details={"filename": filename},
            trigger_alert=False,
        )
