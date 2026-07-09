from __future__ import annotations

import logging
from datetime import datetime, timedelta
from time import perf_counter
from typing import Optional
from uuid import uuid4

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.owner_gesture_record import OwnerGestureRecord
from app.models.user_operation_log import UserOperationLog
from app.models_infer import MediaPipeHands, GestureClassifier
from app.schemas.gesture import ControlPanelState, GestureFrameResult, Keypoint
from app.services.alert_service import AlertService

logger = logging.getLogger(__name__)


class OwnerGestureService:
    """Owner (in-cabin) gesture service backed by MediaPipe Hands."""

    _hands: Optional[MediaPipeHands] = None
    _classifier: Optional[GestureClassifier] = None
    _hold_frame_count = 2
    _trigger_cooldown = timedelta(seconds=2)
    _default_panel_state = {
        "system_awake": False,
        "volume": 32,
        "climate_temperature": 24,
        "phone_call_active": False,
        "current_mode": "home",
    }

    # ------------------------------------------------------------------
    # Lazy-load helpers
    # ------------------------------------------------------------------

    @property
    def hands(self) -> MediaPipeHands:
        """Lazy-initialise MediaPipeHands so the service can be imported
        even when the model file is missing at import time."""
        if self._hands is None:
            self._hands = MediaPipeHands()
            logger.info("OwnerGestureService – MediaPipeHands loaded")
        return self._hands

    @property
    def classifier(self) -> GestureClassifier:
        if self._classifier is None:
            self._classifier = GestureClassifier()
        return self._classifier

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_frame(
        self,
        image_bytes: bytes,
        filename: str,
        *,
        db: Session,
        user_id: int,
        session_id: str | None = None,
    ) -> GestureFrameResult:
        """Run MediaPipe Hands inference on an uploaded image frame.

        Parameters
        ----------
        image_bytes:
            Raw image file bytes (JPEG / PNG / etc.).
        filename:
            Descriptive file name for logging / tracing.

        Returns
        -------
        GestureFrameResult
            Detected keypoints and a placeholder gesture label.
        """
        started_at = perf_counter()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Cannot decode image '{filename}'")
        logger.info("Processing hand-gesture frame '%s' (%dx%d)", filename, frame.shape[1], frame.shape[0])

        result = self.hands.infer(frame)

        raw_kps = result["keypoints"]
        num_hands = result.get("num_hands_detected", 0)

        # ----- Rule-based gesture classification -----
        cls_result = self.classifier.classify(raw_kps, domain="owner")
        gesture_label = cls_result["gesture"]
        cls_conf = cls_result["confidence"]
        if num_hands == 0:
            gesture_label = "未检测到手部"
            cls_conf = 0.0

        active_session_id = session_id or uuid4().hex[:16]
        recent_records = self._recent_session_records(
            db,
            user_id=user_id,
            session_id=active_session_id,
        )
        previous_record = recent_records[0] if recent_records else None

        control_command, _ = self._map_gesture_to_command(gesture_label)
        triggered = self._should_trigger(
            gesture=gesture_label,
            control_command=control_command,
            recent_records=recent_records,
        )
        processing_time_ms = int((perf_counter() - started_at) * 1000)

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

        panel_state = self._build_panel_state(db, user_id=user_id)
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
        """Legacy mock fallback (deprecated — use ``process_frame`` instead)."""
        return GestureFrameResult(
            gesture="手掌张开",
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

    def _map_gesture_to_command(self, gesture: str) -> tuple[str | None, bool]:
        command_map = {
            "open_palm": ("WakeSystem", True),
            "fist": ("ConfirmAction", True),
            "thumbs_up": ("AnswerCall", True),
            "thumbs_down": ("HangUpCall", True),
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

    def _build_panel_state(self, db: Session, *, user_id: int) -> ControlPanelState:
        state = dict(self._default_panel_state)
        last_gesture: str | None = None
        last_command: str | None = None
        updated_at: datetime | None = None

        records = db.scalars(
            select(OwnerGestureRecord)
            .where(OwnerGestureRecord.user_id == user_id)
            .order_by(OwnerGestureRecord.created_at.asc())
        ).all()

        for record in records:
            last_gesture = record.gesture
            last_command = record.control_action if record.control_action != "None" else None
            updated_at = record.created_at
            self._apply_record_to_state(state, record)

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

    def _apply_record_to_state(self, state: dict[str, object], record: OwnerGestureRecord) -> None:
        if not record.is_triggered:
            return

        command = record.control_action
        if command == "WakeSystem":
            state["system_awake"] = True
            state["current_mode"] = "home"
        elif command == "ConfirmAction":
            if not state["system_awake"]:
                return
            state["current_mode"] = "control"
        elif command == "AnswerCall":
            if not state["system_awake"]:
                return
            state["phone_call_active"] = True
        elif command == "HangUpCall":
            if not state["system_awake"]:
                return
            state["phone_call_active"] = False

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
