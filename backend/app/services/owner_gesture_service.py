from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from time import perf_counter
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.owner_gesture_record import OwnerGestureRecord
from app.models.user_operation_log import UserOperationLog
from app.schemas.gesture import ControlPanelState, GestureFrameResult, Keypoint
from app.services.alert_service import AlertService
from app.core.config import settings
from app.core.database import SessionLocal
from app.services.monitor_service import MonitorService

if TYPE_CHECKING:
    from app.models_infer.gesture_classifier import GestureClassifier
    from app.models_infer.mediapipe_hands import MediaPipeHands

logger = logging.getLogger(__name__)


class OwnerGestureService:
    _feature_modes = ("home", "media", "comfort", "vehicle")
    _hands: Optional["MediaPipeHands"] = None
    _classifier: Optional["GestureClassifier"] = None
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

    @property
    def hands(self) -> "MediaPipeHands":
        if self._hands is None:
            from app.models_infer.mediapipe_hands import MediaPipeHands

            self._hands = MediaPipeHands()
            logger.info("OwnerGestureService MediaPipeHands 已加载")
        return self._hands

    @property
    def classifier(self) -> "GestureClassifier":
        if self._classifier is None:
            from app.models_infer.gesture_classifier import GestureClassifier

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
        db:
            SQLAlchemy session for database operations.
        user_id:
            ID of the user performing the gesture.
        session_id:
            Optional session identifier; generated if not provided.

        Returns
        -------
        GestureFrameResult
            Detected keypoints, gesture label, and control action.
        """
        started_at = perf_counter()

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            await self._capture_error(filename, "owner_gesture_decode_error", "无法解析图像字节数据。")
            raise ValueError(f"无法解析图像文件：{filename}")

        frame = self._prepare_frame_for_inference(frame)
        logger.info("正在处理车主手势帧 '%s'（%dx%d）", filename, frame.shape[1], frame.shape[0])
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

        panel_state = self._build_panel_state(
            db,
            user_id=user_id,
            session_id=active_session_id,
        )
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

        # ---------- 集成 MonitorService 监控日志（来自另一端） ----------
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

    def current_result(self) -> GestureFrameResult:
        return GestureFrameResult(
            gesture="手掌张开",
            confidence=0.92,
            keypoints=[
                Keypoint(x=0.42, y=0.18, score=0.99),
                Keypoint(x=0.48, y=0.26, score=0.98),
            ],
            updated_at=datetime.utcnow(),
        )

    def control_panel(
        self,
        db: Session,
        user_id: int,
        *,
        session_id: str | None = None,
    ) -> ControlPanelState:
        return self._build_panel_state(db, user_id=user_id, session_id=session_id)

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
            "index_circle": ("AdjustVolume", True),
            "swipe_left": ("SwitchPrevFeature", True),
            "swipe_right": ("SwitchNextFeature", True),
            "thumbs_up": ("AnswerCall", True),
            "thumbs_down": ("HangUpCall", True),
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
        if not record.is_triggered:
            return

        command = record.control_action
        if command == "WakeSystem":
            state["system_awake"] = True
            state["phone_call_active"] = False
            state["current_mode"] = "home"
            state["focus_tile"] = "home"
            state["last_feedback"] = "CMC 已唤醒，主页信息恢复显示。"
        elif command == "ConfirmAction":
            if not state["system_awake"]:
                return
            self._apply_confirm_action(state)
        elif command == "AdjustVolume":
            if not state["system_awake"]:
                return
            state["current_mode"] = "media"
            state["focus_tile"] = "media"
            state["media_playing"] = True
            state["volume"] = min(100, int(state["volume"]) + 6)
            state["last_feedback"] = f"媒体音量已调至 {state['volume']}%。"
        elif command == "SwitchPrevFeature":
            if not state["system_awake"] or state["phone_call_active"]:
                return
            next_mode = self._shift_mode(str(state["current_mode"]), direction=-1)
            state["current_mode"] = next_mode
            state["focus_tile"] = next_mode
            state["last_feedback"] = f"已切换至{self._mode_label(next_mode)}界面。"
        elif command == "SwitchNextFeature":
            if not state["system_awake"] or state["phone_call_active"]:
                return
            next_mode = self._shift_mode(str(state["current_mode"]), direction=1)
            state["current_mode"] = next_mode
            state["focus_tile"] = next_mode
            state["last_feedback"] = f"已切换至{self._mode_label(next_mode)}界面。"
        elif command == "AnswerCall":
            if not state["system_awake"]:
                return
            state["phone_call_active"] = True
            state["current_mode"] = "call"
            state["focus_tile"] = "call"
            state["last_feedback"] = "蓝牙电话已接通，通话界面接管前台。"
        elif command == "HangUpCall":
            if not state["system_awake"]:
                return
            state["phone_call_active"] = False
            state["current_mode"] = "home"
            state["focus_tile"] = "home"
            state["last_feedback"] = "通话已挂断，系统已回到主页。"
        elif command == "ReturnHome":
            if not state["system_awake"] or state["phone_call_active"]:
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

    # ---------- 以下方法从另一端合并而来（监控日志） ----------
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
