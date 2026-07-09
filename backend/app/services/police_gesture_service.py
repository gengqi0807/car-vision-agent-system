from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from app.core.config import settings
from app.core.database import SessionLocal
from app.models_infer import GestureClassifier, MediaPipePose
from app.schemas.gesture import GestureFrameResult, GestureHistoryItem, Keypoint
from app.services.alert_service import AlertService
from app.services.monitor_service import MonitorService

logger = logging.getLogger(__name__)

NO_POSE_GESTURE = "\u672a\u68c0\u6d4b\u5230\u4eba\u4f53"


class PoliceGestureService:
    """Police (traffic) gesture service backed by MediaPipe Pose."""

    _pose: Optional[MediaPipePose] = None
    _classifier: Optional[GestureClassifier] = None
    _unrecognized_behavior_window_seconds = 30

    @property
    def pose(self) -> MediaPipePose:
        if self._pose is None:
            self._pose = MediaPipePose()
            logger.info("PoliceGestureService loaded MediaPipePose")
        return self._pose

    @property
    def classifier(self) -> GestureClassifier:
        if self._classifier is None:
            self._classifier = GestureClassifier()
        return self._classifier

    async def process_frame(self, image_bytes: bytes, filename: str) -> GestureFrameResult:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            await self._capture_error(
                filename=filename,
                event_type="police_gesture_decode_error",
                summary="Cannot decode image bytes.",
            )
            raise ValueError(f"Cannot decode image '{filename}'")

        logger.info("Processing police-pose frame '%s' (%dx%d)", filename, frame.shape[1], frame.shape[0])
        result = self.pose.infer(frame)

        raw_kps = result["keypoints"]
        num_poses = result.get("num_poses_detected", 0)

        cls_result = self.classifier.classify(raw_kps, domain="police")
        gesture_label = cls_result["gesture"]
        cls_conf = cls_result["confidence"]
        if num_poses == 0:
            gesture_label = NO_POSE_GESTURE
            cls_conf = 0.0

        keypoints = [
            Keypoint(
                x=kp["x"],
                y=kp["y"],
                score=kp.get("visibility", kp.get("z", 0.0)),
            )
            for kp in raw_kps
        ]

        is_unrecognized = self._is_unrecognized_result(gesture=gesture_label, num_poses=num_poses)
        if is_unrecognized:
            with SessionLocal() as session:
                AlertService(session).record_behavior_once(
                    source="police-gesture",
                    title="Police gesture not recognized",
                    summary=self._build_unrecognized_behavior_summary(
                        filename=filename,
                        gesture=gesture_label,
                        num_detections=num_poses,
                    ),
                    window_seconds=self._unrecognized_behavior_window_seconds,
                )
        else:
            await self._capture_monitor_log(
                event_type=(
                    "police_gesture_success"
                    if cls_conf >= settings.alert_low_confidence_threshold
                    else "police_gesture_low_confidence"
                ),
                title="Police gesture frame processed",
                summary=(
                    f"{filename} processed: gesture={gesture_label}, "
                    f"confidence={cls_conf:.2f}, poses={num_poses}."
                ),
                confidence=cls_conf,
                details={
                    "filename": filename,
                    "num_poses_detected": num_poses,
                    "frame_width": int(frame.shape[1]),
                    "frame_height": int(frame.shape[0]),
                    "gesture": gesture_label,
                },
                trigger_alert=cls_conf < settings.alert_low_confidence_threshold,
                level="info" if cls_conf >= settings.alert_low_confidence_threshold else "warning",
            )

        return GestureFrameResult(
            gesture=gesture_label,
            confidence=round(cls_conf, 4),
            keypoints=keypoints,
            updated_at=datetime.utcnow(),
        )

    def current_result(self) -> GestureFrameResult:
        return GestureFrameResult(
            gesture="stop",
            confidence=0.88,
            keypoints=[
                Keypoint(x=0.46, y=0.22, score=0.98),
                Keypoint(x=0.51, y=0.34, score=0.97),
            ],
            updated_at=datetime.utcnow(),
        )

    def history(self) -> list[GestureHistoryItem]:
        return [
            GestureHistoryItem(
                gesture="stop",
                confidence=0.88,
                updated_at=datetime.utcnow(),
            )
        ]

    def _is_unrecognized_result(self, *, gesture: str, num_poses: int) -> bool:
        return num_poses == 0 or gesture in {"unknown", NO_POSE_GESTURE}

    def _build_unrecognized_behavior_summary(
        self,
        *,
        filename: str,
        gesture: str,
        num_detections: int,
    ) -> str:
        return (
            f"{filename} did not produce a recognized police gesture. "
            f"gesture={gesture}, poses={num_detections}."
        )

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
                category="police_gesture",
                source="police-gesture",
                event_type=event_type,
                title=title,
                summary=summary,
                level=level,
                status="processed" if confidence and confidence > 0 else "empty",
                confidence=confidence,
                details=details,
                trigger_alert=trigger_alert,
            )

    async def _capture_error(
        self,
        *,
        filename: str,
        event_type: str,
        summary: str,
    ) -> None:
        with SessionLocal() as session:
            await MonitorService(session).capture_event(
                category="police_gesture",
                source="police-gesture",
                event_type=event_type,
                title="Police gesture frame processing failed",
                summary=f"{filename}: {summary}",
                level="warning",
                status="failed",
                details={"filename": filename},
                trigger_alert=False,
            )
