from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from app.core.config import settings
from app.core.database import SessionLocal
from app.models_infer import MediaPipeHands
from app.schemas.gesture import ControlPanelState, GestureFrameResult, Keypoint
from app.services.monitor_service import MonitorService

logger = logging.getLogger(__name__)


class OwnerGestureService:
    _hands: Optional[MediaPipeHands] = None

    @property
    def hands(self) -> MediaPipeHands:
        if self._hands is None:
            self._hands = MediaPipeHands()
            logger.info("OwnerGestureService MediaPipeHands 已加载")
        return self._hands

    async def process_frame(self, image_bytes: bytes, filename: str) -> GestureFrameResult:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            await self._capture_error(filename, "owner_gesture_decode_error", "无法解析图像字节数据。")
            raise ValueError(f"无法解析图像文件：{filename}")

        logger.info("正在处理车主手势帧 '%s'（%dx%d）", filename, frame.shape[1], frame.shape[0])
        result = self.hands.infer(frame)
        keypoints = [
            Keypoint(x=kp["x"], y=kp["y"], score=kp.get("z", 0.0))
            for kp in result["keypoints"]
        ]
        num_hands = result.get("num_hands_detected", 0)
        confidence = 0.99 if num_hands > 0 else 0.0
        gesture_label = f"检测到 {num_hands} 只手" if num_hands > 0 else "未检测到手部"

        await self._capture_monitor_log(
            event_type=(
                "owner_gesture_success"
                if confidence >= settings.alert_low_confidence_threshold
                else "owner_gesture_low_confidence"
            ),
            title="车主手势帧处理完成",
            summary=f"{filename} 已处理完成，置信度为 {confidence:.2f}，检测到 {num_hands} 只手。",
            confidence=confidence,
            details={
                "filename": filename,
                "num_hands_detected": num_hands,
                "frame_width": int(frame.shape[1]),
                "frame_height": int(frame.shape[0]),
            },
            trigger_alert=confidence < settings.alert_low_confidence_threshold,
            level="info" if confidence >= settings.alert_low_confidence_threshold else "warning",
        )

        return GestureFrameResult(
            gesture=gesture_label,
            confidence=confidence,
            keypoints=keypoints,
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

    def control_panel(self) -> ControlPanelState:
        return ControlPanelState(
            volume=32,
            climate_temperature=24,
            phone_call_active=False,
            current_mode="media",
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
