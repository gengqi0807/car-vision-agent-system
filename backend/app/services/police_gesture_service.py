from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

# HEAD 原有导入
from app.models_infer import MediaPipePose, GestureClassifier
from app.schemas.gesture import GestureFrameResult, GestureHistoryItem, Keypoint

# 另一端新增导入（监控相关）
from app.core.config import settings
from app.core.database import SessionLocal
from app.services.monitor_service import MonitorService

logger = logging.getLogger(__name__)


class PoliceGestureService:
    _pose: Optional[MediaPipePose] = None
    _classifier: Optional[GestureClassifier] = None

    @property
    def pose(self) -> MediaPipePose:
        if self._pose is None:
            self._pose = MediaPipePose()
            logger.info("PoliceGestureService MediaPipePose 已加载")
        return self._pose

    @property
    def classifier(self) -> GestureClassifier:
        if self._classifier is None:
            self._classifier = GestureClassifier()
        return self._classifier

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_frame(self, image_bytes: bytes, filename: str) -> GestureFrameResult:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            await self._capture_error(filename, "police_gesture_decode_error", "无法解析图像字节数据。")
            raise ValueError(f"无法解析图像文件：{filename}")

        logger.info("正在处理交警手势帧 '%s'（%dx%d）", filename, frame.shape[1], frame.shape[0])
        result = self.pose.infer(frame)

        raw_kps = result["keypoints"]
        num_poses = result.get("num_poses_detected", 0)

        # ----- Rule-based gesture classification -----
        cls_result = self.classifier.classify(raw_kps, domain="police")
        gesture_label = cls_result["gesture"]
        cls_conf = cls_result["confidence"]
        if num_poses == 0:
            gesture_label = "未检测到人体"
            cls_conf = 0.0

        keypoints = [
            Keypoint(
                x=kp["x"],
                y=kp["y"],
                score=kp.get("visibility", kp.get("z", 0.0)),
            )
            for kp in raw_kps
        ]

        # ---------- 集成 MonitorService 监控日志（来自另一端） ----------
        await self._capture_monitor_log(
            event_type=(
                "police_gesture_success"
                if cls_conf >= settings.alert_low_confidence_threshold
                else "police_gesture_low_confidence"
            ),
            title="交警手势帧处理完成",
            summary=f"{filename} 已处理完成，置信度为 {cls_conf:.2f}，检测到 {num_poses} 个人体姿态。",
            confidence=cls_conf,
            details={
                "filename": filename,
                "num_poses_detected": num_poses,
                "frame_width": int(frame.shape[1]),
                "frame_height": int(frame.shape[0]),
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
            gesture="停止手势",
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
                gesture="停止手势",
                confidence=0.88,
                updated_at=datetime.utcnow(),
            )
        ]

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

    async def _capture_error(self, filename: str, event_type: str, summary: str) -> None:
        with SessionLocal() as session:
            await MonitorService(session).capture_event(
                category="police_gesture",
                source="police-gesture",
                event_type=event_type,
                title="交警手势帧处理失败",
                summary=f"{filename}: {summary}",
                level="warning",
                status="failed",
                details={"filename": filename},
            )