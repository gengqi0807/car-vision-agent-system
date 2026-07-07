from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from app.models_infer import MediaPipePose
from app.schemas.gesture import GestureFrameResult, GestureHistoryItem, Keypoint

logger = logging.getLogger(__name__)


class PoliceGestureService:
    """Police (traffic) gesture service backed by MediaPipe Pose."""

    _pose: Optional[MediaPipePose] = None

    # ------------------------------------------------------------------
    # Lazy-load helpers
    # ------------------------------------------------------------------

    @property
    def pose(self) -> MediaPipePose:
        """Lazy-initialise MediaPipePose so the service can be imported
        even when the model file is missing at import time."""
        if self._pose is None:
            self._pose = MediaPipePose()
            logger.info("PoliceGestureService – MediaPipePose loaded")
        return self._pose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_frame(self, image_bytes: bytes, filename: str) -> GestureFrameResult:
        """Run MediaPipe Pose inference on an uploaded image frame.

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
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Cannot decode image '{filename}'")
        logger.info("Processing police-pose frame '%s' (%dx%d)", filename, frame.shape[1], frame.shape[0])

        result = self.pose.infer(frame)

        keypoints = [
            Keypoint(
                x=kp["x"],
                y=kp["y"],
                score=kp.get("visibility", kp.get("z", 0.0)),
            )
            for kp in result["keypoints"]
        ]
        num_poses = result.get("num_poses_detected", 0)
        gesture_label = f"检测到 {num_poses} 人" if num_poses > 0 else "未检测到人体"

        return GestureFrameResult(
            gesture=gesture_label,
            confidence=0.99 if num_poses > 0 else 0.0,
            keypoints=keypoints,
            updated_at=datetime.utcnow(),
        )

    def current_result(self) -> GestureFrameResult:
        """Legacy mock fallback (deprecated — use ``process_frame`` instead)."""
        return GestureFrameResult(
            gesture="停止信号",
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
                gesture="停止信号",
                confidence=0.88,
                updated_at=datetime.utcnow(),
            )
        ]
