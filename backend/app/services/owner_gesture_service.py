from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from app.models_infer import MediaPipeHands, GestureClassifier
from app.schemas.gesture import ControlPanelState, GestureFrameResult, Keypoint

logger = logging.getLogger(__name__)


class OwnerGestureService:
    """Owner (in-cabin) gesture service backed by MediaPipe Hands."""

    _hands: Optional[MediaPipeHands] = None
    _classifier: Optional[GestureClassifier] = None

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

    async def process_frame(self, image_bytes: bytes, filename: str) -> GestureFrameResult:
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

        keypoints = [
            Keypoint(x=kp["x"], y=kp["y"], score=kp.get("z", 0.0))
            for kp in raw_kps
        ]

        return GestureFrameResult(
            gesture=gesture_label,
            confidence=round(cls_conf, 4),
            keypoints=keypoints,
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

    def control_panel(self) -> ControlPanelState:
        return ControlPanelState(
            volume=32,
            climate_temperature=24,
            phone_call_active=False,
            current_mode="media",
        )
