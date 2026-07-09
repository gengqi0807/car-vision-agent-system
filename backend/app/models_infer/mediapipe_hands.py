"""MediaPipe Hands–based inference.

Loads the official hand landmarker model and exposes a callable
``infer()`` method that accepts an image file path or a numpy
array and returns 21 hand keypoints.

Model download (once):
    python scripts/download_models.py --model hand_landmarker

Expected model location: ``backend/models/hand_landmarker.task``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from app.core.config import settings
from app.models_infer.mediapipe_compat import patch_windows_mediapipe_free_symbol

logger = logging.getLogger(__name__)

HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandLandmarkerResult = vision.HandLandmarkerResult
RunningMode = vision.RunningMode


class MediaPipeHands:
    """Real-time hand landmark detection via MediaPipe Hands.

    Usage::

        detector = MediaPipeHands()
        result: dict = detector.infer("path/to/frame.jpg")
        # result["keypoints"] -> list[dict]  (21 points)
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        model_path:
            Local path to ``hand_landmarker.task``.  When omitted the
            path is read from application settings.
        """
        resolved = model_path or settings.resolved_hand_model_path
        if not Path(resolved).is_file():
            raise FileNotFoundError(
                f"Hand landmarker model not found at {resolved}. "
                f"Run `python scripts/download_models.py --model hand_landmarker` first."
            )

        self._model_path = resolved
        self._options = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=resolved),
            running_mode=RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker: Optional[HandLandmarker] = None
        logger.info("MediaPipeHands initialised with model %s", resolved)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_landmarker(self) -> HandLandmarker:
        if self._landmarker is None:
            patch_windows_mediapipe_free_symbol()
            self._landmarker = HandLandmarker.create_from_options(self._options)
        return self._landmarker

    @staticmethod
    def _load_image(source: Union[str, np.ndarray]) -> mp.Image:
        """Load an image from file path or numpy array into a MediaPipe Image."""
        if isinstance(source, (str, Path)):
            image_bgr = cv2.imread(str(source))
            if image_bgr is None:
                raise ValueError(f"Cannot read image from {source}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(source, np.ndarray):
            image_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB) if source.shape[-1] == 3 else source
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    @staticmethod
    def _result_to_dict(result: HandLandmarkerResult, source: str) -> dict:
        """Convert MediaPipe result into a JSON-serialisable dict."""
        keypoints: list[dict] = []
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                for lm in hand_landmarks:
                    keypoints.append(
                        {"x": round(lm.x, 6), "y": round(lm.y, 6), "z": round(lm.z, 6)}
                    )
        return {
            "source": source,
            "num_hands_detected": len(result.hand_landmarks),
            "keypoints": keypoints,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer(self, source: Union[str, np.ndarray]) -> dict:
        """Detect hand landmarks in a single image.

        Parameters
        ----------
        source:
            File path (str) or BGR/RGB numpy array of shape ``(H, W, 3)``.

        Returns
        -------
        dict
            ``{"source": ..., "num_hands_detected": ..., "keypoints": [...]}``
        """
        landmarker = self._ensure_landmarker()
        mp_image = self._load_image(source)
        result = landmarker.detect(mp_image)
        return self._result_to_dict(result, str(source))

    def close(self) -> None:
        """Release the underlying MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            logger.info("MediaPipeHands landmarker closed.")

    def __enter__(self) -> "MediaPipeHands":
        return self

    def __exit__(self, *args) -> None:
        self.close()
