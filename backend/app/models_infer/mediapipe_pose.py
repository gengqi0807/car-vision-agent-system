"""MediaPipe Pose–based inference.

Loads the official Pose Landmarker (Lite) model and exposes a callable
``infer()`` method that accepts an image file path or a numpy array and
returns 33 body keypoints.

Model download (once):
    python scripts/download_models.py --model pose_landmarker

Expected model location: ``backend/models/pose_landmarker_lite.task``
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

logger = logging.getLogger(__name__)

PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
PoseLandmarkerResult = vision.PoseLandmarkerResult
RunningMode = vision.RunningMode


class MediaPipePose:
    """Real-time pose landmark detection via MediaPipe Pose.

    Usage::

        detector = MediaPipePose()
        result: dict = detector.infer("path/to/frame.jpg")
        # result["keypoints"] -> list[dict]  (33 points)
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        model_path:
            Local path to ``pose_landmarker_lite.task``.  When omitted the
            path is read from application settings.
        """
        resolved = model_path or settings.resolved_pose_model_path
        if not Path(resolved).is_file():
            raise FileNotFoundError(
                f"Pose landmarker model not found at {resolved}. "
                f"Run `python scripts/download_models.py --model pose_landmarker` first."
            )

        self._model_path = resolved
        self._options = PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=resolved),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker: Optional[PoseLandmarker] = None
        logger.info("MediaPipePose initialised with model %s", resolved)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_landmarker(self) -> PoseLandmarker:
        if self._landmarker is None:
            self._landmarker = PoseLandmarker.create_from_options(self._options)
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
    def _result_to_dict(result: PoseLandmarkerResult, source: str) -> dict:
        """Convert MediaPipe result into a JSON-serialisable dict."""
        keypoints: list[dict] = []
        if result.pose_landmarks:
            for pose_landmarks in result.pose_landmarks:
                for lm in pose_landmarks:
                    keypoints.append(
                        {
                            "x": round(lm.x, 6),
                            "y": round(lm.y, 6),
                            "z": round(lm.z, 6),
                            "visibility": round(lm.visibility, 4) if hasattr(lm, "visibility") else 1.0,
                        }
                    )
        return {
            "source": source,
            "num_poses_detected": len(result.pose_landmarks),
            "keypoints": keypoints,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer(self, source: Union[str, np.ndarray]) -> dict:
        """Detect pose landmarks in a single image.

        Parameters
        ----------
        source:
            File path (str) or BGR/RGB numpy array of shape ``(H, W, 3)``.

        Returns
        -------
        dict
            ``{"source": ..., "num_poses_detected": ..., "keypoints": [...]}``
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
            logger.info("MediaPipePose landmarker closed.")

    def __enter__(self) -> "MediaPipePose":
        return self

    def __exit__(self, *args) -> None:
        self.close()
