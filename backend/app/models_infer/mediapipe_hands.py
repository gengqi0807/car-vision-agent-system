"""MediaPipe Hands inference helpers for image and video modes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar, Optional, Union

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
    """MediaPipe hand landmark detector.

    Supports:
    - instance-based image inference for uploaded snapshots
    - class-level video inference for backend pull-stream mode
    """

    _video_model_path: ClassVar[str | None] = None
    _video_detector: ClassVar[HandLandmarker | None] = None
    _video_num_hands: ClassVar[int] = 2
    _video_min_detection_confidence: ClassVar[float] = 0.5
    _video_min_presence_confidence: ClassVar[float] = 0.5
    _video_min_tracking_confidence: ClassVar[float] = 0.5
    _frame_timestamp_ms: ClassVar[int] = 0

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        resolved = model_path or settings.resolved_hand_model_path
        if not Path(resolved).is_file():
            raise FileNotFoundError(
                f"Hand landmarker model not found at {resolved}. "
                "Please place hand_landmarker.task under backend/models/."
            )

        self._model_path = resolved
        self._options = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=resolved),
            running_mode=RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker: Optional[HandLandmarker] = None
        logger.info("MediaPipeHands initialised with model %s", resolved)

    def _ensure_landmarker(self) -> HandLandmarker:
        if self._landmarker is None:
            patch_windows_mediapipe_free_symbol()
            self._landmarker = HandLandmarker.create_from_options(self._options)
        return self._landmarker

    @staticmethod
    def _load_image(source: Union[str, np.ndarray]) -> mp.Image:
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

    def infer(self, source: Union[str, np.ndarray]) -> dict:
        landmarker = self._ensure_landmarker()
        mp_image = self._load_image(source)
        result = landmarker.detect(mp_image)
        return self._result_to_dict(result, str(source))

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            logger.info("MediaPipeHands landmarker closed.")

    def __enter__(self) -> "MediaPipeHands":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @classmethod
    def configure(
        cls,
        model_path: str,
        *,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        cls._video_model_path = model_path
        cls._video_num_hands = num_hands
        cls._video_min_detection_confidence = min_detection_confidence
        cls._video_min_presence_confidence = min_presence_confidence
        cls._video_min_tracking_confidence = min_tracking_confidence

    @classmethod
    def reset(cls) -> None:
        if cls._video_detector is not None:
            cls._video_detector.close()
            cls._video_detector = None
        cls._frame_timestamp_ms = 0

    @classmethod
    def _get_video_detector(cls) -> HandLandmarker:
        if cls._video_detector is not None:
            return cls._video_detector

        model_path = cls._video_model_path or settings.resolved_hand_model_path
        if not model_path or not Path(model_path).is_file():
            raise FileNotFoundError(
                f"Hand landmarker model not found at {model_path}. "
                "Please place hand_landmarker.task under backend/models/."
            )

        patch_windows_mediapipe_free_symbol()
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = HandLandmarkerOptions(
            base_options=base_options,
            num_hands=cls._video_num_hands,
            min_hand_detection_confidence=cls._video_min_detection_confidence,
            min_hand_presence_confidence=cls._video_min_presence_confidence,
            min_tracking_confidence=cls._video_min_tracking_confidence,
            running_mode=RunningMode.VIDEO,
        )
        cls._video_detector = HandLandmarker.create_from_options(options)
        return cls._video_detector

    @classmethod
    def infer_video(
        cls,
        image_bgr: np.ndarray,
        *,
        timestamp_ms: int | None = None,
    ) -> list[list[dict]]:
        detector = cls._get_video_detector()
        image_rgb = image_bgr[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        if timestamp_ms is None:
            cls._frame_timestamp_ms += 33
            timestamp_ms = cls._frame_timestamp_ms

        result = detector.detect_for_video(mp_image, timestamp_ms)
        hands: list[list[dict]] = []
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                hands.append(
                    [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks]
                )
        return hands
