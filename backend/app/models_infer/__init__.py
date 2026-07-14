"""Inference wrappers for pretrained models.

Each module wraps a computer-vision model and exposes a simple
``infer()`` / ``detect()`` / ``classify()`` / ``recognize()`` interface
that can be consumed by the service layer.
"""

from app.models_infer.gesture_classifier import GestureClassifier
from app.models_infer.mediapipe_hands import MediaPipeHands
from app.models_infer.mediapipe_pose import MediaPipePose
from app.models_infer.yolo_detector import YoloDetector

__all__ = [
    "GestureClassifier",
    "MediaPipeHands",
    "MediaPipePose",
    "YoloDetector",
]
