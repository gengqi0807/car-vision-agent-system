"""
MediaPipe Hands 推理封装 — 后端拉流模式。

设计:
  - lazy-load: 模型仅在首次 infer() 调用时加载（缺文件时服务照常启动，调用时报错）
  - configure() 注入模型路径后，detector 为类级别单例
  - infer() 接收 BGR numpy 数组，返回 21 关键点列表
"""

import os
import time
from typing import ClassVar

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


class MediaPipeHands:
    """MediaPipe 手部关键点检测器（类级别单例）"""

    _model_path: ClassVar[str | None] = None
    _detector: ClassVar[vision.HandLandmarker | None] = None
    _num_hands: ClassVar[int] = 2
    _min_detection_confidence: ClassVar[float] = 0.5
    _min_presence_confidence: ClassVar[float] = 0.5
    _min_tracking_confidence: ClassVar[float] = 0.5
    _frame_timestamp_ms: ClassVar[int] = 0

    # ----------------------------------------------------------------
    # 配置
    # ----------------------------------------------------------------

    @classmethod
    def configure(
        cls,
        model_path: str,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """
        注入模型路径和推理参数。
        调用后已有的 detector 不会自动重建——需显式调用 reset()。
        """
        cls._model_path = model_path
        cls._num_hands = num_hands
        cls._min_detection_confidence = min_detection_confidence
        cls._min_presence_confidence = min_presence_confidence
        cls._min_tracking_confidence = min_tracking_confidence

    @classmethod
    def reset(cls) -> None:
        """强制重建 detector（切换模型时使用）。"""
        if cls._detector is not None:
            cls._detector.close()
            cls._detector = None
        cls._frame_timestamp_ms = 0

    # ----------------------------------------------------------------
    # 内部 — 懒加载
    # ----------------------------------------------------------------

    @classmethod
    def _get_detector(cls) -> vision.HandLandmarker:
        if cls._detector is not None:
            return cls._detector

        if cls._model_path is None or not os.path.exists(cls._model_path):
            raise FileNotFoundError(
                f"手部关键点模型未找到: {cls._model_path}\n"
                "请先下载 hand_landmarker.task 到 backend/models/ 目录"
            )

        base_options = mp_python.BaseOptions(model_asset_path=cls._model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=cls._num_hands,
            min_hand_detection_confidence=cls._min_detection_confidence,
            min_hand_presence_confidence=cls._min_presence_confidence,
            min_tracking_confidence=cls._min_tracking_confidence,
            running_mode=vision.RunningMode.VIDEO,
        )
        cls._detector = vision.HandLandmarker.create_from_options(options)
        return cls._detector

    # ----------------------------------------------------------------
    # 推理
    # ----------------------------------------------------------------

    @classmethod
    def infer(cls, image_bgr: np.ndarray, timestamp_ms: int | None = None) -> list[list[dict]]:
        """
        对单帧 BGR 图像进行手部关键点检测。

        Args:
            image_bgr: OpenCV BGR 格式的帧 (H, W, 3)
            timestamp_ms: 毫秒时间戳（VIDEO 模式必需，None 时自动递增）

        Returns:
            list[list[dict]]  — 每只手一个 list，每个关键点 {"x","y","z"}
        """
        detector = cls._get_detector()

        # BGR → RGB，MediaPipe 要求 RGB
        image_rgb = cv2_cvt_color_bgr_to_rgb(image_bgr)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # 时间戳管理
        if timestamp_ms is None:
            cls._frame_timestamp_ms += 33  # 约 30 fps
            timestamp_ms = cls._frame_timestamp_ms

        result = detector.detect_for_video(mp_image, timestamp_ms)

        hands: list[list[dict]] = []
        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                keypoints = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_lms]
                hands.append(keypoints)

        return hands


def cv2_cvt_color_bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """OpenCV BGR → RGB（避免直接 import cv2 造成循环依赖）。"""
    return image_bgr[:, :, ::-1].copy()
