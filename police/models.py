"""
models.py — MediaPipe Pose/Hand 模型加载与推理

包含：
  - create_pose_detector : 创建 PoseLandmarker
  - create_hand_detector : 创建 HandLandmarker
  - detect_pose          : 执行 Pose 推理
  - detect_hand          : 执行 Hand 推理

使用 mediapipe.tasks API（新版），返回世界坐标和归一化关键点。
"""

import ctypes
import os
import platform
from importlib import resources

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import mediapipe_c_bindings

from . import config


def _patch_mediapipe_windows_free() -> None:
    """Patch MediaPipe Tasks on Windows when libmediapipe.dll lacks free()."""
    if os.name != "nt":
        return
    if getattr(mediapipe_c_bindings, "_windows_free_patch_applied", False):
        return

    def load_raw_library_compat(signatures=()):
        shared_lib = mediapipe_c_bindings._shared_lib
        if shared_lib is None:
            if os.name == "posix":
                if platform.system() == "Darwin":
                    lib_filename = "libmediapipe.dylib"
                else:
                    lib_filename = "libmediapipe.so"
            else:
                lib_filename = "libmediapipe.dll"
            lib_path_context = resources.files("mediapipe.tasks.c")
            absolute_lib_path = str(lib_path_context / lib_filename)
            shared_lib = ctypes.CDLL(absolute_lib_path)
            mediapipe_c_bindings._shared_lib = shared_lib

        for signature in signatures:
            c_func = getattr(shared_lib, signature.func_name)
            c_func.argtypes = signature.argtypes
            c_func.restype = signature.restype

        try:
            free_func = shared_lib.free
        except AttributeError:
            # MediaPipe 0.10.30 on Windows may not export free from libmediapipe.dll.
            free_func = ctypes.CDLL("ucrtbase.dll").free
            shared_lib.free = free_func

        free_func.argtypes = [ctypes.c_void_p]
        free_func.restype = None
        return shared_lib

    mediapipe_c_bindings.load_raw_library = load_raw_library_compat
    mediapipe_c_bindings._windows_free_patch_applied = True


_patch_mediapipe_windows_free()


# ============================================================
# 模型创建
# ============================================================

def create_pose_detector(model_path: str) -> vision.PoseLandmarker:
    """
    创建 MediaPipe PoseLandmarker。

    Args:
        model_path: Pose 模型文件路径（.task）

    Returns:
        配置好的 PoseLandmarker 实例
    """
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=config.NUM_POSES,
        min_pose_detection_confidence=config.POSE_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=config.POSE_PRESENCE_CONFIDENCE,
        min_tracking_confidence=config.POSE_TRACKING_CONFIDENCE,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


def create_hand_detector(model_path: str) -> vision.HandLandmarker:
    """
    创建 MediaPipe HandLandmarker（VIDEO 模式，启用帧间跟踪）。

    VIDEO 模式的跟踪器在帧间保持手部状态，配合 detect_hand_video()
    大幅减少断断续续的问题。需要传入单调递增的时间戳(ms)。

    Args:
        model_path: Hand 模型文件路径（.task）

    Returns:
        配置好的 HandLandmarker 实例
    """
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=config.NUM_HANDS,
        min_hand_detection_confidence=config.HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=config.HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=config.HAND_TRACKING_CONFIDENCE,
    )
    return vision.HandLandmarker.create_from_options(options)


# ============================================================
# 推理执行
# ============================================================

# 首次异常时打印一次调试信息
_debug_hand_once = True


def detect_pose(pose_detector: vision.PoseLandmarker,
                frame: np.ndarray) -> vision.PoseLandmarkerResult:
    """
    对单帧图像执行 Pose 推理。

    自动完成 BGR→RGB 转换及缩放（使用 config.INFER_SCALE）。

    Args:
        pose_detector: PoseLandmarker 实例
        frame:         原始 BGR 图像（全尺寸）

    Returns:
        PoseLandmarkerResult，包含 pose_landmarks 和 pose_world_landmarks
    """
    small_frame = cv2.resize(frame, (0, 0),
                             fx=config.INFER_SCALE,
                             fy=config.INFER_SCALE)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)
    return pose_detector.detect(mp_image)


def detect_hand(hand_detector: vision.HandLandmarker,
                frame: np.ndarray,
                timestamp_ms: int = 0) -> vision.HandLandmarkerResult | None:
    """
    对单帧图像执行 Hand 推理（VIDEO 模式）。

    使用独立于 Pose 的缩放比（HAND_INFER_SCALE），默认全分辨率，
    确保手部细节不被压缩抹平。VIDEO 模式配合帧间跟踪器，大幅减少丢帧。

    Args:
        hand_detector: HandLandmarker 实例，或 None
        frame:         原始 BGR 图像（全尺寸）
        timestamp_ms:  单调递增的时间戳(ms)，VIDEO 模式必需

    Returns:
        HandLandmarkerResult，或 None（无 hand_detector 或推理失败）
    """
    if hand_detector is None:
        return None
    try:
        hs = config.HAND_INFER_SCALE
        if hs < 1.0:
            frame = cv2.resize(frame, (0, 0), fx=hs, fy=hs)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return hand_detector.detect_for_video(mp_image, timestamp_ms)
    except Exception as e:
        global _debug_hand_once
        if _debug_hand_once:
            print(f"[Hand] 推理异常: {e}")
            _debug_hand_once = False
        return None
