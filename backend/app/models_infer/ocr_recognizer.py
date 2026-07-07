from __future__ import annotations

import re
from io import BytesIO
from typing import Any

from app.core.config import settings
from app.models_infer.errors import InferenceDependencyError


class OCRRecognizer:
    def __init__(self) -> None:
        self._ocr = None

    def recognize(self, image_source: Any, bbox: list[int] | None = None) -> dict:
        image = self._load_image_array(image_source)
        plate_region = self._crop_plate_region(image, bbox)
        plate_color = self._classify_plate_color(plate_region)
        ocr = self._load_ocr()
        result = ocr.ocr(plate_region, cls=settings.paddleocr_use_angle_cls)

        lines = result[0] if result else []
        text_fragments: list[str] = []
        confidences: list[float] = []

        for line in lines or []:
            if len(line) < 2:
                continue
            text = str(line[1][0]).strip()
            score = float(line[1][1]) if len(line[1]) > 1 else 0.0
            if text:
                text_fragments.append(text)
                confidences.append(score)

        plate_number = self._normalize_plate_text("".join(text_fragments))
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return {
            "plate_number": plate_number,
            "plate_color": plate_color,
            "confidence": confidence,
            "source": "paddleocr",
        }

    def _load_ocr(self):
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise InferenceDependencyError(
                "缺少 PaddleOCR 或 PaddlePaddle，请先安装 OCR 依赖。"
            ) from exc

        if self._ocr is None:
            self._ocr = PaddleOCR(
                use_angle_cls=settings.paddleocr_use_angle_cls,
                lang=settings.paddleocr_language,
                show_log=False,
            )

        return self._ocr

    def _load_image_array(self, image_source: Any) -> np.ndarray:
        try:
            import numpy as np
            from PIL import Image, UnidentifiedImageError
        except ImportError as exc:
            raise InferenceDependencyError(
                "缺少 numpy 或 Pillow，请先安装图像处理依赖。"
            ) from exc

        try:
            if isinstance(image_source, np.ndarray):
                return image_source
            if isinstance(image_source, (bytes, bytearray)):
                image = Image.open(BytesIO(image_source))
            else:
                image = Image.open(image_source)
            return np.asarray(image.convert("RGB"))
        except (FileNotFoundError, UnidentifiedImageError, OSError, TypeError) as exc:
            raise ValueError("无法解析上传图片，请确认文件是有效的图片格式。") from exc

    def _crop_plate_region(self, image: np.ndarray, bbox: list[int] | None) -> np.ndarray:
        if not bbox:
            return image

        image_height, image_width = image.shape[:2]
        x, y, width, height = bbox
        x1 = max(int(x), 0)
        y1 = max(int(y), 0)
        x2 = min(x1 + max(int(width), 1), image_width)
        y2 = min(y1 + max(int(height), 1), image_height)

        if x1 >= x2 or y1 >= y2:
            return image

        return image[y1:y2, x1:x2]

    def _normalize_plate_text(self, text: str) -> str:
        normalized = text.upper().replace(" ", "")
        normalized = re.sub(r"[^0-9A-Z\u4e00-\u9fff]", "", normalized)
        return normalized

    def _classify_plate_color(self, plate_region: np.ndarray) -> str:
        try:
            import cv2
        except ImportError:
            return "未知"
        import numpy as np

        hsv_image = cv2.cvtColor(plate_region, cv2.COLOR_RGB2HSV)
        hue_channel = hsv_image[:, :, 0]
        saturation_channel = hsv_image[:, :, 1]
        value_channel = hsv_image[:, :, 2]

        colorful_mask = saturation_channel > 40
        if not np.any(colorful_mask):
            brightness = float(value_channel.mean())
            return "白牌" if brightness >= 170 else "黑牌"

        hue_values = hue_channel[colorful_mask]
        color_scores = {
            "蓝牌": int(np.count_nonzero((hue_values >= 90) & (hue_values <= 130))),
            "黄牌": int(np.count_nonzero((hue_values >= 15) & (hue_values <= 40))),
            "绿牌": int(np.count_nonzero((hue_values >= 35) & (hue_values <= 89))),
        }
        color_name, color_score = max(color_scores.items(), key=lambda item: item[1])
        if color_score == 0:
            return "未知"
        return color_name
