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
        if plate_region is None or getattr(plate_region, "size", 0) == 0:
            return {
                "plate_number": "",
                "plate_color": plate_color,
                "confidence": 0.0,
                "source": "paddleocr",
            }

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
                "\u7f3a\u5c11 PaddleOCR \u6216 PaddlePaddle\uff0c\u8bf7\u5148\u5b89\u88c5 OCR \u4f9d\u8d56\u3002"
            ) from exc

        if self._ocr is None:
            self._ocr = PaddleOCR(
                use_angle_cls=settings.paddleocr_use_angle_cls,
                lang=settings.paddleocr_language,
                show_log=False,
            )

        return self._ocr

    def _load_image_array(self, image_source: Any):
        try:
            import numpy as np
            from PIL import Image, UnidentifiedImageError
        except ImportError as exc:
            raise InferenceDependencyError(
                "\u7f3a\u5c11 numpy \u6216 Pillow\uff0c\u8bf7\u5148\u5b89\u88c5\u56fe\u50cf\u5904\u7406\u4f9d\u8d56\u3002"
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
            raise ValueError(
                "\u65e0\u6cd5\u89e3\u6790\u4e0a\u4f20\u56fe\u7247\uff0c\u8bf7\u786e\u8ba4\u6587\u4ef6\u662f\u6709\u6548\u7684\u56fe\u7247\u683c\u5f0f\u3002"
            ) from exc

    def _crop_plate_region(self, image, bbox: list[int] | None):
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

        cropped = image[y1:y2, x1:x2]
        return cropped if getattr(cropped, "size", 0) > 0 else image

    def _normalize_plate_text(self, text: str) -> str:
        normalized = text.upper().replace(" ", "")
        normalized = re.sub(r"[^0-9A-Z\u4e00-\u9fff]", "", normalized)
        return normalized

    def _focus_plate_region(self, plate_region):
        if plate_region is None or getattr(plate_region, "size", 0) == 0:
            return plate_region

        height, width = plate_region.shape[:2]
        if height < 6 or width < 12:
            return plate_region

        top = max(int(round(height * 0.12)), 0)
        bottom = min(int(round(height * 0.88)), height)
        left = max(int(round(width * 0.08)), 0)
        right = min(int(round(width * 0.92)), width)
        focused = plate_region[top:bottom, left:right]
        return focused if getattr(focused, "size", 0) > 0 else plate_region

    def _classify_plate_color(self, plate_region) -> str:
        try:
            import cv2
            import numpy as np
        except ImportError:
            return "\u672a\u77e5"

        if plate_region is None or getattr(plate_region, "size", 0) == 0:
            return "\u672a\u77e5"
        if getattr(plate_region, "ndim", 0) < 3:
            return "\u672a\u77e5"
        if plate_region.shape[0] == 0 or plate_region.shape[1] == 0:
            return "\u672a\u77e5"

        focus_region = self._focus_plate_region(plate_region)
        hsv_image = cv2.cvtColor(focus_region, cv2.COLOR_RGB2HSV)
        hue_channel = hsv_image[:, :, 0]
        saturation_channel = hsv_image[:, :, 1]
        value_channel = hsv_image[:, :, 2]

        colorful_mask = (saturation_channel >= 45) & (value_channel >= 40)
        if not any(bool(value) for value in colorful_mask.flatten()):
            brightness = float(value_channel.mean()) if value_channel.size else 0.0
            return "\u767d\u724c" if brightness >= 170 else "\u9ed1\u724c"

        hue_values = hue_channel[colorful_mask]
        saturation_values = saturation_channel[colorful_mask]
        value_values = value_channel[colorful_mask]
        if hue_values.size == 0:
            return "\u672a\u77e5"

        vivid_mask = (saturation_values >= 75) & (value_values >= 60)
        if vivid_mask.any():
            hue_values = hue_values[vivid_mask]
            saturation_values = saturation_values[vivid_mask]
            value_values = value_values[vivid_mask]

        yellow_mask = (hue_values >= 10) & (hue_values <= 40) & (value_values >= 70)
        green_mask = (hue_values >= 43) & (hue_values <= 92) & (saturation_values >= 65)
        blue_mask = (hue_values >= 94) & (hue_values <= 138) & (saturation_values >= 60)

        color_scores = {
            "\u84dd\u724c": int(blue_mask.sum()),
            "\u9ec4\u724c": int(yellow_mask.sum()),
            "\u7eff\u724c": int(green_mask.sum()),
        }
        if not any(color_scores.values()):
            return "\u672a\u77e5"

        yellow_score = color_scores["\u9ec4\u724c"]
        green_score = color_scores["\u7eff\u724c"]
        if yellow_score >= max(green_score + 8, int(green_score * 1.08), 5):
            return "\u9ec4\u724c"

        color_name, color_score = max(color_scores.items(), key=lambda item: item[1])
        return color_name if color_score > 0 else "\u672a\u77e5"
