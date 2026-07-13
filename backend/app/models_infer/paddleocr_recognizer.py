from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import io
import numbers
import os
import re
from typing import Any
import warnings

from app.core.config import settings
from app.models_infer.errors import InferenceDependencyError


PROVINCE_PREFIXES = (
    "\u4eac\u6caa\u6d25\u6e1d\u5180\u664b\u8499\u8fbd\u5409\u9ed1\u82cf\u6d59\u7696"
    "\u95fd\u8d63\u9c81\u8c6b\u9102\u6e58\u7ca4\u6842\u743c\u5ddd\u8d35\u4e91\u85cf"
    "\u9655\u7518\u9752\u5b81\u65b0"
)


@dataclass
class PaddleOCRDetection:
    plate_number: str
    plate_color: str
    confidence: float
    bbox: list[int]
    source: str = "paddleocr"


class PaddleOCRRecognizer:
    def __init__(self) -> None:
        self._ocr = None
        self._text_recognizer = None

    def reset_runtime(self) -> None:
        self._ocr = None
        self._text_recognizer = None

    def is_available(self) -> bool:
        if not settings.plate_ocr_enabled:
            return False
        try:
            from paddleocr import PaddleOCR  # noqa: F401
        except ImportError:
            return False
        return True

    def warmup(self, *, silent: bool = True) -> None:
        if not self.is_available():
            return

        def _load() -> None:
            self._configure_paddle_runtime()
            self._load_text_recognizer()
            self._load_ocr()

        if not silent:
            _load()
            return

        buffer = io.StringIO()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*No ccache found.*")
            with redirect_stdout(buffer), redirect_stderr(buffer):
                _load()

    def recognize(
        self,
        image_source: Any,
        confidence_threshold: float | None = None,
        *,
        allow_ocr_fallback: bool = True,
    ) -> PaddleOCRDetection | None:
        image = self._load_image_array(image_source)
        if image is None or getattr(image, "size", 0) == 0:
            return None

        threshold = self._resolve_confidence_threshold(confidence_threshold)
        result = self._predict_text(image)
        best = self._pick_best_text_recognition_candidate(result, threshold)
        bbox: list[int] | None = None
        if best is None:
            if not allow_ocr_fallback:
                return None
            fallback = self._pick_best_text_candidate(self._predict(image), max(threshold * 0.92, 0.22))
            if fallback is None:
                return None
            plate_number, confidence, bbox = fallback
        else:
            plate_number, confidence = best

        height, width = image.shape[:2]
        return PaddleOCRDetection(
            plate_number=plate_number,
            plate_color=self._classify_plate_color(image),
            confidence=confidence,
            bbox=[0, 0, max(int(width), 1), max(int(height), 1)],
        )

    def recognize_all(
        self,
        image_source: Any,
        *,
        max_side_override: int | None = None,
        aggressive: bool = False,
        heavy_scan: bool = False,
        confidence_threshold: float | None = None,
    ) -> list[PaddleOCRDetection]:
        source_image = self._load_image_array(image_source)
        if source_image is None or getattr(source_image, "size", 0) == 0:
            return []

        limit = settings.plate_max_image_side if max_side_override is None else max_side_override
        if isinstance(limit, int) and limit > 0:
            image, scale_ratio = self._resize_image_to_limit(source_image, limit)
        else:
            image = source_image
            scale_ratio = 1.0

        threshold = self._resolve_confidence_threshold(confidence_threshold)
        if aggressive:
            threshold = min(threshold, settings.plate_ocr_confidence_threshold * 0.85)
        if heavy_scan:
            threshold = min(threshold, settings.plate_ocr_confidence_threshold * 0.8)

        result = self._predict(image)
        detections: list[PaddleOCRDetection] = []
        for item in self._collect_detection_items(result, scale_ratio=scale_ratio):
            bbox, raw_text, raw_confidence = item
            plate_number = self._normalize_plate_text(raw_text)
            confidence = float(raw_confidence)
            if confidence < threshold or not self._is_valid_plate_text(plate_number):
                continue

            crop = self._crop_with_bbox(source_image, bbox)
            if crop is None:
                continue

            detections.append(
                PaddleOCRDetection(
                    plate_number=plate_number,
                    plate_color=self._classify_plate_color(crop),
                    confidence=confidence,
                    bbox=bbox,
                )
            )

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return self._deduplicate_detections(detections)

    def _load_ocr(self):
        self._configure_paddle_runtime()
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise InferenceDependencyError(
                "Missing PaddleOCR or PaddlePaddle. Install OCR dependencies in the backend environment first."
            ) from exc

        if self._ocr is None:
            self._ocr = PaddleOCR(
                lang=settings.paddleocr_language,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=settings.paddleocr_use_angle_cls,
            )

        return self._ocr

    def _load_text_recognizer(self):
        self._configure_paddle_runtime()
        try:
            from paddleocr import TextRecognition
        except ImportError as exc:
            raise InferenceDependencyError(
                "Missing PaddleOCR or PaddlePaddle. Install OCR dependencies in the backend environment first."
            ) from exc

        if self._text_recognizer is None:
            kwargs: dict[str, Any] = {}
            configured_model_name = settings.paddleocr_text_recognition_model_name.strip()
            if configured_model_name:
                kwargs["model_name"] = configured_model_name
            self._text_recognizer = TextRecognition(**kwargs)

        return self._text_recognizer

    def _predict(self, image) -> Any:
        return self._load_ocr().predict(
            image,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=settings.paddleocr_use_angle_cls,
        )

    def _predict_text(self, image) -> Any:
        return self._load_text_recognizer().predict(image)

    def _load_image_array(self, image_source: Any):
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise InferenceDependencyError(
                "Missing opencv-python-headless or numpy. Install image-processing dependencies first."
            ) from exc

        if isinstance(image_source, np.ndarray):
            image = image_source
        elif isinstance(image_source, (bytes, bytearray)):
            encoded = np.frombuffer(image_source, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(str(image_source), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to parse the input image.")
        if getattr(image, "ndim", 0) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def _resize_image_to_limit(self, image, limit: int):
        try:
            import cv2
        except ImportError as exc:
            raise InferenceDependencyError("Missing OpenCV for PaddleOCR preprocessing.") from exc

        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= 0 or longest <= limit:
            return image, 1.0

        scale_ratio = longest / float(limit)
        resized = cv2.resize(
            image,
            (max(int(round(width / scale_ratio)), 1), max(int(round(height / scale_ratio)), 1)),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale_ratio

    def _collect_detection_items(self, value: Any, *, scale_ratio: float) -> list[tuple[list[int], str, float]]:
        collected: list[tuple[list[int], str, float]] = []
        for item in self._iterate_prediction_results(value):
            texts = self._normalize_string_list(self._result_field(item, "rec_texts"))
            scores = self._normalize_score_list(self._result_field(item, "rec_scores"), len(texts))
            polygons = self._normalize_polygon_list(
                self._result_field(item, "rec_polys") or self._result_field(item, "dt_polys"),
                len(texts),
            )
            for index, text in enumerate(texts):
                if not text:
                    continue
                bbox = self._polygon_to_bbox(polygons[index], scale_ratio) if index < len(polygons) else None
                if bbox is None:
                    bbox = self._full_image_bbox(item, scale_ratio)
                if bbox is None:
                    continue
                collected.append((bbox, text, scores[index]))
        return collected

    def _collect_text_items(self, value: Any) -> list[tuple[str, float, list[int] | None]]:
        collected: list[tuple[str, float, list[int] | None]] = []
        for item in self._iterate_prediction_results(value):
            texts = self._normalize_string_list(self._result_field(item, "rec_texts"))
            scores = self._normalize_score_list(self._result_field(item, "rec_scores"), len(texts))
            polygons = self._normalize_polygon_list(self._result_field(item, "rec_polys"), len(texts))
            for index, text in enumerate(texts):
                if not text:
                    continue
                bbox = self._polygon_to_bbox(polygons[index], 1.0) if index < len(polygons) else None
                collected.append((text, scores[index], bbox))
        return collected

    def _pick_best_text_candidate(self, result: Any, threshold: float) -> tuple[str, float, list[int] | None] | None:
        best: tuple[str, float, list[int] | None] | None = None
        for raw_text, confidence, bbox in self._collect_text_items(result):
            plate_number = self._normalize_plate_text(raw_text)
            if confidence < threshold or not self._is_valid_plate_text(plate_number):
                continue
            if best is None or confidence > best[1]:
                best = (plate_number, confidence, bbox)
        return best

    def _pick_best_text_recognition_candidate(self, result: Any, threshold: float) -> tuple[str, float] | None:
        best: tuple[str, float] | None = None
        for item in self._iterate_prediction_results(result):
            raw_text = str(self._result_field(item, "rec_text") or "").strip()
            raw_confidence = self._result_field(item, "rec_score")
            confidence = float(raw_confidence) if isinstance(raw_confidence, numbers.Real) else 0.0
            plate_number = self._normalize_plate_text(raw_text)
            if confidence < threshold or not self._is_valid_plate_text(plate_number):
                continue
            if best is None or confidence > best[1]:
                best = (plate_number, confidence)
        return best

    def _iterate_prediction_results(self, value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    def _result_field(self, item: Any, field: str):
        if isinstance(item, dict):
            return item.get(field)
        return getattr(item, field, None)

    def _normalize_string_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value]
        return [str(value).strip()]

    def _normalize_score_list(self, value: Any, expected_length: int) -> list[float]:
        if value is None:
            return [0.0] * expected_length
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            scores = [float(item) for item in value]
        else:
            scores = [float(value)]
        if len(scores) < expected_length:
            scores.extend([0.0] * (expected_length - len(scores)))
        return scores[:expected_length]

    def _normalize_polygon_list(self, value: Any, expected_length: int) -> list[list[list[float]]]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            return []

        polygons: list[list[list[float]]] = []
        for item in value:
            polygon = self._as_polygon(item)
            if polygon is not None:
                polygons.append(polygon)
        return polygons[:expected_length] if expected_length > 0 else polygons

    def _as_polygon(self, value: Any) -> list[list[float]] | None:
        if not isinstance(value, (list, tuple)):
            return None
        points: list[list[float]] = []
        for point in value:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                return None
            if not isinstance(point[0], numbers.Real) or not isinstance(point[1], numbers.Real):
                return None
            points.append([float(point[0]), float(point[1])])
        return points if points else None

    def _polygon_to_bbox(self, polygon: list[list[float]] | None, scale_ratio: float) -> list[int] | None:
        if not polygon:
            return None
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        left = int(round(min(xs) * scale_ratio))
        top = int(round(min(ys) * scale_ratio))
        right = int(round(max(xs) * scale_ratio))
        bottom = int(round(max(ys) * scale_ratio))
        return [left, top, max(right - left, 1), max(bottom - top, 1)]

    def _full_image_bbox(self, item: Any, scale_ratio: float) -> list[int] | None:
        image = self._result_field(item, "input_img")
        if image is None or not hasattr(image, "shape") or len(image.shape) < 2:
            return None
        height, width = image.shape[:2]
        return [0, 0, max(int(round(width * scale_ratio)), 1), max(int(round(height * scale_ratio)), 1)]

    def _crop_with_bbox(self, image, bbox: list[int]):
        x, y, width, height = bbox
        image_height, image_width = image.shape[:2]
        left = max(int(x), 0)
        top = max(int(y), 0)
        right = min(left + max(int(width), 1), image_width)
        bottom = min(top + max(int(height), 1), image_height)
        if left >= right or top >= bottom:
            return None
        crop = image[top:bottom, left:right]
        return crop if getattr(crop, "size", 0) > 0 else None

    def _deduplicate_detections(self, detections: list[PaddleOCRDetection]) -> list[PaddleOCRDetection]:
        kept: list[PaddleOCRDetection] = []
        for detection in detections:
            duplicate = False
            for existing in kept:
                if detection.plate_number != existing.plate_number:
                    continue
                if self._compute_iou(detection.bbox, existing.bbox) >= 0.15:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(detection)
        return kept

    def _compute_iou(self, bbox_a: list[int], bbox_b: list[int]) -> float:
        ax1, ay1, aw, ah = bbox_a
        bx1, by1, bw, bh = bbox_b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        inter_left = max(ax1, bx1)
        inter_top = max(ay1, by1)
        inter_right = min(ax2, bx2)
        inter_bottom = min(ay2, by2)
        inter_width = max(0, inter_right - inter_left)
        inter_height = max(0, inter_bottom - inter_top)
        inter_area = inter_width * inter_height

        area_a = max(aw, 0) * max(ah, 0)
        area_b = max(bw, 0) * max(bh, 0)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def _classify_plate_color(self, plate_region) -> str:
        try:
            import cv2
        except ImportError:
            return "\u672a\u77e5"

        if plate_region is None or getattr(plate_region, "size", 0) == 0:
            return "\u672a\u77e5"
        if getattr(plate_region, "ndim", 0) < 3:
            return "\u672a\u77e5"
        if plate_region.shape[0] == 0 or plate_region.shape[1] == 0:
            return "\u672a\u77e5"

        focus_region = self._focus_plate_region(plate_region)
        hsv_image = cv2.cvtColor(focus_region, cv2.COLOR_BGR2HSV)
        hue_channel = hsv_image[:, :, 0]
        saturation_channel = hsv_image[:, :, 1]
        value_channel = hsv_image[:, :, 2]

        colorful_mask = (saturation_channel >= 45) & (value_channel >= 40)
        if not colorful_mask.any():
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

        yellow_mask = (hue_values >= 10) & (hue_values <= 40) & (value_values >= 74)
        green_mask = (hue_values >= 43) & (hue_values <= 92) & (saturation_values >= 65)
        blue_mask = (hue_values >= 92) & (hue_values <= 140) & (saturation_values >= 58)
        deep_blue_mask = (hue_values >= 98) & (hue_values <= 132) & (saturation_values >= 72)

        color_scores = {
            "\u84dd\u724c": int(blue_mask.sum() + deep_blue_mask.sum() * 0.18),
            "\u9ec4\u724c": int(yellow_mask.sum()),
            "\u7eff\u724c": int(green_mask.sum()),
        }
        if not any(color_scores.values()):
            return "\u672a\u77e5"

        blue_score = color_scores["\u84dd\u724c"]
        yellow_score = color_scores["\u9ec4\u724c"]
        green_score = color_scores["\u7eff\u724c"]
        if blue_score >= max(yellow_score + 3, int(yellow_score * 0.9), 4):
            return "\u84dd\u724c"
        if yellow_score >= max(green_score + 8, int(green_score * 1.08), 5):
            return "\u9ec4\u724c"

        color_name, color_score = max(color_scores.items(), key=lambda item: item[1])
        return color_name if color_score > 0 else "\u672a\u77e5"

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

    def _normalize_plate_text(self, text: str) -> str:
        normalized = str(text or "").strip().upper().replace(" ", "")
        return re.sub(r"[^0-9A-Z\u4e00-\u9fff]", "", normalized)

    def _is_valid_plate_text(self, plate_number: str) -> bool:
        if len(plate_number) not in (7, 8):
            return False
        if plate_number[0] not in PROVINCE_PREFIXES:
            return False
        if not re.fullmatch(r"[A-Z]", plate_number[1]):
            return False
        return re.fullmatch(r"[A-Z0-9]{5,6}", plate_number[2:]) is not None

    def _resolve_confidence_threshold(self, confidence_threshold: float | None) -> float:
        if confidence_threshold is not None:
            return float(confidence_threshold)
        return settings.plate_ocr_confidence_threshold

    def _configure_paddle_runtime(self) -> None:
        os.environ.setdefault("GLOG_minloglevel", "2")
        os.environ.setdefault("FLAGS_logtostderr", "0")
