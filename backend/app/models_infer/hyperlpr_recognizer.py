from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any
import warnings

from app.core.config import settings
from app.models_infer.errors import InferenceDependencyError


PROVINCE_PREFIXES = (
    "\u4eac\u6d25\u6caa\u6e1d\u5180\u8c6b\u4e91\u8fbd\u9ed1\u6e58\u7696\u9c81"
    "\u65b0\u82cf\u6d59\u8d63\u9102\u6842\u7518\u664b\u8499\u9655\u5409\u95fd\u8d35"
    "\u7ca4\u9752\u85cf\u5ddd\u5b81\u743c"
)

PLATE_TYPE_TO_COLOR = {
    0: "\u84dd\u724c",
    1: "\u7eff\u724c",
    2: "\u9ec4\u724c",
    3: "\u7eff\u724c",
    4: "\u9ed1\u724c",
    5: "\u9ed1\u724c",
    6: "\u9ed1\u724c",
    7: "\u9ed1\u724c",
    8: "\u9ed1\u724c",
    9: "\u9ec4\u724c",
}


@dataclass
class HyperLPRDetection:
    plate_number: str
    plate_color: str
    confidence: float
    bbox: list[int]


class HyperLPRRecognizer:
    def __init__(self) -> None:
        self._catcher = None

    def recognize_all(
        self,
        image_source: bytes | bytearray | Any,
        max_side_override: int | None = None,
        aggressive: bool = False,
        heavy_scan: bool = True,
        confidence_threshold: float | None = None,
    ) -> list[HyperLPRDetection]:
        image, scale_ratio = self._load_image(image_source, max_side_override=max_side_override)
        threshold = settings.plate_confidence_threshold if confidence_threshold is None else confidence_threshold

        detections = self._run_recognition(image, scale_ratio, confidence_threshold=threshold)
        if detections and not aggressive:
            return detections

        enhanced_image, enhanced_ratio = self._build_enhanced_variant(image, scale_ratio)
        if enhanced_image is not image:
            detections.extend(
                self._run_recognition(
                    enhanced_image,
                    enhanced_ratio,
                    confidence_threshold=threshold,
                )
            )

        if aggressive:
            detections.extend(
                self._run_focus_scan(
                    image,
                    scale_ratio,
                    confidence_threshold=threshold,
                )
            )
            if heavy_scan:
                detections.extend(self._run_tile_scan(image, scale_ratio, confidence_threshold=threshold))
                if enhanced_image is not image:
                    detections.extend(
                        self._run_tile_scan(
                            enhanced_image,
                            enhanced_ratio,
                            confidence_threshold=threshold,
                        )
                    )

        return self._deduplicate_detections(detections)

    def _run_recognition(
        self,
        image,
        scale_ratio: float,
        *,
        confidence_threshold: float,
    ) -> list[HyperLPRDetection]:
        catcher = self._load_catcher()
        raw_results = self._call_catcher(catcher, image)
        return self._normalize_results(raw_results, scale_ratio, confidence_threshold=confidence_threshold)

    def _run_tile_scan(
        self,
        image,
        scale_ratio: float,
        *,
        confidence_threshold: float,
    ) -> list[HyperLPRDetection]:
        height, width = image.shape[:2]
        if width < 320 or height < 180:
            return []

        tile_plans: list[tuple[int, int]] = [(2, 2), (3, 2)]
        if max(width, height) >= 1100:
            tile_plans.append((3, 3))

        collected: list[HyperLPRDetection] = []
        for cols, rows in tile_plans:
            tile_width = min(width, max(int(round(width / cols * 1.45)), int(round(width / cols))))
            tile_height = min(height, max(int(round(height / rows * 1.45)), int(round(height / rows))))
            x_positions = self._build_tile_positions(width, tile_width, cols)
            y_positions = self._build_tile_positions(height, tile_height, rows)

            for offset_y in y_positions:
                for offset_x in x_positions:
                    crop = image[offset_y : offset_y + tile_height, offset_x : offset_x + tile_width]
                    if crop.size == 0:
                        continue
                    collected.extend(
                        self._run_recognition_with_offset(
                            crop,
                            scale_ratio,
                            offset_x=offset_x,
                            offset_y=offset_y,
                            confidence_threshold=confidence_threshold,
                        )
                    )

        return self._deduplicate_detections(collected)

    def _run_focus_scan(
        self,
        image,
        scale_ratio: float,
        *,
        confidence_threshold: float,
    ) -> list[HyperLPRDetection]:
        height, width = image.shape[:2]
        if width < 240 or height < 140:
            return []

        collected: list[HyperLPRDetection] = []
        for offset_x, offset_y, crop_width, crop_height in self._build_focus_crops(width, height):
            crop = image[offset_y : offset_y + crop_height, offset_x : offset_x + crop_width]
            if crop.size == 0:
                continue
            collected.extend(
                self._run_recognition_with_offset(
                    crop,
                    scale_ratio,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    confidence_threshold=confidence_threshold,
                )
            )

        return self._deduplicate_detections(collected)

    def _build_tile_positions(self, full_size: int, tile_size: int, target_count: int) -> list[int]:
        if full_size <= tile_size or target_count <= 1:
            return [0]

        last_start = max(full_size - tile_size, 0)
        positions: list[int] = []
        for index in range(target_count):
            ratio = index / max(target_count - 1, 1)
            position = int(round(last_start * ratio))
            if not positions or abs(position - positions[-1]) > 4:
                positions.append(position)

        if positions[-1] != last_start:
            positions.append(last_start)
        return positions

    def _build_focus_crops(self, width: int, height: int) -> list[tuple[int, int, int, int]]:
        if width <= 0 or height <= 0:
            return []

        crops: list[tuple[int, int, int, int]] = []
        lower_top = int(round(height * 0.28))
        lower_height = max(height - lower_top, 1)
        crops.append((0, lower_top, width, lower_height))

        if width >= 480:
            half_width = int(round(width * 0.62))
            left_positions = self._build_tile_positions(width, half_width, 2)
            for left in left_positions:
                crops.append((left, lower_top, half_width, lower_height))

        if height >= 480:
            mid_top = int(round(height * 0.18))
            mid_height = max(int(round(height * 0.58)), 1)
            crops.append((0, mid_top, width, mid_height))

        unique: list[tuple[int, int, int, int]] = []
        seen: set[tuple[int, int, int, int]] = set()
        for crop in crops:
            if crop not in seen:
                seen.add(crop)
                unique.append(crop)
        return unique

    def _run_recognition_with_offset(
        self,
        image,
        scale_ratio: float,
        *,
        offset_x: int,
        offset_y: int,
        confidence_threshold: float,
    ) -> list[HyperLPRDetection]:
        catcher = self._load_catcher()
        raw_results = self._call_catcher(catcher, image)
        detections = self._normalize_results(raw_results, scale_ratio, confidence_threshold=confidence_threshold)
        offset_scale = 1.0 / scale_ratio if scale_ratio > 0 else 1.0
        translated_x = int(round(offset_x * offset_scale))
        translated_y = int(round(offset_y * offset_scale))

        for detection in detections:
            detection.bbox[0] += translated_x
            detection.bbox[1] += translated_y
        return detections

    def _call_catcher(self, catcher, image):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice.*", category=RuntimeWarning)
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in scalar divide.*",
                category=RuntimeWarning,
            )
            return catcher(image)

    def _normalize_results(
        self,
        raw_results: Any,
        scale_ratio: float,
        *,
        confidence_threshold: float,
    ) -> list[HyperLPRDetection]:
        detections: list[HyperLPRDetection] = []
        for item in raw_results or []:
            normalized = self._normalize_result(item, scale_ratio)
            if normalized is None:
                continue
            if normalized.confidence < confidence_threshold:
                continue
            detections.append(normalized)

        detections.sort(key=lambda item: (item.confidence, item.bbox[2] * item.bbox[3]), reverse=True)
        return self._deduplicate_detections(detections)

    def _load_catcher(self):
        model_root = self._configure_runtime_home()
        try:
            import hyperlpr3 as lpr3
        except ImportError as exc:
            raise InferenceDependencyError(
                "\u7f3a\u5c11 hyperlpr3\uff0c\u8bf7\u5148\u5728\u540e\u7aef\u73af\u5883\u4e2d\u5b89\u88c5 HyperLPR3\u3002"
            ) from exc
        except Exception as exc:
            raise InferenceDependencyError(
                "HyperLPR3 \u521d\u59cb\u5316\u5931\u8d25\u3002\u9996\u6b21\u8fd0\u884c\u9700\u8981\u4e0b\u8f7d\u6a21\u578b\u8d44\u6e90\uff0c"
                "\u8bf7\u786e\u8ba4\u7f51\u7edc\u53ef\u7528\uff0c\u6216\u68c0\u67e5 backend/runtime/.hyperlpr3 \u662f\u5426\u5b8c\u6574\u3002"
            ) from exc

        if self._catcher is not None:
            return self._catcher

        detect_level = self._resolve_detect_level(lpr3)
        self._catcher = lpr3.LicensePlateCatcher(
            folder=str(model_root),
            detect_level=detect_level,
        )
        return self._catcher

    def _configure_runtime_home(self) -> Path:
        runtime_home = Path(__file__).resolve().parents[2] / settings.hyperlpr_home_dir
        runtime_home.mkdir(parents=True, exist_ok=True)
        os.environ["HOMEPATH"] = str(runtime_home)
        os.environ.setdefault("HOME", str(runtime_home))
        model_root = runtime_home / ".hyperlpr3"
        model_root.mkdir(parents=True, exist_ok=True)
        return model_root

    def _resolve_detect_level(self, lpr3):
        if settings.hyperlpr_detect_level.lower() == "high":
            return getattr(lpr3, "DETECT_LEVEL_HIGH")
        return getattr(lpr3, "DETECT_LEVEL_LOW")

    def _load_image(self, image_source: bytes | bytearray | Any, max_side_override: int | None = None):
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise InferenceDependencyError(
                "\u7f3a\u5c11 opencv-python-headless \u6216 numpy\uff0c\u8bf7\u5148\u5b89\u88c5\u56fe\u50cf\u5904\u7406\u4f9d\u8d56\u3002"
            ) from exc

        if not isinstance(image_source, (bytes, bytearray)):
            return self._resize_if_needed(image_source, max_side_override=max_side_override)

        encoded = np.frombuffer(image_source, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(
                "\u65e0\u6cd5\u89e3\u6790\u4e0a\u4f20\u56fe\u7247\uff0c\u8bf7\u786e\u8ba4\u6587\u4ef6\u662f\u6709\u6548\u7684\u56fe\u7247\u683c\u5f0f\u3002"
            )
        return self._resize_if_needed(image, max_side_override=max_side_override)

    def _resize_if_needed(self, image, max_side_override: int | None = None):
        try:
            import cv2
        except ImportError as exc:
            raise InferenceDependencyError(
                "\u7f3a\u5c11 opencv-python-headless\uff0c\u8bf7\u5148\u5b89\u88c5\u56fe\u50cf\u5904\u7406\u4f9d\u8d56\u3002"
            ) from exc

        image_height, image_width = image.shape[:2]
        max_side = max(image_width, image_height)
        limit = settings.plate_max_image_side if max_side_override is None else max_side_override
        if limit <= 0 or max_side <= limit:
            return image, 1.0

        scale_ratio = limit / max_side
        resized = cv2.resize(
            image,
            (max(int(image_width * scale_ratio), 1), max(int(image_height * scale_ratio), 1)),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale_ratio

    def _build_enhanced_variant(self, image, scale_ratio: float):
        try:
            import cv2
        except ImportError as exc:
            raise InferenceDependencyError(
                "\u7f3a\u5c11 opencv-python-headless\uff0c\u8bf7\u5148\u5b89\u88c5\u56fe\u50cf\u5904\u7406\u4f9d\u8d56\u3002"
            ) from exc

        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            return image, scale_ratio

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        softened = cv2.GaussianBlur(enhanced, (0, 0), 1.1)
        sharpened = cv2.addWeighted(enhanced, 1.35, softened, -0.35, 0.0)
        upscale_ratio = 1.8 if max(width, height) <= 960 else 1.5
        enlarged = cv2.resize(
            sharpened,
            (max(int(width * upscale_ratio), 1), max(int(height * upscale_ratio), 1)),
            interpolation=cv2.INTER_CUBIC,
        )
        return enlarged, scale_ratio * upscale_ratio

    def _deduplicate_detections(self, detections: list[HyperLPRDetection]) -> list[HyperLPRDetection]:
        kept: list[HyperLPRDetection] = []
        for detection in sorted(
            detections,
            key=lambda item: (item.confidence, item.bbox[2] * item.bbox[3]),
            reverse=True,
        ):
            duplicate = False
            for existing in kept:
                overlap = self._compute_iou(detection.bbox, existing.bbox)
                if overlap >= 0.55:
                    duplicate = True
                    break
                if detection.plate_number == existing.plate_number and overlap >= 0.15:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(detection)
        return kept

    def _normalize_result(self, item: Any, scale_ratio: float) -> HyperLPRDetection | None:
        if isinstance(item, (list, tuple)):
            if len(item) < 4:
                return None
            plate_number = self._normalize_plate_text(str(item[0]))
            confidence = float(item[1])
            plate_type = int(item[2])
            raw_bbox = item[3]
            raw_vertex = item[4] if len(item) > 4 else None
        elif isinstance(item, dict):
            plate_number = self._normalize_plate_text(str(item.get("plate_code", "")))
            confidence = float(item.get("rec_confidence", 0.0))
            plate_type = int(item.get("plate_type", -1))
            raw_bbox = item.get("det_bound_box")
            raw_vertex = item.get("vertex")
        else:
            return None

        if not plate_number:
            return None

        bbox = self._normalize_bbox(raw_bbox, raw_vertex, scale_ratio)
        if not self._is_valid_bbox(bbox):
            return None
        if not self._is_valid_plate_text(plate_number):
            return None

        return HyperLPRDetection(
            plate_number=plate_number,
            plate_color=self._plate_type_to_color(plate_type),
            confidence=max(0.0, min(confidence, 1.0)),
            bbox=bbox,
        )

    def _normalize_bbox(self, raw_bbox: Any, raw_vertex: Any, scale_ratio: float) -> list[int]:
        inverse_ratio = 1.0 / scale_ratio if scale_ratio > 0 else 1.0

        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
            x1, y1, x2, y2 = [int(round(float(value) * inverse_ratio)) for value in raw_bbox]
            return [x1, y1, max(x2 - x1, 1), max(y2 - y1, 1)]

        if isinstance(raw_vertex, (list, tuple)) and len(raw_vertex) == 4:
            xs = [int(round(float(point[0]) * inverse_ratio)) for point in raw_vertex]
            ys = [int(round(float(point[1]) * inverse_ratio)) for point in raw_vertex]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            return [x1, y1, max(x2 - x1, 1), max(y2 - y1, 1)]

        return [0, 0, 0, 0]

    def _plate_type_to_color(self, plate_type: int) -> str:
        return PLATE_TYPE_TO_COLOR.get(plate_type, "\u672a\u77e5")

    def _normalize_plate_text(self, plate_number: str) -> str:
        normalized = plate_number.strip().upper().replace(" ", "")
        normalized = re.sub(r"[^A-Z0-9\u4e00-\u9fff]", "", normalized)
        return normalized

    def _is_valid_plate_text(self, plate_number: str) -> bool:
        if len(plate_number) not in (7, 8):
            return False
        if plate_number[0] not in PROVINCE_PREFIXES:
            return False
        if not re.fullmatch(r"[A-Z]", plate_number[1]):
            return False

        suffix = plate_number[2:]
        if not re.fullmatch(r"[A-Z0-9]{5,6}", suffix):
            return False
        if suffix.isdigit():
            return False
        if re.fullmatch(r"[1I]{5,6}", suffix):
            return False

        return True

    def _is_valid_bbox(self, bbox: list[int]) -> bool:
        if len(bbox) != 4:
            return False

        _, _, width, height = bbox
        if width <= 0 or height <= 0:
            return False

        aspect_ratio = width / max(height, 1)
        area = width * height
        if aspect_ratio < 1.7 or aspect_ratio > 7.8:
            return False
        if area < 60:
            return False
        return True

    def _compute_iou(self, bbox_a: list[int], bbox_b: list[int]) -> float:
        ax1, ay1, aw, ah = bbox_a
        bx1, by1, bw, bh = bbox_b
        ax2 = ax1 + aw
        ay2 = ay1 + ah
        bx2 = bx1 + bw
        by2 = by1 + bh

        inter_left = max(ax1, bx1)
        inter_top = max(ay1, by1)
        inter_right = min(ax2, bx2)
        inter_bottom = min(ay2, by2)
        if inter_right <= inter_left or inter_bottom <= inter_top:
            return 0.0

        intersection = (inter_right - inter_left) * (inter_bottom - inter_top)
        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - intersection
        if union <= 0:
            return 0.0
        return intersection / union
