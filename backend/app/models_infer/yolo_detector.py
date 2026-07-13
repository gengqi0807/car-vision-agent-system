from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.models_infer.errors import InferenceConfigurationError, InferenceDependencyError


class YoloDetector:
    def __init__(self) -> None:
        self._model = None
        self._vehicle_model = None
        self._backend_dir = Path(__file__).resolve().parents[2]

    def warmup(self) -> None:
        if not self.is_available():
            return
        import numpy as np

        # Match the real image fast path so the first upload does not pay the
        # Ultralytics graph/input-shape initialization cost.
        warmup_image = np.zeros((384, 640, 3), dtype=np.uint8)
        plate_model = self._load_model()
        self._predict_with_model(
            plate_model,
            warmup_image,
            conf=settings.plate_yolo_confidence,
            imgsz=min(settings.plate_yolo_imgsz, 512),
        )
        detect_vehicle_classes = getattr(self, "detect_vehicle_classes", None)
        if callable(detect_vehicle_classes) and settings.plate_vehicle_detector_fallback_enabled:
            try:
                vehicle_model = self._load_vehicle_model()
                self._predict_with_model(
                    vehicle_model,
                    warmup_image,
                    conf=settings.plate_vehicle_yolo_confidence,
                    imgsz=min(settings.plate_vehicle_yolo_imgsz, 512),
                )
            except Exception:
                # Vehicle fallback is optional during warmup.
                pass

    def is_available(self) -> bool:
        if not settings.plate_detector_enabled:
            return False
        self._configure_ultralytics_runtime()
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            return False

        model_path = settings.plate_yolo_model_path.strip()
        if self._is_builtin_model_name(model_path):
            return True
        return self._resolve_model_path(model_path).exists()

    def detect(self, source: Any) -> list[dict]:
        image = self._load_image_array(source)
        return self._detect_impl(
            image,
            merge_second_pass=False,
            include_vehicle_with_plates=False,
            enable_small_target_pass=False,
        )

    def detect_video(self, source: Any) -> list[dict]:
        image = self._load_image_array(source)
        return self._detect_impl(
            image,
            merge_second_pass=True,
            include_vehicle_with_plates=True,
            enable_small_target_pass=(
                settings.plate_detector_small_target_enabled
                or settings.plate_video_small_target_detector_enabled
            ),
            second_pass_variant_limit=1,
            small_target_variant_limit=1,
        )

    def _detect_impl(
        self,
        image,
        *,
        merge_second_pass: bool,
        include_vehicle_with_plates: bool,
        enable_small_target_pass: bool,
        second_pass_variant_limit: int | None = None,
        small_target_variant_limit: int | None = None,
    ) -> list[dict]:
        selected = self._run_detection_pass(
            image,
            conf=settings.plate_yolo_confidence,
            imgsz=settings.plate_yolo_imgsz,
            include_vehicle_with_plates=include_vehicle_with_plates,
        )
        if not settings.plate_detector_second_pass_enabled:
            return selected

        merged = list(selected)
        if selected and not merge_second_pass:
            return selected
        if merge_second_pass and self._should_skip_video_refinement_passes(selected):
            return self._deduplicate_hits(merged)

        second_pass_variants = self._build_second_pass_variants(image)
        if second_pass_variant_limit is not None:
            second_pass_variants = second_pass_variants[: max(second_pass_variant_limit, 0)]

        for variant_image, scale_ratio in second_pass_variants:
            second_pass = self._run_detection_pass(
                variant_image,
                conf=settings.plate_detector_second_pass_confidence,
                imgsz=settings.plate_detector_second_pass_imgsz,
                scale_ratio=scale_ratio,
                include_vehicle_with_plates=include_vehicle_with_plates,
            )
            if not second_pass:
                continue
            if not merge_second_pass:
                return second_pass
            merged.extend(second_pass)

        if enable_small_target_pass:
            small_target_variants = self._build_small_target_variants(image)
            if small_target_variant_limit is not None:
                small_target_variants = small_target_variants[: max(small_target_variant_limit, 0)]
            for variant_image, scale_ratio in small_target_variants:
                small_target_pass = self._run_detection_pass(
                    variant_image,
                    conf=settings.plate_detector_small_target_confidence,
                    imgsz=settings.plate_detector_small_target_imgsz,
                    scale_ratio=scale_ratio,
                    include_vehicle_with_plates=include_vehicle_with_plates,
                )
                if not small_target_pass:
                    continue
                merged.extend(small_target_pass)

        if not merge_second_pass:
            return []
        return self._deduplicate_hits(merged)

    def _should_skip_video_refinement_passes(self, selected: list[dict]) -> bool:
        if not selected:
            return False

        plate_hits = [item for item in selected if str(item.get("kind", "plate")) == "plate"]
        if not plate_hits:
            return False

        for hit in plate_hits:
            bbox = list(hit.get("bbox", []))
            if len(bbox) != 4:
                continue
            width = int(bbox[2])
            height = int(bbox[3])
            confidence = float(hit.get("confidence", 0.0))
            if confidence >= 0.74 and width >= 34 and height >= 10 and width * height >= 420:
                return True
        return False

    def detect_fast(self, source: Any) -> list[dict]:
        image = self._load_image_array(source)
        return self._run_detection_pass(
            image,
            conf=settings.plate_yolo_confidence,
            imgsz=min(settings.plate_yolo_imgsz, 512),
        )

    def detect_fast_with_vehicles(self, source: Any) -> list[dict]:
        image = self._load_image_array(source)
        return self._run_detection_pass(
            image,
            conf=settings.plate_yolo_confidence,
            imgsz=min(settings.plate_yolo_imgsz, 512),
            include_vehicle_with_plates=True,
        )

    def detect_image_detailed(self, source: Any) -> list[dict]:
        image = self._load_image_array(source)
        merged = self._run_detection_pass(
            image,
            conf=min(settings.plate_yolo_confidence, 0.14),
            imgsz=max(settings.plate_yolo_imgsz, 1280),
            include_vehicle_with_plates=True,
        )

        for variant_image, scale_ratio in self._build_second_pass_variants(image):
            second_pass = self._run_detection_pass(
                variant_image,
                conf=min(settings.plate_detector_second_pass_confidence, 0.10),
                imgsz=max(settings.plate_detector_second_pass_imgsz, 1536),
                scale_ratio=scale_ratio,
                include_vehicle_with_plates=True,
            )
            if second_pass:
                merged.extend(second_pass)

        for variant_image, scale_ratio in self._build_aggressive_image_small_target_variants(image):
            small_target_pass = self._run_detection_pass(
                variant_image,
                conf=min(settings.plate_detector_small_target_confidence, 0.08),
                imgsz=max(settings.plate_detector_small_target_imgsz, 1792),
                scale_ratio=scale_ratio,
                include_vehicle_with_plates=True,
            )
            if small_target_pass:
                merged.extend(small_target_pass)

        return self._deduplicate_hits(merged)

    def detect_vehicle_classes(self, source: Any, *, fast_mode: bool = False) -> list[dict]:
        image = self._load_image_array(source)
        imgsz = settings.plate_vehicle_yolo_imgsz
        if fast_mode:
            imgsz = min(imgsz, 512)
        model = self._load_vehicle_model()
        results = self._predict_with_model(
            model,
            image,
            conf=settings.plate_vehicle_yolo_confidence,
            imgsz=imgsz,
        )
        vehicle_detections: list[dict] = []
        for result in results:
            names = result.names or getattr(model, "names", {}) or {}
            for box in result.boxes:
                cls_id = int(box.cls.item()) if box.cls is not None else -1
                label = str(names.get(cls_id, cls_id))
                if not self._is_vehicle_label(label):
                    continue
                raw_x1, raw_y1, raw_x2, raw_y2 = [float(value) for value in box.xyxy[0].tolist()]
                x1 = int(round(raw_x1))
                y1 = int(round(raw_y1))
                x2 = int(round(raw_x2))
                y2 = int(round(raw_y2))
                vehicle_detections.append(
                    {
                        "label": label,
                        "kind": "vehicle",
                        "bbox": [x1, y1, max(x2 - x1, 1), max(y2 - y1, 1)],
                        "confidence": float(box.conf.item()) if box.conf is not None else 0.0,
                    }
                )
        return self._deduplicate_hits(vehicle_detections)

    def _run_detection_pass(
        self,
        image,
        *,
        conf: float,
        imgsz: int,
        scale_ratio: float = 1.0,
        include_vehicle_with_plates: bool = False,
    ) -> list[dict]:
        model = self._load_model()
        results = self._predict(model, image, conf=conf, imgsz=imgsz)
        plate_detections, vehicle_detections = self._collect_detections(
            model,
            results,
            scale_ratio=scale_ratio,
        )
        if include_vehicle_with_plates:
            selected = plate_detections + vehicle_detections
        else:
            selected = plate_detections or vehicle_detections
        selected.sort(key=lambda item: item["confidence"], reverse=True)
        return selected

    def _predict(self, model, image, *, conf: float, imgsz: int):
        return self._predict_with_model(model, image, conf=conf, imgsz=imgsz)

    def _predict_with_model(self, model, image, *, conf: float, imgsz: int):
        device = settings.plate_yolo_device.strip()
        try:
            return model.predict(
                source=image,
                conf=conf,
                imgsz=imgsz,
                max_det=settings.plate_yolo_max_det,
                device=None if device.lower() in {"", "auto"} else device,
                verbose=False,
            )
        except Exception as exc:
            raise InferenceConfigurationError(
                "YOLO model inference failed. The current plate detector weights may be incompatible "
                "with the installed ultralytics version."
            ) from exc

    def _collect_detections(self, model, results, *, scale_ratio: float) -> tuple[list[dict], list[dict]]:
        plate_detections: list[dict] = []
        vehicle_detections: list[dict] = []
        for result in results:
            names = result.names or getattr(model, "names", {}) or {}
            for box in result.boxes:
                cls_id = int(box.cls.item()) if box.cls is not None else -1
                label = str(names.get(cls_id, cls_id))
                raw_x1, raw_y1, raw_x2, raw_y2 = [float(value) for value in box.xyxy[0].tolist()]
                x1 = int(round(raw_x1 / scale_ratio))
                y1 = int(round(raw_y1 / scale_ratio))
                x2 = int(round(raw_x2 / scale_ratio))
                y2 = int(round(raw_y2 / scale_ratio))
                detection = {
                    "label": label,
                    "bbox": [x1, y1, max(x2 - x1, 1), max(y2 - y1, 1)],
                    "confidence": float(box.conf.item()) if box.conf is not None else 0.0,
                }
                if self._is_plate_label(label, names):
                    plate_detections.append({**detection, "kind": "plate"})
                elif settings.plate_vehicle_detector_fallback_enabled and self._is_vehicle_label(label):
                    vehicle_detections.append({**detection, "kind": "vehicle"})
        return plate_detections, vehicle_detections

    def _build_second_pass_variants(self, image) -> list[tuple[object, float]]:
        try:
            import cv2
        except ImportError:
            return []

        variants: list[tuple[object, float]] = []
        height, width = image.shape[:2]
        longest_side = max(height, width, 1)
        mean_brightness = float(image.mean()) if getattr(image, "size", 0) else 0.0

        upscale_ratio = 1.5 if longest_side >= 720 else 2.0
        if longest_side <= 1280:
            upscaled = cv2.resize(
                image,
                None,
                fx=upscale_ratio,
                fy=upscale_ratio,
                interpolation=cv2.INTER_CUBIC,
            )
            variants.append((upscaled, upscale_ratio))

        if mean_brightness <= 118.0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            brightened = cv2.addWeighted(
                enhanced_gray,
                1.28,
                cv2.GaussianBlur(enhanced_gray, (0, 0), 1.1),
                -0.28,
                8.0,
            )
            enhanced = cv2.cvtColor(brightened, cv2.COLOR_GRAY2BGR)
            variants.append((enhanced, 1.0))

            enhanced_upscaled = cv2.resize(
                enhanced,
                None,
                fx=upscale_ratio,
                fy=upscale_ratio,
                interpolation=cv2.INTER_CUBIC,
            )
            variants.append((enhanced_upscaled, upscale_ratio))

        return variants

    def _build_small_target_variants(self, image) -> list[tuple[object, float]]:
        try:
            import cv2
        except ImportError:
            return []

        variants: list[tuple[object, float]] = []
        upscale_ratio = max(settings.plate_detector_small_target_upscale, 1.0)
        if upscale_ratio > 1.0:
            upscaled = cv2.resize(
                image,
                None,
                fx=upscale_ratio,
                fy=upscale_ratio,
                interpolation=cv2.INTER_CUBIC,
            )
            variants.append((upscaled, upscale_ratio))
        else:
            variants.append((image, 1.0))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        boosted_gray = cv2.addWeighted(
            enhanced_gray,
            1.34,
            cv2.GaussianBlur(enhanced_gray, (0, 0), 1.2),
            -0.34,
            10.0,
        )
        enhanced = cv2.cvtColor(boosted_gray, cv2.COLOR_GRAY2BGR)
        variants.append((enhanced, 1.0))
        if upscale_ratio > 1.0:
            enhanced_upscaled = cv2.resize(
                enhanced,
                None,
                fx=upscale_ratio,
                fy=upscale_ratio,
                interpolation=cv2.INTER_CUBIC,
            )
            variants.append((enhanced_upscaled, upscale_ratio))

        return variants

    def _build_aggressive_image_small_target_variants(self, image) -> list[tuple[object, float]]:
        try:
            import cv2
        except ImportError:
            return []

        variants = list(self._build_small_target_variants(image))
        seen_ratios = {round(float(scale_ratio), 3) for _, scale_ratio in variants}
        for scale_ratio in (2.8, 3.2):
            if round(scale_ratio, 3) in seen_ratios:
                continue
            upscaled = cv2.resize(
                image,
                None,
                fx=scale_ratio,
                fy=scale_ratio,
                interpolation=cv2.INTER_CUBIC,
            )
            variants.append((upscaled, scale_ratio))
            seen_ratios.add(round(scale_ratio, 3))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.6, tileGridSize=(6, 6))
        enhanced_gray = clahe.apply(gray)
        sharpened = cv2.addWeighted(
            enhanced_gray,
            1.42,
            cv2.GaussianBlur(enhanced_gray, (0, 0), 1.0),
            -0.42,
            12.0,
        )
        enhanced = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        variants.append((enhanced, 1.0))
        for scale_ratio in (2.4, 3.0):
            enhanced_upscaled = cv2.resize(
                enhanced,
                None,
                fx=scale_ratio,
                fy=scale_ratio,
                interpolation=cv2.INTER_CUBIC,
            )
            variants.append((enhanced_upscaled, scale_ratio))

        return variants

    def _deduplicate_hits(self, hits: list[dict]) -> list[dict]:
        deduplicated: list[dict] = []
        ranked_hits = sorted(
            hits,
            key=lambda item: (
                1 if str(item.get("kind", "plate")) == "plate" else 0,
                float(item.get("confidence", 0.0)),
            ),
            reverse=True,
        )
        for hit in ranked_hits:
            bbox = list(hit.get("bbox", []))
            if len(bbox) != 4:
                continue

            duplicate = False
            for existing in deduplicated:
                if self._compute_iou(bbox, list(existing.get("bbox", []))) >= 0.45:
                    duplicate = True
                    break
            if not duplicate:
                deduplicated.append(hit)

        return deduplicated

    def _compute_iou(self, bbox_a: list[int], bbox_b: list[int]) -> float:
        if len(bbox_a) != 4 or len(bbox_b) != 4:
            return 0.0
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
        union = aw * ah + bw * bh - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    def _load_model(self):
        self._configure_ultralytics_runtime()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise InferenceDependencyError(
                "Missing ultralytics. Install YOLO dependencies in the backend environment first."
            ) from exc

        if self._model is not None:
            return self._model

        configured_path = settings.plate_yolo_model_path.strip()
        if self._is_builtin_model_name(configured_path):
            model_arg = configured_path
        else:
            model_path = self._resolve_model_path(configured_path)
            if not model_path.exists():
                raise InferenceConfigurationError(
                    f"YOLO model not found: {model_path}. Update PLATE_YOLO_MODEL_PATH or place weights there."
                )
            model_arg = str(model_path)

        try:
            self._model = YOLO(model_arg)
        except Exception as exc:
            raise InferenceConfigurationError(
                "YOLO model failed to load. The detector weights may be incompatible with the installed ultralytics version."
            ) from exc
        return self._model

    def _load_vehicle_model(self):
        self._configure_ultralytics_runtime()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise InferenceDependencyError(
                "Missing ultralytics. Install YOLO dependencies in the backend environment first."
            ) from exc

        if self._vehicle_model is not None:
            return self._vehicle_model

        configured_path = settings.plate_vehicle_yolo_model_path.strip()
        resolved_path = self._resolve_model_path(configured_path)
        if resolved_path.exists():
            model_arg = str(resolved_path)
        elif self._is_builtin_model_name(configured_path):
            model_arg = configured_path
        else:
            raise InferenceConfigurationError(
                f"Vehicle YOLO model not found: {resolved_path}. Update PLATE_VEHICLE_YOLO_MODEL_PATH or place weights there."
            )

        try:
            self._vehicle_model = YOLO(model_arg)
        except Exception as exc:
            raise InferenceConfigurationError(
                "Vehicle YOLO model failed to load. The coarse-classification weights may be incompatible with the installed ultralytics version."
            ) from exc
        return self._vehicle_model

    def _configure_ultralytics_runtime(self) -> None:
        runtime_dir = self._backend_dir / settings.inference_runtime_dir / "ultralytics"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(runtime_dir))

    def _is_builtin_model_name(self, configured_path: str) -> bool:
        path = configured_path.strip()
        if not path.lower().endswith(".pt"):
            return False
        return "/" not in path and "\\" not in path and ":" not in path

    def _resolve_model_path(self, configured_path: str) -> Path:
        model_path = Path(configured_path)
        if model_path.is_absolute():
            return model_path
        return (self._backend_dir / model_path).resolve()

    def _load_image_array(self, source: Any):
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise InferenceDependencyError(
                "Missing opencv-python-headless or numpy. Install image-processing dependencies first."
            ) from exc

        if isinstance(source, np.ndarray):
            return source

        if isinstance(source, (bytes, bytearray)):
            encoded = np.frombuffer(source, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode the uploaded image. Please verify the file is a valid image format.")
            return image

        image = cv2.imread(str(source))
        if image is None:
            raise ValueError("Failed to read the image from the given path.")
        return image

    def _is_plate_label(self, label: str, names: dict[Any, Any]) -> bool:
        normalized = label.strip().lower().replace("_", "-")
        if any(keyword in normalized for keyword in settings.plate_detector_class_names):
            return True
        return len(names) <= 1

    def _is_vehicle_label(self, label: str) -> bool:
        normalized = label.strip().lower().replace("_", "-")
        return any(keyword == normalized for keyword in settings.plate_vehicle_class_names)
