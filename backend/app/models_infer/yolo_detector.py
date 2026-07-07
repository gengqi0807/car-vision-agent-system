from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.models_infer.errors import InferenceConfigurationError, InferenceDependencyError


class YoloDetector:
    def __init__(self) -> None:
        self._model = None
        self._backend_dir = Path(__file__).resolve().parents[2]

    def detect(self, source: Any) -> list[dict]:
        image = self._load_image_array(source)
        model = self._load_model()
        results = model.predict(
            source=image,
            conf=settings.plate_yolo_confidence,
            device=settings.plate_yolo_device,
            verbose=False,
        )

        detections: list[dict] = []
        matched_detections: list[dict] = []

        for result in results:
            names = result.names or getattr(model, "names", {}) or {}
            current_result_detections: list[dict] = []
            current_result_matches: list[dict] = []

            for box in result.boxes:
                cls_id = int(box.cls.item()) if box.cls is not None else -1
                label = str(names.get(cls_id, cls_id))
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                detection = {
                    "label": label,
                    "bbox": [x1, y1, max(x2 - x1, 1), max(y2 - y1, 1)],
                    "confidence": float(box.conf.item()) if box.conf is not None else 0.0,
                }
                current_result_detections.append(detection)
                if self._is_plate_label(label):
                    current_result_matches.append(detection)

            detections.extend(current_result_detections)
            if current_result_matches:
                matched_detections.extend(current_result_matches)
            elif len(names) <= 1:
                matched_detections.extend(current_result_detections)

        selected = matched_detections or detections
        selected.sort(key=lambda item: item["confidence"], reverse=True)
        return selected

    def _load_model(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise InferenceDependencyError(
                "缺少 ultralytics，请先在后端环境中安装 YOLOv8 依赖。"
            ) from exc

        if self._model is not None:
            return self._model

        model_path = self._resolve_model_path(settings.plate_yolo_model_path)
        if not model_path.exists():
            raise InferenceConfigurationError(
                f"未找到车牌检测模型：{model_path}。请将 YOLOv8 车牌权重放到该路径，或在 .env 中修改 PLATE_YOLO_MODEL_PATH。"
            )

        self._model = YOLO(str(model_path))
        return self._model

    def _resolve_model_path(self, configured_path: str) -> Path:
        model_path = Path(configured_path)
        if model_path.is_absolute():
            return model_path
        return (self._backend_dir / model_path).resolve()

    def _load_image_array(self, source: Any) -> np.ndarray:
        try:
            import numpy as np
            from PIL import Image, UnidentifiedImageError
        except ImportError as exc:
            raise InferenceDependencyError(
                "缺少 numpy 或 Pillow，请先安装图像处理依赖。"
            ) from exc

        try:
            if isinstance(source, np.ndarray):
                return source
            if isinstance(source, (bytes, bytearray)):
                image = Image.open(BytesIO(source))
            else:
                image = Image.open(source)
            return np.asarray(image.convert("RGB"))
        except (FileNotFoundError, UnidentifiedImageError, OSError, TypeError) as exc:
            raise ValueError("无法解析上传图片，请确认文件是有效的图片格式。") from exc

    def _is_plate_label(self, label: str) -> bool:
        normalized = label.strip().lower().replace("_", "-")
        return any(keyword in normalized for keyword in settings.plate_detector_class_names)
