from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.models_infer.errors import InferenceConfigurationError, InferenceDependencyError


class YoloDetector:
    def __init__(self) -> None:
        self._model = None
        self._backend_dir = Path(__file__).resolve().parents[2]

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
        model = self._load_model()
        device = settings.plate_yolo_device.strip()
        try:
            results = model.predict(
                source=image,
                conf=settings.plate_yolo_confidence,
                imgsz=settings.plate_yolo_imgsz,
                max_det=settings.plate_yolo_max_det,
                device=None if device.lower() in {"", "auto"} else device,
                verbose=False,
            )
        except Exception as exc:
            raise InferenceConfigurationError(
                "YOLO 模型推理失败。当前车牌检测权重可能与本地 ultralytics 版本不兼容，"
                "请尝试升级 ultralytics 后再重启后端。"
            ) from exc

        plate_detections: list[dict] = []
        vehicle_detections: list[dict] = []
        for result in results:
            names = result.names or getattr(model, "names", {}) or {}
            for box in result.boxes:
                cls_id = int(box.cls.item()) if box.cls is not None else -1
                label = str(names.get(cls_id, cls_id))
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                detection = {
                    "label": label,
                    "bbox": [x1, y1, max(x2 - x1, 1), max(y2 - y1, 1)],
                    "confidence": float(box.conf.item()) if box.conf is not None else 0.0,
                }
                if self._is_plate_label(label, names):
                    plate_detections.append({**detection, "kind": "plate"})
                elif settings.plate_vehicle_detector_fallback_enabled and self._is_vehicle_label(label):
                    vehicle_detections.append({**detection, "kind": "vehicle"})

        selected = plate_detections or vehicle_detections
        selected.sort(key=lambda item: item["confidence"], reverse=True)
        return selected

    def _load_model(self):
        self._configure_ultralytics_runtime()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise InferenceDependencyError("缺少 ultralytics，请先在后端环境中安装 YOLO 依赖。") from exc

        if self._model is not None:
            return self._model

        configured_path = settings.plate_yolo_model_path.strip()
        model_arg: str
        if self._is_builtin_model_name(configured_path):
            model_arg = configured_path
        else:
            model_path = self._resolve_model_path(configured_path)
            if not model_path.exists():
                raise InferenceConfigurationError(
                    f"未找到 YOLO 模型：{model_path}。请将模型文件放到该路径，或在 .env 中修改 PLATE_YOLO_MODEL_PATH。"
                )
            model_arg = str(model_path)

        try:
            self._model = YOLO(model_arg)
        except Exception as exc:
            raise InferenceConfigurationError(
                "YOLO 模型加载失败。当前车牌检测权重可能与本地 ultralytics 版本不兼容，"
                "请尝试升级 ultralytics 后再重启后端。"
            ) from exc
        return self._model

    def _configure_ultralytics_runtime(self) -> None:
        runtime_dir = self._backend_dir / settings.hyperlpr_home_dir / "ultralytics"
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
            raise InferenceDependencyError("缺少 opencv-python-headless 或 numpy，请先安装图像处理依赖。") from exc

        if isinstance(source, np.ndarray):
            return source

        if isinstance(source, (bytes, bytearray)):
            encoded = np.frombuffer(source, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("无法解析上传图片，请确认文件是有效的图片格式。")
            return image

        image = cv2.imread(str(source))
        if image is None:
            raise ValueError("无法解析上传图片，请确认文件路径有效且格式受支持。")
        return image

    def _is_plate_label(self, label: str, names: dict[Any, Any]) -> bool:
        normalized = label.strip().lower().replace("_", "-")
        if any(keyword in normalized for keyword in settings.plate_detector_class_names):
            return True
        return len(names) <= 1

    def _is_vehicle_label(self, label: str) -> bool:
        normalized = label.strip().lower().replace("_", "-")
        return any(keyword == normalized for keyword in settings.plate_vehicle_class_names)
