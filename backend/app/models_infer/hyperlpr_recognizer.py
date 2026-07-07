from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.models_infer.errors import InferenceDependencyError


@dataclass
class HyperLPRDetection:
    plate_number: str
    plate_color: str
    confidence: float
    bbox: list[int]


class HyperLPRRecognizer:
    def __init__(self) -> None:
        self._catcher = None

    def recognize_all(self, image_source: bytes | bytearray | Any) -> list[HyperLPRDetection]:
        image = self._load_image(image_source)
        catcher = self._load_catcher()
        raw_results = catcher(image)

        detections: list[HyperLPRDetection] = []
        for item in raw_results or []:
            normalized = self._normalize_result(item)
            if normalized is None:
                continue
            if normalized.confidence < settings.plate_confidence_threshold:
                continue
            detections.append(normalized)

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections

    def _load_catcher(self):
        model_root = self._configure_runtime_home()
        try:
            import hyperlpr3 as lpr3
        except ImportError as exc:
            raise InferenceDependencyError(
                "缺少 hyperlpr3，请先在后端环境中安装 HyperLPR3。"
            ) from exc
        except Exception as exc:
            raise InferenceDependencyError(
                "HyperLPR3 初始化失败。首次运行需要下载模型资源，请确认网络可用，或检查 backend/runtime/.hyperlpr3 是否完整。"
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

    def _load_image(self, image_source: bytes | bytearray | Any):
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise InferenceDependencyError(
                "缺少 opencv-python-headless 或 numpy，请先安装图像处理依赖。"
            ) from exc

        if not isinstance(image_source, (bytes, bytearray)):
            return image_source

        encoded = np.frombuffer(image_source, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法解析上传图片，请确认文件是有效的图片格式。")
        return image

    def _normalize_result(self, item: Any) -> HyperLPRDetection | None:
        if isinstance(item, (list, tuple)):
            if len(item) < 4:
                return None
            plate_number = str(item[0]).strip()
            confidence = float(item[1])
            plate_type = int(item[2])
            raw_bbox = item[3]
            raw_vertex = item[4] if len(item) > 4 else None
        elif isinstance(item, dict):
            plate_number = str(item.get("plate_code", "")).strip()
            confidence = float(item.get("rec_confidence", 0.0))
            plate_type = int(item.get("plate_type", -1))
            raw_bbox = item.get("det_bound_box")
            raw_vertex = item.get("vertex")
        else:
            return None

        if not plate_number:
            return None

        bbox = self._normalize_bbox(raw_bbox, raw_vertex)
        return HyperLPRDetection(
            plate_number=plate_number,
            plate_color=self._plate_type_to_color(plate_type),
            confidence=max(0.0, min(confidence, 1.0)),
            bbox=bbox,
        )

    def _normalize_bbox(self, raw_bbox: Any, raw_vertex: Any) -> list[int]:
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
            x1, y1, x2, y2 = [int(value) for value in raw_bbox]
            return [x1, y1, max(x2 - x1, 1), max(y2 - y1, 1)]

        if isinstance(raw_vertex, (list, tuple)) and len(raw_vertex) == 4:
            xs = [int(point[0]) for point in raw_vertex]
            ys = [int(point[1]) for point in raw_vertex]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            return [x1, y1, max(x2 - x1, 1), max(y2 - y1, 1)]

        return [0, 0, 0, 0]

    def _plate_type_to_color(self, plate_type: int) -> str:
        plate_color_map = {
            0: "蓝牌",
            1: "绿牌",
            2: "黄牌",
            3: "绿牌",
            4: "黑牌",
            5: "黑牌",
            6: "黑牌",
            7: "黑牌",
            8: "黑牌",
            9: "黄牌",
        }
        return plate_color_map.get(plate_type, "未知")
