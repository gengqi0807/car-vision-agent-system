from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.models_infer.errors import InferenceConfigurationError, InferenceDependencyError


@dataclass
class VehicleTypePrediction:
    label: str
    confidence: float


class VehicleTypeClassifier:
    def __init__(self) -> None:
        self._model = None
        self._classes: list[str] = []
        self._backend_dir = Path(__file__).resolve().parents[2]

    def warmup(self) -> None:
        if not self.is_available():
            return
        self._load_model_bundle()

    def is_available(self) -> bool:
        if not settings.plate_vehicle_classifier_enabled:
            return False
        return self._resolve_weights_path().exists()

    def classify(self, source: Any) -> VehicleTypePrediction | None:
        if not self.is_available():
            return None

        image = self._load_image_array(source)
        model, classes = self._load_model_bundle()
        logits = self._predict_logits(model, image)
        probabilities = self._softmax(logits)
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        if best_index < 0 or best_index >= len(classes):
            return None
        if confidence < settings.plate_vehicle_classifier_confidence_threshold:
            return None
        return VehicleTypePrediction(label=str(classes[best_index]), confidence=confidence)

    def _resolve_weights_path(self) -> Path:
        configured_path = Path(settings.plate_vehicle_classifier_weights_path)
        if configured_path.is_absolute():
            return configured_path
        return (self._backend_dir / configured_path).resolve()

    def _load_model_bundle(self):
        if self._model is not None and self._classes:
            return self._model, self._classes

        weights_path = self._resolve_weights_path()
        if not weights_path.exists():
            raise InferenceConfigurationError(
                f"Vehicle classifier weights not found: {weights_path}. Train the classifier first."
            )

        try:
            import torch
            from torchvision import models
        except ImportError as exc:
            raise InferenceDependencyError(
                "Missing torch/torchvision. Install model-training dependencies first."
            ) from exc

        checkpoint = torch.load(weights_path, map_location="cpu")
        classes = checkpoint.get("classes") or []
        if not classes:
            raise InferenceConfigurationError("Vehicle classifier checkpoint does not contain class metadata.")

        architecture = str(checkpoint.get("architecture") or "mobilenet_v3_small")
        if architecture != "mobilenet_v3_small":
            raise InferenceConfigurationError(f"Unsupported vehicle classifier architecture: {architecture}")

        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, len(classes))
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        self._model = model
        self._classes = [str(item) for item in classes]
        return self._model, self._classes

    def _predict_logits(self, model, image) -> np.ndarray:
        try:
            import torch
        except ImportError as exc:
            raise InferenceDependencyError(
                "Missing torch. Install model-training dependencies first."
            ) from exc

        batch = self._preprocess_image(image)
        with torch.inference_mode():
            logits = model(batch).detach().cpu().numpy()[0]
        return logits

    def _preprocess_image(self, image):
        try:
            import torch
            from PIL import Image
            from torchvision import transforms
        except ImportError as exc:
            raise InferenceDependencyError(
                "Missing Pillow/torchvision. Install model-training dependencies first."
            ) from exc

        rgb_image = image[:, :, ::-1]
        pil_image = Image.fromarray(rgb_image)
        transform = transforms.Compose(
            [
                transforms.Resize((settings.plate_vehicle_classifier_input_size, settings.plate_vehicle_classifier_input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transform(pil_image).unsqueeze(0).to(torch.float32)

    def _load_image_array(self, source: Any):
        try:
            import cv2
        except ImportError as exc:
            raise InferenceDependencyError(
                "Missing opencv-python-headless. Install image-processing dependencies first."
            ) from exc

        if isinstance(source, np.ndarray):
            return source

        if isinstance(source, (bytes, bytearray)):
            encoded = np.frombuffer(source, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode the uploaded image. Please verify the file is a valid image format.")
            return image

        file_bytes = np.fromfile(str(source), dtype=np.uint8)
        if file_bytes.size == 0:
            raise ValueError("Failed to read the image from the given path.")
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to read the image from the given path.")
        return image

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exp_values = np.exp(shifted)
        total = float(exp_values.sum())
        if total <= 0:
            return np.zeros_like(exp_values)
        return exp_values / total
