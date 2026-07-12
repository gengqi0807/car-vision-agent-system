from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from app.core.config import settings
from app.models_infer.errors import InferenceConfigurationError, InferenceDependencyError


CHARS = (
    "\u4eac",
    "\u6caa",
    "\u6d25",
    "\u6e1d",
    "\u5180",
    "\u664b",
    "\u8499",
    "\u8fbd",
    "\u5409",
    "\u9ed1",
    "\u82cf",
    "\u6d59",
    "\u7696",
    "\u95fd",
    "\u8d63",
    "\u9c81",
    "\u8c6b",
    "\u9102",
    "\u6e58",
    "\u7ca4",
    "\u6842",
    "\u743c",
    "\u5ddd",
    "\u8d35",
    "\u4e91",
    "\u85cf",
    "\u9655",
    "\u7518",
    "\u9752",
    "\u5b81",
    "\u65b0",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "I",
    "O",
    "-",
)
BLANK_CHAR = "-"
PROVINCE_PREFIXES = (
    "\u4eac\u6caa\u6d25\u6e1d\u5180\u664b\u8499\u8fbd\u5409\u9ed1\u82cf\u6d59\u7696"
    "\u95fd\u8d63\u9c81\u8c6b\u9102\u6e58\u7ca4\u6842\u743c\u5ddd\u8d35\u4e91\u85cf"
    "\u9655\u7518\u9752\u5b81\u65b0"
)


@dataclass
class OpenTrafficFlowLPRResult:
    plate_number: str
    plate_color: str
    confidence: float
    source: str = "open-traffic-flow-lprnet"


class OpenTrafficFlowLPRRecognizer:
    def __init__(self) -> None:
        self._model = None
        self._torch = None
        self._load_error: Exception | None = None
        self._backend_dir = Path(__file__).resolve().parents[2]

    def is_available(self) -> bool:
        if not settings.plate_open_traffic_flow_lpr_enabled:
            return False
        try:
            import torch  # noqa: F401
        except ImportError:
            return False
        return self._resolve_model_path().exists()

    def recognize(self, image_source: Any) -> OpenTrafficFlowLPRResult | None:
        image = self._load_image_array(image_source)
        if image is None or getattr(image, "size", 0) == 0:
            return None

        model, torch = self._load_model()
        best: OpenTrafficFlowLPRResult | None = None
        for candidate in self._build_candidates(image):
            result = self._infer_once(model, torch, candidate)
            if result is None:
                continue
            if best is None or result.confidence > best.confidence:
                best = result
        return best

    def _load_model(self):
        if self._load_error is not None:
            raise self._load_error
        if self._model is not None and self._torch is not None:
            return self._model, self._torch

        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as f
        except ImportError as exc:
            raise InferenceDependencyError("Missing PyTorch; install torch to enable OpenTrafficFlow OCR.") from exc

        model_path = self._resolve_model_path()
        if not model_path.exists():
            raise InferenceConfigurationError(
                f"OpenTrafficFlow LPR model not found: {model_path}. "
                "Place Final_LPRNet_model.pth there or update PLATE_OPEN_TRAFFIC_FLOW_LPR_MODEL_PATH."
            )

        class SmallBasicBlock(nn.Module):
            def __init__(self, ch_in: int, ch_out: int) -> None:
                super().__init__()
                self.block = nn.Sequential(
                    nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
                )

            def forward(self, x):
                return self.block(x)

        class LPRNet(nn.Module):
            def __init__(self, class_num: int, dropout_rate: float = 0.0) -> None:
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
                    nn.BatchNorm2d(num_features=64),
                    nn.ReLU(),
                    nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
                    SmallBasicBlock(ch_in=64, ch_out=128),
                    nn.BatchNorm2d(num_features=128),
                    nn.ReLU(),
                    nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
                    SmallBasicBlock(ch_in=64, ch_out=256),
                    nn.BatchNorm2d(num_features=256),
                    nn.ReLU(),
                    SmallBasicBlock(ch_in=256, ch_out=256),
                    nn.BatchNorm2d(num_features=256),
                    nn.ReLU(),
                    nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),
                    nn.Dropout(dropout_rate),
                    nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),
                    nn.BatchNorm2d(num_features=256),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1),
                    nn.BatchNorm2d(num_features=class_num),
                    nn.ReLU(),
                )
                self.container = nn.Sequential(
                    nn.Conv2d(in_channels=516, out_channels=class_num, kernel_size=(1, 1), stride=(1, 1)),
                )

            def forward(self, x):
                keep_features = []
                for index, layer in enumerate(self.backbone.children()):
                    x = layer(x)
                    if index in [2, 6, 13, 22]:
                        keep_features.append(x)

                global_context = []
                for index, feature in enumerate(keep_features):
                    if index in [0, 1]:
                        feature = f.avg_pool2d(feature, kernel_size=5, stride=5)
                    elif index == 2:
                        feature = f.avg_pool2d(feature, kernel_size=(4, 10), stride=(4, 2))

                    feature = feature / (torch.mean(torch.pow(feature, 2)) ** 0.5 + 1e-8)
                    global_context.append(feature)

                x = torch.cat(global_context, 1)
                x = self.container(x)
                return torch.mean(x, dim=2)

        device = self._resolve_torch_device(torch)
        model = LPRNet(class_num=len(CHARS), dropout_rate=0.0)

        try:
            checkpoint = torch.load(str(model_path), map_location=device)
            state_dict = checkpoint
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
            if not isinstance(state_dict, dict):
                raise InferenceConfigurationError("Unexpected OpenTrafficFlow LPRNet checkpoint format.")

            normalized_state = {}
            for key, value in state_dict.items():
                normalized_key = str(key)[7:] if str(key).startswith("module.") else str(key)
                normalized_state[normalized_key] = value

            model.load_state_dict(normalized_state, strict=True)
            model.to(device)
            model.eval()
        except Exception as exc:
            error = InferenceConfigurationError(
                "OpenTrafficFlow LPRNet weights do not match the current network structure."
            )
            error.__cause__ = exc
            self._load_error = error
            raise error

        self._model = model
        self._torch = torch
        return self._model, self._torch

    def _resolve_model_path(self) -> Path:
        configured_path = settings.plate_open_traffic_flow_lpr_model_path.strip()
        path = Path(configured_path)
        if path.is_absolute():
            return path
        return (self._backend_dir / path).resolve()

    def _resolve_torch_device(self, torch):
        configured = settings.plate_open_traffic_flow_lpr_device.strip().lower()
        if configured in {"", "auto"}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if configured.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(configured)

    def _load_image_array(self, image_source: Any):
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise InferenceDependencyError("Missing OpenCV or numpy for OpenTrafficFlow OCR.") from exc

        if isinstance(image_source, np.ndarray):
            return image_source
        if isinstance(image_source, (bytes, bytearray)):
            encoded = np.frombuffer(image_source, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image bytes.")
            return image

        image = cv2.imread(str(image_source))
        if image is None:
            raise ValueError("Failed to read image for OpenTrafficFlow OCR.")
        return image

    def _build_candidates(self, image):
        candidates = [image]
        enhanced = self._build_enhanced_candidate(image)
        if enhanced is not None:
            candidates.append(enhanced)
        return candidates

    def _build_enhanced_candidate(self, image):
        try:
            import cv2
        except ImportError:
            return None

        if image is None or getattr(image, "size", 0) == 0:
            return None

        resized = cv2.resize(
            image,
            (max(image.shape[1] * 2, 94), max(image.shape[0] * 2, 24)),
            interpolation=cv2.INTER_CUBIC,
        )
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced_l = clahe.apply(l_channel)
        merged = cv2.merge((enhanced_l, a_channel, b_channel))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return cv2.addWeighted(enhanced, 1.15, cv2.GaussianBlur(enhanced, (0, 0), 0.8), -0.15, 0.0)

    def _infer_once(self, model, torch, image) -> OpenTrafficFlowLPRResult | None:
        probs = self._predict_probabilities(model, torch, image)
        if probs is None:
            return None

        plate_number, confidence = self._decode_probs(probs)
        if confidence < settings.plate_open_traffic_flow_lpr_confidence_threshold:
            return None
        if not self._is_valid_plate_text(plate_number):
            return None

        return OpenTrafficFlowLPRResult(
            plate_number=plate_number,
            plate_color=self._classify_plate_color(image, plate_number),
            confidence=confidence,
        )

    def _predict_probabilities(self, model, torch, image):
        tensor = self._preprocess_image(image, torch)
        device = next(model.parameters()).device
        tensor = tensor.to(device)

        with torch.no_grad():
            logits = model(tensor)
            if logits.ndim != 3:
                return None
            if logits.shape[1] == len(CHARS):
                return torch.softmax(logits, dim=1)[0]
            if logits.shape[2] == len(CHARS):
                return torch.softmax(logits, dim=2)[0].transpose(0, 1)
        return None

    def _preprocess_image(self, image, torch):
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise InferenceDependencyError("Missing OpenCV or numpy for OpenTrafficFlow OCR.") from exc

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        resized = cv2.resize(image, (94, 24), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        array = rgb.astype(np.float32)
        array = (array - 127.5) * 0.0078125
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array).unsqueeze(0)

    def _decode_probs(self, probs) -> tuple[str, float]:
        best_indices = probs.argmax(dim=0).detach().cpu().tolist()
        best_scores = probs.max(dim=0).values.detach().cpu().tolist()

        decoded_chars: list[str] = []
        selected_scores: list[float] = []
        previous_index = None
        for index, score in zip(best_indices, best_scores):
            if index == previous_index:
                continue
            previous_index = index
            if index >= len(CHARS):
                continue
            char = CHARS[index]
            if char == BLANK_CHAR:
                continue
            decoded_chars.append(char)
            selected_scores.append(float(score))

        plate_number = self._normalize_plate_text("".join(decoded_chars))
        confidence = sum(selected_scores) / len(selected_scores) if selected_scores else 0.0
        return plate_number, confidence

    def _normalize_plate_text(self, text: str) -> str:
        normalized = text.strip().upper().replace(" ", "")
        return re.sub(r"[^0-9A-Z\u4e00-\u9fff]", "", normalized)

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

    def _classify_plate_color(self, plate_region, plate_number: str | None = None) -> str:
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
        hsv_image = cv2.cvtColor(focus_region, cv2.COLOR_BGR2HSV)
        hue_channel = hsv_image[:, :, 0]
        saturation_channel = hsv_image[:, :, 1]
        value_channel = hsv_image[:, :, 2]

        colorful_mask = (saturation_channel >= 45) & (value_channel >= 40)
        if not np.any(colorful_mask):
            brightness = float(value_channel.mean()) if value_channel.size else 0.0
            return "\u767d\u724c" if brightness >= 170 else "\u9ed1\u724c"

        hue_values = hue_channel[colorful_mask]
        saturation_values = saturation_channel[colorful_mask]
        value_values = value_channel[colorful_mask]
        if hue_values.size == 0:
            return "\u672a\u77e5"

        vivid_mask = (saturation_values >= 75) & (value_values >= 60)
        if np.any(vivid_mask):
            hue_values = hue_values[vivid_mask]
            saturation_values = saturation_values[vivid_mask]
            value_values = value_values[vivid_mask]

        yellow_mask = (hue_values >= 10) & (hue_values <= 40) & (value_values >= 70)
        green_mask = (hue_values >= 43) & (hue_values <= 92) & (saturation_values >= 65)
        blue_mask = (hue_values >= 94) & (hue_values <= 138) & (saturation_values >= 60)

        color_scores = {
            "\u84dd\u724c": int(np.count_nonzero(blue_mask)),
            "\u9ec4\u724c": int(np.count_nonzero(yellow_mask)),
            "\u7eff\u724c": int(np.count_nonzero(green_mask)),
        }
        if not any(color_scores.values()):
            return "\u672a\u77e5"

        plate_length = len(plate_number or "")
        yellow_score = color_scores["\u9ec4\u724c"]
        green_score = color_scores["\u7eff\u724c"]
        blue_score = color_scores["\u84dd\u724c"]

        if plate_length == 8 and green_score >= max(int(yellow_score * 0.8), int(blue_score * 0.8), 6):
            return "\u7eff\u724c"
        if yellow_score >= max(green_score + 8, int(green_score * 1.08), 5):
            return "\u9ec4\u724c"
        if plate_length == 7 and yellow_score >= max(int(green_score * 0.72), 4):
            return "\u9ec4\u724c"

        color_name, color_score = max(color_scores.items(), key=lambda item: item[1])
        if color_score <= 0:
            return "\u672a\u77e5"
        if plate_length == 7 and color_name == "\u7eff\u724c":
            return "\u9ec4\u724c"
        if plate_length == 8 and color_name in {"\u84dd\u724c", "\u9ec4\u724c"}:
            return "\u7eff\u724c"
        return color_name
