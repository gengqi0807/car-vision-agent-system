"""Owner and police gesture classifier with optional ML/static+dynamic fusion."""

from __future__ import annotations

import logging
import math
import os
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

_Domain = Literal["owner", "police"]

_OWNER_GESTURE_ALIASES = {
    "palm": "open_palm",
    "open_palm": "open_palm",
    "thumb_up": "thumbs_up",
    "thumbs_up": "thumbs_up",
    "thumb_down": "thumbs_down",
    "thumbs_down": "thumbs_down",
    "pointing": "point",
    "point": "point",
    "circle_cw": "point",
    "circle_ccw": "point",
    "fist": "fist",
    "wave": "wave",
    "swipe_left": "swipe_left",
    "swipe_right": "swipe_right",
    "idle": "idle",
    "unknown": "unknown",
}


class GestureClassifier:
    """Gesture classifier for owner and police domains."""

    DYNAMIC_COOLDOWN_FRAMES = 18
    NO_HAND_REARM_FRAMES = 3
    STATIC_STABLE_FRAMES = 3
    _DYNAMIC_LABELS = {"wave", "swipe_left", "swipe_right", "index_circle", "idle"}
    _STATIC_LABELS = {"open_palm", "fist", "point", "thumbs_up", "thumbs_down", "unknown"}

    _WRIST = 0
    _THUMB_TIP = 4
    _INDEX_TIP = 8
    _MIDDLE_TIP = 12
    _RING_TIP = 16
    _PINKY_TIP = 20
    _THUMB_MCP = 2
    _INDEX_MCP = 5
    _MIDDLE_MCP = 9
    _RING_MCP = 13
    _PINKY_MCP = 17
    _THUMB_IP = 3
    _INDEX_PIP = 6
    _MIDDLE_PIP = 10
    _RING_PIP = 14
    _PINKY_PIP = 18
    _PER_HAND = 21

    _NOSE = 0
    _LEFT_SHOULDER = 11
    _RIGHT_SHOULDER = 12
    _LEFT_ELBOW = 13
    _RIGHT_ELBOW = 14
    _LEFT_WRIST = 15
    _RIGHT_WRIST = 16
    _LEFT_HIP = 23
    _RIGHT_HIP = 24
    _PER_PERSON = 33

    def __init__(self, domain: _Domain = "owner") -> None:
        self.domain = domain
        self.tracker: HandGestureTracker | None = HandGestureTracker() if domain == "owner" else None
        self._ml_model = None
        self._ml_scaler = None
        self._ml_labels = None
        self._cooldown = 0
        self._no_hand = 0
        self._stable_gesture: str | None = None
        self._stable_count = 0

        if domain == "owner":
            self._load_ml_model()

    def classify(self, keypoints: list[dict], domain: _Domain | None = None) -> dict:
        target_domain = domain or self.domain
        if not keypoints:
            return {"domain": target_domain, "gesture": "unknown", "confidence": 0.0}

        if target_domain == "owner":
            gesture, confidence = self.classify_static(keypoints)
            return {"domain": "owner", "gesture": gesture, "confidence": confidence}
        if target_domain == "police":
            return self._classify_police(keypoints)
        return {"domain": target_domain, "gesture": "unknown", "confidence": 0.0}

    def classify_frame(self, keypoints: list[dict] | None) -> tuple[str, float]:
        if keypoints is None:
            if self.tracker:
                self.tracker.reset()
            self._no_hand += 1
            self._stable_gesture = None
            self._stable_count = 0
            if self._no_hand >= self.NO_HAND_REARM_FRAMES:
                self._cooldown = 0
            return "unknown", 0.0

        if len(keypoints) != self._PER_HAND:
            return "unknown", 0.0

        self._no_hand = 0
        static_gesture, static_conf = self.classify_static(keypoints)
        dynamic_gesture = self.tracker.update(keypoints) if self.tracker else None

        if dynamic_gesture:
            if static_gesture in {"open_palm", "fist", "thumbs_up", "thumbs_down", "unknown"}:
                dynamic_gesture = None
            elif dynamic_gesture in {"wave", "swipe_left", "swipe_right"} and static_gesture != "point":
                dynamic_gesture = None

        if dynamic_gesture:
            self._cooldown = self.DYNAMIC_COOLDOWN_FRAMES
            self._stable_gesture = None
            self._stable_count = 0
            return dynamic_gesture, 0.85

        if self._cooldown > 0:
            self._cooldown -= 1
            self._stable_gesture = None
            self._stable_count = 0
            return "idle", 0.0

        if static_gesture == self._stable_gesture:
            self._stable_count += 1
        else:
            self._stable_gesture = static_gesture
            self._stable_count = 1

        if self._stable_count >= self.STATIC_STABLE_FRAMES:
            return static_gesture, static_conf
        return "idle", 0.0

    def classify_static(self, keypoints: list[dict]) -> tuple[str, float]:
        if len(keypoints) != self._PER_HAND:
            return "unknown", 0.0

        heuristic_result = self._classify_heuristic(keypoints)
        ml_result = self._classify_ml(keypoints)
        if ml_result is None:
            return heuristic_result
        return self._merge_static_predictions(heuristic_result, ml_result)

    def _load_ml_model(self) -> None:
        try:
            from app.core.config import settings

            model_path = settings.resolved_gesture_classifier_model_path
            if not os.path.exists(model_path):
                logger.info("Owner gesture classifier model not found, fallback to heuristics: %s", model_path)
                return

            import joblib

            bundle = joblib.load(model_path)
            self._ml_model = bundle["model"]
            self._ml_scaler = bundle["scaler"]
            self._ml_labels = bundle["label_names"]
            normalized_labels = {
                _OWNER_GESTURE_ALIASES.get(str(label), str(label))
                for label in self._ml_labels
            }
            if not {"open_palm", "point", "thumbs_down"}.issubset(normalized_labels):
                logger.warning(
                    "Owner gesture classifier labels are incomplete for the current feature set: %s. "
                    "Static recognition will prefer heuristic decisions for unsupported gestures.",
                    sorted(normalized_labels),
                )
            logger.info("Loaded owner-gesture classifier model from %s", model_path)
        except Exception as exc:
            logger.warning("Failed to load owner-gesture classifier model, fallback to heuristics: %s", exc)
            self._ml_model = None
            self._ml_scaler = None
            self._ml_labels = None

    def _merge_static_predictions(
        self,
        heuristic_result: tuple[str, float],
        ml_result: tuple[str, float],
    ) -> tuple[str, float]:
        heuristic_gesture, heuristic_conf = heuristic_result
        ml_gesture, ml_conf = ml_result

        if ml_gesture in self._DYNAMIC_LABELS:
            return heuristic_result
        if ml_gesture not in self._STATIC_LABELS:
            return heuristic_result

        if heuristic_gesture == ml_gesture:
            return ml_gesture, max(heuristic_conf, ml_conf)

        if heuristic_gesture == "open_palm":
            return heuristic_result
        if heuristic_gesture == "point":
            if ml_gesture == "thumbs_up" and ml_conf >= 0.92:
                return ml_result
            return heuristic_result
        if heuristic_gesture == "thumbs_down":
            return heuristic_result
        if heuristic_gesture == "thumbs_up":
            if ml_gesture == "thumbs_up":
                return ml_gesture, max(heuristic_conf, ml_conf)
            return heuristic_result
        if heuristic_gesture == "fist":
            if ml_gesture == "fist":
                return ml_gesture, max(heuristic_conf, ml_conf)
            if ml_gesture == "thumbs_up" and ml_conf >= 0.94:
                return ml_result
            return heuristic_result
        if heuristic_gesture == "unknown":
            return ml_result
        return heuristic_result

    def _classify_ml(self, keypoints: list[dict]) -> tuple[str, float] | None:
        if self._ml_model is None or self._ml_scaler is None or self._ml_labels is None:
            return None
        if len(keypoints) != self._PER_HAND:
            return None

        try:
            from app.models_infer.hand_utils import normalize_hand_landmarks_array

            feature = normalize_hand_landmarks_array(keypoints).reshape(1, -1)
            feature_scaled = self._ml_scaler.transform(feature)

            if hasattr(self._ml_model, "predict_proba"):
                proba = self._ml_model.predict_proba(feature_scaled)[0]
                idx = int(np.argmax(proba))
                raw_gesture = str(self._ml_labels[idx])
                confidence = float(proba[idx])
            else:
                idx = int(self._ml_model.predict(feature_scaled)[0])
                raw_gesture = str(self._ml_labels[idx])
                confidence = 0.75

            gesture = _OWNER_GESTURE_ALIASES.get(raw_gesture, raw_gesture)
            return gesture, confidence
        except Exception as exc:
            logger.debug("Owner gesture ML inference failed: %s", exc)
            return None

    def _classify_heuristic(self, keypoints: list[dict]) -> tuple[str, float]:
        hand = keypoints[: self._PER_HAND]

        def _is_extended(tip_idx: int, mcp_idx: int) -> bool:
            return hand[tip_idx]["y"] <= hand[mcp_idx]["y"]

        def _is_curled(tip_idx: int, pip_idx: int) -> bool:
            return hand[tip_idx]["y"] > hand[pip_idx]["y"]

        index_ext = _is_extended(self._INDEX_TIP, self._INDEX_MCP)
        middle_ext = _is_extended(self._MIDDLE_TIP, self._MIDDLE_MCP)
        ring_ext = _is_extended(self._RING_TIP, self._RING_MCP)
        pinky_ext = _is_extended(self._PINKY_TIP, self._PINKY_MCP)
        thumb_ext = _is_extended(self._THUMB_TIP, self._THUMB_MCP)

        index_curled = _is_curled(self._INDEX_TIP, self._INDEX_PIP)
        middle_curled = _is_curled(self._MIDDLE_TIP, self._MIDDLE_PIP)
        ring_curled = _is_curled(self._RING_TIP, self._RING_PIP)
        pinky_curled = _is_curled(self._PINKY_TIP, self._PINKY_PIP)

        all_four_ext = index_ext and middle_ext and ring_ext and pinky_ext
        all_four_curled = index_curled and middle_curled and ring_curled and pinky_curled

        thumb_y = hand[self._THUMB_TIP]["y"]
        wrist_y = hand[self._WRIST]["y"]

        if thumb_ext and all_four_curled and (thumb_y < wrist_y - 0.05):
            return "thumbs_up", 0.88
        if thumb_ext and all_four_curled and (thumb_y > wrist_y + 0.05):
            return "thumbs_down", 0.88
        if all_four_ext:
            return "open_palm", 0.92
        if index_ext and middle_curled and ring_curled and pinky_curled:
            return "point", 0.84
        if all_four_curled:
            return "fist", 0.90

        ext_count = sum([index_ext, middle_ext, ring_ext, pinky_ext])
        if ext_count >= 3:
            return "open_palm", 0.65
        if ext_count <= 1:
            return "fist", 0.60
        return "unknown", 0.40

    def _classify_police(self, keypoints: list[dict]) -> dict:
        n_poses = len(keypoints) // self._PER_PERSON
        if n_poses == 0:
            return {"domain": "police", "gesture": "unknown", "confidence": 0.0}

        pose = keypoints[: self._PER_PERSON]

        def _pt(idx: int) -> dict:
            return pose[idx]

        l_wrist_y = _pt(self._LEFT_WRIST)["y"]
        r_wrist_y = _pt(self._RIGHT_WRIST)["y"]
        l_shoulder_y = _pt(self._LEFT_SHOULDER)["y"]
        r_shoulder_y = _pt(self._RIGHT_SHOULDER)["y"]

        l_raised = l_wrist_y < (l_shoulder_y - 0.08)
        r_raised = r_wrist_y < (r_shoulder_y - 0.08)

        l_wrist_x = _pt(self._LEFT_WRIST)["x"]
        r_wrist_x = _pt(self._RIGHT_WRIST)["x"]
        l_shoulder_x = _pt(self._LEFT_SHOULDER)["x"]
        r_shoulder_x = _pt(self._RIGHT_SHOULDER)["x"]
        l_elbow_x = _pt(self._LEFT_ELBOW)["x"]
        r_elbow_x = _pt(self._RIGHT_ELBOW)["x"]

        l_ext_left = (l_wrist_x < l_elbow_x < l_shoulder_x) and (l_shoulder_x - l_wrist_x > 0.12)
        r_ext_right = (r_wrist_x > r_elbow_x > r_shoulder_x) and (r_wrist_x - r_shoulder_x > 0.12)

        both_raised = l_raised and r_raised
        if both_raised:
            return {"domain": "police", "gesture": "stop", "confidence": 0.82}
        if l_raised and not r_raised:
            return {"domain": "police", "gesture": "left_turn", "confidence": 0.75}
        if r_raised and not l_raised:
            return {"domain": "police", "gesture": "right_turn", "confidence": 0.75}
        if l_ext_left and not r_ext_right:
            return {"domain": "police", "gesture": "left_turn", "confidence": 0.70}
        if r_ext_right and not l_ext_left:
            return {"domain": "police", "gesture": "right_turn", "confidence": 0.70}
        if not l_raised and not r_raised:
            return {"domain": "police", "gesture": "go_straight", "confidence": 0.55}
        return {"domain": "police", "gesture": "unknown", "confidence": 0.40}


class HandGestureTracker:
    """Track hand motion and detect dynamic gestures."""

    def __init__(self) -> None:
        self.history: list[tuple[int, float, float, float, float]] = []
        self.frame_count = 0
        self._max_history = 60

    def reset(self) -> None:
        self.history.clear()
        self.frame_count = 0

    def update(self, keypoints: list[dict]) -> str | None:
        if len(keypoints) != 21:
            return None

        self.frame_count += 1
        wrist = keypoints[0]
        index_tip = keypoints[8]
        self.history.append(
            (
                self.frame_count,
                wrist["x"],
                wrist["y"],
                index_tip["x"],
                index_tip["y"],
            )
        )
        if len(self.history) > self._max_history:
            self.history = self.history[-self._max_history :]

        if len(self.history) < 8:
            return None

        wave = self._detect_wave()
        if wave:
            self.reset()
            return wave

        circle = self._detect_circle()
        if circle:
            self.reset()
            return circle

        swipe = self._detect_swipe()
        if swipe:
            self.reset()
            return swipe

        return None

    def _detect_circle(self) -> str | None:
        if len(self.history) < 15:
            return None

        points = [(item[3], item[4]) for item in self.history[-30:]]
        center_x = sum(point[0] for point in points) / len(points)
        center_y = sum(point[1] for point in points) / len(points)

        total_angle = 0.0
        for index in range(1, len(points)):
            angle_a = math.atan2(points[index - 1][1] - center_y, points[index - 1][0] - center_x)
            angle_b = math.atan2(points[index][1] - center_y, points[index][0] - center_x)
            diff = angle_b - angle_a
            while diff > math.pi:
                diff -= 2 * math.pi
            while diff < -math.pi:
                diff += 2 * math.pi
            total_angle += diff

        radii = [math.hypot(point[0] - center_x, point[1] - center_y) for point in points]
        mean_radius = sum(radii) / len(radii)
        if mean_radius < 0.015:
            return None

        max_deviation = max(abs(radius - mean_radius) for radius in radii) / mean_radius
        if abs(total_angle) >= math.radians(300) and max_deviation < 0.55:
            return "index_circle"
        return None

    def _detect_swipe(self) -> str | None:
        if len(self.history) < 8:
            return None

        start_x = self.history[0][1]
        end_x = self.history[-1][1]
        displacement = end_x - start_x
        if abs(displacement) < 0.16:
            return None

        y_positions = [item[2] for item in self.history]
        net_dy = y_positions[-1] - y_positions[0]
        if abs(net_dy) > 0.12:
            return None

        x_deltas = [right[1] - left[1] for left, right in zip(self.history, self.history[1:])]
        meaningful_deltas = [delta for delta in x_deltas if abs(delta) > 0.025]
        if len(meaningful_deltas) < 2:
            return None
        if max(abs(delta) for delta in meaningful_deltas) < 0.06:
            return None

        if displacement > 0.16:
            return "swipe_right"
        if displacement < -0.16:
            return "swipe_left"
        return None

    def _detect_wave(self) -> str | None:
        if len(self.history) < 15:
            return None

        wrist_x = [item[1] for item in self.history]
        reversals = 0
        previous_direction = 0
        x_span = max(wrist_x) - min(wrist_x)
        if x_span < 0.16:
            return None

        for index in range(5, len(wrist_x)):
            diff = wrist_x[index] - wrist_x[index - 5]
            if abs(diff) < 0.03:
                continue
            current_direction = 1 if diff > 0 else -1
            if previous_direction != 0 and current_direction != previous_direction:
                reversals += 1
                if reversals >= 2:
                    return "wave"
            previous_direction = current_direction
        return None
