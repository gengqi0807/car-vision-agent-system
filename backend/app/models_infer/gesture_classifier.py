"""Owner and police gesture classifier with optional static and dynamic fusion."""

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
    "index_circle": "index_circle",
    "circle_cw": "circle_cw",
    "circle_ccw": "circle_ccw",
    "fist": "fist",
    "wave": "wave",
    "swipe_left": "swipe_left",
    "swipe_right": "swipe_right",
    "idle": "idle",
    "unknown": "unknown",
}


def _normalize_owner_gesture(gesture: str) -> str:
    return _OWNER_GESTURE_ALIASES.get(gesture, gesture)


class GestureClassifier:
    """Gesture classifier for owner and police domains."""

    DYNAMIC_COOLDOWN_FRAMES = 18
    NO_HAND_REARM_FRAMES = 3
    STATIC_STABLE_FRAMES = 3
    DYNAMIC_CONF_MARGIN = 0.05
    MOTION_FIRE_THRESHOLD = 0.0004
    SUSTAINED_MOTION_FRAMES = 4
    WAVE_STABLE_FRAMES = 5

    _DYNAMIC_LABELS = {
        "wave",
        "swipe_left",
        "swipe_right",
        "index_circle",
        "circle_cw",
        "circle_ccw",
        "idle",
    }
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
        self._dynamic_lstm = None
        self._cooldown = 0
        self._no_hand = 0
        self._stable_gesture: str | None = None
        self._stable_count = 0
        self._wave_count = 0
        self._motion_history: list[tuple[float, float]] = []
        self._motion_history_max = 15
        self._motion_frames = 0

        if domain == "owner":
            self._load_ml_model()
            self._load_dynamic_lstm()

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
            boundary_result = self._handle_boundary()
            self._reset_runtime_trackers()
            self._no_hand += 1
            self._stable_gesture = None
            self._stable_count = 0
            self._wave_count = 0
            self._motion_history.clear()
            self._motion_frames = 0
            if self._no_hand >= self.NO_HAND_REARM_FRAMES:
                self._cooldown = 0
            if boundary_result[0] != "unknown":
                self._cooldown = self.DYNAMIC_COOLDOWN_FRAMES
                return boundary_result
            return "unknown", 0.0

        if len(keypoints) != self._PER_HAND:
            return "unknown", 0.0

        self._no_hand = 0
        static_gesture, static_conf = self.classify_static(keypoints)
        dynamic_gesture, dynamic_conf = self._classify_dynamic(keypoints)

        if dynamic_gesture:
            motion_energy = self._update_motion(keypoints)
            should_fire_dynamic = False

            if static_gesture in {"point", "unknown"}:
                should_fire_dynamic = True
                self._motion_frames = min(self._motion_frames + 1, 10)
            elif motion_energy > self.MOTION_FIRE_THRESHOLD:
                self._motion_frames += 1
                adjusted_static_conf = static_conf * (0.35 if static_gesture in {"open_palm", "fist"} else 0.65)
                if self._motion_frames >= self.SUSTAINED_MOTION_FRAMES:
                    should_fire_dynamic = dynamic_conf >= 0.20
                else:
                    should_fire_dynamic = dynamic_conf > adjusted_static_conf + 0.02
            elif dynamic_conf > static_conf + self.DYNAMIC_CONF_MARGIN:
                should_fire_dynamic = True
                self._motion_frames = max(0, self._motion_frames - 1)
            else:
                self._motion_frames = max(0, self._motion_frames - 1)

            if should_fire_dynamic:
                if dynamic_gesture == "wave":
                    self._wave_count += 1
                    if self._wave_count < self.WAVE_STABLE_FRAMES:
                        self._stable_gesture = None
                        self._stable_count = 0
                        return "idle", 0.0
                self._cooldown = self.DYNAMIC_COOLDOWN_FRAMES
                self._stable_gesture = None
                self._stable_count = 0
                self._wave_count = 0
                self._reset_dynamic_sequence()
                return dynamic_gesture, round(dynamic_conf, 4)

            self._stable_gesture = None
            self._stable_count = 0
            return "idle", 0.0

        self._wave_count = 0

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
            return static_gesture, round(static_conf, 4)
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
            normalized_labels = {_normalize_owner_gesture(str(label)) for label in self._ml_labels}
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

    def _load_dynamic_lstm(self) -> None:
        try:
            from app.models_infer.dynamic_lstm import DynamicLSTMClassifier

            self._dynamic_lstm = DynamicLSTMClassifier()
            if not self._dynamic_lstm.is_loaded:
                logger.info("Owner gesture dynamic LSTM model not loaded, fallback to heuristics tracker")
        except Exception as exc:
            logger.warning("Failed to initialize owner gesture dynamic LSTM, fallback to heuristics: %s", exc)
            self._dynamic_lstm = None

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
            # Some fist poses expose the thumb tip above the wrist, which can fool
            # the heuristic into "thumbs_up". If the trained classifier strongly
            # prefers fist, trust the model for this ambiguous boundary.
            if ml_gesture == "fist" and ml_conf >= 0.84:
                return ml_result
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

            gesture = _normalize_owner_gesture(raw_gesture)
            return gesture, confidence
        except Exception as exc:
            logger.debug("Owner gesture ML inference failed: %s", exc)
            return None

    def _classify_dynamic(self, keypoints: list[dict]) -> tuple[str | None, float]:
        if self._dynamic_lstm is not None and self._dynamic_lstm.is_loaded:
            try:
                gesture, confidence = self._dynamic_lstm.classify(keypoints, is_boundary=False)
                gesture = _normalize_owner_gesture(gesture)
                if gesture != "unknown":
                    return gesture, confidence
            except Exception as exc:
                logger.debug("Owner gesture dynamic LSTM inference failed: %s", exc)

        if self.tracker is None:
            return None, 0.0

        gesture = self.tracker.update(keypoints)
        if gesture is None:
            return None, 0.0
        return _normalize_owner_gesture(gesture), 0.85

    def _handle_boundary(self) -> tuple[str, float]:
        if self._dynamic_lstm is None or not self._dynamic_lstm.is_loaded:
            return "unknown", 0.0
        try:
            trajectory = self._dynamic_lstm.get_trajectory()
            if trajectory.shape[0] < self._dynamic_lstm.MIN_SEQUENCE_LENGTH:
                self._dynamic_lstm.reset_trajectory()
                return "unknown", 0.0
            gesture, confidence = self._dynamic_lstm.classify_sequence(trajectory)
            self._dynamic_lstm.reset_trajectory()
            return _normalize_owner_gesture(gesture), confidence
        except Exception as exc:
            logger.debug("Owner gesture dynamic boundary inference failed: %s", exc)
            self._dynamic_lstm.reset_trajectory()
            return "unknown", 0.0

    def _reset_runtime_trackers(self) -> None:
        if self.tracker:
            self.tracker.reset()

    def _reset_dynamic_sequence(self) -> None:
        if self.tracker:
            self.tracker.reset()
        if self._dynamic_lstm is not None:
            self._dynamic_lstm.reset_trajectory()

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

    def _update_motion(self, keypoints: list[dict]) -> float:
        if len(keypoints) != self._PER_HAND:
            return 0.0

        wrist_x = float(keypoints[self._WRIST]["x"])
        wrist_y = float(keypoints[self._WRIST]["y"])
        self._motion_history.append((wrist_x, wrist_y))
        if len(self._motion_history) > self._motion_history_max:
            self._motion_history = self._motion_history[-self._motion_history_max :]

        if len(self._motion_history) < 5:
            return 0.0

        xs = [point[0] for point in self._motion_history]
        ys = [point[1] for point in self._motion_history]
        return float(np.var(xs)) + float(np.var(ys))

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
                float(wrist["x"]),
                float(wrist["y"]),
                float(index_tip["x"]),
                float(index_tip["y"]),
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
        if len(self.history) < 12:
            return None

        xs = np.array([item[3] for item in self.history[-30:]], dtype=np.float64)
        ys = np.array([item[4] for item in self.history[-30:]], dtype=np.float64)

        if len(xs) >= 3:
            kernel = np.ones(3) / 3.0
            xs_smooth = np.convolve(xs, kernel, mode="valid")
            ys_smooth = np.convolve(ys, kernel, mode="valid")
        else:
            xs_smooth = xs
            ys_smooth = ys

        total_angle = 0.0
        valid_steps = 0
        for index in range(1, len(xs_smooth) - 1):
            v1_x = xs_smooth[index] - xs_smooth[index - 1]
            v1_y = ys_smooth[index] - ys_smooth[index - 1]
            v2_x = xs_smooth[index + 1] - xs_smooth[index]
            v2_y = ys_smooth[index + 1] - ys_smooth[index]

            mag1 = math.hypot(v1_x, v1_y)
            mag2 = math.hypot(v2_x, v2_y)
            if mag1 < 0.0015 or mag2 < 0.0015:
                continue

            cos_angle = (v1_x * v2_x + v1_y * v2_y) / (mag1 * mag2)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle = math.acos(cos_angle)
            cross = v1_x * v2_y - v1_y * v2_x
            if cross > 0:
                angle = -angle

            total_angle += angle
            valid_steps += 1

        if valid_steps < 6 or abs(total_angle) <= 5.0:
            return None

        net_x = float(xs[-1] - xs[0])
        net_y = float(ys[-1] - ys[0])
        net_dist = math.hypot(net_x, net_y)
        span = max(float(np.max(xs) - np.min(xs)), float(np.max(ys) - np.min(ys)))
        if net_dist >= 0.05 or span <= 0.03:
            return None

        return "circle_ccw" if total_angle > 0 else "circle_cw"

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
