"""Rule-based hand / pose gesture classifier.

Classifies MediaPipe keypoints into named gestures using geometric
rules (no ML model required).  Supports two domains:

- ``owner`` — hand landmarks (21 points per hand)
- ``police`` — pose landmarks (33 body keypoints)
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

logger = logging.getLogger(__name__)

_Domain = Literal["owner", "police"]


class GestureClassifier:
    """Rule-based gesture classifier for two domains."""

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def classify(self, keypoints: list[dict], domain: _Domain) -> dict:
        """Classify a set of keypoints into a gesture label.

        Parameters
        ----------
        keypoints:
            Raw MediaPipe keypoints, each ``{"x": float, "y": float, "z": float}``.
            For hands: 21 points per hand (21, 42, 63 … total).
            For pose:  33 points per person.
        domain:
            ``"owner"`` → in-cabin hand gestures (6 types).
            ``"police"`` → traffic police body-gesture rules (8 types).

        Returns
        -------
        dict
            ``{"domain": ..., "gesture": ..., "confidence": ...}``
        """
        if not keypoints:
            return {"domain": domain, "gesture": "unknown", "confidence": 0.0}

        if domain == "owner":
            return self._classify_hand(keypoints)
        elif domain == "police":
            return self._classify_police(keypoints)
        return {"domain": domain, "gesture": "unknown", "confidence": 0.0}

    # ----------------------------------------------------------------
    # Hand (owner) – 21 landmarks per hand
    # ----------------------------------------------------------------
    # Landmark indices
    _WRIST = 0
    _THUMB_TIP = 4
    _INDEX_TIP = 8
    _MIDDLE_TIP = 12
    _RING_TIP = 16
    _PINKY_TIP = 20
    # MCP (knuckle) for each finger
    _THUMB_MCP = 2
    _INDEX_MCP = 5
    _MIDDLE_MCP = 9
    _RING_MCP = 13
    _PINKY_MCP = 17
    # PIP for each finger
    _THUMB_IP = 3
    _INDEX_PIP = 6
    _MIDDLE_PIP = 10
    _RING_PIP = 14
    _PINKY_PIP = 18

    _PER_HAND = 21

    def _classify_hand(self, keypoints: list[dict]) -> dict:
        n_hands = len(keypoints) // self._PER_HAND
        if n_hands == 0:
            return {"domain": "owner", "gesture": "unknown", "confidence": 0.0}

        # Use first detected hand
        hand = keypoints[: self._PER_HAND]

        # Helper: is a finger *extended*?  True if tip_y < mcp_y (image coords: y ↓)
        def _is_extended(tip_idx: int, mcp_idx: int) -> bool:
            return hand[tip_idx]["y"] <= hand[mcp_idx]["y"]

        # Helper: is a finger *curled*?  True if tip_y > pip_y
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

        # Thumb direction relative to wrist
        thumb_y = hand[self._THUMB_TIP]["y"]
        wrist_y = hand[self._WRIST]["y"]
        thumb_x = hand[self._THUMB_TIP]["x"]
        wrist_x = hand[self._WRIST]["x"]

        if thumb_ext and all_four_curled and (thumb_y < wrist_y - 0.05):
            return {"domain": "owner", "gesture": "thumbs_up", "confidence": 0.88}
        if thumb_ext and all_four_curled and (thumb_y > wrist_y + 0.05):
            return {"domain": "owner", "gesture": "thumbs_down", "confidence": 0.88}

        if all_four_ext:
            return {"domain": "owner", "gesture": "open_palm", "confidence": 0.92}

        if index_ext and middle_curled and ring_curled and pinky_curled:
            return {"domain": "owner", "gesture": "point", "confidence": 0.84}

        if all_four_curled:
            return {"domain": "owner", "gesture": "fist", "confidence": 0.90}

        # Fallback – partial extension
        ext_count = sum([index_ext, middle_ext, ring_ext, pinky_ext])
        if ext_count >= 3:
            return {"domain": "owner", "gesture": "open_palm", "confidence": 0.65}
        if ext_count <= 1:
            return {"domain": "owner", "gesture": "fist", "confidence": 0.60}

        return {"domain": "owner", "gesture": "unknown", "confidence": 0.40}

    # ----------------------------------------------------------------
    # Police (pose) – 33 landmarks
    # ----------------------------------------------------------------
    # Key landmark indices (MediaPipe Pose)
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

    def _classify_police(self, keypoints: list[dict]) -> dict:
        n_poses = len(keypoints) // self._PER_PERSON
        if n_poses == 0:
            return {"domain": "police", "gesture": "unknown", "confidence": 0.0}

        pose = keypoints[: self._PER_PERSON]

        def _pt(idx: int) -> dict:
            return pose[idx]

        # Arm raised high: wrist_y < shoulder_y by a margin
        l_wrist_y = _pt(self._LEFT_WRIST)["y"]
        r_wrist_y = _pt(self._RIGHT_WRIST)["y"]
        l_shoulder_y = _pt(self._LEFT_SHOULDER)["y"]
        r_shoulder_y = _pt(self._RIGHT_SHOULDER)["y"]

        l_raised = l_wrist_y < (l_shoulder_y - 0.08)
        r_raised = r_wrist_y < (r_shoulder_y - 0.08)

        # Arm extended horizontally (elbow ≈ wrist_x far from shoulder)
        l_wrist_x = _pt(self._LEFT_WRIST)["x"]
        r_wrist_x = _pt(self._RIGHT_WRIST)["x"]
        l_shoulder_x = _pt(self._LEFT_SHOULDER)["x"]
        r_shoulder_x = _pt(self._RIGHT_SHOULDER)["x"]
        l_elbow_x = _pt(self._LEFT_ELBOW)["x"]
        r_elbow_x = _pt(self._RIGHT_ELBOW)["x"]

        # Left arm extended to left
        l_ext_left = (l_wrist_x < l_elbow_x < l_shoulder_x) and (
            l_shoulder_x - l_wrist_x > 0.12
        )
        # Right arm extended to right
        r_ext_right = (r_wrist_x > r_elbow_x > r_shoulder_x) and (
            r_wrist_x - r_shoulder_x > 0.12
        )

        # Both arms raised forward-ish
        both_raised = l_raised and r_raised

        # Classification priority
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

        # Go-straight: both arms somewhere forward, not raised
        if not l_raised and not r_raised:
            # Check if arms are reasonably forward
            return {"domain": "police", "gesture": "go_straight", "confidence": 0.55}

        return {"domain": "police", "gesture": "unknown", "confidence": 0.40}
