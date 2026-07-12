from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from police import config as police_cfg
from police.features import associate_hands, classify_palm_orientation, extract_features
from police.geometry import calc_dist, get_hand_region, setup_local_frame
from police.gesture_classifier import GestureStateMachine
from police.models import create_hand_detector, create_pose_detector, detect_hand, detect_pose
from police.visualization import draw_chinese_text, draw_hand_landmarks, draw_pose_landmarks, draw_wrist_marker

try:
    from ctpgr_engine import CTPGREngine
except Exception:  # pragma: no cover - runtime import guard
    CTPGREngine = None


NO_VIDEO_GESTURE = "no_gesture"
NO_POSE_GESTURE = "未检测到人体"
FILTERED_GESTURE = "误触过滤"

VIDEO_GESTURE_MAP: dict[str, str] = {
    "无手势": NO_VIDEO_GESTURE,
    "停止": "stop",
    "停止信号": "stop",
    "直行": "go_straight",
    "直行信号": "go_straight",
    "左转弯": "left_turn",
    "左转弯信号": "left_turn",
    "左待转": "left_wait_turn",
    "左待转信号": "left_wait_turn",
    "右转弯": "right_turn",
    "右转弯信号": "right_turn",
    "变道": "lane_change",
    "变道信号": "lane_change",
    "减速": "slow_down",
    "减速慢行": "slow_down",
    "减速慢行信号": "slow_down",
    "靠边停车": "pull_over",
    "靠边停车信号": "pull_over",
}
VIDEO_GESTURE_TEXT: dict[str, str] = {
    NO_VIDEO_GESTURE: "无手势",
    "stop": "停止",
    "go_straight": "直行",
    "left_turn": "左转弯",
    "left_wait_turn": "左待转",
    "right_turn": "右转弯",
    "lane_change": "变道",
    "slow_down": "减速",
    "pull_over": "靠边停车",
}


@dataclass
class LocalPoliceGestureResult:
    gesture: str
    confidence: float
    keypoints: list[dict[str, float]]
    annotated_frame: np.ndarray
    display_label: str


class PoliceGestureVideoSession:
    def __init__(self) -> None:
        self.pose_detector = create_pose_detector(police_cfg.POSE_MODEL_PATH)
        self.hand_detector = None
        if os.path.exists(police_cfg.HAND_MODEL_PATH):
            self.hand_detector = create_hand_detector(police_cfg.HAND_MODEL_PATH)
        self.dl_engine = CTPGREngine() if CTPGREngine is not None else None

        self.state_machine = GestureStateMachine()
        self.frame_counter = 0
        self.global_frame = 0
        self.last_landmarks = None
        self.last_world_landmarks = None
        self.last_feat = None
        self.last_hand_left = None
        self.last_hand_right = None
        self.display_result: str | None = None
        self.display_confidence = 0.0
        self.result_display_timer = 0
        self.dl_gesture = "loading..."
        self.dl_confidence = 0.0
        self.dl_error_once = False

    def __enter__(self) -> PoliceGestureVideoSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.pose_detector.close()
        if self.hand_detector is not None:
            self.hand_detector.close()

    def process_frame(self, frame: np.ndarray) -> LocalPoliceGestureResult:
        source_frame = frame.copy()
        annotated = frame.copy()
        self.frame_counter += 1
        self.global_frame += 1

        should_infer = self.frame_counter % police_cfg.SKIP_FRAMES == 0
        height, width = source_frame.shape[:2]

        if should_infer:
            pose_result = detect_pose(self.pose_detector, source_frame)
            hand_result = (
                detect_hand(self.hand_detector, source_frame, timestamp_ms=self.global_frame * 33)
                if self.hand_detector is not None
                else None
            )
        else:
            pose_result = None
            hand_result = None

        if should_infer and pose_result and pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks[0]
            self.last_landmarks = landmarks

            if pose_result.pose_world_landmarks:
                world_landmarks = pose_result.pose_world_landmarks[0]
                self.last_world_landmarks = world_landmarks
            else:
                world_landmarks = self.last_world_landmarks

            draw_pose_landmarks(annotated, landmarks, height, width)

            def px(index: int) -> tuple[float, float]:
                landmark = landmarks[index]
                return (landmark.x * width, landmark.y * height)

            left_shoulder = px(11)
            right_shoulder = px(12)
            nose = px(0)
            left_hip = px(23)
            right_hip = px(24)

            _ = calc_dist(left_shoulder, right_shoulder)
            _, body_right_2d, body_up_2d = setup_local_frame(
                left_shoulder,
                right_shoulder,
                nose,
                left_hip,
                right_hip,
            )

            if world_landmarks is not None:
                feat = extract_features(world_landmarks, landmarks, self.last_hand_left, self.last_hand_right)
                self.last_feat = feat
            else:
                feat = self.last_feat

            hands = associate_hands(
                hand_result,
                (landmarks[15].x, landmarks[15].y),
                (landmarks[16].x, landmarks[16].y),
            )
            self.last_hand_left = hands["left"]
            self.last_hand_right = hands["right"]

            if self.last_hand_left:
                draw_hand_landmarks(annotated, self.last_hand_left, height, width)
            else:
                draw_wrist_marker(annotated, landmarks, 15, height, width, side="L")
            if self.last_hand_right:
                draw_hand_landmarks(annotated, self.last_hand_right, height, width)
            else:
                draw_wrist_marker(annotated, landmarks, 16, height, width, side="R")

            left_palm_ori = classify_palm_orientation(self.last_hand_left, body_right_2d, body_up_2d)
            right_palm_ori = classify_palm_orientation(self.last_hand_right, body_right_2d, body_up_2d)

            if feat is not None:
                result = self.state_machine.update(
                    feat,
                    left_palm_ori,
                    right_palm_ori,
                    self.global_frame,
                    feat.get("shoulder_width", 0.35),
                )
                if result is not None:
                    self.display_result, self.display_confidence = result
                    self.result_display_timer = police_cfg.RESULT_DISPLAY_FRAMES
        elif should_infer:
            self.state_machine.cancel_action(self.global_frame)
            self.last_landmarks = None
            self.last_world_landmarks = None
            self.last_feat = None
            self.last_hand_left = None
            self.last_hand_right = None

        if self.dl_engine is not None and self.frame_counter % 3 == 0:
            try:
                dl_frame = cv2.resize(source_frame, (512, 512))
                dl_result = self.dl_engine.predict_frame(dl_frame)
                self.dl_gesture = str(dl_result.get("gesture", "无手势"))
                self.dl_confidence = float(dl_result.get("confidence", 0.0) or 0.0)
            except Exception:
                if not self.dl_error_once:
                    self.dl_error_once = True
                self.dl_gesture = "DL error"
                self.dl_confidence = 0.0

        if not should_infer and self.last_landmarks is not None:
            draw_pose_landmarks(annotated, self.last_landmarks, height, width)
            if self.last_hand_left:
                draw_hand_landmarks(annotated, self.last_hand_left, height, width)
            else:
                draw_wrist_marker(annotated, self.last_landmarks, 15, height, width, side="L")
            if self.last_hand_right:
                draw_hand_landmarks(annotated, self.last_hand_right, height, width)
            else:
                draw_wrist_marker(annotated, self.last_landmarks, 16, height, width, side="R")

        if self.result_display_timer > 0:
            self.result_display_timer -= 1
        else:
            self.display_result = None
            self.display_confidence = 0.0

        status_y = 30
        if self.last_landmarks is None:
            annotated = draw_chinese_text(annotated, "未检测到人体", (10, status_y), (0, 0, 255), 36)
        elif self.state_machine.state == police_cfg.STATE_ACTIVE:
            annotated = draw_chinese_text(annotated, "动作识别中...", (10, status_y), (0, 255, 255), 30)
        elif self.state_machine.cooldown_counter > 0:
            annotated = draw_chinese_text(
                annotated,
                f"冷却中 {self.state_machine.cooldown_counter}",
                (10, status_y),
                (128, 128, 128),
                28,
            )
        elif self.display_result and self.display_result != FILTERED_GESTURE:
            annotated = draw_chinese_text(
                annotated,
                f"交警手势: {self.display_result}",
                (10, status_y),
                (0, 255, 0),
                36,
            )
            if self.display_confidence > 0:
                annotated = draw_chinese_text(
                    annotated,
                    f"置信度: {self.display_confidence:.0%}",
                    (10, 75),
                    (0, 200, 0),
                    22,
                )
        else:
            annotated = draw_chinese_text(annotated, "等待动作...", (10, status_y), (255, 255, 255), 30)

        if self.dl_engine is not None:
            panel_w, panel_h = 260, 80
            panel_x, panel_y = width - panel_w - 15, height - panel_h - 15
            overlay = annotated.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 50), -1)
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 200, 255), 2)
            annotated = cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0)
            cv2.putText(
                annotated,
                "-- DL Model --",
                (panel_x + 10, panel_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
            annotated = draw_chinese_text(annotated, self.dl_gesture, (panel_x + 10, panel_y + 32), (0, 215, 255), 28)
            cv2.putText(
                annotated,
                f"conf: {self.dl_confidence:.1%}",
                (panel_x + 10, panel_y + 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        keypoints = _pose_landmarks_to_keypoints(self.last_landmarks)
        if self.display_result and self.display_result != FILTERED_GESTURE:
            gesture = normalize_gesture(self.display_result)
            confidence = round(float(self.display_confidence), 4)
            display_label = self.display_result
        else:
            gesture = NO_VIDEO_GESTURE
            confidence = 0.0
            display_label = VIDEO_GESTURE_TEXT[NO_VIDEO_GESTURE]

        return LocalPoliceGestureResult(
            gesture=gesture,
            confidence=confidence,
            keypoints=keypoints,
            annotated_frame=annotated,
            display_label=display_label,
        )


class PoliceGestureLocalRuntime:
    def __init__(self) -> None:
        self._image_lock = threading.Lock()
        self._image_pose_detector = None
        self._image_hand_detector = None

    def create_video_session(self) -> PoliceGestureVideoSession:
        return PoliceGestureVideoSession()

    def recognize_image(self, frame: np.ndarray) -> LocalPoliceGestureResult:
        with self._image_lock:
            pose_detector = self._get_image_pose_detector()
            hand_detector = self._get_image_hand_detector()

            annotated = frame.copy()
            height, width = annotated.shape[:2]

            pose_result = detect_pose(pose_detector, frame)
            hand_result = detect_hand(hand_detector, frame, timestamp_ms=33) if hand_detector is not None else None

            if not pose_result or not pose_result.pose_landmarks:
                annotated = draw_chinese_text(annotated, "未检测到人体", (18, 18), (0, 0, 255), 34)
                return LocalPoliceGestureResult(
                    gesture=NO_POSE_GESTURE,
                    confidence=0.0,
                    keypoints=[],
                    annotated_frame=annotated,
                    display_label=NO_POSE_GESTURE,
                )

            landmarks = pose_result.pose_landmarks[0]
            world_landmarks = pose_result.pose_world_landmarks[0] if pose_result.pose_world_landmarks else None
            draw_pose_landmarks(annotated, landmarks, height, width)

            if hand_result and getattr(hand_result, "hand_landmarks", None):
                for hand_landmarks in hand_result.hand_landmarks:
                    draw_hand_landmarks(annotated, hand_landmarks, height, width)

            left_region = "?"
            right_region = "?"
            left_raise = 0.0
            right_raise = 0.0
            if world_landmarks is not None:
                feat = extract_features(world_landmarks, landmarks, None, None)
                left_region = feat.get("left_region", "?")
                right_region = feat.get("right_region", "?")
                left_raise = feat.get("left_raise", 0.0)
                right_raise = feat.get("right_raise", 0.0)

                left_wrist = world_landmarks[15]
                right_wrist = world_landmarks[16]
                shoulder_y_avg = (world_landmarks[11].y + world_landmarks[12].y) / 2.0
                left_hand_y = left_wrist.y + (0.08 if left_raise >= 0 else -0.08)
                right_hand_y = right_wrist.y + (0.08 if right_raise >= 0 else -0.08)
                left_region = get_hand_region(left_hand_y, shoulder_y_avg, 0, 0)
                right_region = get_hand_region(right_hand_y, shoulder_y_avg, 0, 0)

            title_bar_h = 40
            cv2.rectangle(annotated, (0, 0), (width, title_bar_h), (30, 30, 30), -1)
            cv2.putText(
                annotated,
                "Image Gesture Recognition",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2,
            )

            rule_x, rule_y = 10, title_bar_h + 15
            box_w, box_h = 280, 90
            overlay = annotated.copy()
            cv2.rectangle(overlay, (rule_x, rule_y), (rule_x + box_w, rule_y + box_h), (40, 40, 40), -1)
            annotated = cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0)
            cv2.putText(
                annotated,
                "Rule Model",
                (rule_x + 10, rule_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                annotated,
                f"L: {left_region}  ({left_raise:+.2f})",
                (rule_x + 10, rule_y + 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 255),
                2,
            )
            cv2.putText(
                annotated,
                f"R: {right_region} ({right_raise:+.2f})",
                (rule_x + 10, rule_y + 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 100, 255),
                2,
            )

            info_x = max(width - 220, 10)
            cv2.putText(
                annotated,
                f"Size: {width}x{height}",
                (info_x, title_bar_h + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
            )

            keypoints = _pose_landmarks_to_keypoints(landmarks)
            return LocalPoliceGestureResult(
                gesture=NO_VIDEO_GESTURE,
                confidence=0.0,
                keypoints=keypoints,
                annotated_frame=annotated,
                display_label=VIDEO_GESTURE_TEXT[NO_VIDEO_GESTURE],
            )

    def _get_image_pose_detector(self):
        if self._image_pose_detector is None:
            self._image_pose_detector = create_pose_detector(police_cfg.POSE_MODEL_PATH)
        return self._image_pose_detector

    def _get_image_hand_detector(self):
        if self._image_hand_detector is None and os.path.exists(police_cfg.HAND_MODEL_PATH):
            self._image_hand_detector = create_hand_detector(police_cfg.HAND_MODEL_PATH)
        return self._image_hand_detector


def normalize_gesture(raw_gesture: str | None) -> str:
    gesture = str(raw_gesture or "").strip()
    if gesture == FILTERED_GESTURE:
        return NO_VIDEO_GESTURE
    if gesture == NO_POSE_GESTURE:
        return NO_POSE_GESTURE
    return VIDEO_GESTURE_MAP.get(gesture, gesture or NO_VIDEO_GESTURE)


def _pose_landmarks_to_keypoints(landmarks: Any) -> list[dict[str, float]]:
    if landmarks is None:
        return []
    return [
        {
            "x": float(point.x),
            "y": float(point.y),
            "score": float(getattr(point, "visibility", 1.0)),
        }
        for point in landmarks
    ]
