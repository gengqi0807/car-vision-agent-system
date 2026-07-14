from __future__ import annotations

import os
import sys
import threading
import time
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
from police.visualization import (
    draw_chinese_text,
    draw_chinese_text_lines,
    draw_hand_landmarks,
    draw_pose_landmarks,
    draw_wrist_marker,
)

try:
    from ctpgr_engine import CTPGREngine, mediapipe_to_aic14
except Exception:  # pragma: no cover - runtime import guard
    CTPGREngine = None
    mediapipe_to_aic14 = None


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
    completed_gesture: str | None = None
    completed_confidence: float = 0.0


def _person_tracking_info(landmarks) -> dict[str, float] | None:
    if landmarks is None or len(landmarks) <= 24:
        return None
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    shoulder_width = (
        (float(left_shoulder.x) - float(right_shoulder.x)) ** 2
        + (float(left_shoulder.y) - float(right_shoulder.y)) ** 2
    ) ** 0.5
    center_x = sum(float(landmarks[index].x) for index in (11, 12, 23, 24)) / 4.0
    center_y = sum(float(landmarks[index].y) for index in (11, 12, 23, 24)) / 4.0
    distance_to_center = ((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5
    return {
        "sw": shoulder_width,
        "cx": center_x,
        "cy": center_y,
        "dist_center": distance_to_center,
    }


def _select_target_person(pose_landmarks_list) -> tuple[int, dict[str, float] | None]:
    best_index = 0
    best_score = float("-inf")
    best_info = None
    for index, landmarks in enumerate(pose_landmarks_list or []):
        person_info = _person_tracking_info(landmarks)
        if person_info is None:
            continue
        score = person_info["sw"] * 3.0 - person_info["dist_center"] * 0.5
        if score > best_score:
            best_index = index
            best_score = score
            best_info = person_info
    return best_index, best_info


def _track_target_person(
    pose_landmarks_list,
    last_target_info: dict[str, float] | None,
    track_threshold: float,
) -> tuple[int, dict[str, float] | None]:
    if not pose_landmarks_list or last_target_info is None:
        return _select_target_person(pose_landmarks_list)

    best_index = 0
    best_distance = float("inf")
    best_info = None
    for index, landmarks in enumerate(pose_landmarks_list):
        person_info = _person_tracking_info(landmarks)
        if person_info is None:
            continue
        distance = (
            (person_info["cx"] - last_target_info["cx"]) ** 2
            + (person_info["cy"] - last_target_info["cy"]) ** 2
        ) ** 0.5
        if distance < best_distance:
            best_index = index
            best_distance = distance
            best_info = person_info

    if best_info is None or best_distance > track_threshold:
        return _select_target_person(pose_landmarks_list)
    return best_index, best_info


class PoliceGestureVideoSession:
    def __init__(self, *, realtime: bool = False) -> None:
        self.realtime = realtime
        self.pose_detector = create_pose_detector(
            police_cfg.POSE_MODEL_PATH,
            num_poses=police_cfg.NUM_POSES_MULTI,
        )
        self.hand_detector = None
        if os.path.exists(police_cfg.HAND_MODEL_PATH):
            self.hand_detector = create_hand_detector(police_cfg.HAND_MODEL_PATH)
        self.dl_engine = CTPGREngine(load_pose_model=False) if CTPGREngine is not None else None

        self.state_machine = GestureStateMachine(verbose=False)
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
        self.dl_gesture = "预热中..." if realtime else "loading..."
        self.dl_confidence = 0.0
        self.dl_keypoints: list[dict[str, float]] = []
        self.dl_error_once = False
        self.dl_skip = 3
        self.dl_warmed_up = not realtime
        self.warmup_started_at = time.time()
        self.warmup_seconds = 2.0
        self.action_state = "idle"
        self.action_frame_count = 0
        self.hip_both_frames = 0
        self.hip_stop_threshold = 8
        self.dl_window: list[tuple[str, float]] = []
        self.dl_filtered_gesture = "无手势"
        self.dl_filtered_confidence = 0.0
        self.gesture_min_confidence = 0.60
        self.gesture_min_run = 3
        self.first_guess_frames = 5
        self.action_flash_text = ""
        self.action_flash_remaining = 0
        self.action_flash_frames = 30
        self.completed_gesture: str | None = None
        self.completed_confidence = 0.0
        self.target_locked = False
        self.target_info: dict[str, float] | None = None
        self.target_lock_counter = 0

    def __enter__(self) -> PoliceGestureVideoSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.pose_detector.close()
        if self.hand_detector is not None:
            self.hand_detector.close()

    def process_frame(self, frame: np.ndarray) -> LocalPoliceGestureResult:
        source_frame = frame
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
            target_index = self._select_or_track_target(pose_result.pose_landmarks)
            landmarks = pose_result.pose_landmarks[target_index]
            self.last_landmarks = landmarks

            if pose_result.pose_world_landmarks and target_index < len(pose_result.pose_world_landmarks):
                world_landmarks = pose_result.pose_world_landmarks[target_index]
                self.last_world_landmarks = world_landmarks
            else:
                world_landmarks = self.last_world_landmarks

            if len(pose_result.pose_landmarks) > 1:
                for index, other_landmarks in enumerate(pose_result.pose_landmarks):
                    if index == target_index:
                        continue
                    draw_pose_landmarks(
                        annotated,
                        other_landmarks,
                        height,
                        width,
                        point_color=(100, 100, 100),
                        line_color=(80, 80, 80),
                    )

            draw_pose_landmarks(annotated, landmarks, height, width)

            def px(index: int) -> tuple[float, float]:
                landmark = landmarks[index]
                return (landmark.x * width, landmark.y * height)

            left_shoulder = px(11)
            right_shoulder = px(12)
            left_elbow = px(13)
            right_elbow = px(14)
            nose = px(0)
            left_hip = px(23)
            right_hip = px(24)

            cv2.putText(
                annotated,
                "L",
                (int(left_shoulder[0]) - 15, int(left_shoulder[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                3,
            )
            cv2.putText(
                annotated,
                "R",
                (int(right_shoulder[0]) + 5, int(right_shoulder[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                3,
            )

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
                if self.last_feat is not None:
                    left_visibility = landmarks[15].visibility if landmarks else 0.0
                    right_visibility = landmarks[16].visibility if landmarks else 0.0
                    left_keys = (
                        "left_raise",
                        "left_stretch",
                        "left_z_diff",
                        "left_wx",
                        "left_wy",
                        "left_sx",
                        "left_sy",
                        "left_orient",
                        "left_pose",
                        "left_region",
                        "left_fwd",
                        "left_lat",
                        "left_dir_raw",
                        "left_arm_angle",
                    )
                    right_keys = (
                        "right_raise",
                        "right_stretch",
                        "right_z_diff",
                        "right_wx",
                        "right_wy",
                        "right_sx",
                        "right_sy",
                        "right_orient",
                        "right_pose",
                        "right_region",
                        "right_fwd",
                        "right_lat",
                        "right_dir_raw",
                        "right_arm_angle",
                    )
                    if left_visibility < 0.3:
                        for key in left_keys:
                            if key in feat and key in self.last_feat:
                                feat[key] = self.last_feat[key]
                        feat["left_visible"] = False
                    else:
                        feat["left_visible"] = True
                    if right_visibility < 0.3:
                        for key in right_keys:
                            if key in feat and key in self.last_feat:
                                feat[key] = self.last_feat[key]
                        feat["right_visible"] = False
                    else:
                        feat["right_visible"] = True
                else:
                    feat["left_visible"] = True
                    feat["right_visible"] = True
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
                self.state_machine.update(
                    feat,
                    left_palm_ori,
                    right_palm_ori,
                    self.global_frame,
                    feat.get("shoulder_width", 0.35),
                )
                self._update_action_state(feat)
        elif should_infer:
            self.state_machine.cancel_action(self.global_frame)
            self.last_landmarks = None
            self.last_world_landmarks = None
            self.last_feat = None
            self.last_hand_left = None
            self.last_hand_right = None
            self._reset_target_tracking()
            self._reset_action_state()

        if self.dl_engine is not None and should_infer:
            if self.realtime and not self.dl_warmed_up and (time.time() - self.warmup_started_at) >= self.warmup_seconds:
                self.dl_warmed_up = True
                self.dl_engine.reset_state()
                self.dl_gesture = "无手势"
                self.dl_confidence = 0.0

            if self.last_landmarks is not None and (not self.realtime or self.dl_warmed_up):
                try:
                    coord_aic = mediapipe_to_aic14(self.last_landmarks)
                    dl_result = self.dl_engine.predict_from_keypoints(coord_aic)
                    raw_gesture = str(dl_result.get("gesture", "无手势"))
                    raw_confidence = float(dl_result.get("confidence", 0.0) or 0.0)
                    self.dl_keypoints = [
                        {"x": float(item.get("x", 0.0)), "y": float(item.get("y", 0.0)), "score": 1.0}
                        for item in dl_result.get("keypoints", [])
                    ]
                except Exception:
                    if not self.dl_error_once:
                        self.dl_error_once = True
                    raw_gesture = "DL error"
                    raw_confidence = 0.0
                    self.dl_keypoints = []

                self._update_filtered_dl_result(raw_gesture, raw_confidence)

        show_gesture = self.dl_gesture
        show_confidence = self.dl_confidence

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

        text_lines: list[tuple[str, tuple[int, int], tuple[int, int, int], int]] = []

        if self.realtime and self.dl_engine is not None and not self.dl_warmed_up:
            remaining = self.warmup_seconds - (time.time() - self.warmup_started_at)
            if remaining > 0:
                overlay = annotated.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
                annotated = cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0)
                text_lines.extend([
                    (
                        "请保持站立，倒计时结束后开始识别",
                        (max(width // 2 - 220, 20), max(height // 2 - 10, 20)),
                        (0, 255, 255),
                        26,
                    ),
                    (
                        f"预热中 {max(remaining, 0.0):.0f}s",
                        (max(width // 2 - 80, 20), max(height // 2 + 28, 20)),
                        (255, 255, 255),
                        24,
                    ),
                ])

        if self.last_landmarks is None:
            text_lines.append(("未检测到人体", (10, 110), (0, 0, 255), 36))

        if self.action_flash_remaining > 0:
            banner_width, banner_height = 300, 50
            banner_x = (width - banner_width) // 2
            banner_y = height // 8
            overlay = annotated.copy()
            cv2.rectangle(
                overlay,
                (banner_x, banner_y),
                (banner_x + banner_width, banner_y + banner_height),
                (0, 0, 0),
                -1,
            )
            annotated = cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0)
            flash_color = (0, 255, 0) if self.action_flash_text == "动作开始" else (0, 140, 255)
            text_lines.append((self.action_flash_text, (banner_x + 65, banner_y + 8), flash_color, 32))
            self.action_flash_remaining -= 1

        if self.last_feat is not None:
            info_x = width - 200
            left_region = self.last_feat.get("left_region", "?")
            right_region = self.last_feat.get("right_region", "?")
            text_lines.append((f"左手: {left_region}", (info_x, 15), (0, 200, 255), 20))
            text_lines.append((f"右手: {right_region}", (info_x, 42), (200, 100, 255), 20))

        if self.dl_engine is not None:
            panel_w, panel_h = 260, 80
            panel_x, panel_y = 15, 15
            overlay = annotated.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 50), -1)
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 200, 255), 2)
            annotated = cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0)
            text_lines.extend([
                ("交警手势识别", (panel_x + 10, panel_y + 5), (0, 255, 255), 22),
                (show_gesture, (panel_x + 10, panel_y + 28), (0, 215, 255), 28),
                (
                    f"置信度: {show_confidence:.1%}" if show_confidence > 0 else "置信度: —",
                    (panel_x + 10, panel_y + 60),
                    (255, 255, 255),
                    18,
                ),
            ])

        if text_lines:
            annotated = draw_chinese_text_lines(annotated, text_lines)

        keypoints = _pose_landmarks_to_keypoints(self.last_landmarks)
        if self.dl_engine is not None and self.dl_warmed_up:
            gesture = normalize_gesture(show_gesture)
            confidence = round(float(show_confidence), 4)
            display_label = VIDEO_GESTURE_TEXT.get(gesture, show_gesture)
        elif self.display_result and self.display_result != FILTERED_GESTURE:
            gesture = normalize_gesture(self.display_result)
            confidence = round(float(self.display_confidence), 4)
            display_label = VIDEO_GESTURE_TEXT.get(gesture, self.display_result)
        elif self.last_landmarks is None:
            gesture = NO_POSE_GESTURE
            confidence = 0.0
            display_label = NO_POSE_GESTURE
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
            completed_gesture=self._consume_completed_gesture(),
            completed_confidence=self.completed_confidence,
        )

    def _update_action_state(self, feat: dict[str, Any]) -> None:
        both_hip = feat.get("left_region") == "hip" and feat.get("right_region") == "hip"
        if self.action_state == "idle":
            if not both_hip:
                self.action_state = "active"
                self.action_frame_count = 0
                self.completed_gesture = None
                self.completed_confidence = 0.0
                self.dl_window.clear()
                self.dl_filtered_gesture = "无手势"
                self.dl_filtered_confidence = 0.0
                self.hip_both_frames = 0
                if self.dl_engine is not None:
                    self.dl_engine.reset_state()
                self.action_flash_text = "动作开始"
                self.action_flash_remaining = self.action_flash_frames
            return

        self.action_frame_count += 1
        if both_hip:
            self.hip_both_frames += 1
            if self.hip_both_frames >= self.hip_stop_threshold:
                self._reset_action_state(show_flash=True)
        else:
            self.hip_both_frames = 0

    def _select_or_track_target(self, pose_landmarks_list) -> int:
        if not self.target_locked:
            target_index, person_info = _select_target_person(pose_landmarks_list)
            self.target_lock_counter += 1
            if self.target_lock_counter >= police_cfg.TARGET_LOCK_FRAMES:
                self.target_locked = True
                self.target_info = person_info
            return target_index

        target_index, person_info = _track_target_person(
            pose_landmarks_list,
            self.target_info,
            police_cfg.TARGET_TRACK_THRESH,
        )
        if person_info is not None:
            if self.target_info is None:
                self.target_info = person_info
            else:
                alpha = 0.7
                self.target_info["cx"] = alpha * person_info["cx"] + (1.0 - alpha) * self.target_info["cx"]
                self.target_info["cy"] = alpha * person_info["cy"] + (1.0 - alpha) * self.target_info["cy"]
                self.target_info["sw"] = person_info.get("sw", self.target_info.get("sw", 0.0))
        return target_index

    def _reset_target_tracking(self) -> None:
        self.target_locked = False
        self.target_info = None
        self.target_lock_counter = 0

    def _reset_action_state(self, *, show_flash: bool = False) -> None:
        was_active = self.action_state == "active"
        if show_flash and was_active:
            self.completed_gesture = normalize_gesture(self.dl_filtered_gesture)
            self.completed_confidence = round(float(self.dl_filtered_confidence), 4)
        self.action_state = "idle"
        self.action_frame_count = 0
        self.hip_both_frames = 0
        self.dl_window.clear()
        self.dl_filtered_gesture = "无手势"
        self.dl_filtered_confidence = 0.0
        self.dl_gesture = "无手势"
        self.dl_confidence = 0.0
        if self.dl_engine is not None:
            self.dl_engine.reset_state()
        if show_flash and was_active:
            self.action_flash_text = "动作结束"
            self.action_flash_remaining = self.action_flash_frames

    def _consume_completed_gesture(self) -> str | None:
        completed_gesture = self.completed_gesture
        self.completed_gesture = None
        return completed_gesture

    def _update_filtered_dl_result(self, raw_gesture: str, raw_confidence: float) -> None:
        if self.action_state != "active":
            self.dl_gesture = "无手势"
            self.dl_confidence = 0.0
            self.dl_window.clear()
            return

        self.dl_window.append((raw_gesture, raw_confidence))
        if len(self.dl_window) > 60:
            self.dl_window = self.dl_window[-40:]

        if self.action_frame_count <= self.first_guess_frames:
            candidates = [
                (gesture, confidence)
                for gesture, confidence in self.dl_window
                if gesture not in {"无手势", "DL error", "loading...", "预热中..."} and confidence > 0
            ]
            if candidates:
                self.dl_filtered_gesture, self.dl_filtered_confidence = max(candidates, key=lambda item: item[1])
        else:
            runs: list[tuple[str, int, float]] = []
            recent = self.dl_window[-15:]
            if recent:
                current_gesture = recent[0][0]
                confidences = [recent[0][1]]
                for gesture, confidence in recent[1:]:
                    if gesture == current_gesture:
                        confidences.append(confidence)
                    else:
                        runs.append((current_gesture, len(confidences), sum(confidences) / len(confidences)))
                        current_gesture = gesture
                        confidences = [confidence]
                runs.append((current_gesture, len(confidences), sum(confidences) / len(confidences)))
            valid = [
                item
                for item in runs
                if item[0] not in {"无手势", "DL error", "loading...", "预热中..."}
                and item[2] >= self.gesture_min_confidence
                and item[1] >= self.gesture_min_run
            ]
            if valid:
                gesture, _, confidence = valid[-1]
                self.dl_filtered_gesture = gesture
                self.dl_filtered_confidence = confidence

        self.dl_gesture = self.dl_filtered_gesture
        self.dl_confidence = self.dl_filtered_confidence


class PoliceGestureLocalRuntime:
    def __init__(self) -> None:
        self._image_lock = threading.Lock()
        self._image_pose_detector = None
        self._image_hand_detector = None

    def create_video_session(self) -> PoliceGestureVideoSession:
        return PoliceGestureVideoSession()

    def create_camera_session(self) -> PoliceGestureVideoSession:
        return PoliceGestureVideoSession(realtime=True)

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


def _crop_person(frame: np.ndarray, landmarks: Any, target_size: int = 512, padding_ratio: float = 0.25) -> np.ndarray:
    height, width = frame.shape[:2]
    xs = [int(point.x * width) for point in landmarks]
    ys = [int(point.y * height) for point in landmarks]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    box_w, box_h = x2 - x1, y2 - y1
    pad_w = int(box_w * padding_ratio)
    pad_h = int(box_h * padding_ratio)
    x1 = max(0, x1 - pad_w)
    x2 = min(width, x2 + pad_w)
    y1 = max(0, y1 - pad_h)
    y2 = min(height, y2 + pad_h)

    side = max(x2 - x1, y2 - y1)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    half = side // 2
    x1 = max(0, center_x - half)
    x2 = min(width, center_x + half)
    y1 = max(0, center_y - half)
    y2 = min(height, center_y + half)

    crop_w = x2 - x1
    crop_h = y2 - y1
    side = max(1, min(crop_w, crop_h))
    crop = frame[y1:y1 + side, x1:x1 + side]
    return cv2.resize(crop, (target_size, target_size))
