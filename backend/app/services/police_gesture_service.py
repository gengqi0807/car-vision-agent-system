from __future__ import annotations

import asyncio
import base64
from collections import deque
from dataclasses import dataclass
import logging
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4
import time

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from sqlalchemy import or_, select

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.police_gesture_record import PoliceGestureRecord
from app.models_infer import GestureClassifier, MediaPipePose
from app.schemas.gesture import (
    GestureFrameResult,
    GestureHistoryItem,
    Keypoint,
    PoliceGestureVideoEvent,
    PoliceGestureVideoJobCreateResponse,
    PoliceGestureVideoProgress,
    PoliceGestureVideoResult,
)
from app.services.police_gesture_local_runtime import (
    NO_POSE_GESTURE,
    NO_VIDEO_GESTURE,
    PoliceGestureLocalRuntime,
    PoliceGestureVideoSession,
    VIDEO_GESTURE_MAP,
)
from app.services.alert_service import AlertService
from app.services.monitor_service import MonitorService
from app.services.police_video_preview_publisher import PoliceVideoPreviewPublisher
from police.visualization import draw_chinese_text

logger = logging.getLogger(__name__)

AIC_BONES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (3, 4),
    (4, 5),
    (13, 0),
    (13, 3),
    (0, 6),
    (3, 9),
    (6, 7),
    (7, 8),
    (9, 10),
    (10, 11),
    (12, 13),
)

VIDEO_GESTURE_TEXT: dict[str, str] = {
    NO_VIDEO_GESTURE: "无手势",
    "stop": "停止",
    "停止信号": "停止",
    "go_straight": "直行",
    "直行信号": "直行",
    "left_turn": "左转弯",
    "左转弯信号": "左转弯",
    "left_wait_turn": "左待转",
    "左待转信号": "左待转",
    "right_turn": "右转弯",
    "右转弯信号": "右转弯",
    "lane_change": "变道",
    "变道信号": "变道",
    "slow_down": "减速",
    "减速慢行": "减速",
    "减速慢行信号": "减速",
    "pull_over": "靠边停车",
    "靠边停车信号": "靠边停车",
}


@dataclass
class _CameraRuntimeState:
    session: PoliceGestureVideoSession



class PoliceGestureService:
    """Police (traffic) gesture service with image and video recognition."""

    _legacy_pose: Optional[MediaPipePose] = None
    _legacy_classifier: Optional[GestureClassifier] = None
    _progress_lock = threading.Lock()
    _video_progress: dict[str, PoliceGestureVideoProgress] = {}
    _preview_condition = threading.Condition()
    _video_preview_frames: dict[str, deque[bytes]] = {}
    _video_cancel_events: dict[str, threading.Event] = {}
    _unrecognized_behavior_window_seconds = 30
    _video_event_min_gap_multiplier = 3
    _warmup_lock = threading.Lock()
    _runtime_warmed = False

    def __init__(self) -> None:
        self._backend_dir = Path(__file__).resolve().parents[2]
        self._media_root = (self._backend_dir / settings.plate_upload_dir).resolve()
        self._upload_root = (self._media_root / "police").resolve()
        self._runtime = PoliceGestureLocalRuntime()
        self._camera_session_lock = threading.Lock()
        self._camera_sessions: dict[tuple[int, str], _CameraRuntimeState] = {}

    @property
    def legacy_pose(self) -> MediaPipePose:
        if self._legacy_pose is None:
            self._legacy_pose = MediaPipePose()
        return self._legacy_pose

    @property
    def legacy_classifier(self) -> GestureClassifier:
        if self._legacy_classifier is None:
            self._legacy_classifier = GestureClassifier(domain="police")
        return self._legacy_classifier

    def warmup_runtime(self) -> None:
        if self._runtime_warmed:
            return
        with self._warmup_lock:
            if self._runtime_warmed:
                return

            dummy_frame = np.zeros((256, 256, 3), dtype=np.uint8)
            _ = self.legacy_pose.infer(dummy_frame)
            _ = self.legacy_classifier
            _ = self._runtime.recognize_image(dummy_frame)
            with self._runtime.create_camera_session() as session:
                _ = session.process_frame(dummy_frame)
            self._runtime_warmed = True

    async def process_frame(
        self,
        image_bytes: bytes,
        filename: str,
        user_id: int | None,
        session_id: str | None = None,
        input_mode: str = "image",
    ) -> GestureFrameResult:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            await self._capture_error(
                filename=filename,
                event_type="police_gesture_decode_error",
                summary="无法解析图像字节数据。",
            )
            raise ValueError(f"无法解析图像文件“{filename}”")

        logger.info("Processing police-pose frame '%s' (%dx%d)", filename, frame.shape[1], frame.shape[0])
        if input_mode == "camera":
            return self._process_camera_frame(
                frame=frame,
                filename=filename,
                user_id=user_id,
                session_id=session_id,
            )

        visual_result = self._runtime.recognize_image(frame)
        gesture_label, cls_conf, num_poses = self._recognize_image_with_legacy_model(frame)
        keypoints = [Keypoint(x=kp["x"], y=kp["y"], score=kp.get("score", 1.0)) for kp in visual_result.keypoints]
        annotated_frame = self._annotate_image_result(
            visual_result.annotated_frame,
            gesture=gesture_label,
            confidence=cls_conf,
        )
        annotated_image = self._encode_frame_to_data_url(annotated_frame)

        self._save_record(
            gesture_label,
            cls_conf,
            filename,
            keypoints_payload=[{"x": kp.x, "y": kp.y, "score": kp.score} for kp in keypoints],
            user_id=user_id,
            session_id="police-image",
        )

        is_unrecognized = self._is_unrecognized_result(gesture=gesture_label, num_poses=num_poses)
        if is_unrecognized:
            with SessionLocal() as session:
                AlertService(session).record_behavior_once(
                    source="police-gesture",
                    title="交警手势未识别",
                    summary=self._build_unrecognized_behavior_summary(
                        filename=filename,
                        gesture=gesture_label,
                        num_detections=num_poses,
                    ),
                    window_seconds=self._unrecognized_behavior_window_seconds,
                )
        else:
            is_success = self._is_success_confidence(cls_conf)
            await self._capture_monitor_log(
                event_type=(
                    "police_gesture_success"
                    if is_success
                    else "police_gesture_low_confidence"
                ),
                title="交警手势帧处理完成",
                summary=(
                    f"{filename} 已处理完成：手势={self._gesture_label(gesture_label)}，"
                    f"置信度={cls_conf:.2f}，姿态数={num_poses}。"
                ),
                confidence=cls_conf,
                details={
                    "filename": filename,
                    "num_poses_detected": num_poses,
                    "frame_width": int(frame.shape[1]),
                    "frame_height": int(frame.shape[0]),
                    "gesture": gesture_label,
                },
                trigger_alert=False,
                level="info" if is_success else "warning",
            )

        return GestureFrameResult(
            gesture=gesture_label,
            confidence=round(cls_conf, 4),
            keypoints=keypoints,
            annotated_image=annotated_image,
            updated_at=datetime.utcnow(),
        )

    def _process_camera_frame(
        self,
        *,
        frame: np.ndarray,
        filename: str,
        user_id: int | None,
        session_id: str | None,
    ) -> GestureFrameResult:
        active_session_id = session_id or "police-camera"
        runtime_state = self._camera_runtime_state(user_id=user_id or 0, session_id=active_session_id)
        visual_result = runtime_state.session.process_frame(frame)
        gesture_label = visual_result.gesture
        cls_conf = round(float(visual_result.confidence or 0.0), 4)
        keypoints = [Keypoint(x=kp["x"], y=kp["y"], score=kp.get("score", 1.0)) for kp in visual_result.keypoints]
        updated_at = datetime.utcnow()
        annotated_image = self._encode_camera_frame_to_data_url(visual_result.annotated_frame)

        completed_gesture = visual_result.completed_gesture
        completed_confidence = round(float(visual_result.completed_confidence or 0.0), 4)
        if self._is_recordable_completed_gesture(completed_gesture, completed_confidence):
            self._save_record(
                completed_gesture,
                completed_confidence,
                filename,
                keypoints_payload=[{"x": kp.x, "y": kp.y, "score": kp.score} for kp in keypoints],
                user_id=user_id,
                session_id=active_session_id,
            )

        return GestureFrameResult(
            gesture=gesture_label,
            confidence=cls_conf,
            keypoints=keypoints,
            annotated_image=annotated_image,
            updated_at=updated_at,
        )

    def process_video_bytes(
        self,
        video_bytes: bytes,
        filename: str,
        user_id: int | None,
        task_id: str | None = None,
        cancel_event: threading.Event | None = None,
    ) -> PoliceGestureVideoResult:
        if not video_bytes:
            raise ValueError("Uploaded video is empty.")

        resolved_task_id = task_id or uuid4().hex
        self._set_video_progress(
            resolved_task_id,
            source_filename=filename,
            status="queued",
            progress=0.0,
            message="已接收视频，准备开始识别。",
        )
        with self._preview_condition:
            self._video_preview_frames[resolved_task_id] = deque(
                maxlen=max(settings.police_video_preview_buffer_max_frames, 1)
            )
            self._preview_condition.notify_all()

        logger.info("Starting police gesture video processing: %s", filename)
        source_path, output_path = self._prepare_video_paths(filename)
        capture = None
        writer = None
        preview_publisher: PoliceVideoPreviewPublisher | None = None
        temp_output_path = output_path.with_name(f"{output_path.stem}.raw.mp4")
        processed_frame_count = 0
        total_frames = 0
        best_result = self._empty_video_result()
        last_result = self._empty_video_result()
        preview_interval_seconds = 1.0 / max(settings.police_video_preview_max_fps, 0.1)
        last_preview_sent_at = 0.0
        inference_count = 0
        inference_elapsed_seconds = 0.0
        low_confidence_streak = 0
        last_low_confidence_at: float | None = None
        no_gesture_started_at = 0.0

        try:
            source_path.write_bytes(video_bytes)
            self._set_video_progress(
                resolved_task_id,
                source_filename=filename,
                status="preparing",
                progress=0.06,
                message="视频上传完成，正在加载识别模型。",
            )
            logger.info("Saved uploaded police video to: %s", source_path)

            capture = cv2.VideoCapture(str(source_path))
            if not capture.isOpened():
                raise RuntimeError("无法打开上传的视频文件。")

            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps <= 1:
                fps = 25.0
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            preview_publisher = PoliceVideoPreviewPublisher(resolved_task_id, fps)
            preview_publisher.start()
            progress_log_interval = max(settings.police_video_progress_log_interval_frames, 10)

            logger.info(
                (
                    "Police gesture video runtime configured: %s | fps=%.2f total_frames=%s "
                    "skip=%d infer_scale=%.2f output_max_side=%d preview_max_side=%d"
                ),
                filename,
                fps,
                total_frames if total_frames > 0 else "unknown",
                2,
                0.5,
                settings.police_video_output_max_side,
                settings.police_video_preview_max_side,
            )

            def annotate_and_write_frame(frame_index: int, annotated_source: np.ndarray) -> np.ndarray:
                nonlocal writer, processed_frame_count
                annotated_frame = self._resize_frame_to_limit(
                    annotated_source,
                    settings.police_video_output_max_side,
                )
                if writer is None:
                    height, width = annotated_frame.shape[:2]
                    temp_output_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(temp_output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("无法创建交警手势标注视频文件。")
                writer.write(annotated_frame)
                processed_frame_count = frame_index
                return annotated_frame

            with self._runtime.create_video_session() as session:
                no_gesture_started_at = time.monotonic()
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        raise InterruptedError("视频识别已由用户终止。")
                    ok, current_frame = capture.read()
                    if not ok or current_frame is None:
                        break

                    current_frame_index = processed_frame_count + 1
                    inference_started_at = time.perf_counter()
                    frame_result = session.process_frame(current_frame)
                    inference_elapsed_seconds += max(time.perf_counter() - inference_started_at, 0.0)
                    inference_count += 1
                    last_result = {
                        "gesture": frame_result.gesture,
                        "confidence": frame_result.confidence,
                        "keypoints": frame_result.keypoints,
                        "display_label": frame_result.display_label,
                    }
                    if self._should_replace_best(best_result, last_result):
                        best_result = last_result
                    completed_gesture = frame_result.completed_gesture
                    completed_confidence = round(float(frame_result.completed_confidence or 0.0), 4)
                    if self._is_recordable_completed_gesture(completed_gesture, completed_confidence):
                        self._append_video_event(
                            resolved_task_id,
                            source_filename=filename,
                            gesture=completed_gesture,
                            confidence=completed_confidence,
                            frame_index=current_frame_index,
                            fps=fps,
                        )

                    now = time.monotonic()
                    if self._is_effective_gesture(frame_result.gesture):
                        no_gesture_started_at = now
                    elif now - no_gesture_started_at >= settings.police_gesture_no_gesture_alert_seconds:
                        self._capture_monitor_log_sync(
                            event_type="police_gesture_no_gesture_timeout",
                            title="交警手势长时间无动作",
                            summary=(
                                f"{filename} 在视频识别过程中连续 "
                                f"{settings.police_gesture_no_gesture_alert_seconds} 秒未识别到有效动作。"
                            ),
                            details={
                                "filename": filename,
                                "task_id": resolved_task_id,
                                "frame_index": current_frame_index,
                                "mode": "video",
                            },
                            trigger_alert=True,
                            level="warning",
                        )
                        no_gesture_started_at = now

                    if self._is_effective_gesture(completed_gesture):
                        no_gesture_started_at = now
                        if self._is_success_confidence(completed_confidence):
                            low_confidence_streak = 0
                            last_low_confidence_at = None
                            self._capture_monitor_log_sync(
                                event_type="police_gesture_success",
                                title="交警手势识别成功",
                                summary=(
                                    f"{filename} 成功识别动作 {self._gesture_label(completed_gesture)}，"
                                    f"置信率 {completed_confidence:.2f}。"
                                ),
                                confidence=completed_confidence,
                                details={
                                    "filename": filename,
                                    "task_id": resolved_task_id,
                                    "frame_index": current_frame_index,
                                    "gesture": completed_gesture,
                                    "mode": "video",
                                },
                                trigger_alert=False,
                                level="info",
                            )
                        elif self._is_low_confidence(completed_confidence):
                            if self._should_reset_low_confidence_streak(last_low_confidence_at, now):
                                low_confidence_streak = 0
                            low_confidence_streak += 1
                            last_low_confidence_at = now
                            reached_streak_threshold = (
                                low_confidence_streak >= settings.police_gesture_low_confidence_streak
                            )
                            self._capture_monitor_log_sync(
                                event_type=(
                                    "police_gesture_low_confidence_alert"
                                    if reached_streak_threshold
                                    else "police_gesture_low_confidence"
                                ),
                                title=(
                                    "交警手势连续低置信率告警"
                                    if reached_streak_threshold
                                    else "交警手势低置信率"
                                ),
                                summary=(
                                    f"{filename} 识别到动作 {self._gesture_label(completed_gesture)}，"
                                    f"但置信率仅 {completed_confidence:.2f}，"
                                    f"连续低置信次数 {low_confidence_streak}。"
                                ),
                                confidence=completed_confidence,
                                details={
                                    "filename": filename,
                                    "task_id": resolved_task_id,
                                    "frame_index": current_frame_index,
                                    "gesture": completed_gesture,
                                    "mode": "video",
                                    "low_confidence_streak": low_confidence_streak,
                                },
                                trigger_alert=reached_streak_threshold,
                                level="warning",
                            )
                            if reached_streak_threshold:
                                low_confidence_streak = 0
                                last_low_confidence_at = None
                        else:
                            low_confidence_streak = 0
                            last_low_confidence_at = None

                    annotated = annotate_and_write_frame(current_frame_index, frame_result.annotated_frame)
                    preview_publisher.submit(
                        current_frame_index,
                        self._resize_frame_to_limit(annotated, settings.police_video_preview_max_side)
                    )

                    if processed_frame_count == 1 or processed_frame_count % progress_log_interval == 0:
                        logger.info(
                            "Police gesture video progress: %s frame=%d inferences=%d avg_infer_ms=%.1f",
                            filename,
                            processed_frame_count,
                            inference_count,
                            (inference_elapsed_seconds * 1000 / inference_count) if inference_count > 0 else 0.0,
                        )

                    should_push_preview = (
                        processed_frame_count == 1
                        or now - last_preview_sent_at >= preview_interval_seconds
                    )
                    if should_push_preview:
                        preview_frame = self._resize_frame_to_limit(annotated, settings.police_video_preview_max_side)
                        self._update_preview_frame(resolved_task_id, preview_frame)
                        last_preview_sent_at = now

                    self._set_video_progress(
                        resolved_task_id,
                        source_filename=filename,
                        status="processing",
                        progress=self._build_video_progress(processed_frame_count, total_frames, 0.1, 0.82),
                        message=(
                            f"正在识别视频内容，已处理 {processed_frame_count} 帧。"
                            if last_result["gesture"] == NO_VIDEO_GESTURE
                            else (
                                f"识别到动作：{self._video_gesture_text(last_result['gesture'])}，"
                                f"已处理 {processed_frame_count} 帧。"
                            )
                        ),
                        processed_frame_count=processed_frame_count,
                        total_frames=total_frames if total_frames > 0 else None,
                        gesture=last_result["gesture"] if last_result["gesture"] != NO_VIDEO_GESTURE else None,
                        confidence=last_result["confidence"] if last_result["gesture"] != NO_VIDEO_GESTURE else None,
                        playback_url=preview_publisher.playback_url if preview_publisher.ready else None,
                    )

            if processed_frame_count == 0:
                raise RuntimeError("上传的视频没有可读取的帧。")

            if best_result["gesture"] == NO_VIDEO_GESTURE and last_result["gesture"] != NO_VIDEO_GESTURE:
                best_result = last_result

            if writer is not None:
                writer.release()
                writer = None
            if preview_publisher is not None:
                preview_publisher.close()
                preview_publisher = None

            self._set_video_progress(
                resolved_task_id,
                source_filename=filename,
                status="transcoding",
                progress=0.9,
                message="识别完成，正在生成可播放的标注视频。",
                processed_frame_count=processed_frame_count,
                total_frames=total_frames if total_frames > 0 else None,
                gesture=best_result["gesture"] if best_result["gesture"] != NO_VIDEO_GESTURE else None,
                confidence=best_result["confidence"] if best_result["gesture"] != NO_VIDEO_GESTURE else None,
                playback_url="",
            )

            logger.info("Transcoding processed police video: %s -> %s", temp_output_path.name, output_path.name)
            self._transcode_video_for_web(temp_output_path, output_path)
            try:
                temp_output_path.unlink(missing_ok=True)
            except Exception:
                logger.warning("Failed to remove temporary processed police video: %s", temp_output_path)

            self._save_record(
                best_result["gesture"],
                best_result["confidence"],
                self.public_media_url_for(output_path),
                keypoints_payload=best_result["keypoints"],
                user_id=user_id,
                session_id="police-video",
            )

            duration_seconds = round(processed_frame_count / fps, 2) if fps > 0 else None
            logger.info(
                (
                    "Police gesture video processing finished: %s gesture=%s confidence=%.4f "
                    "frames=%d inferences=%d avg_infer_ms=%.1f"
                ),
                filename,
                best_result["gesture"],
                best_result["confidence"],
                processed_frame_count,
                inference_count,
                (inference_elapsed_seconds * 1000 / inference_count) if inference_count > 0 else 0.0,
            )

            response = PoliceGestureVideoResult(
                source_filename=filename,
                gesture=best_result["gesture"],
                confidence=best_result["confidence"],
                keypoints=[
                    Keypoint(x=item["x"], y=item["y"], score=item.get("score", 1.0))
                    for item in best_result["keypoints"]
                ],
                task_id=resolved_task_id,
                processed_video_url=self.public_media_url_for(output_path),
                processed_frame_count=processed_frame_count,
                duration_seconds=duration_seconds,
                updated_at=datetime.utcnow(),
            )
            self._set_video_progress(
                resolved_task_id,
                source_filename=filename,
                status="completed",
                progress=1.0,
                message="视频识别完成，标注视频已生成。",
                processed_frame_count=processed_frame_count,
                total_frames=total_frames if total_frames > 0 else None,
                gesture=response.gesture,
                confidence=response.confidence,
                processed_video_url=response.processed_video_url,
                duration_seconds=response.duration_seconds,
            )
            return response
        except Exception as exc:
            cancelled = isinstance(exc, InterruptedError)
            self._set_video_progress(
                resolved_task_id,
                source_filename=filename,
                status="cancelled" if cancelled else "failed",
                progress=1.0,
                message=str(exc),
                processed_frame_count=processed_frame_count,
                total_frames=total_frames if total_frames > 0 else None,
            )
            raise
        finally:
            self._finish_preview_stream(resolved_task_id)
            if capture is not None:
                capture.release()
            if writer is not None:
                writer.release()
            if preview_publisher is not None:
                preview_publisher.close()
            if not settings.plate_save_uploads:
                try:
                    source_path.unlink(missing_ok=True)
                except Exception:
                    logger.warning("Failed to remove temporary uploaded police video: %s", source_path)

    def start_video_job(
        self,
        video_bytes: bytes,
        filename: str,
        user_id: int | None,
        task_id: str | None = None,
    ) -> PoliceGestureVideoJobCreateResponse:
        if not video_bytes:
            raise ValueError("Uploaded video is empty.")
        resolved_task_id = task_id or uuid4().hex
        cancel_event = threading.Event()
        with self._progress_lock:
            current = self._video_progress.get(resolved_task_id)
            if current is not None and current.status not in {"completed", "failed", "cancelled", "missing"}:
                raise ValueError("该视频任务正在处理中。")
            self._video_cancel_events[resolved_task_id] = cancel_event

        def worker() -> None:
            try:
                self.process_video_bytes(
                    video_bytes,
                    filename,
                    user_id,
                    resolved_task_id,
                    cancel_event,
                )
            except InterruptedError:
                logger.info("Police gesture video job cancelled: %s", resolved_task_id)
            except Exception:
                logger.exception("Police gesture video job failed: %s", resolved_task_id)
            finally:
                with self._progress_lock:
                    self._video_cancel_events.pop(resolved_task_id, None)

        threading.Thread(target=worker, daemon=True, name=f"police-video-{resolved_task_id[:8]}").start()
        return PoliceGestureVideoJobCreateResponse(task_id=resolved_task_id, status="queued")

    def cancel_video_job(self, task_id: str) -> PoliceGestureVideoProgress:
        with self._progress_lock:
            cancel_event = self._video_cancel_events.get(task_id)
            progress = self._video_progress.get(task_id)
            if cancel_event is not None:
                cancel_event.set()
            if progress is None:
                return PoliceGestureVideoProgress(
                    task_id=task_id,
                    status="missing",
                    message="未找到对应的视频识别任务。",
                    updated_at=datetime.utcnow(),
                )
            if progress.status not in {"completed", "failed", "cancelled"}:
                progress = progress.model_copy(update={"status": "cancelling", "message": "正在终止视频识别..."})
                self._video_progress[task_id] = progress
            return progress.model_copy(deep=True)

    def history(self, user_id: int | None = None) -> list[GestureHistoryItem]:
        try:
            with SessionLocal() as session:
                statement = select(PoliceGestureRecord)
                if user_id is not None:
                    statement = statement.where(
                        or_(PoliceGestureRecord.user_id == user_id, PoliceGestureRecord.user_id.is_(None))
                    )
                records = session.scalars(
                    statement.order_by(PoliceGestureRecord.created_at.desc()).limit(20)
                ).all()
        except Exception as exc:
            logger.warning("Failed to load police gesture history: %s", exc)
            return []

        return [
            GestureHistoryItem(
                gesture=record.gesture,
                confidence=record.confidence,
                source_path=record.source_path,
                updated_at=record.updated_at or record.created_at,
            )
            for record in records
        ]

    def _camera_runtime_state(self, *, user_id: int, session_id: str) -> _CameraRuntimeState:
        key = (user_id, session_id)
        with self._camera_session_lock:
            state = self._camera_sessions.get(key)
            if state is None:
                state = _CameraRuntimeState(session=self._runtime.create_camera_session())
                self._camera_sessions[key] = state
            return state

    @staticmethod
    def _is_recordable_completed_gesture(gesture: str | None, confidence: float) -> bool:
        if gesture in {
            None,
            "",
            NO_VIDEO_GESTURE,
            NO_POSE_GESTURE,
            "no_pose",
            "unknown",
            "other",
            "其他",
            "其他手势",
            "无手势",
        }:
            return False
        if confidence < settings.police_gesture_success_confidence_threshold:
            return False
        return True

    @staticmethod
    def _is_effective_gesture(gesture: str | None) -> bool:
        return gesture not in {
            None,
            "",
            NO_VIDEO_GESTURE,
            NO_POSE_GESTURE,
            "no_pose",
            "unknown",
            "other",
        }

    @staticmethod
    def _is_success_confidence(confidence: float) -> bool:
        return float(confidence) >= settings.police_gesture_success_confidence_threshold

    @staticmethod
    def _is_low_confidence(confidence: float) -> bool:
        return 0.0 < float(confidence) < settings.police_gesture_low_confidence_threshold

    @staticmethod
    def _should_reset_low_confidence_streak(last_low_confidence_at: float | None, current_time: float) -> bool:
        if last_low_confidence_at is None:
            return True
        return (current_time - last_low_confidence_at) > settings.police_gesture_low_confidence_window_seconds

    def _capture_monitor_log_sync(
        self,
        *,
        event_type: str,
        title: str,
        summary: str,
        confidence: float | None = None,
        details: dict | None = None,
        trigger_alert: bool = False,
        level: str = "info",
    ) -> None:
        asyncio.run(
            self._capture_monitor_log(
                event_type=event_type,
                title=title,
                summary=summary,
                confidence=confidence,
                details=details,
                trigger_alert=trigger_alert,
                level=level,
            )
        )

    def get_video_progress(self, task_id: str) -> PoliceGestureVideoProgress:
        with self._progress_lock:
            progress = self._video_progress.get(task_id)
            if progress is None:
                return PoliceGestureVideoProgress(
                    task_id=task_id,
                    status="missing",
                    progress=0.0,
                    message="未找到对应的视频识别任务。",
                    updated_at=datetime.utcnow(),
                )
            return progress

    def _prepare_video_paths(self, filename: str) -> tuple[Path, Path]:
        suffix = Path(filename).suffix.lower() or ".mp4"
        source_dir = self._upload_root / "videos" / "source"
        output_dir = self._upload_root / "videos" / "processed"
        source_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        token = uuid4().hex
        source_path = source_dir / f"{token}{suffix}"
        output_path = output_dir / f"{token}.mp4"
        return source_path, output_path

    def public_media_url_for(self, file_path: Path) -> str:
        relative_path = file_path.resolve().relative_to(self._media_root)
        return f"/media/{relative_path.as_posix()}"

    def _resolve_video_process_interval(self, fps: float) -> int:
        base_interval = max(int(settings.police_video_process_every_n_frames), 1)
        target_inference_fps = max(float(settings.police_video_target_inference_fps), 0.1)
        adaptive_interval = max(int(round(fps / target_inference_fps)), 1)
        return max(base_interval, adaptive_interval)

    def _append_video_event(
        self,
        task_id: str,
        *,
        source_filename: str,
        gesture: str,
        confidence: float,
        frame_index: int,
        fps: float,
    ) -> None:
        with self._progress_lock:
            current = self._video_progress.get(task_id)
            events = list(current.events) if current is not None else []
            events.append(
                PoliceGestureVideoEvent(
                    gesture=gesture,
                    confidence=confidence,
                    frame_index=frame_index,
                    timestamp_seconds=round(frame_index / fps, 2) if fps > 0 else None,
                    message=f"识别到动作：{self._video_gesture_text(gesture)}",
                    updated_at=datetime.utcnow(),
                )
            )
            self._video_progress[task_id] = PoliceGestureVideoProgress(
                task_id=task_id,
                source_filename=source_filename,
                status=current.status if current is not None else "processing",
                progress=current.progress if current is not None else 0.0,
                message=current.message if current is not None else "正在识别视频内容。",
                processed_frame_count=current.processed_frame_count if current is not None else frame_index,
                total_frames=current.total_frames if current is not None else None,
                gesture=gesture,
                confidence=confidence,
                annotated_frame=None,
                events=events[-20:],
                updated_at=datetime.utcnow(),
            )

    def _video_gesture_text(self, gesture: str) -> str:
        return VIDEO_GESTURE_TEXT.get(gesture, gesture or "未知动作")

    def _resize_frame_to_limit(self, frame: np.ndarray, max_side_limit: int) -> np.ndarray:
        if max_side_limit <= 0:
            return frame

        frame_height, frame_width = frame.shape[:2]
        max_side = max(frame_width, frame_height)
        if max_side <= max_side_limit:
            return frame

        scale_ratio = max_side_limit / float(max_side)
        resized_width = max(int(frame_width * scale_ratio), 1)
        resized_height = max(int(frame_height * scale_ratio), 1)
        return cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    def _transcode_video_for_web(self, source_path: Path, output_path: Path) -> None:
        command = [
            settings.plate_push_ffmpeg_bin,
            "-y",
            "-i",
            str(source_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "ultrafast",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        try:
            result = subprocess.run(command, capture_output=True, check=False)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"ffmpeg was not found. Cannot transcode the processed police video. "
                f"Current value: {settings.plate_push_ffmpeg_bin!r}."
            ) from exc

        if result.returncode != 0:
            stderr_text = result.stderr.decode("utf-8", errors="ignore").strip() if result.stderr else ""
            raise RuntimeError(
                f"Processed police video transcode failed. "
                f"{stderr_text or 'Please verify ffmpeg can run normally.'}"
            )

    def _empty_video_result(self) -> dict:
        return {
            "gesture": NO_VIDEO_GESTURE,
            "confidence": 0.0,
            "keypoints": [],
            "display_label": VIDEO_GESTURE_TEXT[NO_VIDEO_GESTURE],
        }

    def _normalize_video_result(self, raw_result: dict) -> dict:
        raw_gesture = str(raw_result.get("gesture", "")).strip()
        normalized_gesture = VIDEO_GESTURE_MAP.get(raw_gesture, raw_gesture or NO_VIDEO_GESTURE)
        confidence = float(raw_result.get("confidence", 0.0) or 0.0)
        keypoints = [
            {
                "x": float(item.get("x", 0.0)),
                "y": float(item.get("y", 0.0)),
                "score": 1.0,
            }
            for item in raw_result.get("keypoints", [])
        ]
        return {
            "gesture": normalized_gesture,
            "confidence": round(confidence, 4),
            "keypoints": keypoints,
            "display_label": VIDEO_GESTURE_TEXT.get(normalized_gesture, raw_gesture or "未知动作"),
        }

    def _should_replace_best(self, current_best: dict, candidate: dict) -> bool:
        if candidate["gesture"] == NO_VIDEO_GESTURE:
            return False
        if current_best["gesture"] == NO_VIDEO_GESTURE:
            return True
        return candidate["confidence"] >= current_best["confidence"]

    def _annotate_video_frame(self, frame, result: dict):
        annotated = frame.copy()
        height, width = annotated.shape[:2]

        for start_index, end_index in AIC_BONES:
            if start_index < len(result["keypoints"]) and end_index < len(result["keypoints"]):
                start = result["keypoints"][start_index]
                end = result["keypoints"][end_index]
                start_point = (int(start["x"] * width), int(start["y"] * height))
                end_point = (int(end["x"] * width), int(end["y"] * height))
                cv2.line(annotated, start_point, end_point, (0, 165, 255), 2, cv2.LINE_AA)

        for keypoint in result["keypoints"]:
            point = (int(keypoint["x"] * width), int(keypoint["y"] * height))
            cv2.circle(annotated, point, 4, (80, 255, 170), -1, cv2.LINE_AA)
            cv2.circle(annotated, point, 6, (10, 40, 90), 1, cv2.LINE_AA)

        label = result["display_label"]
        confidence_text = f"{result['confidence'] * 100:.1f}%"
        panel_width = 260
        panel_height = 96
        panel_right = width - 14
        panel_left = max(panel_right - panel_width, 14)
        panel_top = 14
        panel_bottom = min(panel_top + panel_height, height - 14)

        cv2.rectangle(annotated, (panel_left, panel_top), (panel_right, panel_bottom), (14, 22, 38), -1)
        cv2.rectangle(annotated, (panel_left, panel_top), (panel_right, panel_bottom), (78, 196, 255), 2)
        annotated = draw_chinese_text(annotated, "交警手势", (panel_left + 14, panel_top + 10), (255, 255, 255), 24)
        annotated = draw_chinese_text(annotated, label, (panel_left + 14, panel_top + 38), (80, 255, 170), 28)
        cv2.putText(
            annotated,
            confidence_text,
            (panel_left + 14, panel_top + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 226, 145),
            2,
        )
        return annotated

    def _encode_preview_frame_bytes(self, frame: np.ndarray) -> bytes | None:
        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), settings.police_video_preview_jpeg_quality],
        )
        if not ok:
            return None
        return buffer.tobytes()

    def _encode_frame_to_data_url(self, frame: np.ndarray) -> str | None:
        preview = self._resize_frame_to_limit(frame, 1440)
        ok, buffer = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            return None
        encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _encode_camera_frame_to_data_url(self, frame: np.ndarray) -> str | None:
        preview = self._resize_frame_to_limit(frame, 960)
        ok, buffer = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 74])
        if not ok:
            return None
        encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _recognize_image_with_legacy_model(self, frame: np.ndarray) -> tuple[str, float, int]:
        legacy_result = self.legacy_pose.infer(frame)
        raw_kps = legacy_result["keypoints"]
        num_poses = int(legacy_result.get("num_poses_detected", 0) or 0)
        if num_poses == 0:
            return NO_POSE_GESTURE, 0.0, 0

        cls_result = self.legacy_classifier.classify(raw_kps, domain="police")
        gesture_label = str(cls_result.get("gesture", "unknown") or "unknown")
        cls_conf = float(cls_result.get("confidence", 0.0) or 0.0)
        return gesture_label, round(cls_conf, 4), num_poses

    def _annotate_image_result(self, frame: np.ndarray, *, gesture: str, confidence: float) -> np.ndarray:
        annotated = frame.copy()
        label = self._video_gesture_text(gesture)
        panel_width = 280
        panel_height = 96
        height, width = annotated.shape[:2]
        panel_right = width - 14
        panel_left = max(panel_right - panel_width, 14)
        panel_top = 14
        panel_bottom = min(panel_top + panel_height, height - 14)

        cv2.rectangle(annotated, (panel_left, panel_top), (panel_right, panel_bottom), (18, 24, 42), -1)
        cv2.rectangle(annotated, (panel_left, panel_top), (panel_right, panel_bottom), (92, 210, 255), 2)
        annotated = draw_chinese_text(annotated, "交警手势图片识别", (panel_left + 14, panel_top + 10), (255, 255, 255), 22)
        annotated = draw_chinese_text(annotated, label, (panel_left + 14, panel_top + 38), (95, 255, 180), 28)
        cv2.putText(
            annotated,
            f"{confidence * 100:.1f}%",
            (panel_left + 14, panel_top + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 226, 145),
            2,
        )
        return annotated

    def _update_preview_frame(self, task_id: str, frame: np.ndarray) -> None:
        encoded = self._encode_preview_frame_bytes(frame)
        if encoded is None:
            return
        with self._preview_condition:
            queue = self._video_preview_frames.get(task_id)
            if queue is None:
                queue = deque(maxlen=max(settings.police_video_preview_buffer_max_frames, 1))
                self._video_preview_frames[task_id] = queue
            queue.append(encoded)
            self._preview_condition.notify_all()

    def _finish_preview_stream(self, task_id: str) -> None:
        with self._preview_condition:
            self._preview_condition.notify_all()

    def iter_video_preview_stream(self, task_id: str):
        target_interval = 1.0 / max(settings.police_video_preview_stream_fps, 0.1)
        initial_buffer_frames = max(settings.police_video_preview_initial_buffer_frames, 1)
        waited_seconds = 0.0
        wait_step = 0.1
        started = False
        last_frame_bytes: bytes | None = None

        try:
            while True:
                with self._preview_condition:
                    preview_queue = self._video_preview_frames.get(task_id)
                    progress = self._video_progress.get(task_id)
                    queue_length = len(preview_queue) if preview_queue is not None else 0
                    is_finished = progress is not None and progress.status in {"completed", "failed", "missing"}

                    if not started:
                        enough_frames = queue_length >= initial_buffer_frames
                        waited_long_enough = waited_seconds >= 1.0 and queue_length > 0
                        can_start_early = is_finished and queue_length > 0
                        if not enough_frames and not waited_long_enough and not can_start_early:
                            self._preview_condition.wait(timeout=wait_step)
                            waited_seconds += wait_step
                            if waited_seconds >= 20.0 and progress is None:
                                break
                            continue
                        started = True

                    if preview_queue:
                        frame_bytes = preview_queue.popleft()
                        last_frame_bytes = frame_bytes
                    else:
                        frame_bytes = last_frame_bytes

                if frame_bytes is None:
                    if is_finished:
                        break
                    time.sleep(wait_step)
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-store\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
                time.sleep(target_interval)

                if is_finished and (not preview_queue or len(preview_queue) == 0):
                    break
        finally:
            with self._preview_condition:
                self._video_preview_frames.pop(task_id, None)

    def _save_record(
        self,
        gesture: str,
        confidence: float,
        source_path: str | None,
        *,
        keypoints_payload: list[dict] | None = None,
        user_id: int | None = None,
        session_id: str | None = "police",
        processing_time_ms: int | None = None,
    ) -> None:
        with SessionLocal() as session:
            session.add(
                PoliceGestureRecord(
                    user_id=user_id,
                    session_id=session_id,
                    gesture=gesture,
                    confidence=confidence,
                    keypoints=keypoints_payload,
                    processing_time_ms=processing_time_ms,
                    source_path=source_path,
                )
            )
            session.commit()

    def _set_video_progress(
        self,
        task_id: str,
        *,
        source_filename: str,
        status: str,
        progress: float,
        message: str,
        processed_frame_count: int = 0,
        total_frames: int | None = None,
        gesture: str | None = None,
        confidence: float | None = None,
        annotated_frame: str | None = None,
        playback_url: str | None = None,
        processed_video_url: str | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        with self._progress_lock:
            current = self._video_progress.get(task_id)
            self._video_progress[task_id] = PoliceGestureVideoProgress(
                task_id=task_id,
                source_filename=source_filename,
                status=status,
                progress=max(0.0, min(progress, 1.0)),
                message=message,
                processed_frame_count=processed_frame_count,
                total_frames=total_frames,
                gesture=gesture,
                confidence=confidence,
                annotated_frame=annotated_frame,
                playback_url=playback_url if playback_url is not None else (current.playback_url if current else None),
                processed_video_url=(
                    processed_video_url if processed_video_url is not None else (current.processed_video_url if current else None)
                ),
                duration_seconds=(duration_seconds if duration_seconds is not None else (current.duration_seconds if current else None)),
                events=list(current.events) if current is not None else [],
                updated_at=datetime.utcnow(),
            )

    def _build_video_progress(
        self,
        processed_frame_count: int,
        total_frames: int,
        start: float,
        end: float,
    ) -> float:
        if total_frames <= 0:
            return start
        ratio = min(max(processed_frame_count / total_frames, 0.0), 1.0)
        return start + (end - start) * ratio

    def _is_unrecognized_result(self, *, gesture: str, num_poses: int) -> bool:
        return num_poses == 0 or gesture in {"unknown", NO_POSE_GESTURE}

    def _build_unrecognized_behavior_summary(
        self,
        *,
        filename: str,
        gesture: str,
        num_detections: int,
    ) -> str:
        return (
            f"{filename} 未识别出有效的交警手势。"
            f"手势={self._gesture_label(gesture)}，姿态数={num_detections}。"
        )

    def _gesture_label(self, gesture: str) -> str:
        labels = {
            "stop": "停止",
            "go_straight": "直行",
            "turn_left": "左转",
            "turn_right": "右转",
            "slow_down": "减速",
            "pull_over": "靠边停车",
            "lane_change": "变道",
            "unknown": "未知",
            NO_POSE_GESTURE: NO_POSE_GESTURE,
        }
        return labels.get(gesture, gesture)

    async def _capture_monitor_log(
        self,
        *,
        event_type: str,
        title: str,
        summary: str,
        confidence: float | None = None,
        details: dict | None = None,
        trigger_alert: bool = False,
        level: str = "info",
    ) -> None:
        with SessionLocal() as session:
            await MonitorService(session).capture_event(
                category="police_gesture",
                source="police-gesture",
                event_type=event_type,
                title=title,
                summary=summary,
                level=level,
                status="processed" if confidence and confidence > 0 else "empty",
                confidence=confidence,
                details=details,
                trigger_alert=trigger_alert,
            )

    async def _capture_error(
        self,
        *,
        filename: str,
        event_type: str,
        summary: str,
    ) -> None:
        with SessionLocal() as session:
            await MonitorService(session).capture_event(
                category="police_gesture",
                source="police-gesture",
                event_type=event_type,
                title="交警手势帧处理失败",
                summary=f"{filename}: {summary}",
                level="warning",
                status="failed",
                details={"filename": filename},
                trigger_alert=False,
            )
