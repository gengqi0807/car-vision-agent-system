from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
from time import perf_counter
import re
import subprocess
import time
from uuid import uuid4

from sqlalchemy import select

from app.core.config import settings
from app.core.database import SessionLocal
from app.core.logger import get_logger
from app.models.plate_record import PlateRecord
from app.models.user_operation_log import UserOperationLog
from app.models_infer.errors import (
    InferenceConfigurationError,
    InferenceDependencyError,
    InferenceTimeoutError,
    PlateInferenceError,
)
from app.models_infer.hyperlpr_recognizer import HyperLPRRecognizer
from app.models_infer.open_traffic_flow_lpr_recognizer import OpenTrafficFlowLPRRecognizer
from app.models_infer.yolo_detector import YoloDetector
from app.schemas.plate import (
    PlateDetection,
    PlateRecognitionResponse,
    PlateRecordSummary,
    PlateVideoRecognitionResponse,
)
from app.services.alert_service import AlertService
from app.services.monitor_service import MonitorService

logger = get_logger(__name__)


@dataclass
class PlateTrack:
    track_id: str
    plate_number: str
    plate_color: str
    confidence: float
    bbox: list[int]
    template: object
    last_seen_frame: int
    last_recognized_frame: int
    misses: int = 0
    text_votes: dict[str, float] = field(default_factory=dict)
    color_votes: dict[str, float] = field(default_factory=dict)
    unread_snapshots_saved: int = 0
    last_unread_snapshot_frame: int = 0


@dataclass
class PlateProcessingState:
    frame_index: int = 0
    last_history_saved_at: float = 0.0
    last_recognition_frame: int = field(
        default_factory=lambda: -settings.plate_stream_process_every_n_frames
    )
    last_tracking_frame: int = 0
    last_probe_frame: int = 0
    tracks: list[PlateTrack] = field(default_factory=list)
    recognition_interval: int = settings.plate_stream_process_every_n_frames
    heavy_scan_interval: int = settings.plate_stream_process_every_n_frames
    working_width: int = 0
    working_height: int = 0


@dataclass
class VideoDetectionStats:
    detection: PlateDetection
    fresh_count: int = 0
    display_count: int = 0


@dataclass
class UnreadSample:
    track_id: str
    frame_index: int
    file_path: Path


@dataclass
class BestUnreadCandidate:
    track_id: str
    crop: object
    bbox: list[int]
    frame_index: int
    quality_score: float


class PlateService:
    def __init__(self) -> None:
        self.recognizer = HyperLPRRecognizer()
        self.crop_recognizer = OpenTrafficFlowLPRRecognizer()
        self.detector = YoloDetector()
        self._backend_dir = Path(__file__).resolve().parents[2]
        self._anonymous_history: list[PlateRecordSummary] = []
        self._anonymous_history_id = 0
        self._upload_root = (self._backend_dir / settings.plate_upload_dir).resolve()

    async def recognize_image(self, filename: str, image_bytes: bytes | None = None) -> PlateRecognitionResponse:
        return await self.recognize_image_bytes_async(image_bytes or b"", filename)

    async def recognize_image_bytes_async(
        self,
        image_bytes: bytes,
        filename: str = "unknown.jpg",
        *,
        save_history: bool = False,
        user_id: int | None = None,
    ) -> PlateRecognitionResponse:
        if not image_bytes:
            raise ValueError("上传文件为空。")

        image_path = self._persist_upload(image_bytes, filename) if settings.plate_save_uploads else None
        started_at = perf_counter()
        trace_id = uuid4().hex

        try:
            detections = await asyncio.wait_for(
                asyncio.to_thread(self._recognize_detections, image_bytes),
                timeout=settings.plate_inference_timeout_seconds,
            )
        except TimeoutError as exc:
            elapsed_ms = int((perf_counter() - started_at) * 1000)
            await self._record_failure(
                user_id=user_id,
                filename=filename,
                elapsed_ms=elapsed_ms,
                trace_id=trace_id,
                title="车牌识别超时",
                summary=(
                    f"文件 {filename} 的车牌识别耗时超过 "
                    f"{settings.plate_inference_timeout_seconds:.1f} 秒。"
                ),
                response_status="Timeout",
            )
            raise InferenceTimeoutError("车牌识别超时。") from exc
        except PlateInferenceError as exc:
            elapsed_ms = int((perf_counter() - started_at) * 1000)
            await self._record_failure(
                user_id=user_id,
                filename=filename,
                elapsed_ms=elapsed_ms,
                trace_id=trace_id,
                title="车牌识别依赖异常",
                summary=f"文件 {filename} 的识别流程执行失败：{exc}",
                response_status="Failed",
            )
            raise
        except Exception as exc:
            elapsed_ms = int((perf_counter() - started_at) * 1000)
            await self._record_failure(
                user_id=user_id,
                filename=filename,
                elapsed_ms=elapsed_ms,
                trace_id=trace_id,
                title="车牌识别服务异常",
                summary=f"文件 {filename} 的识别流程出现未预期错误：{exc}",
                response_status="Failed",
            )
            raise PlateInferenceError("车牌识别服务异常。") from exc

        if detections and save_history and user_id is not None:
            self._save_history(detections, image_path=image_path, user_id=user_id)

        elapsed_ms = int((perf_counter() - started_at) * 1000)
        operation_status = "Success" if detections else "NoDetection"
        self._record_operation(user_id, "plate_recognition", operation_status)
        summary = self._build_behavior_summary(filename, detections, elapsed_ms)
        self._record_behavior(
            title="车牌识别完成" if detections else "车牌识别未命中结果",
            summary=summary,
        )
        await self._record_monitor_success(
            user_id=user_id,
            filename=filename,
            elapsed_ms=elapsed_ms,
            trace_id=trace_id,
            detections=detections,
        )
        return PlateRecognitionResponse(frame_id=filename, detections=detections)

    def recognize_image_bytes(
        self,
        image_bytes: bytes,
        filename: str = "unknown.jpg",
        *,
        save_history: bool = False,
        user_id: int | None = None,
    ) -> PlateRecognitionResponse:
        if not image_bytes:
            return PlateRecognitionResponse(frame_id=filename, detections=[])

        image_path = self._persist_upload(image_bytes, filename) if settings.plate_save_uploads else None
        detections = self._recognize_detections(image_bytes)

        if detections and save_history and user_id is not None:
            self._save_history(detections, image_path=image_path, user_id=user_id)
        elif detections and user_id is None:
            self._save_anonymous_history(detections)

        return PlateRecognitionResponse(frame_id=filename, detections=detections)

    def recognize_video_bytes(
        self,
        video_bytes: bytes,
        filename: str = "unknown.mp4",
        *,
        save_history: bool = False,
        user_id: int | None = None,
    ) -> PlateVideoRecognitionResponse:
        if not video_bytes:
            raise ValueError("Uploaded video is empty.")

        source_path, output_path = self._prepare_video_paths(filename)
        source_path.write_bytes(video_bytes)

        try:
            result = self._process_video_file(
                source_path,
                output_path,
                filename,
                save_history=save_history,
                user_id=user_id,
            )
        finally:
            if not settings.plate_save_uploads:
                try:
                    source_path.unlink(missing_ok=True)
                except Exception:
                    logger.warning("Failed to remove temporary uploaded video: %s", source_path)

        return result

    def stream_rtsp(self, rtsp_url: str):
        for frame, detections in self.iter_annotated_stream(rtsp_url):
            yield self._build_stream_payload(frame, detections)

    def iter_annotated_stream(self, rtsp_url: str, stop_event=None):
        capture, pending_frame = self._open_rtsp_capture(rtsp_url)
        state = PlateProcessingState()
        last_sent_at = 0.0

        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break

                if pending_frame is not None:
                    source_frame = pending_frame
                    pending_frame = None
                    ok = True
                else:
                    ok, source_frame = capture.read()

                if not ok or source_frame is None:
                    break

                annotated, scaled_detections, _ = self._process_frame(
                    source_frame,
                    state,
                    save_history=True,
                    force_detect_when_empty=True,
                )

                if settings.plate_stream_max_fps > 0:
                    min_interval = 1.0 / settings.plate_stream_max_fps
                    elapsed = time.monotonic() - last_sent_at
                    if elapsed < min_interval:
                        time.sleep(min_interval - elapsed)
                last_sent_at = time.monotonic()

                yield annotated, scaled_detections
        finally:
            capture.release()

    def list_history(self, user_id: int | None = None) -> list[PlateRecordSummary]:
        try:
            with SessionLocal() as session:
                statement = select(PlateRecord)
                if user_id is not None:
                    statement = statement.where(PlateRecord.user_id == user_id)
                statement = statement.order_by(PlateRecord.created_at.desc()).limit(settings.plate_history_limit)
                records = session.scalars(statement).all()
        except Exception as exc:
            logger.warning("Failed to load plate recognition history: %s", exc)
            records = []

        if not records:
            if user_id is None and self._anonymous_history:
                return self._anonymous_history[: settings.plate_history_limit]
            return [
                PlateRecordSummary(
                    id=1,
                    plate_number="测试123",
                    plate_color="蓝牌",
                    created_at=datetime.utcnow(),
                )
            ]

        return [
            PlateRecordSummary(
                id=record.id,
                plate_number=record.plate_number,
                plate_color=record.plate_color,
                created_at=record.created_at,
            )
            for record in records
        ]

    def _save_anonymous_history(self, detections: list[PlateDetection]) -> None:
        created_at = datetime.utcnow()
        summaries: list[PlateRecordSummary] = []
        for detection in detections:
            if not detection.plate_number:
                continue
            self._anonymous_history_id += 1
            summaries.append(
                PlateRecordSummary(
                    id=self._anonymous_history_id,
                    plate_number=detection.plate_number,
                    plate_color=detection.plate_color,
                    created_at=created_at,
                )
            )

        if not summaries:
            return

        self._anonymous_history = (summaries + self._anonymous_history)[: settings.plate_history_limit]

    def get_upload_root(self) -> Path:
        self._upload_root.mkdir(parents=True, exist_ok=True)
        return self._upload_root

    def public_media_url_for(self, file_path: Path) -> str:
        relative_path = file_path.resolve().relative_to(self.get_upload_root())
        return f"/media/{relative_path.as_posix()}"

    def _process_video_file(
        self,
        source_path: Path,
        output_path: Path,
        filename: str,
        *,
        save_history: bool,
        user_id: int | None,
    ) -> PlateVideoRecognitionResponse:
        cv2 = self._require_cv2()
        capture, pending_frame = self._open_local_video_capture(source_path)
        writer = None
        best_detections: dict[str, VideoDetectionStats] = {}
        processed_frame_count = 0
        temp_output_path = output_path.with_name(f"{output_path.stem}.raw.mp4")
        detection_pass_count = 0
        detection_hit_count = 0
        started_at = time.monotonic()
        unread_samples: list[UnreadSample] = []
        unread_debug_dir = self._prepare_video_debug_dir(output_path)
        best_unread_candidates: dict[str, BestUnreadCandidate] = {}

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1:
            fps = 25.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        recognition_interval = max(
            settings.plate_video_process_every_n_frames,
            max(int(round(fps * 0.6)), 12),
        )
        heavy_scan_interval = max(recognition_interval * 8, 48)
        progress_log_interval = max(recognition_interval * 2, 30)
        summary_merge_interval = 3
        state = PlateProcessingState(
            recognition_interval=recognition_interval,
            heavy_scan_interval=heavy_scan_interval,
        )

        logger.info(
            "Starting plate video processing: %s | fps=%.2f total_frames=%s recognition_interval=%d max_side=%d",
            filename,
            fps,
            total_frames if total_frames > 0 else "unknown",
            recognition_interval,
            settings.plate_video_recognition_max_side,
        )

        try:
            while True:
                if pending_frame is not None:
                    source_frame = pending_frame
                    pending_frame = None
                    ok = True
                else:
                    ok, source_frame = capture.read()

                if not ok or source_frame is None:
                    break

                processed_frame_count += 1
                (
                    annotated,
                    scaled_detections,
                    fresh_detections,
                    active_detections,
                ) = self._process_video_frame_with_tracking(
                    source_frame,
                    state=state,
                )
                self._collect_unread_video_samples(
                    frame=source_frame,
                    state=state,
                    frame_index=processed_frame_count,
                    target_dir=unread_debug_dir,
                    samples=unread_samples,
                )
                self._update_best_unread_candidates(
                    source_frame=source_frame,
                    state=state,
                    frame_index=processed_frame_count,
                    candidates=best_unread_candidates,
                )
                if fresh_detections:
                    detection_pass_count += 1
                    detection_hit_count += 1
                    self._merge_best_detections(best_detections, fresh_detections, recognized=True)
                if processed_frame_count == 1 or processed_frame_count % summary_merge_interval == 0:
                    self._merge_best_detections(best_detections, active_detections, recognized=False)
                if processed_frame_count == 1 or processed_frame_count % progress_log_interval == 0:
                    elapsed = max(time.monotonic() - started_at, 0.001)
                    logger.info(
                        "Video processing progress: %s | frame=%d/%s detection_passes=%d fresh_hits=%d elapsed=%.1fs",
                        filename,
                        processed_frame_count,
                        total_frames if total_frames > 0 else "?",
                        detection_pass_count,
                        detection_hit_count,
                        elapsed,
                    )

                if writer is None:
                    height, width = annotated.shape[:2]
                    temp_output_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(temp_output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise InferenceConfigurationError("Failed to create annotated video output file.")

                writer.write(annotated)
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        rescue_detections = self._recover_unread_video_detections(best_unread_candidates)
        if rescue_detections:
            self._merge_best_detections(best_detections, rescue_detections, recognized=True)
        detections = self._finalize_video_detections(best_detections)
        if detections:
            if save_history and user_id is not None:
                self._save_history(detections, image_path=None, user_id=user_id)
            else:
                self._save_anonymous_history(detections)

        logger.info(
            "Video detection stage finished: %s | processed_frames=%d detection_passes=%d unique_plates=%d elapsed=%.1fs",
            filename,
            processed_frame_count,
            detection_pass_count,
            len(detections),
            time.monotonic() - started_at,
        )

        self._transcode_video_for_web(temp_output_path, output_path)
        try:
            temp_output_path.unlink(missing_ok=True)
        except Exception:
            logger.warning("Failed to remove temporary processed video: %s", temp_output_path)

        duration_seconds = round(processed_frame_count / fps, 2) if fps > 0 else None
        return PlateVideoRecognitionResponse(
            source_filename=filename,
            processed_video_url=self.public_media_url_for(output_path),
            detections=detections,
            unread_samples=[self.public_media_url_for(item.file_path) for item in unread_samples],
            processed_frame_count=processed_frame_count,
            duration_seconds=duration_seconds,
        )

    def _process_video_frame_with_tracking(
        self,
        source_frame,
        *,
        state: PlateProcessingState,
    ) -> tuple[object, list[PlateDetection], list[PlateDetection], list[PlateDetection]]:
        state.frame_index += 1
        display_frame = self._resize_stream_frame(source_frame)
        working_frame = self._resize_frame_to_limit(source_frame, settings.plate_video_recognition_max_side)
        state.working_width = working_frame.shape[1]
        state.working_height = working_frame.shape[0]
        should_update_tracks = (
            state.frame_index == 1
            or not state.tracks
            or any(track.misses > 0 for track in state.tracks)
            or state.frame_index - state.last_tracking_frame >= 2
        )
        if should_update_tracks:
            state.tracks = self._update_tracks(working_frame, state.tracks, state.frame_index)
            state.last_tracking_frame = state.frame_index

        fresh_working_detections: list[PlateDetection] = []
        if self._should_rerecognize_video(state):
            use_heavy_scan = settings.plate_video_detector_full_frame_fallback and self._should_use_heavy_scan(state)
            raw_working_detections = self._recognize_detections(
                working_frame,
                aggressive=False,
                heavy_scan=use_heavy_scan,
                allow_full_frame_fallback=settings.plate_video_detector_full_frame_fallback,
                fast_mode=True,
                preserve_unread=True,
            )
            fresh_working_detections = [item for item in raw_working_detections if item.plate_number]
            state.tracks = self._merge_recognized_tracks(
                working_frame,
                state.tracks,
                raw_working_detections,
                state.frame_index,
            )
            state.last_recognition_frame = state.frame_index
            state.last_probe_frame = state.frame_index
        elif self._should_probe_new_video_tracks(state):
            probe_hits = self._find_untracked_detector_hits(
                working_frame,
                state.tracks,
                max_hits=2,
            )
            if probe_hits:
                raw_probe_detections = self._recognize_detector_hits(
                    working_frame,
                    probe_hits,
                    fast_mode=True,
                    preserve_unread=True,
                )
                fresh_working_detections = [item for item in raw_probe_detections if item.plate_number]
                state.tracks = self._merge_recognized_tracks(
                    working_frame,
                    state.tracks,
                    raw_probe_detections,
                    state.frame_index,
                )
                if fresh_working_detections:
                    state.last_recognition_frame = state.frame_index
            state.last_probe_frame = state.frame_index

        active_working_detections = self._tracks_to_detections(state.tracks)
        fresh_detections = self._scale_detections(
            fresh_working_detections,
            working_frame.shape[1],
            working_frame.shape[0],
            source_frame.shape[1],
            source_frame.shape[0],
        )
        active_detections = self._scale_detections(
            active_working_detections,
            working_frame.shape[1],
            working_frame.shape[0],
            source_frame.shape[1],
            source_frame.shape[0],
        )
        scaled_detections = self._scale_detections(
            active_detections,
            source_frame.shape[1],
            source_frame.shape[0],
            display_frame.shape[1],
            display_frame.shape[0],
        )
        annotated = self._annotate_frame(display_frame, scaled_detections)
        return annotated, scaled_detections, fresh_detections, active_detections

    def _should_rerecognize_video(self, state: PlateProcessingState) -> bool:
        if state.frame_index == 1:
            return True

        interval = max(state.recognition_interval, 1)
        since_last = state.frame_index - state.last_recognition_frame
        if not state.tracks:
            return since_last >= max(interval // 2, 8)
        if any(track.misses > 0 for track in state.tracks):
            return since_last >= max(interval // 2, 8)
        if any(not track.plate_number for track in state.tracks):
            return since_last >= max(interval, 12)
        return since_last >= max(interval, 12)

    def _should_probe_new_video_tracks(self, state: PlateProcessingState) -> bool:
        if state.frame_index <= 1:
            return False
        probe_interval = max(state.recognition_interval, 12)
        return state.frame_index - state.last_probe_frame >= probe_interval
    def _prepare_video_paths(self, filename: str) -> tuple[Path, Path]:
        suffix = Path(filename).suffix.lower() or ".mp4"
        source_dir = self.get_upload_root() / "videos" / "source"
        output_dir = self.get_upload_root() / "videos" / "processed"
        source_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        token = uuid4().hex
        source_path = source_dir / f"{token}{suffix}"
        output_path = output_dir / f"{token}.mp4"
        return source_path, output_path

    def _prepare_video_debug_dir(self, output_path: Path) -> Path:
        debug_dir = self.get_upload_root() / "videos" / "debug" / output_path.stem
        debug_dir.mkdir(parents=True, exist_ok=True)
        return debug_dir

    def _transcode_video_for_web(self, source_path: Path, output_path: Path) -> None:
        logger.info("Starting processed video transcode: %s -> %s", source_path.name, output_path.name)
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
            raise InferenceConfigurationError(
                f"ffmpeg was not found. Cannot transcode the processed video. Current value: {settings.plate_push_ffmpeg_bin!r}."
            ) from exc

        if result.returncode != 0:
            stderr_text = result.stderr.decode("utf-8", errors="ignore").strip() if result.stderr else ""
            raise InferenceConfigurationError(
                f"Processed video transcode failed. {stderr_text or 'Please verify ffmpeg can run normally.'}"
            )

    def _collect_unread_video_samples(
        self,
        *,
        frame,
        state: PlateProcessingState,
        frame_index: int,
        target_dir: Path,
        samples: list[UnreadSample],
    ) -> None:
        cv2 = self._require_cv2()
        max_samples = 8
        min_frame_gap = max(state.recognition_interval, 12)

        if len(samples) >= max_samples:
            return

        for track in state.tracks:
            if len(samples) >= max_samples:
                break
            if track.plate_number:
                continue
            if track.misses > 0:
                continue
            if frame_index - track.last_seen_frame > 1:
                continue
            if track.last_seen_frame < max(3, frame_index - 2):
                continue
            if track.unread_snapshots_saved >= 2:
                continue
            if track.last_unread_snapshot_frame and frame_index - track.last_unread_snapshot_frame < min_frame_gap:
                continue

            crop = self._crop_detection(frame, track.bbox)
            if crop is None or getattr(crop, "size", 0) == 0:
                continue

            crop_height, crop_width = crop.shape[:2]
            if crop_width < 36 or crop_height < 14:
                continue

            sample_path = target_dir / f"{len(samples) + 1:02d}_frame{frame_index:04d}_{track.track_id[:8]}.jpg"
            if not cv2.imwrite(str(sample_path), crop):
                continue

            track.unread_snapshots_saved += 1
            track.last_unread_snapshot_frame = frame_index
            samples.append(
                UnreadSample(
                    track_id=track.track_id,
                    frame_index=frame_index,
                    file_path=sample_path,
                )
            )

    def _update_best_unread_candidates(
        self,
        *,
        source_frame,
        state: PlateProcessingState,
        frame_index: int,
        candidates: dict[str, BestUnreadCandidate],
    ) -> None:
        if state.working_width <= 0 or state.working_height <= 0:
            return

        frame_height, frame_width = source_frame.shape[:2]
        for track in state.tracks:
            if track.plate_number:
                continue
            if track.misses > 0:
                continue

            source_bbox = self._scale_bbox(
                track.bbox,
                state.working_width,
                state.working_height,
                frame_width,
                frame_height,
            )
            if not self._is_reasonable_crop_bbox(source_bbox):
                continue

            crop = self._crop_detection(source_frame, source_bbox)
            if crop is None or getattr(crop, "size", 0) == 0:
                continue

            quality_score = self._score_unread_crop_quality(crop)
            current = candidates.get(track.track_id)
            if current is None or quality_score > current.quality_score:
                candidates[track.track_id] = BestUnreadCandidate(
                    track_id=track.track_id,
                    crop=crop.copy(),
                    bbox=list(source_bbox),
                    frame_index=frame_index,
                    quality_score=quality_score,
                )

    def _recover_unread_video_detections(
        self,
        candidates: dict[str, BestUnreadCandidate],
    ) -> list[PlateDetection]:
        recovered: list[PlateDetection] = []
        for candidate in candidates.values():
            detection = self._recognize_best_unread_candidate(candidate)
            if detection is not None:
                recovered.append(detection)
        return self._deduplicate_plate_detections(recovered)

    def _recognize_best_unread_candidate(self, candidate: BestUnreadCandidate) -> PlateDetection | None:
        crop = candidate.crop
        if crop is None or getattr(crop, "size", 0) == 0:
            return None

        crop_candidates: list[PlateDetection] = []
        for variant in self._build_best_effort_ocr_crop_variants(crop):
            crop_result = self._recognize_crop_with_optional_lprnet(
                variant,
                {"confidence": 0.82},
                candidate.bbox,
            )
            if crop_result is not None:
                crop_candidates.append(crop_result)

            ocr_candidates = self.recognizer.recognize_all(
                variant,
                max_side_override=0,
                aggressive=True,
                heavy_scan=False,
                confidence_threshold=min(settings.plate_confidence_threshold, 0.24),
            )
            best_candidate = self._pick_best_crop_detection(ocr_candidates)
            if best_candidate is None:
                continue
            crop_candidates.append(
                PlateDetection(
                    plate_number=best_candidate.plate_number,
                    plate_color=self._normalize_plate_color_for_plate(
                        best_candidate.plate_color,
                        best_candidate.plate_number,
                    ),
                    confidence=min(1.0, best_candidate.confidence * 0.84 + 0.16 * 0.82),
                    bbox=list(candidate.bbox),
                )
            )

        best_detection = self._pick_best_plate_candidate(crop_candidates)
        if best_detection is None:
            return None
        if best_detection.confidence < 0.34:
            return None
        return best_detection

    def _build_best_effort_ocr_crop_variants(self, crop) -> list[object]:
        variants = list(self._build_ocr_crop_variants(crop))
        cv2 = self._require_cv2()

        try:
            crop_height, crop_width = crop.shape[:2]
            upscale_factor = 4 if crop_height <= 28 or crop_width <= 120 else 3
            boosted = cv2.resize(
                crop,
                (max(crop_width * upscale_factor, 94), max(crop_height * upscale_factor, 24)),
                interpolation=cv2.INTER_CUBIC,
            )
            gray = cv2.cvtColor(boosted, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
            boosted_gray = clahe.apply(gray)
            boosted_gray = cv2.bilateralFilter(boosted_gray, 7, 40, 40)
            boosted_gray = cv2.addWeighted(
                boosted_gray,
                1.34,
                cv2.GaussianBlur(boosted_gray, (0, 0), 1.4),
                -0.34,
                0.0,
            )
            variants.append(cv2.cvtColor(boosted_gray, cv2.COLOR_GRAY2BGR))
        except Exception:
            logger.debug("Failed to build best-effort OCR crop variants.", exc_info=True)

        return variants

    def _score_unread_crop_quality(self, crop) -> float:
        cv2 = self._require_cv2()
        if crop is None or getattr(crop, "size", 0) == 0:
            return 0.0

        height, width = crop.shape[:2]
        if height <= 0 or width <= 0:
            return 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        area_score = float(width * height) / 1000.0
        aspect_ratio = width / max(height, 1)
        aspect_bonus = 0.6 if 2.2 <= aspect_ratio <= 5.8 else 0.0
        return laplacian_var * 0.08 + area_score + aspect_bonus

    def _process_frame(
        self,
        source_frame,
        state: PlateProcessingState,
        *,
        save_history: bool,
        force_detect_when_empty: bool,
        aggressive_detection: bool = False,
        heavy_scan_detection: bool = False,
    ) -> tuple[object, list[PlateDetection], list[PlateDetection]]:
        state.frame_index += 1
        display_frame = self._resize_stream_frame(source_frame)

        state.tracks = self._update_tracks(source_frame, state.tracks, state.frame_index)
        fresh_detections: list[PlateDetection] = []

        if self._should_rerecognize(
            state.tracks,
            state.frame_index,
            state.last_recognition_frame,
            recognition_interval=state.recognition_interval,
            force_when_no_tracks=force_detect_when_empty,
        ):
            should_use_heavy_scan = heavy_scan_detection or (
                aggressive_detection and self._should_use_heavy_scan(state)
            )
            fresh_detections = self._recognize_detections(
                source_frame,
                aggressive=aggressive_detection,
                heavy_scan=should_use_heavy_scan,
            )
            state.tracks = self._merge_recognized_tracks(source_frame, state.tracks, fresh_detections, state.frame_index)
            state.last_recognition_frame = state.frame_index

            if save_history and fresh_detections:
                now = time.monotonic()
                if now - state.last_history_saved_at >= settings.plate_stream_history_interval_seconds:
                    self._save_history(fresh_detections, None)
                    state.last_history_saved_at = now

        active_detections = self._tracks_to_detections(state.tracks)
        scaled_detections = self._scale_detections(
            active_detections,
            source_frame.shape[1],
            source_frame.shape[0],
            display_frame.shape[1],
            display_frame.shape[0],
        )
        annotated = self._annotate_frame(display_frame, scaled_detections)
        return annotated, scaled_detections, fresh_detections

    def _open_rtsp_capture(self, rtsp_url: str):
        cv2 = self._require_cv2()
        attempts: list[tuple[str, int | None, str | None]] = [
            ("FFMPEG", cv2.CAP_FFMPEG, "rtsp_transport;tcp"),
            ("DEFAULT", None, None),
        ]
        errors: list[str] = []

        for backend_name, backend, ffmpeg_options in attempts:
            capture = self._create_video_capture(rtsp_url, backend, ffmpeg_options)
            if not capture.isOpened():
                errors.append(f"{backend_name} backend open failed")
                capture.release()
                continue

            ok, frame = capture.read()
            if ok and frame is not None:
                logger.info("RTSP stream opened via %s backend: %s", backend_name, rtsp_url)
                return capture, frame

            errors.append(f"{backend_name} backend first-frame read failed")
            capture.release()

        detail = "; ".join(errors) if errors else "no extra diagnostics"
        raise InferenceConfigurationError(
            "Failed to open the RTSP stream. Verify the current computer is connected to the required network, the RTSP URL is reachable, and local OpenCV supports RTSP/FFMPEG. "
            f"Diagnostics: {detail}"
        )

    def _open_local_video_capture(self, video_path: Path):
        cv2 = self._require_cv2()
        capture = cv2.VideoCapture(str(video_path))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            capture.release()
            raise InferenceConfigurationError("Failed to open the uploaded video file.")

        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise InferenceConfigurationError("The uploaded video has no readable frames.")

        return capture, frame

    def _create_video_capture(self, rtsp_url: str, backend: int | None, ffmpeg_options: str | None):
        cv2 = self._require_cv2()
        if ffmpeg_options:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_options
        else:
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)

        capture = cv2.VideoCapture(rtsp_url, backend) if backend is not None else cv2.VideoCapture(rtsp_url)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _recognize_detections(
        self,
        image_source,
        aggressive: bool = False,
        heavy_scan: bool = True,
        allow_full_frame_fallback: bool = True,
        fast_mode: bool = False,
        preserve_unread: bool = False,
    ) -> list[PlateDetection]:
        merged_detections: list[PlateDetection] = []
        if self._should_use_detector():
            try:
                detector_detections = self._recognize_detections_via_detector(
                    image_source,
                    aggressive=aggressive,
                    fast_mode=fast_mode,
                    preserve_unread=preserve_unread,
                )
                if detector_detections:
                    merged_detections.extend(detector_detections)
            except (InferenceConfigurationError, InferenceDependencyError):
                if not settings.plate_detector_fallback_to_full_frame:
                    raise
                logger.warning("Plate detector unavailable, falling back to full-frame HyperLPR OCR.", exc_info=True)

        # With a dedicated plate detector loaded, detector hits are usually the best tradeoff for video latency.
        # Full-frame OCR remains a fallback path when detector results are empty.
        if merged_detections:
            return self._deduplicate_plate_detections(merged_detections)
        if not allow_full_frame_fallback:
            return []

        max_side_override = settings.plate_stream_recognition_max_side if not isinstance(image_source, (bytes, bytearray)) else None
        confidence_threshold = 0.26 if aggressive else None
        full_frame_detections = [
            PlateDetection(
                plate_number=item.plate_number,
                plate_color=self._normalize_plate_color_for_plate(item.plate_color, item.plate_number),
                confidence=item.confidence,
                bbox=item.bbox,
            )
            for item in self.recognizer.recognize_all(
                image_source,
                max_side_override=max_side_override,
                aggressive=aggressive,
                heavy_scan=heavy_scan,
                confidence_threshold=confidence_threshold,
            )
        ]
        return full_frame_detections

    def _should_use_detector(self) -> bool:
        return self.detector.is_available()

    def _recognize_detections_via_detector(
        self,
        image_source,
        aggressive: bool = False,
        fast_mode: bool = False,
        preserve_unread: bool = False,
    ) -> list[PlateDetection]:
        image = self._decode_image_source(image_source)
        detector_hits = self.detector.detect(image)
        if not detector_hits:
            return []

        max_candidates = settings.plate_detector_max_candidates
        if aggressive:
            max_candidates = min(max(max_candidates, 8), 10)
        if fast_mode:
            max_candidates = min(max_candidates, 10)

        recognized: list[PlateDetection] = []
        for hit in detector_hits[:max_candidates]:
            recognized.extend(
                self._recognize_detector_hit(
                    image,
                    hit,
                    aggressive=aggressive,
                    fast_mode=fast_mode,
                    preserve_unread=preserve_unread,
                )
            )

        return self._deduplicate_plate_detections(recognized)

    def _recognize_detector_hits(
        self,
        image,
        hits: list[dict],
        *,
        fast_mode: bool,
        preserve_unread: bool = False,
    ) -> list[PlateDetection]:
        recognized: list[PlateDetection] = []
        for hit in hits:
            recognized.extend(
                self._recognize_detector_hit(
                    image,
                    hit,
                    aggressive=False,
                    fast_mode=fast_mode,
                    preserve_unread=preserve_unread,
                )
            )
        return self._deduplicate_plate_detections(recognized)

    def _find_untracked_detector_hits(
        self,
        image,
        tracks: list[PlateTrack],
        *,
        max_hits: int,
    ) -> list[dict]:
        if not self._should_use_detector():
            return []

        try:
            detector_hits = self.detector.detect(image)
        except (InferenceConfigurationError, InferenceDependencyError):
            logger.warning("Plate detector unavailable during video probe scan.", exc_info=True)
            return []

        unmatched: list[dict] = []
        for hit in detector_hits:
            bbox = list(hit.get("bbox", []))
            if not self._is_reasonable_crop_bbox(bbox):
                continue

            matched = False
            for track in tracks:
                if self._compute_iou(bbox, track.bbox) >= 0.18:
                    matched = True
                    break

            if not matched:
                unmatched.append(hit)
            if len(unmatched) >= max_hits:
                break

        return unmatched

    def _recognize_detector_hit(
        self,
        image,
        hit: dict,
        aggressive: bool = False,
        fast_mode: bool = False,
        preserve_unread: bool = False,
    ) -> list[PlateDetection]:
        recognized: list[PlateDetection] = []
        candidate_bboxes = self._resolve_detector_crop_bboxes(image, hit)
        confidence_threshold = min(
            settings.plate_confidence_threshold,
            0.28 if aggressive else settings.plate_confidence_threshold,
        )
        if fast_mode:
            candidate_bboxes = candidate_bboxes[:1]

        for crop_bbox in candidate_bboxes:
            if not self._is_reasonable_crop_bbox(crop_bbox):
                continue
            crop = self._crop_detection(image, crop_bbox)
            if crop is None:
                continue

            crop_candidates: list[PlateDetection] = []
            crop_variants = self._build_ocr_crop_variants(crop)
            use_extra_lprnet_variant = self._should_try_extra_fast_lprnet_variant(crop_bbox)
            if fast_mode:
                lprnet_variants = crop_variants[:2] if use_extra_lprnet_variant else crop_variants[:1]
            else:
                lprnet_variants = crop_variants
            hyperlpr_variants = crop_variants[:1] if fast_mode else crop_variants[:2]

            for crop_variant in lprnet_variants:
                crop_result = self._recognize_crop_with_optional_lprnet(crop_variant, hit, crop_bbox)
                if crop_result is not None:
                    crop_candidates.append(crop_result)

            should_run_hyperlpr = (
                not crop_candidates
                or crop_candidates[0].confidence < 0.88
                or crop_candidates[0].plate_color == "??"
            )
            if fast_mode and crop_candidates:
                should_run_hyperlpr = False
            if should_run_hyperlpr:
                for crop_variant in hyperlpr_variants:
                    ocr_candidates = self.recognizer.recognize_all(
                        crop_variant,
                        max_side_override=0,
                        aggressive=aggressive,
                        heavy_scan=False,
                        confidence_threshold=confidence_threshold,
                    )
                    best_candidate = self._pick_best_crop_detection(ocr_candidates)
                    if best_candidate is None:
                        continue
                    crop_candidates.append(
                        PlateDetection(
                            plate_number=best_candidate.plate_number,
                            plate_color=self._normalize_plate_color_for_plate(
                                best_candidate.plate_color,
                                best_candidate.plate_number,
                            ),
                            confidence=min(1.0, best_candidate.confidence * 0.82 + float(hit["confidence"]) * 0.18),
                            bbox=list(crop_bbox),
                        )
                    )
                    break

            best_detection = self._pick_best_plate_candidate(crop_candidates)
            if best_detection is not None:
                recognized.append(best_detection)
            elif preserve_unread:
                recognized.append(
                    PlateDetection(
                        plate_number="",
                        plate_color="??",
                        confidence=float(hit["confidence"]),
                        bbox=list(crop_bbox),
                    )
                )

        return self._deduplicate_plate_detections(recognized)

    def _recognize_crop_with_optional_lprnet(self, crop, hit: dict, crop_bbox: list[int]) -> PlateDetection | None:
        if not self.crop_recognizer.is_available():
            return None

        try:
            result = self.crop_recognizer.recognize(crop)
        except (InferenceConfigurationError, InferenceDependencyError):
            logger.warning("OpenTrafficFlow LPRNet unavailable, falling back to HyperLPR crop OCR.")
            return None

        if result is None or not result.plate_number:
            return None

        return PlateDetection(
            plate_number=result.plate_number,
            plate_color=self._normalize_plate_color_for_plate(result.plate_color, result.plate_number),
            confidence=min(1.0, result.confidence * 0.82 + float(hit["confidence"]) * 0.18),
            bbox=list(crop_bbox),
        )
    def _is_reasonable_crop_bbox(self, bbox: list[int]) -> bool:
        if len(bbox) != 4:
            return False
        _, _, width, height = bbox
        if width < 20 or height < 10:
            return False
        if width * height < 300:
            return False
        return True

    def _decode_image_source(self, image_source):
        cv2 = self._require_cv2()
        try:
            import numpy as np
        except ImportError as exc:
            raise InferenceDependencyError("Missing numpy. Install image-processing dependencies in the backend environment first.") from exc

        if isinstance(image_source, np.ndarray):
            return image_source
        if isinstance(image_source, (bytes, bytearray)):
            encoded = np.frombuffer(image_source, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode the uploaded image. Please verify the file is a valid image format.")
            return image
        return image_source

    def _crop_detection(self, image, bbox: list[int]):
        x, y, width, height = bbox
        image_height, image_width = image.shape[:2]
        pad_x = int(round(width * settings.plate_detector_crop_padding_x))
        pad_y = int(round(height * settings.plate_detector_crop_padding_y))

        left = max(x - pad_x, 0)
        top = max(y - pad_y, 0)
        right = min(x + width + pad_x, image_width)
        bottom = min(y + height + pad_y, image_height)
        if right <= left or bottom <= top:
            return None
        crop = image[top:bottom, left:right]
        return crop if crop.size > 0 else None

    def _resolve_detector_crop_bboxes(self, image, hit: dict) -> list[list[int]]:
        kind = str(hit.get("kind", "plate"))
        bbox = list(hit["bbox"])
        if kind != "vehicle":
            return self._plate_bbox_to_crop_bboxes(image, bbox)
        return self._vehicle_bbox_to_plate_bboxes(image, bbox)

    def _plate_bbox_to_crop_bboxes(self, image, plate_bbox: list[int]) -> list[list[int]]:
        x, y, width, height = plate_bbox
        image_height, image_width = image.shape[:2]

        variants = [
            [x, y, width, height],
            self._clamp_bbox(
                image_width,
                image_height,
                x - int(round(width * 0.08)),
                y - int(round(height * 0.18)),
                int(round(width * 1.16)),
                int(round(height * 1.36)),
                min_left=0,
                min_top=0,
                max_right=image_width,
                max_bottom=image_height,
            ),
        ]

        unique: list[list[int]] = []
        for candidate in variants:
            if candidate not in unique and self._is_reasonable_crop_bbox(candidate):
                unique.append(candidate)
        return unique

    def _build_ocr_crop_variants(self, crop) -> list[object]:
        cv2 = self._require_cv2()
        variants = [crop]

        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            enhanced = cv2.GaussianBlur(enhanced, (0, 0), 0.8)
            enhanced = cv2.addWeighted(gray, 1.15, enhanced, -0.15, 0.0)
            variants.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))

            crop_height, crop_width = crop.shape[:2]
            if crop_width <= 140 or crop_height <= 40:
                upscaled = cv2.resize(
                    crop,
                    (max(crop_width * 2, 94), max(crop_height * 2, 24)),
                    interpolation=cv2.INTER_CUBIC,
                )
                upscaled_gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
                upscaled_gray = clahe.apply(upscaled_gray)
                upscaled_gray = cv2.bilateralFilter(upscaled_gray, 5, 30, 30)
                upscaled_gray = cv2.addWeighted(upscaled_gray, 1.22, cv2.GaussianBlur(upscaled_gray, (0, 0), 1.1), -0.22, 0.0)
                variants.append(cv2.cvtColor(upscaled_gray, cv2.COLOR_GRAY2BGR))
        except Exception:
            logger.debug("Failed to build enhanced OCR crop variant.", exc_info=True)

        return variants

    def _should_try_extra_fast_lprnet_variant(self, crop_bbox: list[int]) -> bool:
        if len(crop_bbox) != 4:
            return False
        _, _, width, height = crop_bbox
        return width <= 140 or height <= 40 or width * height <= 5200

    def _vehicle_bbox_to_plate_bboxes(self, image, vehicle_bbox: list[int]) -> list[list[int]]:
        x, y, width, height = vehicle_bbox
        image_height, image_width = image.shape[:2]

        pad_x = int(round(width * settings.plate_vehicle_crop_padding_x))
        pad_y = int(round(height * settings.plate_vehicle_crop_padding_y))
        vehicle_left = max(x - pad_x, 0)
        vehicle_top = max(y - pad_y, 0)
        vehicle_right = min(x + width + pad_x, image_width)
        vehicle_bottom = min(y + height + pad_y, image_height)

        padded_width = max(vehicle_right - vehicle_left, 1)
        padded_height = max(vehicle_bottom - vehicle_top, 1)

        base_width = max(int(round(padded_width * settings.plate_vehicle_plate_width_ratio)), 20)
        base_height = max(int(round(padded_height * settings.plate_vehicle_plate_height_ratio)), 12)
        top_anchor = vehicle_top + int(round(padded_height * settings.plate_vehicle_plate_top_ratio))

        width_ratios = [1.0, 0.78, 0.92]
        x_offsets = [0.5, 0.36, 0.64]
        y_offsets = [0.0, -0.08, 0.08]
        candidates: list[list[int]] = []

        for width_ratio, x_center_ratio, y_offset_ratio in zip(width_ratios, x_offsets, y_offsets):
            plate_width = max(int(round(base_width * width_ratio)), 20)
            plate_height = base_height
            plate_left = vehicle_left + int(round((padded_width - plate_width) * x_center_ratio))
            plate_top = top_anchor + int(round(padded_height * y_offset_ratio))
            candidate = self._clamp_bbox(
                image_width,
                image_height,
                plate_left,
                plate_top,
                plate_width,
                plate_height,
                min_left=vehicle_left,
                min_top=vehicle_top,
                max_right=vehicle_right,
                max_bottom=vehicle_bottom,
            )
            candidates.append(candidate)

        # Add a wider low-position fallback for rear/front cars with lower-mounted plates.
        candidates.append(
            self._clamp_bbox(
                image_width,
                image_height,
                vehicle_left + int(round(padded_width * 0.16)),
                vehicle_top + int(round(padded_height * 0.56)),
                int(round(padded_width * 0.68)),
                max(int(round(padded_height * 0.22)), 12),
                min_left=vehicle_left,
                min_top=vehicle_top,
                max_right=vehicle_right,
                max_bottom=vehicle_bottom,
            )
        )

        unique: list[list[int]] = []
        for candidate in candidates:
            if candidate not in unique:
                unique.append(candidate)
        return unique

    def _clamp_bbox(
        self,
        image_width: int,
        image_height: int,
        left: int,
        top: int,
        width: int,
        height: int,
        *,
        min_left: int,
        min_top: int,
        max_right: int,
        max_bottom: int,
    ) -> list[int]:
        clamped_left = max(min_left, min(left, max_right - 1))
        clamped_top = max(min_top, min(top, max_bottom - 1))
        clamped_width = max(min(width, max_right - clamped_left, image_width - clamped_left), 1)
        clamped_height = max(min(height, max_bottom - clamped_top, image_height - clamped_top), 1)
        return [int(clamped_left), int(clamped_top), int(clamped_width), int(clamped_height)]

    def _pick_best_crop_detection(self, detections) -> object | None:
        if not detections:
            return None
        return max(
            detections,
            key=lambda item: (item.confidence, item.bbox[2] * item.bbox[3]),
        )

    def _normalize_plate_color_label(self, plate_color: str) -> str:
        normalized = str(plate_color or "").strip()
        if not normalized:
            return "未知"

        lower = normalized.lower()
        if "蓝" in normalized or lower == "blue":
            return "蓝牌"
        if "黄" in normalized or lower == "yellow":
            return "黄牌"
        if "绿" in normalized or lower == "green":
            return "绿牌"
        if "白" in normalized or lower == "white":
            return "白牌"
        if "黑" in normalized or lower == "black":
            return "黑牌"
        if "未" in normalized or lower == "unknown":
            return "未知"
        return normalized

    def _normalize_plate_color_for_plate(self, plate_color: str, plate_number: str) -> str:
        normalized = self._normalize_plate_color_label(plate_color)
        plate_length = len(plate_number or "")

        if plate_length == 8 and normalized in {"蓝牌", "黄牌"}:
            return "绿牌"
        if plate_length == 7 and normalized == "绿牌":
            return "黄牌"
        return normalized

    def _pick_best_plate_candidate(self, detections: list[PlateDetection]) -> PlateDetection | None:
        if not detections:
            return None

        merged: dict[str, PlateDetection] = {}
        for detection in detections:
            if not detection.plate_number:
                continue

            normalized = PlateDetection(
                plate_number=detection.plate_number,
                plate_color=self._normalize_plate_color_for_plate(detection.plate_color, detection.plate_number),
                confidence=detection.confidence,
                bbox=list(detection.bbox),
            )
            current = merged.get(normalized.plate_number)
            if current is None:
                merged[normalized.plate_number] = normalized
                continue

            merged[normalized.plate_number] = PlateDetection(
                plate_number=normalized.plate_number,
                plate_color=self._pick_better_plate_color(
                    current.plate_color,
                    normalized.plate_color,
                    normalized.plate_number,
                ),
                confidence=min(1.0, max(current.confidence, normalized.confidence) + 0.04),
                bbox=list(normalized.bbox if normalized.confidence >= current.confidence else current.bbox),
            )

        if not merged:
            return None

        return max(
            merged.values(),
            key=lambda item: (
                self._score_plate_candidate(item),
                item.confidence,
                item.bbox[2] * item.bbox[3],
            ),
        )

    def _pick_better_plate_color(self, current_color: str, next_color: str, plate_number: str) -> str:
        current = self._normalize_plate_color_for_plate(current_color, plate_number)
        candidate = self._normalize_plate_color_for_plate(next_color, plate_number)
        if current == candidate:
            return current
        if len(plate_number or "") == 8 and candidate == "绿牌":
            return candidate
        if len(plate_number or "") == 7 and candidate == "黄牌":
            return candidate
        if current == "未知":
            return candidate
        return current

    def _score_plate_candidate(self, detection: PlateDetection) -> float:
        plate_number = detection.plate_number or ""
        suffix = plate_number[2:]
        score = detection.confidence

        if len(plate_number) in (7, 8):
            score += 0.03
        if detection.plate_color != "未知":
            score += 0.02
        if len(plate_number) == 8 and detection.plate_color == "绿牌":
            score += 0.05
        if len(plate_number) == 7 and detection.plate_color == "黄牌":
            score += 0.03
        if len(plate_number) == 7 and detection.plate_color == "绿牌":
            score -= 0.08
        if re.fullmatch(r"[1I]{5,6}", suffix):
            score -= 0.25

        return score

    def _deduplicate_plate_detections(self, detections: list[PlateDetection]) -> list[PlateDetection]:
        kept: list[PlateDetection] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            duplicate = False
            for existing in kept:
                if detection.plate_number == existing.plate_number and self._compute_iou(detection.bbox, existing.bbox) >= 0.1:
                    duplicate = True
                    break
                if self._compute_iou(detection.bbox, existing.bbox) >= 0.6:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(detection)
        return kept

    def _should_use_heavy_scan(self, state: PlateProcessingState) -> bool:
        if not state.tracks:
            return True
        if any(track.misses > 0 for track in state.tracks):
            return True
        interval = max(state.heavy_scan_interval, 1)
        return state.frame_index == 1 or state.frame_index % interval == 0

    def _update_tracks(self, frame, tracks: list[PlateTrack], frame_index: int) -> list[PlateTrack]:
        active_tracks: list[PlateTrack] = []
        for track in tracks:
            tracked = self._track_plate(frame, track)
            if tracked:
                track.last_seen_frame = frame_index
                track.misses = 0
            else:
                track.misses += 1

            if track.misses <= settings.plate_stream_tracking_max_misses:
                active_tracks.append(track)
        return active_tracks

    def _track_plate(self, frame, track: PlateTrack) -> bool:
        cv2 = self._require_cv2()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        x, y, width, height = track.bbox
        template_height, template_width = track.template.shape[:2]
        if width <= 0 or height <= 0 or template_width <= 0 or template_height <= 0:
            return False

        search_rect = self._build_search_rect(
            frame_width=gray_frame.shape[1],
            frame_height=gray_frame.shape[0],
            x=x,
            y=y,
            width=width,
            height=height,
        )
        sx, sy, sw, sh = search_rect
        search_region = gray_frame[sy : sy + sh, sx : sx + sw]
        if search_region.size == 0 or search_region.shape[0] < template_height or search_region.shape[1] < template_width:
            return False

        result = cv2.matchTemplate(search_region, track.template, cv2.TM_CCOEFF_NORMED)
        _, max_score, _, max_loc = cv2.minMaxLoc(result)
        if max_score < settings.plate_stream_tracking_match_threshold:
            return False

        next_x = sx + max_loc[0]
        next_y = sy + max_loc[1]
        next_bbox = [next_x, next_y, width, height]
        next_template = self._extract_template(gray_frame, next_bbox)
        if next_template is None:
            return False

        track.bbox = next_bbox
        track.template = self._blend_template(track.template, next_template)
        return True

    def _should_rerecognize(
        self,
        tracks: list[PlateTrack],
        frame_index: int,
        last_recognition_frame: int,
        *,
        recognition_interval: int,
        force_when_no_tracks: bool,
    ) -> bool:
        if frame_index == 1 or not tracks:
            return frame_index == 1 or force_when_no_tracks
        if any(track.misses > 0 for track in tracks):
            return True
        interval = max(recognition_interval, 1)
        return frame_index - last_recognition_frame >= interval

    def _merge_recognized_tracks(
        self,
        frame,
        tracks: list[PlateTrack],
        detections: list[PlateDetection],
        frame_index: int,
    ) -> list[PlateTrack]:
        merged_tracks = list(tracks)
        matched_indices: set[int] = set()

        for detection in detections:
            match_index = self._find_track_match(detection, merged_tracks, matched_indices)
            template = self._extract_template_from_frame(frame, detection.bbox)
            if template is None:
                continue

            if match_index is None:
                merged_tracks.append(
                    self._build_track_from_detection(
                        detection=detection,
                        template=template,
                        frame_index=frame_index,
                    )
                )
                continue

            track = merged_tracks[match_index]
            if self._should_replace_track(track, detection, frame_index):
                merged_tracks[match_index] = self._build_track_from_detection(
                    detection=detection,
                    template=template,
                    frame_index=frame_index,
                )
                matched_indices.add(match_index)
                continue

            self._apply_detection_vote(track, detection)
            track.bbox = list(detection.bbox)
            track.template = template
            track.last_seen_frame = frame_index
            track.last_recognized_frame = frame_index
            track.misses = 0
            matched_indices.add(match_index)

        return merged_tracks

    def _build_track_from_detection(
        self,
        *,
        detection: PlateDetection,
        template,
        frame_index: int,
    ) -> PlateTrack:
        text_votes = {}
        color_votes = {}
        if detection.plate_number:
            text_votes[detection.plate_number] = max(detection.confidence, 0.1)
        if detection.plate_color:
            color_votes[detection.plate_color] = max(detection.confidence, 0.1)
        return PlateTrack(
            track_id=uuid4().hex,
            plate_number=detection.plate_number,
            plate_color=detection.plate_color,
            confidence=detection.confidence,
            bbox=list(detection.bbox),
            template=template,
            last_seen_frame=frame_index,
            last_recognized_frame=frame_index,
            misses=0,
            text_votes=text_votes,
            color_votes=color_votes,
        )

    def _apply_detection_vote(self, track: PlateTrack, detection: PlateDetection) -> None:
        weight = max(detection.confidence, 0.1)
        if detection.plate_number:
            track.text_votes[detection.plate_number] = track.text_votes.get(detection.plate_number, 0.0) + weight
        if detection.plate_color and detection.plate_color != "未知":
            track.color_votes[detection.plate_color] = track.color_votes.get(detection.plate_color, 0.0) + weight

        best_text = max(track.text_votes.items(), key=lambda item: (item[1], len(item[0])))[0] if track.text_votes else ""
        best_color = max(track.color_votes.items(), key=lambda item: item[1])[0] if track.color_votes else track.plate_color
        current_score = track.text_votes.get(track.plate_number, 0.0)
        next_score = track.text_votes.get(best_text, 0.0)

        if best_text and (not track.plate_number or next_score >= current_score * 1.08):
            track.plate_number = best_text
        if best_color:
            track.plate_color = best_color
        track.confidence = max(track.confidence, detection.confidence)

    def _should_replace_track(
        self,
        track: PlateTrack,
        detection: PlateDetection,
        frame_index: int,
    ) -> bool:
        if not track.plate_number or not detection.plate_number:
            return False
        if track.plate_number == detection.plate_number:
            return False
        if track.confidence < 0.55 or detection.confidence < 0.55:
            return False
        if self._plate_text_distance_ratio(track.plate_number, detection.plate_number) < 0.5:
            return False
        if self._compute_iou(track.bbox, detection.bbox) < 0.2:
            return False
        if frame_index - track.last_recognized_frame < 1 and sum(track.text_votes.values()) < 2.0:
            return False
        return True

    def _plate_text_distance_ratio(self, plate_a: str, plate_b: str) -> float:
        max_len = max(len(plate_a), len(plate_b), 1)
        mismatch = abs(len(plate_a) - len(plate_b))
        for char_a, char_b in zip(plate_a, plate_b):
            if char_a != char_b:
                mismatch += 1
        return mismatch / max_len

    def _find_track_match(
        self,
        detection: PlateDetection,
        tracks: list[PlateTrack],
        matched_indices: set[int],
    ) -> int | None:
        best_index: int | None = None
        best_score = 0.0

        for index, track in enumerate(tracks):
            if index in matched_indices:
                continue

            score = self._compute_iou(detection.bbox, track.bbox)
            if score > best_score:
                best_score = score
                best_index = index

        return best_index if best_score >= 0.1 else None

    def _tracks_to_detections(self, tracks: list[PlateTrack]) -> list[PlateDetection]:
        return [
            PlateDetection(
                plate_number=track.plate_number,
                plate_color=track.plate_color,
                confidence=track.confidence,
                bbox=list(track.bbox),
            )
            for track in tracks
        ]

    def _merge_best_detections(
        self,
        best_detections: dict[str, VideoDetectionStats],
        detections: list[PlateDetection],
        *,
        recognized: bool,
    ) -> None:
        for detection in detections:
            if not detection.plate_number:
                continue
            key = f"{detection.plate_number}:{detection.plate_color}"
            current = best_detections.get(key)
            if current is None:
                best_detections[key] = VideoDetectionStats(
                    detection=detection,
                    fresh_count=1 if recognized else 0,
                    display_count=1,
                )
                continue
            current.display_count += 1
            if recognized:
                current.fresh_count += 1
            if detection.confidence > current.detection.confidence:
                current.detection = detection

    def _finalize_video_detections(self, best_detections: dict[str, VideoDetectionStats]) -> list[PlateDetection]:
        accepted: list[PlateDetection] = []
        fallback: list[PlateDetection] = []
        ranked_stats = sorted(
            best_detections.values(),
            key=lambda item: (item.fresh_count, item.display_count, item.detection.confidence),
            reverse=True,
        )
        for stats in ranked_stats:
            confidence = stats.detection.confidence
            if stats.fresh_count >= 2 and stats.display_count >= 4 and confidence >= 0.30:
                accepted.append(stats.detection)
                continue
            if stats.fresh_count >= 2 and confidence >= 0.38:
                accepted.append(stats.detection)
                continue
            if stats.fresh_count >= 1 and stats.display_count >= 3 and confidence >= 0.40:
                accepted.append(stats.detection)
                continue
            if stats.display_count >= 6 and confidence >= 0.32:
                accepted.append(stats.detection)
                continue
            if stats.display_count >= 9 and confidence >= 0.30:
                accepted.append(stats.detection)
                continue
            if confidence >= 0.52:
                fallback.append(stats.detection)

        if not accepted:
            accepted = fallback[:8]
        accepted.sort(key=lambda item: item.confidence, reverse=True)
        return accepted

    def _build_stream_payload(self, frame, detections: list[PlateDetection]):
        cv2 = self._require_cv2()
        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), settings.plate_stream_jpeg_quality],
        )
        if not success:
            raise InferenceConfigurationError("Video frame encoding failed; unable to return the frame to the frontend.")

        frame_height, frame_width = frame.shape[:2]
        return {
            "frame": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "frame_width": frame_width,
            "frame_height": frame_height,
            "detections": [detection.model_dump() for detection in detections],
        }

    def _scale_detections(
        self,
        detections: list[PlateDetection],
        source_width: int,
        source_height: int,
        target_width: int,
        target_height: int,
    ) -> list[PlateDetection]:
        if source_width <= 0 or source_height <= 0:
            return detections
        if source_width == target_width and source_height == target_height:
            return detections

        scale_x = target_width / source_width
        scale_y = target_height / source_height
        scaled: list[PlateDetection] = []
        for detection in detections:
            x, y, width, height = detection.bbox
            scaled.append(
                PlateDetection(
                    plate_number=detection.plate_number,
                    plate_color=detection.plate_color,
                    confidence=detection.confidence,
                    bbox=[
                        int(round(x * scale_x)),
                        int(round(y * scale_y)),
                        max(int(round(width * scale_x)), 1),
                        max(int(round(height * scale_y)), 1),
                    ],
                )
            )
        return scaled

    def _scale_bbox(
        self,
        bbox: list[int],
        source_width: int,
        source_height: int,
        target_width: int,
        target_height: int,
    ) -> list[int]:
        if source_width <= 0 or source_height <= 0:
            return list(bbox)
        scale_x = target_width / source_width
        scale_y = target_height / source_height
        x, y, width, height = bbox
        return [
            int(round(x * scale_x)),
            int(round(y * scale_y)),
            max(int(round(width * scale_x)), 1),
            max(int(round(height * scale_y)), 1),
        ]

    def _annotate_frame(self, frame, detections: list[PlateDetection]):
        cv2 = self._require_cv2()
        annotated = frame.copy()
        line_color = (191, 212, 45)
        tag_background = (26, 14, 10)
        primary_text = (255, 254, 236)
        secondary_text = (214, 250, 207)

        for detection in detections:
            x, y, width, height = detection.bbox
            x1 = max(int(x), 0)
            y1 = max(int(y), 0)
            x2 = min(int(x + width), annotated.shape[1] - 1)
            y2 = min(int(y + height), annotated.shape[0] - 1)
            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(annotated, (x1, y1), (x2, y2), line_color, 2)

            plate_text = detection.plate_number or "UNREAD"
            confidence_text = f"{detection.confidence * 100:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            plate_scale = 0.52
            meta_scale = 0.40
            plate_thickness = 2
            meta_thickness = 1
            padding_x = 8
            padding_y = 7
            line_gap = 4

            (plate_w, plate_h), _ = cv2.getTextSize(plate_text, font, plate_scale, plate_thickness)
            (meta_w, meta_h), _ = cv2.getTextSize(confidence_text, font, meta_scale, meta_thickness)
            tag_width = max(plate_w, meta_w) + padding_x * 2
            tag_height = plate_h + meta_h + padding_y * 2 + line_gap

            preferred_top = y2 + 8
            if preferred_top + tag_height > annotated.shape[0]:
                preferred_top = max(y1 - tag_height - 8, 0)

            tag_left = min(max(x1, 0), max(annotated.shape[1] - tag_width - 1, 0))
            tag_top = preferred_top
            tag_right = min(tag_left + tag_width, annotated.shape[1] - 1)
            tag_bottom = min(tag_top + tag_height, annotated.shape[0] - 1)

            cv2.rectangle(annotated, (tag_left, tag_top), (tag_right, tag_bottom), tag_background, -1)
            cv2.rectangle(annotated, (tag_left, tag_top), (tag_right, tag_bottom), line_color, 1)

            plate_origin = (tag_left + padding_x, tag_top + padding_y + plate_h)
            meta_origin = (tag_left + padding_x, plate_origin[1] + line_gap + meta_h)
            cv2.putText(annotated, plate_text, plate_origin, font, plate_scale, primary_text, plate_thickness, cv2.LINE_AA)
            cv2.putText(annotated, confidence_text, meta_origin, font, meta_scale, secondary_text, meta_thickness, cv2.LINE_AA)

        return annotated

    def _resize_stream_frame(self, frame):
        return self._resize_frame_to_limit(frame, settings.plate_stream_max_side)

    def _resize_frame_to_limit(self, frame, max_side_limit: int):
        cv2 = self._require_cv2()
        if max_side_limit <= 0:
            return frame

        frame_height, frame_width = frame.shape[:2]
        max_side = max(frame_width, frame_height)
        if max_side <= max_side_limit:
            return frame

        scale_ratio = max_side_limit / max_side
        resized_width = max(int(frame_width * scale_ratio), 1)
        resized_height = max(int(frame_height * scale_ratio), 1)
        return cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    def _build_search_rect(self, frame_width: int, frame_height: int, x: int, y: int, width: int, height: int) -> list[int]:
        expand = max(settings.plate_stream_tracking_search_expand, 1.0)
        center_x = x + width / 2
        center_y = y + height / 2
        search_width = int(round(width * expand))
        search_height = int(round(height * expand))

        left = max(int(round(center_x - search_width / 2)), 0)
        top = max(int(round(center_y - search_height / 2)), 0)
        right = min(left + search_width, frame_width)
        bottom = min(top + search_height, frame_height)
        return [left, top, max(right - left, width), max(bottom - top, height)]

    def _extract_template_from_frame(self, frame, bbox: list[int]):
        cv2 = self._require_cv2()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self._extract_template(gray_frame, bbox)

    def _extract_template(self, gray_frame, bbox: list[int]):
        x, y, width, height = bbox
        frame_height, frame_width = gray_frame.shape[:2]
        left = max(x, 0)
        top = max(y, 0)
        right = min(x + width, frame_width)
        bottom = min(y + height, frame_height)
        if right <= left or bottom <= top:
            return None

        patch = gray_frame[top:bottom, left:right]
        if patch.size == 0 or patch.shape[0] < 8 or patch.shape[1] < 16:
            return None
        return patch.copy()

    def _blend_template(self, previous_template, next_template):
        cv2 = self._require_cv2()
        if previous_template.shape != next_template.shape:
            next_template = cv2.resize(
                next_template,
                (previous_template.shape[1], previous_template.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        alpha = max(0.0, min(settings.plate_stream_tracking_template_update_alpha, 1.0))
        if alpha <= 0:
            return previous_template
        return cv2.addWeighted(previous_template, 1.0 - alpha, next_template, alpha, 0.0)

    def _compute_iou(self, bbox_a: list[int], bbox_b: list[int]) -> float:
        ax1, ay1, aw, ah = bbox_a
        bx1, by1, bw, bh = bbox_b
        ax2 = ax1 + aw
        ay2 = ay1 + ah
        bx2 = bx1 + bw
        by2 = by1 + bh

        inter_left = max(ax1, bx1)
        inter_top = max(ay1, by1)
        inter_right = min(ax2, bx2)
        inter_bottom = min(ay2, by2)
        if inter_right <= inter_left or inter_bottom <= inter_top:
            return 0.0

        intersection = (inter_right - inter_left) * (inter_bottom - inter_top)
        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    def _save_history(
        self,
        detections: list[PlateDetection],
        image_path: str | None,
        user_id: int | None = None,
    ) -> None:
        if user_id is None:
            self._save_anonymous_history(detections)
            return

        records = [
            PlateRecord(
                user_id=user_id,
                plate_number=detection.plate_number,
                plate_color=detection.plate_color,
                bbox=detection.bbox,
                confidence=detection.confidence,
                image_path=image_path,
            )
            for detection in detections
            if detection.plate_number
        ]
        if not records:
            return

        try:
            with SessionLocal() as session:
                session.add_all(records)
                session.commit()
        except Exception as exc:
            logger.warning("Failed to persist plate recognition history: %s", exc)

    def _persist_upload(self, image_bytes: bytes, filename: str) -> str:
        suffix = Path(filename).suffix or ".jpg"
        target_dir = self.get_upload_root()
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{uuid4().hex}{suffix}"
        target_path.write_bytes(image_bytes)
        return str(target_path)

    def _detect_plates(self, image_bytes: bytes) -> list[PlateDetection]:
        return self._recognize_detections(image_bytes)

    async def _record_failure(
        self,
        *,
        user_id: int | None,
        filename: str,
        elapsed_ms: int,
        trace_id: str,
        title: str,
        summary: str,
        response_status: str,
    ) -> None:
        full_summary = f"{summary} 处理耗时：{elapsed_ms} ms。"
        try:
            self._record_operation(user_id, "plate_recognition", response_status)
            self._record_behavior(title=title, summary=full_summary)
            with SessionLocal() as session:
                await MonitorService(session).capture_event(
                    category="plate",
                    source="plate-recognition",
                    event_type=(
                        "plate_recognition_timeout"
                        if response_status.lower() == "timeout"
                        else "plate_recognition_failure"
                    ),
                    title=title,
                    summary=full_summary,
                    level="warning",
                    status=response_status,
                    trace_id=trace_id,
                    user_id=user_id,
                    details={
                        "filename": filename,
                        "elapsed_ms": elapsed_ms,
                        "response_status": response_status,
                    },
                )
        except Exception as exc:
            logger.warning("Failed to persist plate failure log for %s: %s", filename, exc)

    async def _record_monitor_success(
        self,
        *,
        user_id: int | None,
        filename: str,
        elapsed_ms: int,
        trace_id: str,
        detections: list[PlateDetection],
    ) -> None:
        event_type = "plate_recognition_success" if detections else "plate_recognition_no_detection"
        try:
            with SessionLocal() as session:
                await MonitorService(session).capture_event(
                    category="plate",
                    source="plate-recognition",
                    event_type=event_type,
                    title="车牌识别完成" if detections else "车牌识别未命中结果",
                    summary=self._build_behavior_summary(filename, detections, elapsed_ms),
                    level="info" if detections else "warning",
                    status="success" if detections else "no_detection",
                    trace_id=trace_id,
                    user_id=user_id,
                    confidence=detections[0].confidence if detections else None,
                    details={
                        "filename": filename,
                        "elapsed_ms": elapsed_ms,
                        "detection_count": len(detections),
                        "plates": [item.plate_number for item in detections],
                    },
                    trigger_alert=not detections,
                )
        except Exception as exc:
            logger.warning("Failed to persist plate monitor success event for %s: %s", filename, exc)

    def _record_operation(self, user_id: int | None, operation_type: str, response_status: str) -> None:
        if user_id is None:
            return

        try:
            with SessionLocal() as session:
                session.add(
                    UserOperationLog(
                        user_id=user_id,
                        operation_type=operation_type,
                        response_status=response_status,
                    )
                )
                session.commit()
        except Exception as exc:
            logger.warning("Failed to persist plate operation log: %s", exc)

    def _record_behavior(self, *, title: str, summary: str) -> None:
        try:
            with SessionLocal() as session:
                AlertService(session).record_behavior(
                    source="plate-recognition",
                    title=title,
                    summary=summary,
                )
        except Exception as exc:
            logger.warning("Failed to persist plate behavior record: %s", exc)

    def _build_behavior_summary(
        self,
        filename: str,
        detections: list[PlateDetection],
        elapsed_ms: int,
    ) -> str:
        if not detections:
            return f"{filename} 已完成识别，但未匹配到有效车牌。处理耗时：{elapsed_ms} ms。"

        first_detection = detections[0]
        return (
            f"{filename} 共识别到 {len(detections)} 个车牌结果。"
            f"首个结果为：{first_detection.plate_number}（{first_detection.plate_color}）。"
            f"处理耗时：{elapsed_ms} ms。"
        )

    def _require_cv2(self):
        try:
            import cv2
        except ImportError as exc:
            raise InferenceDependencyError("Missing opencv-python-headless. Install video-processing dependencies in the backend environment first.") from exc
        return cv2
