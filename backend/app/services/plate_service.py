from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import threading
from threading import Lock, Thread
import time
from uuid import uuid4

from sqlalchemy import select

from app.core.config import settings
from app.core.database import SessionLocal
from app.core.logger import get_logger
from app.models.plate_record import PlateRecord
from app.models_infer.errors import InferenceConfigurationError, InferenceDependencyError
from app.models_infer.paddleocr_recognizer import PaddleOCRRecognizer
from app.models_infer.vehicle_type_classifier import VehicleTypeClassifier
from app.models_infer.yolo_detector import YoloDetector
from app.schemas.plate import (
    PlateDetection,
    PlateRecognitionResponse,
    PlateRecordSummary,
    PlateVideoJobCreateResponse,
    PlateVideoJobStatusResponse,
    PlateVideoRecognitionResponse,
)

logger = get_logger(__name__)

PLATE_COLOR_BLUE = "\u84dd\u724c"
PLATE_COLOR_YELLOW = "\u9ec4\u724c"
PLATE_COLOR_GREEN = "\u7eff\u724c"
PLATE_COLOR_WHITE = "\u767d\u724c"
PLATE_COLOR_BLACK = "\u9ed1\u724c"
PLATE_COLOR_UNKNOWN = "\u672a\u77e5"
VEHICLE_TYPE_CAR = "\u8f7f\u8f66"
VEHICLE_TYPE_TRUCK = "\u5361\u8f66"
VEHICLE_TYPE_BUS = "\u516c\u4ea4"
VEHICLE_TYPE_CRANE = "\u540a\u8f66"
VEHICLE_TYPE_JEEP = "\u5409\u666e\u8f66"
VEHICLE_TYPE_PICKUP = "\u76ae\u5361"
VEHICLE_TYPE_MOTORCYCLE = "\u6469\u6258\u8f66"
VEHICLE_TYPE_UNKNOWN = "\u672a\u8bc6\u522b"

_CONFUSABLE_PLATE_CHAR_GROUPS = (
    frozenset({"1", "I", "7"}),
    frozenset({"0", "O"}),
)


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
    vehicle_type_votes: dict[str, float] = field(default_factory=dict)
    unread_snapshots_saved: int = 0
    last_unread_snapshot_frame: int = 0
    last_unread_ocr_frame: int = 0
    unread_observations: int = 0
    last_tracking_score: float = 0.0
    vehicle_type: str = VEHICLE_TYPE_UNKNOWN


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
    stream_mode: bool = False


@dataclass
class VideoDetectionStats:
    detection: PlateDetection
    fresh_count: int = 0
    display_count: int = 0
    vehicle_type_votes: dict[str, float] = field(default_factory=dict)


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


@dataclass
class VideoProcessingJob:
    job_id: str
    source_filename: str
    source_path: Path
    output_path: Path
    preview_path: Path
    status: str = "queued"
    progress: float = 0.0
    processed_frame_count: int = 0
    total_frames: int = 0
    detections: list[PlateDetection] = field(default_factory=list)
    unread_samples: list[str] = field(default_factory=list)
    duration_seconds: float | None = None
    processed_video_url: str | None = None
    preview_image_url: str | None = None
    error_message: str | None = None
    updated_at: int = 0


class PlateService:
    _VIDEO_ACTIVE_UNREAD_OCR_INTERVAL = 18
    _VIDEO_ACTIVE_UNREAD_OCR_MAX_TRACKS = 1

    def __init__(self) -> None:
        self.recognizer = PaddleOCRRecognizer()
        self.detector = YoloDetector()
        self.vehicle_classifier = VehicleTypeClassifier()
        self._backend_dir = Path(__file__).resolve().parents[2]
        self._upload_root = (self._backend_dir / settings.plate_upload_dir).resolve()
        self._video_jobs: dict[str, VideoProcessingJob] = {}
        self._video_jobs_lock = Lock()
        self._annotation_font_cache: dict[tuple[str, int], object] = {}
        self._runtime_warmed = False

    def warmup_runtime(self, *, silent: bool = True) -> None:
        if self._runtime_warmed:
            return
        self.detector.warmup()
        self.recognizer.warmup(silent=silent)
        self.vehicle_classifier.warmup()
        self._runtime_warmed = True

    async def recognize_image(self, filename: str, image_bytes: bytes | None = None) -> PlateRecognitionResponse:
        return self.recognize_image_bytes(image_bytes or b"", filename)

    def recognize_image_bytes(self, image_bytes: bytes, filename: str = "unknown.jpg") -> PlateRecognitionResponse:
        if not image_bytes:
            return PlateRecognitionResponse(frame_id=filename, detections=[])

        source_path = self._persist_upload(image_bytes, filename) if settings.plate_save_uploads else None
        detections = self._recognize_image_detections(image_bytes)

        if detections:
            self._save_history(detections, source_path)

        return PlateRecognitionResponse(frame_id=filename, detections=detections)

    def _recognize_image_detections(self, image_bytes: bytes) -> list[PlateDetection]:
        # Image uploads use a lighter path than video: favor detector crops and skip the
        # expensive full-frame OCR fallback unless the detector itself is unavailable.
        if self._should_use_detector():
            image = self._decode_image_source(image_bytes)
            detector_hits = self._detect_image_detector_hits(image)
            detector_hits = self._attach_vehicle_type_context_to_hits(detector_hits)
            self._log_image_detector_stage("fast-pass", detector_hits)
            plate_hits = [hit for hit in detector_hits if str(hit.get("kind", "plate")) != "vehicle"]
            if plate_hits:
                detector_hits = self._augment_image_hits_with_vehicle_detector(image, detector_hits)
                self._log_image_detector_stage("fast-pass+vehicle-context", detector_hits)
                plate_hits = [hit for hit in detector_hits if str(hit.get("kind", "plate")) != "vehicle"]
                limited_plate_hits = plate_hits[: min(settings.plate_detector_max_candidates, 6)]
                fast_detections, failed_plate_hits = self._recognize_image_fast_plate_hits(
                    image,
                    limited_plate_hits,
                )
                boosted_plate_detections = self._recognize_image_failed_plate_hits_with_local_ocr_boost(
                    image,
                    failed_plate_hits,
                )
                supplemental_detections = self._recognize_image_unmatched_vehicle_hits(
                    image,
                    detector_hits,
                    plate_hits,
                )
                merged_detections = self._deduplicate_plate_detections(
                    fast_detections + boosted_plate_detections + supplemental_detections
                )
                if merged_detections:
                    logger.info(
                        "Image recognition resolved on fast path | detections=%d fast=%d boosted=%d supplemental=%d",
                        len(merged_detections),
                        len(fast_detections),
                        len(boosted_plate_detections),
                        len(supplemental_detections),
                    )
                    return merged_detections
                logger.info("Image fast path found detector plate hits but OCR produced no final detections.")
                return []

            detailed_hits = self._detect_image_detector_hits_detailed(image)
            detailed_hits = self._augment_image_hits_with_vehicle_detector(image, detailed_hits)
            detailed_hits = self._attach_vehicle_type_context_to_hits(detailed_hits)
            self._log_image_detector_stage("detailed-pass", detailed_hits)
            detailed_plate_hits = [hit for hit in detailed_hits if str(hit.get("kind", "plate")) != "vehicle"]
            detailed_supplemental_detections = self._recognize_image_unmatched_vehicle_hits(
                image,
                detailed_hits,
                detailed_plate_hits,
            )
            if detailed_plate_hits:
                detailed_detections = self._recognize_detector_hits(
                    image,
                    detailed_plate_hits[: min(settings.plate_detector_max_candidates, 6)],
                    fast_mode=False,
                    preserve_unread=False,
                    allow_ocr_fallback=True,
                )
                merged_detections = self._deduplicate_plate_detections(
                    detailed_detections + detailed_supplemental_detections
                )
                if merged_detections:
                    logger.info(
                        "Image recognition resolved on detailed path | detections=%d detailed=%d supplemental=%d",
                        len(merged_detections),
                        len(detailed_detections),
                        len(detailed_supplemental_detections),
                    )
                    return merged_detections
            elif detailed_supplemental_detections:
                logger.info(
                    "Image recognition resolved via detailed unmatched-vehicle supplemental path | detections=%d",
                    len(detailed_supplemental_detections),
                )
                return detailed_supplemental_detections

            if not self._should_run_aggressive_image_full_frame_fallback(image, detailed_hits):
                logger.info(
                    "Image detailed detector produced local hints but no final OCR result; skipping aggressive full-frame fallback to avoid long stall."
                )
                return []

            logger.info("Image detector passes exhausted; falling back to aggressive full-frame OCR scan.")
            return self._recognize_detections(
                image_bytes,
                aggressive=True,
                heavy_scan=True,
                allow_full_frame_fallback=True,
                fast_mode=False,
            )

        return self._recognize_detections(
            image_bytes,
            heavy_scan=False,
            allow_full_frame_fallback=True,
            fast_mode=True,
        )

    def recognize_video_bytes(self, video_bytes: bytes, filename: str = "unknown.mp4") -> PlateVideoRecognitionResponse:
        if not video_bytes:
            raise ValueError("Uploaded video is empty.")

        source_path, output_path = self._prepare_video_paths(filename)
        source_path.write_bytes(video_bytes)

        try:
            result = self._process_video_file(source_path, output_path, filename)
        finally:
            if not settings.plate_save_uploads:
                try:
                    source_path.unlink(missing_ok=True)
                except Exception:
                    logger.warning("Failed to remove temporary uploaded video: %s", source_path)

        return result

    def start_video_job(self, video_bytes: bytes, filename: str = "unknown.mp4") -> PlateVideoJobCreateResponse:
        if not video_bytes:
            raise ValueError("Uploaded video is empty.")

        source_path, output_path = self._prepare_video_paths(filename)
        preview_dir = self.get_upload_root() / "videos" / "progress"
        preview_dir.mkdir(parents=True, exist_ok=True)
        job_id = uuid4().hex
        preview_path = preview_dir / f"{job_id}.jpg"
        source_path.write_bytes(video_bytes)

        job = VideoProcessingJob(
            job_id=job_id,
            source_filename=filename,
            source_path=source_path,
            output_path=output_path,
            preview_path=preview_path,
            updated_at=int(time.time() * 1000),
        )
        with self._video_jobs_lock:
            self._video_jobs[job_id] = job

        worker = Thread(target=self._run_video_job, args=(job_id,), daemon=True)
        worker.start()
        return PlateVideoJobCreateResponse(job_id=job_id, status=job.status)

    def get_video_job_status(self, job_id: str) -> PlateVideoJobStatusResponse | None:
        with self._video_jobs_lock:
            job = self._video_jobs.get(job_id)
            if job is None:
                return None
            return PlateVideoJobStatusResponse(
                job_id=job.job_id,
                source_filename=job.source_filename,
                status=job.status,
                progress=job.progress,
                processed_frame_count=job.processed_frame_count,
                total_frames=job.total_frames,
                detections=[item.model_copy(deep=True) for item in job.detections],
                preview_image_url=self._build_cache_busted_media_url(job.preview_image_url, job.updated_at),
                processed_video_url=job.processed_video_url,
                unread_samples=list(job.unread_samples),
                duration_seconds=job.duration_seconds,
                error_message=job.error_message,
            )

    def stream_rtsp(self, rtsp_url: str):
        for frame, detections in self.iter_annotated_stream(rtsp_url):
            yield self._build_stream_payload(frame, detections)

    def iter_annotated_stream(self, rtsp_url: str, stop_event=None):
        capture, pending_frame = self._open_rtsp_capture(rtsp_url)
        state = PlateProcessingState(
            recognition_interval=1,
            heavy_scan_interval=max(settings.plate_stream_process_every_n_frames, 4),
            stream_mode=True,
        )
        last_sent_at = 0.0
        min_interval = 1.0 / settings.plate_stream_max_fps if settings.plate_stream_max_fps > 0 else 0.0
        consecutive_read_failures = 0
        last_output_frame = None
        last_output_detections: list[PlateDetection] = []
        recognition_lock = threading.Lock()
        recognition_state = {
            "frame": None,
            "version": 0,
            "processed_version": 0,
            "detections": [],
            "detections_version": 0,
            "detections_updated_at": 0.0,
            "display_width": 0,
            "display_height": 0,
        }
        recognition_submit_interval = self._stream_recognition_submit_interval_seconds(min_interval)
        last_recognition_submit_at = 0.0

        internal_stop_event = stop_event if stop_event is not None else threading.Event()

        def recognition_loop() -> None:
            while not internal_stop_event.is_set():
                local_frame = None
                local_version = 0
                with recognition_lock:
                    if recognition_state["version"] > recognition_state["processed_version"]:
                        local_frame = recognition_state["frame"]
                        local_version = recognition_state["version"]

                if local_frame is None:
                    time.sleep(0.01)
                    continue

                try:
                    _, scaled_detections, _ = self._process_frame(
                        local_frame,
                        state,
                        save_history=True,
                        force_detect_when_empty=True,
                    )
                    display_frame = self._resize_stream_frame(local_frame)
                    with recognition_lock:
                        recognition_state["detections"] = [item.model_copy() for item in scaled_detections]
                        recognition_state["detections_version"] = local_version
                        recognition_state["detections_updated_at"] = time.monotonic()
                        recognition_state["display_width"] = display_frame.shape[1]
                        recognition_state["display_height"] = display_frame.shape[0]
                        recognition_state["processed_version"] = local_version
                except Exception:
                    logger.warning("Async stream recognition step failed; keeping the latest published frame alive.", exc_info=True)
                    with recognition_lock:
                        recognition_state["processed_version"] = local_version

        recognition_thread = threading.Thread(target=recognition_loop, daemon=True)
        recognition_thread.start()

        try:
            while True:
                if internal_stop_event.is_set():
                    break

                if pending_frame is not None:
                    source_frame = pending_frame
                    pending_frame = None
                    ok = True
                else:
                    ok, source_frame = capture.read()

                if not ok or source_frame is None:
                    consecutive_read_failures += 1
                    if consecutive_read_failures > self._stream_read_failure_retry_limit():
                        logger.warning(
                            "RTSP stream read failed repeatedly; stopping processed stream | rtsp_url=%s failures=%d",
                            rtsp_url,
                            consecutive_read_failures,
                        )
                        break

                    if last_output_frame is not None:
                        now = time.monotonic()
                        if not self._should_skip_stream_frame_for_rate_limit(
                            last_sent_at=last_sent_at,
                            current_time=now,
                            min_interval=min_interval,
                        ):
                            last_sent_at = now
                            repeated_detections = [
                                item.model_copy() if hasattr(item, "model_copy") else item
                                for item in last_output_detections
                            ]
                            yield last_output_frame, repeated_detections

                    time.sleep(self._stream_read_failure_retry_sleep_seconds(min_interval))
                    continue

                consecutive_read_failures = 0

                now = time.monotonic()
                if self._should_skip_stream_frame_for_rate_limit(
                    last_sent_at=last_sent_at,
                    current_time=now,
                    min_interval=min_interval,
                ):
                    continue

                if (
                    now - last_recognition_submit_at >= recognition_submit_interval
                    or recognition_state["version"] == 0
                ):
                    with recognition_lock:
                        recognition_state["frame"] = source_frame.copy()
                        recognition_state["version"] += 1
                    last_recognition_submit_at = now

                display_frame = self._resize_stream_frame(source_frame)
                with recognition_lock:
                    cached_detections = [item.model_copy() for item in recognition_state["detections"]]
                    cached_detections_version = int(recognition_state["detections_version"])
                    cached_detections_updated_at = float(recognition_state["detections_updated_at"])
                    cached_display_width = int(recognition_state["display_width"])
                    cached_display_height = int(recognition_state["display_height"])
                    current_version = int(recognition_state["version"])

                if (
                    cached_detections
                    and cached_display_width > 0
                    and cached_display_height > 0
                    and self._should_display_stream_cached_detections(
                        current_version=current_version,
                        detections_version=cached_detections_version,
                        detections_updated_at=cached_detections_updated_at,
                        current_time=now,
                        submit_interval=recognition_submit_interval,
                    )
                ):
                    scaled_detections = self._scale_detections(
                        cached_detections,
                        cached_display_width,
                        cached_display_height,
                        display_frame.shape[1],
                        display_frame.shape[0],
                    )
                else:
                    scaled_detections = []

                annotated = self._annotate_frame(display_frame, scaled_detections)
                last_sent_at = time.monotonic()
                last_output_frame = annotated.copy() if hasattr(annotated, "copy") else annotated
                last_output_detections = [
                    item.model_copy() if hasattr(item, "model_copy") else item
                    for item in scaled_detections
                ]

                yield annotated, scaled_detections
        finally:
            if stop_event is None:
                internal_stop_event.set()
            recognition_thread.join(timeout=1.0)
            capture.release()

    def list_history(self) -> list[PlateRecordSummary]:
        with SessionLocal() as session:
            statement = (
                select(PlateRecord)
                .order_by(PlateRecord.created_at.desc())
                .limit(settings.plate_history_limit)
            )
            records = session.scalars(statement).all()

        if not records:
            return [
                PlateRecordSummary(
                    id=1,
                    plate_number="?A12345",
                    plate_color="??",
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

    def get_upload_root(self) -> Path:
        self._upload_root.mkdir(parents=True, exist_ok=True)
        return self._upload_root

    def public_media_url_for(self, file_path: Path) -> str:
        relative_path = file_path.resolve().relative_to(self.get_upload_root())
        return f"/media/{relative_path.as_posix()}"

    def _build_cache_busted_media_url(self, media_url: str | None, updated_at: int) -> str | None:
        if not media_url:
            return None
        separator = "&" if "?" in media_url else "?"
        return f"{media_url}{separator}ts={updated_at}"

    def _run_video_job(self, job_id: str) -> None:
        with self._video_jobs_lock:
            job = self._video_jobs.get(job_id)
        if job is None:
            return

        self._update_video_job(
            job_id,
            status="processing",
            progress=0.0,
            processed_frame_count=0,
            total_frames=0,
            error_message=None,
        )

        try:
            result = self._process_video_file(
                job.source_path,
                job.output_path,
                job.source_filename,
                progress_callback=lambda **payload: self._handle_video_job_progress(job_id, **payload),
            )
            self._update_video_job(
                job_id,
                status="completed",
                progress=1.0,
                processed_frame_count=result.processed_frame_count,
                detections=result.detections,
                unread_samples=result.unread_samples,
                duration_seconds=result.duration_seconds,
                processed_video_url=result.processed_video_url,
            )
        except Exception as exc:
            logger.exception("Video job failed: %s", job_id)
            self._update_video_job(
                job_id,
                status="failed",
                error_message=str(exc),
            )
        finally:
            if not settings.plate_save_uploads:
                try:
                    job.source_path.unlink(missing_ok=True)
                except Exception:
                    logger.warning("Failed to remove temporary uploaded video: %s", job.source_path)

    def _handle_video_job_progress(
        self,
        job_id: str,
        *,
        processed_frame_count: int,
        total_frames: int,
        detections: list[PlateDetection],
        annotated_frame=None,
    ) -> None:
        preview_image_url: str | None = None
        preview_updated_at = int(time.time() * 1000)
        if annotated_frame is not None:
            preview_image_url = self._write_video_job_preview(job_id, annotated_frame)
        progress = 0.0
        if total_frames > 0:
            progress = min(max(processed_frame_count / total_frames, 0.0), 1.0)
        self._update_video_job(
            job_id,
            status="processing",
            progress=progress,
            processed_frame_count=processed_frame_count,
            total_frames=total_frames,
            detections=detections,
            preview_image_url=preview_image_url,
            updated_at=preview_updated_at,
        )

    def _write_video_job_preview(self, job_id: str, annotated_frame) -> str | None:
        with self._video_jobs_lock:
            job = self._video_jobs.get(job_id)
        if job is None:
            return None

        cv2 = self._require_cv2()
        success = cv2.imwrite(
            str(job.preview_path),
            annotated_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), max(settings.plate_stream_jpeg_quality, 72)],
        )
        if not success:
            return job.preview_image_url
        return self.public_media_url_for(job.preview_path)

    def _update_video_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: float | None = None,
        processed_frame_count: int | None = None,
        total_frames: int | None = None,
        detections: list[PlateDetection] | None = None,
        preview_image_url: str | None = None,
        unread_samples: list[str] | None = None,
        duration_seconds: float | None = None,
        processed_video_url: str | None = None,
        error_message: str | None = None,
        updated_at: int | None = None,
    ) -> None:
        with self._video_jobs_lock:
            job = self._video_jobs.get(job_id)
            if job is None:
                return
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = progress
            if processed_frame_count is not None:
                job.processed_frame_count = processed_frame_count
            if total_frames is not None:
                job.total_frames = total_frames
            if detections is not None:
                job.detections = [item.model_copy(deep=True) for item in detections]
            if preview_image_url is not None:
                job.preview_image_url = preview_image_url
            if unread_samples is not None:
                job.unread_samples = list(unread_samples)
            if duration_seconds is not None:
                job.duration_seconds = duration_seconds
            if processed_video_url is not None:
                job.processed_video_url = processed_video_url
            if error_message is not None or status == "failed":
                job.error_message = error_message
            if updated_at is not None:
                job.updated_at = updated_at
            else:
                job.updated_at = int(time.time() * 1000)

    def _process_video_file(
        self,
        source_path: Path,
        output_path: Path,
        filename: str,
        progress_callback=None,
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

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1:
            fps = 25.0
        output_fps_target = max(
            min(
                self._resolve_video_output_fps_target(fps),
                int(round(fps)) if fps > 0 else settings.plate_video_output_fps,
            ),
            1,
        )
        output_frame_stride = max(int(round(fps / output_fps_target)), 1)
        output_fps = fps / output_frame_stride
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        recognition_interval = max(
            settings.plate_video_process_every_n_frames,
            max(int(round(fps * 0.75)), 16),
        )
        heavy_scan_interval = max(recognition_interval * 8, 48)
        progress_log_interval = max(recognition_interval * 2, 30)
        preview_update_interval = max(recognition_interval * 2, 24)
        summary_merge_interval = 3
        state = PlateProcessingState(
            recognition_interval=recognition_interval,
            heavy_scan_interval=heavy_scan_interval,
        )

        logger.info(
            "Starting plate video processing: %s | fps=%.2f total_frames=%s recognition_interval=%d max_side=%d output_fps=%.2f output_stride=%d",
            filename,
            fps,
            total_frames if total_frames > 0 else "unknown",
            recognition_interval,
            settings.plate_video_recognition_max_side,
            output_fps,
            output_frame_stride,
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
                should_publish_progress = callable(progress_callback) and (
                    processed_frame_count == 1
                    or processed_frame_count % preview_update_interval == 0
                    or (total_frames > 0 and processed_frame_count >= total_frames)
                )
                should_write_output_frame = processed_frame_count == 1 or processed_frame_count % output_frame_stride == 0
                (
                    annotated,
                    preview_detections,
                    fresh_detections,
                    active_detections,
                ) = self._process_video_frame_with_tracking(
                    source_frame,
                    state=state,
                    render=should_publish_progress or should_write_output_frame,
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
                if should_publish_progress:
                    try:
                        progress_callback(
                            processed_frame_count=processed_frame_count,
                            total_frames=total_frames,
                            detections=preview_detections,
                            annotated_frame=annotated,
                        )
                    except Exception:
                        logger.warning("Failed to publish video job progress update.", exc_info=True)
                if should_write_output_frame:
                    if writer is None:
                        height, width = annotated.shape[:2]
                        temp_output_path.parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(
                            str(temp_output_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            output_fps,
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise InferenceConfigurationError("Failed to create annotated video output file.")

                    writer.write(annotated)
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        detections = self._finalize_video_detections(best_detections)
        if detections:
            self._save_history(detections, None)

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
        if callable(progress_callback):
            try:
                progress_callback(
                    processed_frame_count=processed_frame_count,
                    total_frames=total_frames,
                    detections=detections,
                    annotated_frame=None,
                )
            except Exception:
                logger.warning("Failed to publish final video job progress update.", exc_info=True)
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
        render: bool,
    ) -> tuple[object | None, list[PlateDetection], list[PlateDetection], list[PlateDetection]]:
        state.frame_index += 1
        use_fast_large_plate_mode = self._should_use_fast_large_plate_mode(state)
        if self._should_skip_light_video_frame(state=state, render=render, use_fast_large_plate_mode=use_fast_large_plate_mode):
            active_detections = self._scale_detections(
                self._tracks_to_detections(state.tracks),
                state.working_width,
                state.working_height,
                source_frame.shape[1],
                source_frame.shape[0],
            )
            return None, active_detections, [], active_detections

        working_frame = self._resize_frame_to_limit(
            source_frame,
            self._resolve_video_recognition_max_side(source_frame, state),
        )
        state.working_width = working_frame.shape[1]
        state.working_height = working_frame.shape[0]
        should_update_tracks = (
            state.frame_index == 1
            or not state.tracks
            or any(track.misses > 0 for track in state.tracks)
            or state.frame_index - state.last_tracking_frame >= self._video_track_update_interval(use_fast_large_plate_mode)
        )
        if should_update_tracks:
            state.tracks = self._update_tracks(working_frame, state.tracks, state.frame_index)
            state.last_tracking_frame = state.frame_index

        fresh_working_detections: list[PlateDetection] = []
        ran_recognize = False
        if self._should_rerecognize_video(state):
            ran_recognize = True
            use_heavy_scan = settings.plate_video_detector_full_frame_fallback and self._should_use_heavy_scan(state)
            raw_working_detections = self._recognize_detections(
                working_frame,
                aggressive=False,
                heavy_scan=use_heavy_scan,
                allow_full_frame_fallback=settings.plate_video_detector_full_frame_fallback,
                fast_mode=True,
                preserve_unread=not use_fast_large_plate_mode,
                video_fast_detector=use_fast_large_plate_mode,
            )
            fresh_working_detections = [item for item in raw_working_detections if item.plate_number]
            state.tracks = self._merge_recognized_tracks(
                working_frame,
                state.tracks,
                raw_working_detections,
                state.frame_index,
            )
            self._log_video_frame_summary(
                frame_index=state.frame_index,
                phase="recognize",
                detections=raw_working_detections,
                tracks=state.tracks,
            )
            state.last_recognition_frame = state.frame_index
        if self._should_probe_new_video_tracks(state):
            probe_hits = self._find_untracked_detector_hits(
                working_frame,
                state.tracks,
                max_hits=1 if use_fast_large_plate_mode else 2,
            )
            if probe_hits:
                raw_probe_detections = self._recognize_detector_hits(
                    working_frame,
                    probe_hits,
                    fast_mode=True,
                    preserve_unread=not use_fast_large_plate_mode,
                    video_mode=True,
                )
                fresh_working_detections = [item for item in raw_probe_detections if item.plate_number]
                state.tracks = self._merge_recognized_tracks(
                    working_frame,
                    state.tracks,
                    raw_probe_detections,
                    state.frame_index,
                )
                if ran_recognize:
                    fresh_working_detections.extend(
                        item for item in raw_probe_detections if item.plate_number
                    )
                else:
                    fresh_working_detections = [item for item in raw_probe_detections if item.plate_number]
                self._log_video_frame_summary(
                    frame_index=state.frame_index,
                    phase="probe",
                    detections=raw_probe_detections,
                    tracks=state.tracks,
                )
                if fresh_working_detections:
                    state.last_recognition_frame = state.frame_index
            state.last_probe_frame = state.frame_index

        unread_ocr_detections: list[PlateDetection] = []
        if self._should_attempt_active_unread_ocr(state=state, fresh_detections=fresh_working_detections):
            unread_ocr_detections = self._recover_active_unread_tracks(
                source_frame=source_frame,
                working_frame=working_frame,
                state=state,
                fresh_detections=fresh_working_detections,
            )
        if unread_ocr_detections:
            fresh_working_detections.extend(unread_ocr_detections)
            state.tracks = self._merge_recognized_tracks(
                working_frame,
                state.tracks,
                unread_ocr_detections,
                state.frame_index,
            )
            self._log_video_frame_summary(
                frame_index=state.frame_index,
                phase="unread-ocr",
                detections=unread_ocr_detections,
                tracks=state.tracks,
            )
            state.last_recognition_frame = state.frame_index

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
        if not render:
            return None, active_detections, fresh_detections, active_detections

        display_frame = self._resize_stream_frame(source_frame)
        scaled_detections = self._scale_detections(
            active_detections,
            source_frame.shape[1],
            source_frame.shape[0],
            display_frame.shape[1],
            display_frame.shape[0],
        )
        annotated = self._annotate_frame(display_frame, scaled_detections)
        return annotated, active_detections, fresh_detections, active_detections

    def _resolve_video_output_fps_target(self, fps: float) -> int:
        configured = max(int(settings.plate_video_output_fps), 1)
        if fps >= 20:
            return min(configured, 6)
        return configured

    def _video_track_update_interval(self, use_fast_large_plate_mode: bool) -> int:
        return 5 if use_fast_large_plate_mode else 2

    def _should_skip_light_video_frame(
        self,
        *,
        state: PlateProcessingState,
        render: bool,
        use_fast_large_plate_mode: bool,
    ) -> bool:
        if render:
            return False
        if not use_fast_large_plate_mode:
            return False
        if state.working_width <= 0 or state.working_height <= 0:
            return False
        if not state.tracks:
            return False
        if any(track.misses > 0 or not track.plate_number for track in state.tracks):
            return False
        return state.frame_index % 2 == 0

    def _should_rerecognize_video(self, state: PlateProcessingState) -> bool:
        if state.frame_index == 1:
            return True

        interval = max(state.recognition_interval, 1)
        since_last = state.frame_index - state.last_recognition_frame
        if state.stream_mode:
            if not state.tracks:
                return since_last >= max(interval, 2)
            if any(track.misses > 0 for track in state.tracks):
                return since_last >= max(interval, 2)
            if any(not track.plate_number for track in state.tracks):
                return since_last >= max(interval * 2, 4)
            if any(self._is_large_recognized_track(track) for track in state.tracks):
                return since_last >= max(interval * 3, 6)
            return since_last >= max(interval, 4)
        if not state.tracks:
            return since_last >= max(interval // 2, 8)
        if any(track.misses > 0 for track in state.tracks):
            return since_last >= max(interval // 2, 8)
        if any(not track.plate_number for track in state.tracks):
            return since_last >= max(interval * 2, 20)
        if any(self._is_large_recognized_track(track) for track in state.tracks):
            return since_last >= max(interval * 2, 24)
        return since_last >= max(interval, 12)

    def _should_probe_new_video_tracks(self, state: PlateProcessingState) -> bool:
        if state.frame_index <= 1:
            return False
        unread_track_count = sum(1 for track in state.tracks if not track.plate_number and track.misses == 0)
        if unread_track_count >= 2:
            probe_interval = max(state.recognition_interval * 2, 24)
        elif unread_track_count >= 1:
            if any(self._is_large_recognized_track(track) for track in state.tracks):
                probe_interval = max(state.recognition_interval * 2, 20)
            else:
                probe_interval = max(state.recognition_interval // 2, 4)
        elif any(self._is_large_recognized_track(track) for track in state.tracks):
            probe_interval = max(state.recognition_interval * 6, 72)
        else:
            probe_interval = max(state.recognition_interval, 12)
        return state.frame_index - state.last_probe_frame >= probe_interval

    def _is_large_recognized_track(self, track: PlateTrack) -> bool:
        if not track.plate_number or track.misses > 0:
            return False
        _, _, width, height = track.bbox
        return width >= 76 and height >= 18 and width * height >= 1400

    def _is_small_target_track_bbox(self, bbox: list[int]) -> bool:
        if len(bbox) != 4:
            return False
        _, _, width, height = bbox
        area = width * height
        return width <= 60 and height <= 18 and 220 <= area <= 960

    def _resolve_video_recognition_max_side(self, source_frame, state: PlateProcessingState) -> int:
        configured_limit = settings.plate_video_recognition_max_side
        if configured_limit <= 0:
            return configured_limit

        frame_height, frame_width = source_frame.shape[:2]
        source_longest_side = max(frame_width, frame_height)
        if source_longest_side <= 960:
            return configured_limit

        active_unread_tracks = [
            track
            for track in state.tracks
            if not track.plate_number and track.misses == 0
        ]
        if any(self._is_small_target_track_bbox(track.bbox) for track in active_unread_tracks):
            return configured_limit
        if any(self._is_large_recognized_track(track) for track in state.tracks):
            return min(configured_limit, 960)
        if active_unread_tracks:
            return min(configured_limit, 1120)
        return min(configured_limit, 1024)

    def _should_use_fast_large_plate_mode(self, state: PlateProcessingState) -> bool:
        return any(self._is_large_recognized_track(track) for track in state.tracks) and not any(
            not track.plate_number and track.misses == 0 for track in state.tracks
        )

    def _should_attempt_active_unread_ocr(
        self,
        *,
        state: PlateProcessingState,
        fresh_detections: list[PlateDetection],
    ) -> bool:
        if self._should_use_fast_large_plate_mode(state):
            return False
        candidates = [
            track
            for track in state.tracks
            if self._is_active_unread_ocr_candidate(track) and track.misses == 0
        ]
        if not candidates:
            return False
        if any(self._is_large_recognized_track(track) for track in state.tracks) and not any(
            self._should_prioritize_unread_ocr_track(track) for track in candidates
        ):
            return False
        return not any(detection.plate_number for detection in fresh_detections)

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

    def _should_collect_unread_video_artifacts(self, state: PlateProcessingState) -> bool:
        if not any(not track.plate_number and track.misses == 0 for track in state.tracks):
            return False
        if state.frame_index <= 1:
            return True
        interval = max(state.recognition_interval, 12)
        return state.frame_index % interval == 0

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
            if crop_width < 20 or crop_height < 8:
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
        prioritized_candidates = sorted(
            candidates.values(),
            key=lambda item: (item.quality_score, item.frame_index),
            reverse=True,
        )[:1]
        for candidate in prioritized_candidates:
            detection = self._recognize_best_unread_candidate(candidate)
            if detection is not None:
                recovered.append(detection)
        return self._deduplicate_plate_detections(recovered)

    def _recover_active_unread_tracks(
        self,
        *,
        source_frame,
        working_frame,
        state: PlateProcessingState,
        fresh_detections: list[PlateDetection],
    ) -> list[PlateDetection]:
        if not self.recognizer.is_available():
            return []

        candidates = self._select_active_unread_ocr_tracks(
            state=state,
            fresh_detections=fresh_detections,
        )
        if not candidates or state.working_width <= 0 or state.working_height <= 0:
            return []

        source_height, source_width = source_frame.shape[:2]
        recovered: list[PlateDetection] = []
        for track in candidates:
            track.last_unread_ocr_frame = state.frame_index
            source_bbox = self._scale_bbox(
                track.bbox,
                state.working_width,
                state.working_height,
                source_width,
                source_height,
            )
            source_bbox = self._expand_active_unread_source_bbox(
                source_bbox,
                image_width=source_width,
                image_height=source_height,
            )
            detection = self._recognize_active_unread_track(
                source_frame=source_frame,
                track=track,
                source_bbox=source_bbox,
            )
            if detection is None:
                continue
            recovered.append(
                PlateDetection(
                    plate_number=detection.plate_number,
                    plate_color=detection.plate_color,
                    vehicle_type=track.vehicle_type,
                    confidence=detection.confidence,
                    bbox=self._scale_bbox(
                        detection.bbox,
                        source_width,
                        source_height,
                        state.working_width,
                        state.working_height,
                    ),
                )
            )

        return self._deduplicate_plate_detections(recovered)

    def _select_active_unread_ocr_tracks(
        self,
        *,
        state: PlateProcessingState,
        fresh_detections: list[PlateDetection],
    ) -> list[PlateTrack]:
        if state.working_width <= 0 or state.working_height <= 0:
            return []

        candidates: list[PlateTrack] = []
        for track in state.tracks:
            if track.plate_number or track.misses > 0:
                continue
            if not self._is_active_unread_ocr_candidate(track):
                continue
            if (
                state.frame_index - track.last_unread_ocr_frame
                < self._active_unread_ocr_interval_for_track(track)
            ):
                continue
            if self._is_likely_side_shadow_bbox_for_recognized_track(track.bbox, state.tracks):
                continue
            if any(
                detection.plate_number and self._compute_iou(track.bbox, detection.bbox) >= 0.35
                for detection in fresh_detections
            ):
                continue
            candidates.append(track)

        candidates.sort(
            key=lambda item: (
                item.unread_observations,
                item.bbox[2] * item.bbox[3],
                item.confidence,
                item.last_seen_frame,
            ),
            reverse=True,
        )
        return candidates[: self._VIDEO_ACTIVE_UNREAD_OCR_MAX_TRACKS]

    def _is_active_unread_ocr_candidate(self, track: PlateTrack) -> bool:
        if track.plate_number:
            return False
        _, _, width, height = track.bbox
        area = width * height
        return width >= 26 and height >= 8 and area >= 220 and width <= 60 and height <= 18 and area <= 900

    def _should_prioritize_unread_ocr_track(self, track: PlateTrack) -> bool:
        if not self._is_active_unread_ocr_candidate(track):
            return False
        return track.unread_observations >= 2 and self._is_small_target_track_bbox(track.bbox)

    def _active_unread_ocr_interval_for_track(self, track: PlateTrack) -> int:
        if self._should_prioritize_unread_ocr_track(track):
            return self._VIDEO_ACTIVE_UNREAD_OCR_INTERVAL
        return max(self._VIDEO_ACTIVE_UNREAD_OCR_INTERVAL + 10, 28)

    def _expand_active_unread_source_bbox(
        self,
        bbox: list[int],
        *,
        image_width: int,
        image_height: int,
    ) -> list[int]:
        x, y, width, height = bbox
        expand_x = int(round(width * 0.18))
        expand_y = int(round(height * 0.42))
        return self._clamp_bbox(
            image_width,
            image_height,
            x - expand_x,
            y - expand_y,
            width + expand_x * 2,
            height + expand_y * 2,
            min_left=0,
            min_top=0,
            max_right=image_width,
            max_bottom=image_height,
        )

    def _recognize_active_unread_track(
        self,
        *,
        source_frame,
        track: PlateTrack,
        source_bbox: list[int],
    ) -> PlateDetection | None:
        crop = self._crop_detection(source_frame, source_bbox)
        if crop is None:
            return None

        fast_crop = self._build_fast_active_unread_ocr_crop(crop)
        crop_result = self._recognize_crop_with_paddleocr(
            fast_crop,
            {"confidence": track.confidence, "kind": "plate"},
            source_bbox,
            confidence_threshold=min(settings.plate_ocr_confidence_threshold, 0.24),
        )
        if crop_result is None:
            return None
        return crop_result

    def _build_fast_active_unread_ocr_crop(self, crop):
        cv2 = self._require_cv2()
        try:
            crop_height, crop_width = crop.shape[:2]
            if crop_width >= 72 and crop_height >= 18:
                return crop
            return cv2.resize(
                crop,
                (max(crop_width * 2, 72), max(crop_height * 2, 20)),
                interpolation=cv2.INTER_CUBIC,
            )
        except Exception:
            logger.debug("Failed to build fast unread OCR crop.", exc_info=True)
            return crop

    def _recognize_best_unread_candidate(self, candidate: BestUnreadCandidate) -> PlateDetection | None:
        crop = candidate.crop
        if crop is None or getattr(crop, "size", 0) == 0:
            return None

        crop_candidates: list[PlateDetection] = []
        for variant in self._build_best_effort_ocr_crop_variants(crop):
            crop_result = self._recognize_crop_with_paddleocr(
                variant,
                {"confidence": 0.82},
                candidate.bbox,
                confidence_threshold=min(settings.plate_ocr_confidence_threshold, 0.24),
            )
            if crop_result is None:
                continue
            crop_candidates.append(
                crop_result
            )

        best_detection = self._pick_best_plate_candidate(crop_candidates)
        if best_detection is None:
            return None
        if best_detection.confidence < 0.30:
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
        video_fast_detector: bool = False,
    ) -> list[PlateDetection]:
        merged_detections: list[PlateDetection] = []
        if self._should_use_detector():
            try:
                detector_detections = self._recognize_detections_via_detector(
                    image_source,
                    aggressive=aggressive,
                    fast_mode=fast_mode,
                    preserve_unread=preserve_unread,
                    video_mode=not isinstance(image_source, (bytes, bytearray)),
                    video_fast_detector=video_fast_detector,
                )
                if detector_detections:
                    merged_detections.extend(detector_detections)
            except (InferenceConfigurationError, InferenceDependencyError):
                if not settings.plate_detector_fallback_to_full_frame:
                    raise
                logger.warning("Plate detector unavailable, falling back to full-frame PaddleOCR.", exc_info=True)

        # With a dedicated plate detector loaded, detector hits are usually the best tradeoff for video latency.
        # Full-frame OCR remains a fallback path when detector results are empty.
        if merged_detections:
            return self._deduplicate_plate_detections(merged_detections)
        if not allow_full_frame_fallback:
            return []

        max_side_override = settings.plate_stream_recognition_max_side if not isinstance(image_source, (bytes, bytearray)) else None
        confidence_threshold = 0.26 if aggressive else None
        try:
            recognized_items = self.recognizer.recognize_all(
                image_source,
                max_side_override=max_side_override,
                aggressive=aggressive,
                heavy_scan=heavy_scan,
                confidence_threshold=confidence_threshold,
            )
        except ValueError:
            if isinstance(image_source, (bytes, bytearray)):
                raise
            logger.warning("Skipping full-frame OCR for the current frame because PaddleOCR raised ValueError.", exc_info=True)
            return []

        full_frame_detections = [
            PlateDetection(
                plate_number=item.plate_number,
                plate_color=self._normalize_plate_color_for_plate(item.plate_color, item.plate_number),
                vehicle_type=VEHICLE_TYPE_UNKNOWN,
                confidence=item.confidence,
                bbox=item.bbox,
            )
            for item in recognized_items
        ]
        return full_frame_detections

    def _should_use_detector(self) -> bool:
        return self.detector.is_available()

    def _detect_image_detector_hits(self, image) -> list[dict]:
        detector = self.detector
        detect_fast_with_vehicles = getattr(detector, "detect_fast_with_vehicles", None)
        if callable(detect_fast_with_vehicles):
            return detect_fast_with_vehicles(image)
        detect_fast = getattr(detector, "detect_fast", None)
        if callable(detect_fast):
            return detect_fast(image)
        return detector.detect(image)

    def _detect_image_detector_hits_detailed(self, image) -> list[dict]:
        detector = self.detector
        detect_image_detailed = getattr(detector, "detect_image_detailed", None)
        if callable(detect_image_detailed):
            return detect_image_detailed(image)
        return detector.detect(image)

    def _augment_image_hits_with_vehicle_detector(self, image, hits: list[dict]) -> list[dict]:
        if not hits:
            hits = []
        if any(str(hit.get("kind", "plate")) == "vehicle" for hit in hits):
            return list(hits)

        detect_vehicle_classes = getattr(self.detector, "detect_vehicle_classes", None)
        if not callable(detect_vehicle_classes):
            return list(hits)

        try:
            vehicle_hits = detect_vehicle_classes(image, fast_mode=False)
        except (InferenceConfigurationError, InferenceDependencyError, ValueError):
            logger.warning("Vehicle detector unavailable during image vehicle-assisted scan.", exc_info=True)
            return list(hits)

        if not vehicle_hits:
            return list(hits)
        return list(hits) + list(vehicle_hits)

    def _select_image_unmatched_vehicle_hits(self, hits: list[dict], plate_hits: list[dict]) -> list[dict]:
        vehicle_hits = [hit for hit in hits if str(hit.get("kind", "plate")) == "vehicle"]
        if not vehicle_hits or not plate_hits:
            return vehicle_hits if vehicle_hits and not plate_hits else []

        unmatched: list[dict] = []
        for vehicle_hit in vehicle_hits:
            vehicle_bbox = list(vehicle_hit.get("bbox", []))
            if len(vehicle_bbox) != 4:
                continue

            matched = False
            for plate_hit in plate_hits:
                plate_bbox = list(plate_hit.get("bbox", []))
                if len(plate_bbox) != 4:
                    continue
                if self._compute_iou(plate_bbox, vehicle_bbox) >= 0.01:
                    matched = True
                    break
                offset_x_ratio, offset_y_ratio = self._compute_bbox_center_offset_ratio(plate_bbox, vehicle_bbox)
                if (
                    self._compute_plate_vehicle_zone_score(plate_bbox, vehicle_bbox) >= 0.34
                    and offset_y_ratio <= 0.9
                    and offset_x_ratio <= 1.4
                ):
                    matched = True
                    break

            if not matched:
                unmatched.append(vehicle_hit)

        unmatched.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        return unmatched

    def _recognize_image_unmatched_vehicle_hits(
        self,
        image,
        hits: list[dict],
        plate_hits: list[dict],
    ) -> list[PlateDetection]:
        unmatched_vehicle_hits = self._select_image_unmatched_vehicle_hits(hits, plate_hits)
        if not unmatched_vehicle_hits:
            logger.info("Image unmatched vehicle scan | candidates=0")
            return []
        logger.info(
            "Image unmatched vehicle scan | candidates=%d sample_bboxes=%s",
            len(unmatched_vehicle_hits),
            [list(hit.get("bbox", [])) for hit in unmatched_vehicle_hits[:2]],
        )
        recognized: list[PlateDetection] = []
        for hit in unmatched_vehicle_hits[:2]:
            detector_detections = self._recognize_detector_hits(
                image,
                [hit],
                fast_mode=False,
                preserve_unread=False,
                allow_ocr_fallback=True,
                video_mode=True,
            )
            if detector_detections:
                logger.info(
                    "Image unmatched vehicle recognized via detector crops | bbox=%s detections=%d",
                    list(hit.get("bbox", [])),
                    len(detector_detections),
                )
                recognized.extend(detector_detections)
                continue

            boosted_detection = self._recognize_image_vehicle_hit_with_local_ocr_boost(image, hit)
            if boosted_detection is not None:
                logger.info(
                    "Image unmatched vehicle recognized via local OCR boost | bbox=%s plate=%s confidence=%.2f",
                    list(hit.get("bbox", [])),
                    boosted_detection.plate_number,
                    boosted_detection.confidence,
                )
                recognized.append(boosted_detection)
            else:
                logger.info(
                    "Image unmatched vehicle detected but OCR failed | bbox=%s confidence=%.2f",
                    list(hit.get("bbox", [])),
                    float(hit.get("confidence", 0.0)),
                )

        return self._deduplicate_plate_detections(recognized)

    def _recognize_image_fast_plate_hits(
        self,
        image,
        plate_hits: list[dict],
    ) -> tuple[list[PlateDetection], list[dict]]:
        recognized: list[PlateDetection] = []
        failed_hits: list[dict] = []

        for hit in plate_hits:
            hit_detections = self._recognize_detector_hits(
                image,
                [hit],
                fast_mode=True,
                preserve_unread=False,
                allow_ocr_fallback=False,
            )
            if hit_detections:
                recognized.extend(hit_detections)
            else:
                failed_hits.append(hit)

        if failed_hits:
            logger.info(
                "Image fast plate OCR misses | failed_hits=%d sample_bboxes=%s",
                len(failed_hits),
                [list(hit.get("bbox", [])) for hit in failed_hits[:2]],
            )
        return self._deduplicate_plate_detections(recognized), failed_hits

    def _recognize_image_failed_plate_hits_with_local_ocr_boost(
        self,
        image,
        failed_plate_hits: list[dict],
    ) -> list[PlateDetection]:
        recognized: list[PlateDetection] = []
        for hit in failed_plate_hits[:2]:
            boosted_detection = self._recognize_image_plate_hit_with_local_ocr_boost(image, hit)
            if boosted_detection is not None:
                logger.info(
                    "Image failed plate hit recovered via local OCR boost | bbox=%s plate=%s confidence=%.2f",
                    list(hit.get("bbox", [])),
                    boosted_detection.plate_number,
                    boosted_detection.confidence,
                )
                recognized.append(boosted_detection)
            else:
                logger.info(
                    "Image failed plate hit stayed unread after local OCR boost | bbox=%s confidence=%.2f",
                    list(hit.get("bbox", [])),
                    float(hit.get("confidence", 0.0)),
                )
        return self._deduplicate_plate_detections(recognized)

    def _recognize_image_plate_hit_with_local_ocr_boost(self, image, hit: dict) -> PlateDetection | None:
        if str(hit.get("kind", "plate")) == "vehicle":
            return None

        candidate_bboxes = self._build_failed_image_plate_crop_bboxes(image, hit)
        if not candidate_bboxes:
            return None

        candidates: list[PlateDetection] = []
        for crop_bbox in candidate_bboxes[:4]:
            if not self._is_reasonable_crop_bbox(crop_bbox) or not self._is_plausible_plate_bbox(crop_bbox):
                continue

            crop = self._crop_detection(image, crop_bbox)
            if crop is None:
                continue

            crop_variants = self._build_best_effort_ocr_crop_variants(crop)
            selected_variants: list[object] = []
            if crop_variants:
                selected_variants.append(crop_variants[0])
            if len(crop_variants) > 1:
                selected_variants.append(crop_variants[-1])
            if len(crop_variants) > 2:
                selected_variants.append(crop_variants[1])

            crop_candidates: list[PlateDetection] = []
            for index, crop_variant in enumerate(selected_variants):
                crop_result = self._recognize_crop_with_paddleocr(
                    crop_variant,
                    hit,
                    crop_bbox,
                    confidence_threshold=min(settings.plate_ocr_confidence_threshold, 0.21),
                    allow_ocr_fallback=index == len(selected_variants) - 1,
                )
                if crop_result is None:
                    continue
                crop_candidates.append(crop_result)
                if crop_result.confidence >= 0.9:
                    break

            best_detection = self._pick_best_plate_candidate(crop_candidates)
            if best_detection is not None:
                candidates.append(best_detection)

        best_detection = self._pick_best_plate_candidate(candidates)
        if best_detection is None or best_detection.confidence < 0.26:
            return None

        vehicle_type = self._vehicle_type_from_hit(hit)
        if vehicle_type == VEHICLE_TYPE_UNKNOWN:
            vehicle_type = self._resolve_vehicle_type_for_plate_detection(
                image,
                best_detection,
                fast_mode=False,
                video_mode=False,
            )
        return best_detection.model_copy(update={"vehicle_type": vehicle_type})

    def _build_failed_image_plate_crop_bboxes(self, image, hit: dict) -> list[list[int]]:
        candidate_bboxes = list(self._resolve_detector_crop_bboxes(image, hit, video_mode=False))
        bbox = list(hit.get("bbox", []))
        if len(bbox) != 4 or not hasattr(image, "shape"):
            return candidate_bboxes

        x, y, width, height = bbox
        if width <= 0 or height <= 0:
            return candidate_bboxes

        image_height, image_width = image.shape[:2]
        extra_candidates: list[list[int]] = []
        if width <= 136 or height <= 42:
            extra_candidates.extend(
                [
                    self._clamp_bbox(
                        image_width,
                        image_height,
                        x - int(round(width * 0.18)),
                        y - int(round(height * 0.34)),
                        int(round(width * 1.38)),
                        int(round(height * 1.86)),
                        min_left=0,
                        min_top=0,
                        max_right=image_width,
                        max_bottom=image_height,
                    ),
                    self._clamp_bbox(
                        image_width,
                        image_height,
                        x - int(round(width * 0.28)),
                        y - int(round(height * 0.46)),
                        int(round(width * 1.56)),
                        int(round(height * 2.12)),
                        min_left=0,
                        min_top=0,
                        max_right=image_width,
                        max_bottom=image_height,
                    ),
                    self._clamp_bbox(
                        image_width,
                        image_height,
                        x - int(round(width * 0.10)),
                        y - int(round(height * 0.48)),
                        int(round(width * 1.20)),
                        int(round(height * 2.18)),
                        min_left=0,
                        min_top=0,
                        max_right=image_width,
                        max_bottom=image_height,
                    ),
                ]
            )

        unique: list[list[int]] = []
        for candidate in candidate_bboxes + extra_candidates:
            if candidate not in unique and self._is_reasonable_crop_bbox(candidate) and self._is_plausible_plate_bbox(candidate):
                unique.append(candidate)
        return unique

    def _recognize_image_vehicle_hit_with_local_ocr_boost(self, image, hit: dict) -> PlateDetection | None:
        if str(hit.get("kind", "plate")) != "vehicle":
            return None

        candidate_bboxes = self._resolve_detector_crop_bboxes(image, hit, video_mode=True)
        if not candidate_bboxes:
            return None

        ranked_bboxes = sorted(
            candidate_bboxes,
            key=lambda bbox: ((bbox[1] + bbox[3] * 0.5), bbox[2] * bbox[3]),
            reverse=True,
        )

        candidates: list[PlateDetection] = []
        for crop_bbox in ranked_bboxes[:2]:
            if not self._is_reasonable_crop_bbox(crop_bbox):
                continue

            crop = self._crop_detection(image, crop_bbox)
            if crop is None:
                continue

            crop_variants = self._build_best_effort_ocr_crop_variants(crop)
            selected_variants: list[object] = []
            if crop_variants:
                selected_variants.append(crop_variants[0])
            if len(crop_variants) > 1:
                selected_variants.append(crop_variants[-1])
            if len(crop_variants) > 2:
                selected_variants.append(crop_variants[1])

            crop_candidates: list[PlateDetection] = []
            for index, crop_variant in enumerate(selected_variants):
                crop_result = self._recognize_crop_with_paddleocr(
                    crop_variant,
                    hit,
                    crop_bbox,
                    confidence_threshold=min(settings.plate_ocr_confidence_threshold, 0.23),
                    allow_ocr_fallback=index == len(selected_variants) - 1,
                )
                if crop_result is None:
                    continue
                crop_candidates.append(crop_result)
                if crop_result.confidence >= 0.9:
                    break

            best_detection = self._pick_best_plate_candidate(crop_candidates)
            if best_detection is not None:
                candidates.append(best_detection)

        best_detection = self._pick_best_plate_candidate(candidates)
        if best_detection is None or best_detection.confidence < 0.26:
            return None

        vehicle_type = self._vehicle_type_from_hit(hit)
        if vehicle_type == VEHICLE_TYPE_UNKNOWN:
            vehicle_type = self._resolve_vehicle_type_for_plate_detection(
                image,
                best_detection,
                fast_mode=False,
                video_mode=True,
            )
        return best_detection.model_copy(update={"vehicle_type": vehicle_type})

    def _detect_video_detector_hits(self, image, *, fast_mode: bool = False) -> list[dict]:
        detector = self.detector
        if fast_mode:
            detect_fast = getattr(detector, "detect_fast", None)
            if callable(detect_fast):
                return detect_fast(image)
        detect_video = getattr(detector, "detect_video", None)
        if callable(detect_video):
            return detect_video(image)
        return detector.detect(image)

    def _recognize_detections_via_detector(
        self,
        image_source,
        aggressive: bool = False,
        fast_mode: bool = False,
        preserve_unread: bool = False,
        video_mode: bool = False,
        video_fast_detector: bool = False,
    ) -> list[PlateDetection]:
        image = self._decode_image_source(image_source)
        detector_hits = (
            self._detect_video_detector_hits(image, fast_mode=video_fast_detector)
            if video_mode
            else self.detector.detect(image)
        )
        if not detector_hits:
            return []
        detector_hits = self._attach_vehicle_type_context_to_hits(detector_hits)

        if video_mode:
            detector_hits = self._select_video_detector_hits_for_recognition(
                detector_hits,
                preserve_unread=preserve_unread,
            )
            if not detector_hits:
                return []

        max_candidates = settings.plate_detector_max_candidates
        if aggressive:
            max_candidates = min(max(max_candidates, 8), 10)
        if fast_mode:
            max_candidates = min(max_candidates, 10)
        if video_mode:
            plate_hits_total, vehicle_hits_total = self._count_detector_hit_kinds(detector_hits)
            if fast_mode:
                max_candidates = min(max_candidates, 4)
            elif plate_hits_total > 0:
                max_candidates = min(max_candidates, 6)
            else:
                max_candidates = min(max_candidates, 4)

        if video_mode:
            plate_hits, vehicle_hits = self._count_detector_hit_kinds(detector_hits[:max_candidates])
            logger.info(
                "Video detector pass | hits=%d plate_hits=%d vehicle_hits=%d fast_mode=%s",
                min(len(detector_hits), max_candidates),
                plate_hits,
                vehicle_hits,
                fast_mode,
            )

        recognized: list[PlateDetection] = []
        for hit in detector_hits[:max_candidates]:
            recognized.extend(
                self._recognize_detector_hit(
                    image,
                    hit,
                    aggressive=aggressive,
                    fast_mode=fast_mode,
                    preserve_unread=preserve_unread,
                    video_mode=video_mode,
                )
            )

        return self._deduplicate_plate_detections(recognized)

    def _select_video_detector_hits_for_recognition(
        self,
        hits: list[dict],
        *,
        preserve_unread: bool,
    ) -> list[dict]:
        plate_hits = [hit for hit in hits if str(hit.get("kind", "plate")) != "vehicle"]
        if plate_hits:
            return plate_hits
        if not preserve_unread:
            return []
        return hits
    def _recognize_detector_hits(
        self,
        image,
        hits: list[dict],
        *,
        fast_mode: bool,
        preserve_unread: bool = False,
        allow_ocr_fallback: bool = True,
        video_mode: bool = False,
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
                    allow_ocr_fallback=allow_ocr_fallback,
                    video_mode=video_mode,
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

        use_fast_probe = self._should_use_fast_video_probe(tracks)
        try:
            detector_hits = self._detect_video_detector_hits(image, fast_mode=use_fast_probe)
        except (InferenceConfigurationError, InferenceDependencyError):
            logger.warning("Plate detector unavailable during video probe scan.", exc_info=True)
            return []

        masked_detector_hits: list[dict] = []
        if tracks and not use_fast_probe:
            masked_image = self._mask_video_track_regions(image, tracks)
            if masked_image is not None:
                try:
                    masked_detector_hits = self._detect_video_detector_hits(masked_image)
                except (InferenceConfigurationError, InferenceDependencyError):
                    logger.warning("Plate detector unavailable during masked video probe scan.", exc_info=True)

        unmatched: list[dict] = []
        for hit in self._merge_detector_hits_for_probe(detector_hits, masked_detector_hits):
            bbox = list(hit.get("bbox", []))
            if not self._is_reasonable_crop_bbox(bbox):
                continue
            if self._is_likely_side_shadow_bbox_for_recognized_track(bbox, tracks):
                continue

            matched = False
            for track in tracks:
                if self._is_same_unread_track_bbox(bbox, track.bbox):
                    matched = True
                    break

            if not matched:
                unmatched.append(hit)
            if len(unmatched) >= max_hits:
                break

        plate_hits, vehicle_hits = self._count_detector_hit_kinds(detector_hits)
        masked_plate_hits, masked_vehicle_hits = self._count_detector_hit_kinds(masked_detector_hits)
        unmatched_plate_hits, unmatched_vehicle_hits = self._count_detector_hit_kinds(unmatched)
        logger.info(
            "Video probe hits | total=%d plate_hits=%d vehicle_hits=%d masked_total=%d masked_plate=%d masked_vehicle=%d unmatched=%d unmatched_plate=%d unmatched_vehicle=%d",
            len(detector_hits),
            plate_hits,
            vehicle_hits,
            len(masked_detector_hits),
            masked_plate_hits,
            masked_vehicle_hits,
            len(unmatched),
            unmatched_plate_hits,
            unmatched_vehicle_hits,
        )

        return unmatched

    def _should_use_fast_video_probe(self, tracks: list[PlateTrack]) -> bool:
        if any(not track.plate_number and track.misses == 0 for track in tracks):
            return False
        return any(self._is_large_recognized_track(track) for track in tracks)

    def _recognize_detector_hit(
        self,
        image,
        hit: dict,
        aggressive: bool = False,
        fast_mode: bool = False,
        preserve_unread: bool = False,
        allow_ocr_fallback: bool = True,
        video_mode: bool = False,
    ) -> list[PlateDetection]:
        recognized: list[PlateDetection] = []
        candidate_bboxes = self._resolve_detector_crop_bboxes(image, hit, video_mode=video_mode)
        confidence_threshold = min(
            settings.plate_confidence_threshold,
            0.28 if aggressive else settings.plate_confidence_threshold,
        )
        hit_kind = str(hit.get("kind", "plate"))
        if fast_mode:
            if video_mode and hit_kind == "vehicle":
                candidate_bboxes = candidate_bboxes[:3]
            else:
                candidate_bboxes = candidate_bboxes[:1]

        unread_fallback_bbox: list[int] | None = None
        crop_allow_ocr_fallback = allow_ocr_fallback
        if video_mode and fast_mode:
            crop_allow_ocr_fallback = False
        for crop_bbox in candidate_bboxes:
            if not self._is_reasonable_crop_bbox(crop_bbox):
                continue
            if hit_kind == "plate" and not self._is_plausible_plate_bbox(crop_bbox):
                continue
            crop = self._crop_detection(image, crop_bbox)
            if crop is None:
                continue
            if unread_fallback_bbox is None:
                unread_fallback_bbox = list(crop_bbox)

            crop_candidates: list[PlateDetection] = []
            crop_variants = self._build_ocr_crop_variants(crop)
            if fast_mode:
                ocr_variants = crop_variants[:1]
            else:
                use_extra_ocr_variant = self._should_try_extra_fast_ocr_variant(crop_bbox)
                ocr_variants = crop_variants[:2] if use_extra_ocr_variant else crop_variants[:1]

            for crop_variant in ocr_variants:
                crop_result = self._recognize_crop_with_paddleocr(
                    crop_variant,
                    hit,
                    crop_bbox,
                    allow_ocr_fallback=crop_allow_ocr_fallback,
                )
                if crop_result is not None:
                    crop_candidates.append(crop_result)
                    if fast_mode and crop_result.confidence >= 0.88:
                        break

            best_detection = self._pick_best_plate_candidate(crop_candidates)
            if best_detection is not None:
                vehicle_type = self._vehicle_type_from_hit(hit)
                if vehicle_type == VEHICLE_TYPE_UNKNOWN:
                    vehicle_type = self._resolve_vehicle_type_for_plate_detection(
                        image,
                        best_detection,
                        fast_mode=fast_mode,
                        video_mode=video_mode,
                    )
                recognized.append(
                    best_detection.model_copy(
                        update={"vehicle_type": vehicle_type}
                    )
                )
            elif preserve_unread and not (video_mode and hit_kind == "vehicle"):
                recognized.append(
                    PlateDetection(
                        plate_number="",
                        plate_color="??",
                        vehicle_type=self._vehicle_type_from_hit(hit),
                        confidence=float(hit["confidence"]),
                        bbox=list(crop_bbox),
                    )
                )

        if not recognized and preserve_unread and video_mode and hit_kind == "vehicle" and unread_fallback_bbox is not None:
            fallback_bbox = self._pick_video_vehicle_unread_bbox(candidate_bboxes, unread_fallback_bbox)
            recognized.append(
                PlateDetection(
                    plate_number="",
                    plate_color="??",
                    vehicle_type=self._vehicle_type_from_hit(hit),
                    confidence=float(hit["confidence"]),
                    bbox=list(fallback_bbox),
                )
            )

        return self._deduplicate_plate_detections(recognized)

    def _vehicle_type_from_hit(self, hit: dict) -> str:
        return self._normalize_vehicle_type_from_label(str(hit.get("vehicle_type") or hit.get("label", "")))

    def _resolve_vehicle_type_for_plate_detection(
        self,
        image,
        detection: PlateDetection,
        *,
        fast_mode: bool,
        video_mode: bool,
    ) -> str:
        vehicle_type = self._classify_vehicle_type_near_plate_bbox(
            image,
            detection.bbox,
            fast_mode=fast_mode,
            video_mode=video_mode,
        )
        if self._should_default_small_plate_to_car(detection.bbox, video_mode=video_mode):
            vehicle_type = self._pick_better_vehicle_type(vehicle_type, VEHICLE_TYPE_CAR)

        refined_vehicle_type = self._classify_vehicle_type_with_classifier(
            image,
            detection.bbox,
            base_vehicle_type=vehicle_type,
            fast_mode=fast_mode,
            video_mode=video_mode,
        )
        if refined_vehicle_type != VEHICLE_TYPE_UNKNOWN:
            return refined_vehicle_type
        return vehicle_type

    def _classify_vehicle_type_with_classifier(
        self,
        image,
        plate_bbox: list[int],
        *,
        base_vehicle_type: str,
        fast_mode: bool,
        video_mode: bool,
    ) -> str:
        classifier = getattr(self, "vehicle_classifier", None)
        if classifier is None or not classifier.is_available():
            return VEHICLE_TYPE_UNKNOWN

        normalized_base_vehicle_type = self._normalize_vehicle_type_from_label(base_vehicle_type)
        if normalized_base_vehicle_type not in {VEHICLE_TYPE_CAR, VEHICLE_TYPE_TRUCK, VEHICLE_TYPE_UNKNOWN}:
            return VEHICLE_TYPE_UNKNOWN

        vehicle_crop = self._crop_vehicle_candidate_for_classifier(
            image,
            plate_bbox,
            video_mode=video_mode,
        )
        if vehicle_crop is None:
            return VEHICLE_TYPE_UNKNOWN

        try:
            prediction = classifier.classify(vehicle_crop)
        except (InferenceConfigurationError, InferenceDependencyError, ValueError):
            logger.warning("Vehicle fine-classifier unavailable while classifying plate context.", exc_info=True)
            return VEHICLE_TYPE_UNKNOWN

        if prediction is None:
            return VEHICLE_TYPE_UNKNOWN
        predicted_vehicle_type = self._normalize_vehicle_type_from_label(prediction.label)
        if not self._should_accept_vehicle_classifier_prediction(
            normalized_base_vehicle_type,
            predicted_vehicle_type,
            confidence=prediction.confidence,
        ):
            return VEHICLE_TYPE_UNKNOWN
        return predicted_vehicle_type

    def _crop_vehicle_candidate_for_classifier(
        self,
        image,
        plate_bbox: list[int],
        *,
        video_mode: bool,
    ):
        if len(plate_bbox) != 4:
            return None
        x, y, width, height = plate_bbox
        if width <= 0 or height <= 0:
            return None
        image_height, image_width = image.shape[:2]

        if video_mode:
            expand_left = 3.8
            expand_right = 4.2
            expand_top = 7.2
            expand_bottom = 4.8
        else:
            expand_left = 4.2
            expand_right = 4.6
            expand_top = 8.0
            expand_bottom = 5.2

        left = max(int(round(x - width * expand_left)), 0)
        top = max(int(round(y - height * expand_top)), 0)
        right = min(int(round(x + width * (1 + expand_right))), image_width)
        bottom = min(int(round(y + height * (1 + expand_bottom))), image_height)
        if right - left < max(width * 2, 32) or bottom - top < max(height * 3, 32):
            return None
        return image[top:bottom, left:right]

    def _classify_vehicle_type_near_plate_bbox(
        self,
        image,
        plate_bbox: list[int],
        *,
        fast_mode: bool,
        video_mode: bool,
    ) -> str:
        detect_vehicle_classes = getattr(self.detector, "detect_vehicle_classes", None)
        if not callable(detect_vehicle_classes):
            return VEHICLE_TYPE_UNKNOWN

        context = self._crop_vehicle_context_region(image, plate_bbox, video_mode=video_mode)
        if context is None:
            return VEHICLE_TYPE_UNKNOWN
        roi_image, roi_left, roi_top = context
        roi_plate_bbox = [
            plate_bbox[0] - roi_left,
            plate_bbox[1] - roi_top,
            plate_bbox[2],
            plate_bbox[3],
        ]

        try:
            vehicle_hits = detect_vehicle_classes(roi_image, fast_mode=fast_mode)
        except (InferenceConfigurationError, InferenceDependencyError, ValueError):
            logger.warning("Vehicle ROI coarse-classification detector unavailable.", exc_info=True)
            return VEHICLE_TYPE_UNKNOWN

        best_type = VEHICLE_TYPE_UNKNOWN
        best_score = 0.0
        for vehicle_hit in vehicle_hits:
            vehicle_type = self._normalize_vehicle_type_from_label(str(vehicle_hit.get("label", "")))
            if vehicle_type == VEHICLE_TYPE_UNKNOWN:
                continue
            vehicle_bbox = list(vehicle_hit.get("bbox", []))
            if len(vehicle_bbox) != 4:
                continue
            iou = self._compute_iou(roi_plate_bbox, vehicle_bbox)
            offset_x_ratio, offset_y_ratio = self._compute_bbox_center_offset_ratio(roi_plate_bbox, vehicle_bbox)
            center_score = max(0.0, 1.0 - offset_x_ratio * 0.35 - offset_y_ratio * 0.2)
            zone_score = self._compute_plate_vehicle_zone_score(roi_plate_bbox, vehicle_bbox)
            score = zone_score * 1.25 + center_score * 0.55 + iou * 0.45 + float(vehicle_hit.get("confidence", 0.0)) * 0.25
            if score > best_score and (zone_score >= 0.38 or center_score >= 0.50 or iou >= 0.01):
                best_score = score
                best_type = vehicle_type
        return best_type

    def _should_accept_vehicle_classifier_prediction(
        self,
        base_vehicle_type: str,
        predicted_vehicle_type: str,
        *,
        confidence: float,
    ) -> bool:
        if predicted_vehicle_type == VEHICLE_TYPE_UNKNOWN:
            return False

        normalized_base = self._normalize_vehicle_type_from_label(base_vehicle_type)
        normalized_predicted = self._normalize_vehicle_type_from_label(predicted_vehicle_type)

        if normalized_predicted == VEHICLE_TYPE_BUS:
            return False

        if normalized_base == VEHICLE_TYPE_CAR:
            if normalized_predicted == VEHICLE_TYPE_JEEP:
                return confidence >= 0.72
            if normalized_predicted == VEHICLE_TYPE_PICKUP:
                return confidence >= 0.84
            if normalized_predicted == VEHICLE_TYPE_MOTORCYCLE:
                return confidence >= 0.80
            return False

        if normalized_base == VEHICLE_TYPE_TRUCK:
            return normalized_predicted == VEHICLE_TYPE_CRANE and confidence >= 0.76

        if normalized_base == VEHICLE_TYPE_UNKNOWN:
            return normalized_predicted in {
                VEHICLE_TYPE_JEEP,
                VEHICLE_TYPE_PICKUP,
                VEHICLE_TYPE_MOTORCYCLE,
                VEHICLE_TYPE_CRANE,
            } and confidence >= 0.82

        return False

    def _crop_vehicle_context_region(
        self,
        image,
        plate_bbox: list[int],
        *,
        video_mode: bool,
    ) -> tuple[object, int, int] | None:
        if len(plate_bbox) != 4:
            return None
        x, y, width, height = plate_bbox
        if width <= 0 or height <= 0:
            return None
        image_height, image_width = image.shape[:2]

        if video_mode:
            expand_left = 4.2
            expand_right = 4.8
            expand_top = 8.5
            expand_bottom = 6.2
        else:
            expand_left = 4.8
            expand_right = 5.4
            expand_top = 9.0
            expand_bottom = 6.8

        left = max(int(round(x - width * expand_left)), 0)
        top = max(int(round(y - height * expand_top)), 0)
        right = min(int(round(x + width * (1 + expand_right))), image_width)
        bottom = min(int(round(y + height * (1 + expand_bottom))), image_height)
        if right - left < max(width * 2, 24) or bottom - top < max(height * 3, 24):
            return None
        return image[top:bottom, left:right], left, top

    def _should_default_small_plate_to_car(self, plate_bbox: list[int], *, video_mode: bool) -> bool:
        if not video_mode or len(plate_bbox) != 4:
            return False
        _, _, width, height = plate_bbox
        area = width * height
        return width <= 80 and height <= 26 and 180 <= area <= 1700

    def _fallback_vehicle_type_for_bbox(self, vehicle_type: str, bbox: list[int], *, video_mode: bool) -> str:
        normalized = self._normalize_vehicle_type_from_label(vehicle_type)
        if normalized != VEHICLE_TYPE_UNKNOWN:
            return normalized
        if self._should_default_small_plate_to_car(bbox, video_mode=video_mode):
            return VEHICLE_TYPE_CAR
        if video_mode and len(bbox) == 4:
            _, _, width, height = bbox
            area = width * height
            if width <= 116 and height <= 34 and 220 <= area <= 3200:
                return VEHICLE_TYPE_CAR
        return VEHICLE_TYPE_UNKNOWN

    def _augment_hits_with_secondary_vehicle_context(
        self,
        image,
        hits: list[dict],
        *,
        fast_mode: bool,
    ) -> list[dict]:
        if not hits:
            return []
        if any(str(hit.get("kind", "plate")) == "vehicle" for hit in hits):
            return list(hits)
        if not any(str(hit.get("kind", "plate")) == "plate" for hit in hits):
            return list(hits)

        detect_vehicle_classes = getattr(self.detector, "detect_vehicle_classes", None)
        if not callable(detect_vehicle_classes):
            return list(hits)

        try:
            vehicle_hits = detect_vehicle_classes(image, fast_mode=fast_mode)
        except (InferenceConfigurationError, InferenceDependencyError, ValueError):
            logger.warning("Vehicle coarse-classification detector unavailable.", exc_info=True)
            return list(hits)

        if not vehicle_hits:
            return list(hits)
        return list(hits) + list(vehicle_hits)

    def _attach_vehicle_type_context_to_hits(self, hits: list[dict]) -> list[dict]:
        vehicle_hits = [
            hit for hit in hits if self._normalize_vehicle_type_from_label(str(hit.get("label", ""))) != VEHICLE_TYPE_UNKNOWN
        ]
        if not vehicle_hits:
            return [dict(hit) for hit in hits]

        enriched_hits: list[dict] = []
        for raw_hit in hits:
            hit = dict(raw_hit)
            hit["vehicle_type"] = self._resolve_vehicle_type_for_hit(hit, vehicle_hits)
            enriched_hits.append(hit)
        return enriched_hits

    def _resolve_vehicle_type_for_hit(self, hit: dict, vehicle_hits: list[dict]) -> str:
        direct_vehicle_type = self._normalize_vehicle_type_from_label(str(hit.get("label", "")))
        if direct_vehicle_type != VEHICLE_TYPE_UNKNOWN:
            return direct_vehicle_type

        bbox = list(hit.get("bbox", []))
        if len(bbox) != 4:
            return VEHICLE_TYPE_UNKNOWN

        best_type = VEHICLE_TYPE_UNKNOWN
        best_score = 0.0
        for vehicle_hit in vehicle_hits:
            vehicle_type = self._normalize_vehicle_type_from_label(str(vehicle_hit.get("label", "")))
            if vehicle_type == VEHICLE_TYPE_UNKNOWN:
                continue
            vehicle_bbox = list(vehicle_hit.get("bbox", []))
            if len(vehicle_bbox) != 4:
                continue
            iou = self._compute_iou(bbox, vehicle_bbox)
            offset_x_ratio, offset_y_ratio = self._compute_bbox_center_offset_ratio(bbox, vehicle_bbox)
            center_score = max(0.0, 1.0 - offset_x_ratio * 0.35 - offset_y_ratio * 0.2)
            zone_score = self._compute_plate_vehicle_zone_score(bbox, vehicle_bbox)
            score = (
                iou * 1.8
                + center_score * 0.75
                + zone_score * 0.9
                + float(vehicle_hit.get("confidence", 0.0)) * 0.15
            )
            if score > best_score and (iou >= 0.01 or center_score >= 0.44 or zone_score >= 0.58):
                best_score = score
                best_type = vehicle_type
        return best_type

    def _compute_plate_vehicle_zone_score(self, plate_bbox: list[int], vehicle_bbox: list[int]) -> float:
        if len(plate_bbox) != 4 or len(vehicle_bbox) != 4:
            return 0.0
        px, py, pw, ph = plate_bbox
        vx, vy, vw, vh = vehicle_bbox
        if vw <= 0 or vh <= 0:
            return 0.0

        plate_center_x = px + pw / 2
        plate_center_y = py + ph / 2
        x_ratio = (plate_center_x - vx) / max(vw, 1)
        y_ratio = (plate_center_y - vy) / max(vh, 1)
        width_ratio = pw / max(vw, 1)

        horizontal_score = max(0.0, 1.0 - abs(x_ratio - 0.5) / 0.38)
        vertical_score = max(0.0, 1.0 - abs(y_ratio - 0.68) / 0.34)
        width_score = max(0.0, 1.0 - abs(width_ratio - 0.30) / 0.28)

        inside_x = -0.08 <= x_ratio <= 1.08
        inside_y = 0.24 <= y_ratio <= 1.08
        containment_bonus = 0.18 if inside_x and inside_y else 0.0

        return min(
            1.0,
            horizontal_score * 0.34
            + vertical_score * 0.46
            + width_score * 0.20
            + containment_bonus,
        )

    def _normalize_vehicle_type_from_label(self, label: str) -> str:
        raw = str(label or "").strip()
        if not raw:
            return VEHICLE_TYPE_UNKNOWN
        if raw in {
            VEHICLE_TYPE_CAR,
            VEHICLE_TYPE_TRUCK,
            VEHICLE_TYPE_BUS,
            VEHICLE_TYPE_CRANE,
            VEHICLE_TYPE_JEEP,
            VEHICLE_TYPE_PICKUP,
            VEHICLE_TYPE_MOTORCYCLE,
            VEHICLE_TYPE_UNKNOWN,
        }:
            return raw
        normalized = raw.lower()
        if normalized in {"car", "sedan", "suv", "coupe", "hatchback", "van"}:
            return VEHICLE_TYPE_CAR
        if normalized in {"jeep"}:
            return VEHICLE_TYPE_JEEP
        if normalized in {"pickup", "pick-up"}:
            return VEHICLE_TYPE_PICKUP
        if normalized in {"truck", "lorry"}:
            return VEHICLE_TYPE_TRUCK
        if normalized in {"bus", "coach"}:
            return VEHICLE_TYPE_BUS
        if normalized in {"crane"}:
            return VEHICLE_TYPE_CRANE
        if normalized in {"motorcycle", "motorbike", "bike"}:
            return VEHICLE_TYPE_MOTORCYCLE
        return VEHICLE_TYPE_UNKNOWN

    def _recognize_crop_with_paddleocr(
        self,
        crop,
        hit: dict,
        crop_bbox: list[int],
        confidence_threshold: float | None = None,
        allow_ocr_fallback: bool = True,
    ) -> PlateDetection | None:
        if not self.recognizer.is_available():
            return None

        try:
            result = self.recognizer.recognize(
                crop,
                confidence_threshold=confidence_threshold,
                allow_ocr_fallback=allow_ocr_fallback,
            )
        except (InferenceConfigurationError, InferenceDependencyError):
            logger.warning("PaddleOCR unavailable while reading detector crop.", exc_info=True)
            return None
        except ValueError:
            logger.warning("Skipping detector crop because PaddleOCR raised ValueError.", exc_info=True)
            return None
        except RuntimeError:
            logger.warning("Skipping detector crop because PaddleOCR runtime crashed on this crop.", exc_info=True)
            reset_runtime = getattr(self.recognizer, "reset_runtime", None)
            if callable(reset_runtime):
                try:
                    reset_runtime()
                except Exception:
                    logger.debug("Failed to reset PaddleOCR runtime after crop crash.", exc_info=True)
            return None

        if result is None or not result.plate_number:
            return None

        return PlateDetection(
            plate_number=result.plate_number,
            plate_color=self._normalize_plate_color_for_plate(result.plate_color, result.plate_number),
            vehicle_type=VEHICLE_TYPE_UNKNOWN,
            confidence=min(1.0, result.confidence * 0.82 + float(hit["confidence"]) * 0.18),
            bbox=list(crop_bbox),
        )
    def _is_reasonable_crop_bbox(self, bbox: list[int]) -> bool:
        if len(bbox) != 4:
            return False
        _, _, width, height = bbox
        if width < 12 or height < 6:
            return False
        if width * height < 96:
            return False
        return True

    def _is_plausible_plate_bbox(self, bbox: list[int]) -> bool:
        if len(bbox) != 4:
            return False
        _, _, width, height = bbox
        aspect_ratio = width / max(height, 1)
        return 1.5 <= aspect_ratio <= 7.8

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

    def _resolve_detector_crop_bboxes(self, image, hit: dict, *, video_mode: bool = False) -> list[list[int]]:
        kind = str(hit.get("kind", "plate"))
        bbox = list(hit["bbox"])
        if kind != "vehicle":
            return self._plate_bbox_to_crop_bboxes(image, bbox)
        if video_mode:
            return self._video_vehicle_bbox_to_plate_bboxes(image, bbox)
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
            self._clamp_bbox(
                image_width,
                image_height,
                x - int(round(width * 0.16)),
                y - int(round(height * 0.28)),
                int(round(width * 1.34)),
                int(round(height * 1.62)),
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

            dark_boost = cv2.addWeighted(enhanced, 1.28, cv2.GaussianBlur(enhanced, (0, 0), 1.2), -0.28, 8.0)
            variants.append(cv2.cvtColor(dark_boost, cv2.COLOR_GRAY2BGR))

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
            if crop_width <= 96 or crop_height <= 28:
                boosted = cv2.resize(
                    dark_boost,
                    (max(crop_width * 3, 96), max(crop_height * 3, 24)),
                    interpolation=cv2.INTER_CUBIC,
                )
                boosted = cv2.bilateralFilter(boosted, 5, 28, 28)
                variants.append(cv2.cvtColor(boosted, cv2.COLOR_GRAY2BGR))
        except Exception:
            logger.debug("Failed to build enhanced OCR crop variant.", exc_info=True)

        return variants

    def _should_try_extra_fast_ocr_variant(self, crop_bbox: list[int]) -> bool:
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

    def _video_vehicle_bbox_to_plate_bboxes(self, image, vehicle_bbox: list[int]) -> list[list[int]]:
        base_candidates = self._vehicle_bbox_to_plate_bboxes(image, vehicle_bbox)
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

        video_candidates = list(base_candidates)
        for width_ratio, height_ratio, x_ratio, y_ratio in [
            (0.84, 0.24, 0.50, 0.52),
            (0.88, 0.26, 0.50, 0.58),
            (0.78, 0.22, 0.50, 0.64),
            (0.72, 0.20, 0.36, 0.58),
            (0.72, 0.20, 0.64, 0.58),
        ]:
            plate_width = max(int(round(padded_width * width_ratio)), 20)
            plate_height = max(int(round(padded_height * height_ratio)), 12)
            plate_left = vehicle_left + int(round((padded_width - plate_width) * x_ratio))
            plate_top = vehicle_top + int(round(padded_height * y_ratio))
            video_candidates.append(
                self._clamp_bbox(
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
            )

        unique: list[list[int]] = []
        for candidate in video_candidates:
            if candidate not in unique:
                unique.append(candidate)
        return unique

    def _pick_video_vehicle_unread_bbox(
        self,
        candidate_bboxes: list[list[int]],
        default_bbox: list[int],
    ) -> list[int]:
        best_bbox = list(default_bbox)
        best_score = float("-inf")
        for bbox in candidate_bboxes:
            if not self._is_reasonable_crop_bbox(bbox):
                continue
            x, y, width, height = bbox
            score = (y + height * 0.5) + (width * height) / 1000.0
            if score > best_score:
                best_score = score
                best_bbox = list(bbox)
        return best_bbox

    def _count_detector_hit_kinds(self, hits: list[dict]) -> tuple[int, int]:
        plate_hits = 0
        vehicle_hits = 0
        for hit in hits:
            if str(hit.get("kind", "plate")) == "vehicle":
                vehicle_hits += 1
            else:
                plate_hits += 1
        return plate_hits, vehicle_hits

    def _log_image_detector_stage(self, stage: str, hits: list[dict]) -> None:
        plate_hits, vehicle_hits = self._count_detector_hit_kinds(hits)
        logger.info(
            "Image detector stage %s | total=%d plate_hits=%d vehicle_hits=%d sample_bboxes=%s",
            stage,
            len(hits),
            plate_hits,
            vehicle_hits,
            [list(hit.get("bbox", [])) for hit in hits[:3]],
        )

    def _should_run_aggressive_image_full_frame_fallback(self, image, hits: list[dict]) -> bool:
        if hits:
            return False
        if not hasattr(image, "shape"):
            return True
        image_height, image_width = image.shape[:2]
        return max(image_height, image_width) <= 1280

    def _merge_detector_hits_for_probe(self, primary_hits: list[dict], masked_hits: list[dict]) -> list[dict]:
        merged: list[dict] = []
        for hit in list(primary_hits) + list(masked_hits):
            bbox = list(hit.get("bbox", []))
            if len(bbox) != 4:
                continue
            duplicate = False
            for existing in merged:
                if self._compute_iou(bbox, list(existing.get("bbox", []))) >= 0.45:
                    duplicate = True
                    break
            if not duplicate:
                merged.append(hit)
        return merged

    def _mask_video_track_regions(self, image, tracks: list[PlateTrack]):
        if not tracks:
            return None
        cv2 = self._require_cv2()
        masked = image.copy()
        image_height, image_width = masked.shape[:2]
        for track in tracks:
            x, y, width, height = track.bbox
            expand_x = max(int(round(width * 0.35)), 6)
            expand_y = max(int(round(height * 0.55)), 4)
            left = max(x - expand_x, 0)
            top = max(y - expand_y, 0)
            right = min(x + width + expand_x, image_width)
            bottom = min(y + height + expand_y, image_height)
            if right <= left or bottom <= top:
                continue
            cv2.rectangle(masked, (left, top), (right, bottom), (0, 0, 0), -1)
        return masked

    def _count_unread_detections(self, detections: list[PlateDetection]) -> int:
        return sum(1 for detection in detections if not detection.plate_number)

    def _log_video_frame_summary(
        self,
        *,
        frame_index: int,
        phase: str,
        detections: list[PlateDetection],
        tracks: list[PlateTrack],
    ) -> None:
        unread_count = self._count_unread_detections(detections)
        track_unread_count = sum(1 for track in tracks if not track.plate_number)
        logger.info(
            "Video frame %d | phase=%s detections=%d unread=%d tracks=%d unread_tracks=%d",
            frame_index,
            phase,
            len(detections),
            unread_count,
            len(tracks),
            track_unread_count,
        )

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

    def _normalize_plate_color_label(self, plate_color: str) -> str:
        normalized = str(plate_color or "").strip()
        if not normalized:
            return PLATE_COLOR_UNKNOWN

        lower = normalized.lower()
        if PLATE_COLOR_BLUE in normalized or lower == "blue":
            return PLATE_COLOR_BLUE
        if PLATE_COLOR_YELLOW in normalized or lower == "yellow":
            return PLATE_COLOR_YELLOW
        if PLATE_COLOR_GREEN in normalized or lower == "green":
            return PLATE_COLOR_GREEN
        if PLATE_COLOR_WHITE in normalized or lower == "white":
            return PLATE_COLOR_WHITE
        if PLATE_COLOR_BLACK in normalized or lower == "black":
            return PLATE_COLOR_BLACK
        if PLATE_COLOR_UNKNOWN in normalized or lower == "unknown":
            return PLATE_COLOR_UNKNOWN
        return normalized
    def _normalize_plate_color_for_plate(self, plate_color: str, plate_number: str) -> str:
        normalized = self._normalize_plate_color_label(plate_color)
        plate_length = len(plate_number or "")

        if plate_length == 8 and normalized in {PLATE_COLOR_BLUE, PLATE_COLOR_YELLOW, PLATE_COLOR_BLACK, PLATE_COLOR_WHITE}:
            return PLATE_COLOR_GREEN
        if plate_length == 7 and normalized in {PLATE_COLOR_BLACK, PLATE_COLOR_WHITE}:
            return PLATE_COLOR_BLUE
        if plate_length == 8 and normalized in {PLATE_COLOR_BLUE, PLATE_COLOR_YELLOW}:
            return PLATE_COLOR_GREEN
        if plate_length == 7 and normalized == PLATE_COLOR_GREEN:
            return PLATE_COLOR_YELLOW
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
                vehicle_type=detection.vehicle_type,
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
                vehicle_type=self._pick_better_vehicle_type(current.vehicle_type, normalized.vehicle_type),
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
        if len(plate_number or "") == 8 and candidate == PLATE_COLOR_GREEN:
            return candidate
        if len(plate_number or "") == 7 and candidate == PLATE_COLOR_YELLOW:
            return candidate
        if current == PLATE_COLOR_UNKNOWN:
            return candidate
        return current

    def _pick_better_vehicle_type(self, current_type: str, next_type: str) -> str:
        current = self._normalize_vehicle_type_from_label(current_type)
        candidate = self._normalize_vehicle_type_from_label(next_type)
        if current == candidate:
            return current
        if current == VEHICLE_TYPE_UNKNOWN:
            return candidate
        if current == VEHICLE_TYPE_CAR and candidate in {
            VEHICLE_TYPE_JEEP,
            VEHICLE_TYPE_PICKUP,
            VEHICLE_TYPE_MOTORCYCLE,
        }:
            return candidate
        if current == VEHICLE_TYPE_TRUCK and candidate == VEHICLE_TYPE_CRANE:
            return candidate
        return current
    def _score_plate_candidate(self, detection: PlateDetection) -> float:
        plate_number = detection.plate_number or ""
        suffix = plate_number[2:]
        score = detection.confidence

        if len(plate_number) in (7, 8):
            score += 0.03
        if detection.plate_color != PLATE_COLOR_UNKNOWN:
            score += 0.02
        if len(plate_number) == 8 and detection.plate_color == PLATE_COLOR_GREEN:
            score += 0.05
        if len(plate_number) == 7 and detection.plate_color == PLATE_COLOR_YELLOW:
            score += 0.03
        if len(plate_number) == 7 and detection.plate_color == PLATE_COLOR_GREEN:
            score -= 0.08
        if plate_number:
            ambiguous_count = sum(1 for char in plate_number if char in {"1", "I"})
            score -= ambiguous_count * 0.012
            if ambiguous_count >= max(3, len(plate_number) - 2):
                score -= 0.04
        if re.fullmatch(r"[1I]{5,6}", suffix):
            score -= 0.25

        return score
    def _deduplicate_plate_detections(self, detections: list[PlateDetection]) -> list[PlateDetection]:
        kept: list[PlateDetection] = []
        ranked_detections = sorted(
            detections,
            key=lambda item: (self._score_plate_candidate(item), item.confidence, item.bbox[2] * item.bbox[3]),
            reverse=True,
        )
        for detection in ranked_detections:
            duplicate = False
            for existing in kept:
                if detection.plate_number == existing.plate_number and self._compute_iou(detection.bbox, existing.bbox) >= 0.1:
                    duplicate = True
                    break
                if self._compute_iou(detection.bbox, existing.bbox) >= 0.6:
                    duplicate = True
                    break
                if self._should_merge_similar_plate_detections(detection, existing):
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
            if tracked and self._should_reject_stale_tracked_plate(track, frame_index):
                tracked = False
                track.misses += 2
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
        track.last_tracking_score = float(max_score)
        return True

    def _should_reject_stale_tracked_plate(self, track: PlateTrack, frame_index: int) -> bool:
        if not track.plate_number:
            return False

        stale_frames = frame_index - track.last_recognized_frame
        if stale_frames <= 8:
            return False

        score = track.last_tracking_score
        if stale_frames >= 20 and score < 0.60:
            return True
        if stale_frames >= 12 and score < 0.66:
            return True
        return False

    def _should_rerecognize(
        self,
        tracks: list[PlateTrack],
        frame_index: int,
        last_recognition_frame: int,
        *,
        recognition_interval: int,
        force_when_no_tracks: bool,
    ) -> bool:
        if frame_index == 1:
            return True
        if not tracks:
            if force_when_no_tracks:
                return True
            interval = max(recognition_interval, 1)
            return frame_index - last_recognition_frame >= interval
        if any(track.misses > 0 for track in tracks):
            return True
        interval = max(recognition_interval, 1)
        return frame_index - last_recognition_frame >= interval

    def _should_skip_stream_frame_for_rate_limit(
        self,
        *,
        last_sent_at: float,
        current_time: float,
        min_interval: float,
    ) -> bool:
        if min_interval <= 0 or last_sent_at <= 0:
            return False
        return current_time - last_sent_at < min_interval

    def _stream_read_failure_retry_limit(self) -> int:
        return max(settings.plate_stream_max_fps * 2, 8)

    def _stream_read_failure_retry_sleep_seconds(self, min_interval: float) -> float:
        if min_interval > 0:
            return min(max(min_interval * 0.5, 0.03), 0.12)
        return 0.05

    def _stream_recognition_submit_interval_seconds(self, min_interval: float) -> float:
        base_interval = min_interval if min_interval > 0 else 0.125
        return min(max(base_interval * 1.5, 0.12), 0.3)

    def _should_display_stream_cached_detections(
        self,
        *,
        current_version: int,
        detections_version: int,
        detections_updated_at: float,
        current_time: float,
        submit_interval: float,
    ) -> bool:
        if detections_version <= 0 or detections_updated_at <= 0:
            return False
        if current_version - detections_version > 1:
            return False
        max_age = max(submit_interval * 3.0, 0.8)
        return current_time - detections_updated_at <= max_age

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
            if detection.plate_number:
                track.unread_observations = 0
            else:
                track.unread_observations += 1
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
        vehicle_type_votes = {}
        if detection.plate_number:
            text_votes[detection.plate_number] = max(detection.confidence, 0.1)
        if detection.plate_color:
            color_votes[detection.plate_color] = max(detection.confidence, 0.1)
        normalized_vehicle_type = self._normalize_vehicle_type_from_label(detection.vehicle_type)
        if normalized_vehicle_type != VEHICLE_TYPE_UNKNOWN:
            vehicle_type_votes[normalized_vehicle_type] = self._vehicle_type_vote_weight(detection)
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
            vehicle_type_votes=vehicle_type_votes,
            unread_observations=1 if not detection.plate_number else 0,
            last_tracking_score=1.0,
            vehicle_type=self._fallback_vehicle_type_for_bbox(
                detection.vehicle_type,
                detection.bbox,
                video_mode=bool(detection.plate_number),
            ),
        )

    def _apply_detection_vote(self, track: PlateTrack, detection: PlateDetection) -> None:
        weight = max(detection.confidence, 0.1)
        if detection.plate_number:
            track.text_votes[detection.plate_number] = track.text_votes.get(detection.plate_number, 0.0) + weight
        if detection.plate_color and detection.plate_color != PLATE_COLOR_UNKNOWN:
            color_weight = weight
            plate_length = len(detection.plate_number or track.plate_number or "")
            if plate_length == 7:
                if detection.plate_color == PLATE_COLOR_BLUE:
                    color_weight *= 1.18
                elif detection.plate_color == PLATE_COLOR_YELLOW:
                    color_weight *= 0.94
            track.color_votes[detection.plate_color] = track.color_votes.get(detection.plate_color, 0.0) + color_weight

        best_text = max(track.text_votes.items(), key=lambda item: (item[1], len(item[0])))[0] if track.text_votes else ""
        best_color = self._pick_stable_track_color(track.color_votes, best_text, fallback_color=track.plate_color)
        current_score = track.text_votes.get(track.plate_number, 0.0)
        next_score = track.text_votes.get(best_text, 0.0)

        if best_text and (not track.plate_number or next_score >= current_score * 1.08):
            track.plate_number = best_text
        if best_color:
            track.plate_color = best_color
        self._apply_vehicle_type_vote(track, detection)
        if self._normalize_vehicle_type_from_label(track.vehicle_type) == VEHICLE_TYPE_UNKNOWN and track.plate_number:
            track.vehicle_type = self._fallback_vehicle_type_for_bbox(
                track.vehicle_type,
                track.bbox,
                video_mode=True,
            )
        track.confidence = max(track.confidence, detection.confidence)

    def _apply_vehicle_type_vote(self, track: PlateTrack, detection: PlateDetection) -> None:
        normalized_vehicle_type = self._normalize_vehicle_type_from_label(detection.vehicle_type)
        if normalized_vehicle_type != VEHICLE_TYPE_UNKNOWN:
            vote_weight = self._vehicle_type_vote_weight(detection)
            if vote_weight > 0:
                track.vehicle_type_votes[normalized_vehicle_type] = (
                    track.vehicle_type_votes.get(normalized_vehicle_type, 0.0) + vote_weight
                )

        best_vehicle_type = self._pick_stable_track_vehicle_type(
            track.vehicle_type_votes,
            fallback_type=track.vehicle_type,
        )
        track.vehicle_type = self._fallback_vehicle_type_for_bbox(
            best_vehicle_type,
            detection.bbox,
            video_mode=bool(detection.plate_number or track.plate_number),
        )

    def _vehicle_type_vote_weight(self, detection: PlateDetection) -> float:
        normalized_vehicle_type = self._normalize_vehicle_type_from_label(detection.vehicle_type)
        if normalized_vehicle_type == VEHICLE_TYPE_UNKNOWN:
            return 0.0

        _, _, width, height = detection.bbox
        area = max(width * height, 1)
        weight = max(detection.confidence, 0.1)

        if not detection.plate_number:
            weight *= 0.45
        else:
            weight *= 1.05

        if area < 420:
            weight *= 0.42
        elif area < 900:
            weight *= 0.72
        elif area >= 2200:
            weight *= 1.12

        if normalized_vehicle_type in {VEHICLE_TYPE_TRUCK, VEHICLE_TYPE_BUS}:
            if not detection.plate_number:
                weight *= 0.42
            if area < 1200:
                weight *= 0.44
        elif normalized_vehicle_type in {
            VEHICLE_TYPE_JEEP,
            VEHICLE_TYPE_PICKUP,
            VEHICLE_TYPE_MOTORCYCLE,
            VEHICLE_TYPE_CRANE,
        }:
            if not detection.plate_number:
                weight *= 0.48
            if area < 900:
                weight *= 0.46
            elif area < 1800:
                weight *= 0.78
            elif area >= 2600:
                weight *= 1.08
        elif normalized_vehicle_type == VEHICLE_TYPE_CAR:
            if detection.plate_number:
                weight *= 1.34
            else:
                weight *= 1.12

        return weight

    def _pick_stable_track_vehicle_type(
        self,
        vehicle_type_votes: dict[str, float],
        *,
        fallback_type: str,
    ) -> str:
        if not vehicle_type_votes:
            return self._normalize_vehicle_type_from_label(fallback_type)

        normalized_votes: dict[str, float] = {}
        for raw_vehicle_type, score in vehicle_type_votes.items():
            normalized = self._normalize_vehicle_type_from_label(raw_vehicle_type)
            if normalized == VEHICLE_TYPE_UNKNOWN or score <= 0:
                continue
            normalized_votes[normalized] = normalized_votes.get(normalized, 0.0) + float(score)

        if not normalized_votes:
            return self._normalize_vehicle_type_from_label(fallback_type)

        car_score = normalized_votes.get(VEHICLE_TYPE_CAR, 0.0)
        truck_score = normalized_votes.get(VEHICLE_TYPE_TRUCK, 0.0)
        bus_score = normalized_votes.get(VEHICLE_TYPE_BUS, 0.0)

        if bus_score >= max(car_score * 2.4, 1.6) and bus_score >= truck_score * 1.35:
            return VEHICLE_TYPE_BUS
        if truck_score >= max(car_score * 2.0, 1.4):
            return VEHICLE_TYPE_TRUCK

        for candidate in (VEHICLE_TYPE_JEEP, VEHICLE_TYPE_PICKUP, VEHICLE_TYPE_MOTORCYCLE, VEHICLE_TYPE_CRANE):
            candidate_score = normalized_votes.get(candidate, 0.0)
            if candidate_score >= max(car_score * 1.45, 1.15):
                return candidate

        if car_score > 0:
            return VEHICLE_TYPE_CAR

        return max(normalized_votes.items(), key=lambda item: item[1])[0]
    def _pick_stable_track_color(
        self,
        color_votes: dict[str, float],
        plate_number: str,
        *,
        fallback_color: str,
    ) -> str:
        if not color_votes:
            return fallback_color

        best_color = max(color_votes.items(), key=lambda item: item[1])[0]
        if len(plate_number or "") != 7:
            return best_color

        blue_votes = color_votes.get(PLATE_COLOR_BLUE, 0.0)
        yellow_votes = color_votes.get(PLATE_COLOR_YELLOW, 0.0)
        if blue_votes <= 0 and yellow_votes <= 0:
            return best_color
        if blue_votes >= yellow_votes * 0.84:
            return PLATE_COLOR_BLUE
        return PLATE_COLOR_YELLOW
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

    def _should_merge_similar_plate_detections(
        self,
        detection: PlateDetection,
        existing: PlateDetection,
    ) -> bool:
        if not detection.plate_number or not existing.plate_number:
            return False
        if len(detection.plate_number) != len(existing.plate_number):
            return False
        if self._compute_iou(detection.bbox, existing.bbox) < 0.35:
            return False
        return self._is_confusable_plate_text_pair(detection.plate_number, existing.plate_number)

    def _is_confusable_plate_text_pair(self, plate_a: str, plate_b: str) -> bool:
        mismatch_count = 0
        for char_a, char_b in zip(plate_a, plate_b):
            if char_a == char_b:
                continue
            mismatch_count += 1
            if not self._is_confusable_plate_char_pair(char_a, char_b):
                return False
        return 0 < mismatch_count <= max(3, len(plate_a) - 2)

    def _is_confusable_plate_char_pair(self, char_a: str, char_b: str) -> bool:
        for group in _CONFUSABLE_PLATE_CHAR_GROUPS:
            if char_a in group and char_b in group:
                return True
        return False

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
            if detection.plate_number or track.plate_number:
                if score < 0.1:
                    continue
            else:
                if not self._is_same_unread_track_bbox(detection.bbox, track.bbox):
                    continue
            if score > best_score:
                best_score = score
                best_index = index

        return best_index if best_score >= 0.1 else None

    def _is_same_unread_track_bbox(self, bbox_a: list[int], bbox_b: list[int]) -> bool:
        iou = self._compute_iou(bbox_a, bbox_b)
        if iou >= 0.2:
            return True
        if iou < 0.08:
            return False

        offset_x_ratio, offset_y_ratio = self._compute_bbox_center_offset_ratio(bbox_a, bbox_b)
        if iou >= 0.14:
            return offset_x_ratio <= 1.05 and offset_y_ratio <= 0.52
        return offset_x_ratio <= 0.65 and offset_y_ratio <= 0.32

    def _compute_bbox_center_offset_ratio(self, bbox_a: list[int], bbox_b: list[int]) -> tuple[float, float]:
        ax, ay, aw, ah = bbox_a
        bx, by, bw, bh = bbox_b
        center_ax = ax + aw / 2
        center_ay = ay + ah / 2
        center_bx = bx + bw / 2
        center_by = by + bh / 2
        width_scale = max(min(aw, bw), 1)
        height_scale = max(min(ah, bh), 1)
        return abs(center_ax - center_bx) / width_scale, abs(center_ay - center_by) / height_scale

    def _is_likely_side_shadow_bbox_for_recognized_track(
        self,
        bbox: list[int],
        tracks: list[PlateTrack],
    ) -> bool:
        if not self._is_plausible_plate_bbox(bbox):
            return False

        _, by, bw, bh = bbox
        bbox_area = bw * bh
        bbox_center_y = by + bh / 2
        for track in tracks:
            if not track.plate_number:
                continue
            _, ty, tw, th = track.bbox
            lateral_ratio, _ = self._compute_bbox_center_offset_ratio(bbox, track.bbox)
            if lateral_ratio < 0.72 or lateral_ratio > 3.2:
                continue

            track_area = max(tw * th, 1)
            area_ratio = bbox_area / track_area
            if area_ratio < 0.38 or area_ratio > 1.85:
                continue

            upper_bound = ty - th * 2.1
            lower_bound = ty + th * 1.05
            if not (upper_bound <= bbox_center_y <= lower_bound):
                continue

            return True
        return False

    def _tracks_to_detections(self, tracks: list[PlateTrack]) -> list[PlateDetection]:
        detections: list[PlateDetection] = []
        for track in tracks:
            if not track.plate_number and not self._should_display_unread_track(track, tracks):
                continue
            display_vehicle_type = self._fallback_vehicle_type_for_bbox(
                track.vehicle_type,
                track.bbox,
                video_mode=bool(track.plate_number),
            )
            detections.append(
                PlateDetection(
                    plate_number=track.plate_number,
                    plate_color=track.plate_color,
                    vehicle_type=display_vehicle_type,
                    confidence=track.confidence,
                    bbox=list(track.bbox),
                )
            )
        return detections

    def _should_display_unread_track(self, track: PlateTrack, tracks: list[PlateTrack]) -> bool:
        if track.plate_number:
            return True
        if not self._is_plausible_plate_bbox(track.bbox):
            return False
        if self._is_likely_side_shadow_bbox_for_recognized_track(track.bbox, tracks):
            return False
        _, _, width, height = track.bbox
        area = width * height
        is_small_target = self._is_small_target_track_bbox(track.bbox)
        if track.confidence < 0.34:
            return False
        if track.unread_observations >= 2:
            if is_small_target:
                if area < 260 and track.confidence < 0.46 and track.unread_observations < 3:
                    return False
                return True
            if track.unread_observations < 3:
                return area >= 760 and track.confidence >= 0.60
            if area < 360 and track.confidence < 0.54:
                return False
            return True
        return area >= 320 and track.confidence >= 0.42

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
                    vehicle_type_votes=self._build_video_detection_vehicle_type_votes(detection, recognized=recognized),
                )
                continue
            current.display_count += 1
            if recognized:
                current.fresh_count += 1
            self._merge_vehicle_type_vote_dict(
                current.vehicle_type_votes,
                self._build_video_detection_vehicle_type_votes(detection, recognized=recognized),
            )
            if detection.confidence > current.detection.confidence:
                current.detection = detection

    def _build_video_detection_vehicle_type_votes(
        self,
        detection: PlateDetection,
        *,
        recognized: bool,
    ) -> dict[str, float]:
        normalized_vehicle_type = self._normalize_vehicle_type_from_label(detection.vehicle_type)
        if normalized_vehicle_type == VEHICLE_TYPE_UNKNOWN:
            return {}

        weight = self._vehicle_type_vote_weight(detection)
        if recognized:
            weight *= 1.15
        else:
            weight *= 0.6
        return {normalized_vehicle_type: weight}

    def _merge_vehicle_type_vote_dict(
        self,
        target_votes: dict[str, float],
        incoming_votes: dict[str, float],
    ) -> None:
        for vehicle_type, weight in incoming_votes.items():
            normalized_vehicle_type = self._normalize_vehicle_type_from_label(vehicle_type)
            if normalized_vehicle_type == VEHICLE_TYPE_UNKNOWN or weight <= 0:
                continue
            target_votes[normalized_vehicle_type] = target_votes.get(normalized_vehicle_type, 0.0) + float(weight)

    def _merge_video_detection_stats_by_number(
        self,
        best_detections: dict[str, VideoDetectionStats],
    ) -> list[VideoDetectionStats]:
        grouped: dict[str, dict[str, object]] = {}
        for stats in best_detections.values():
            plate_number = stats.detection.plate_number
            if not plate_number:
                continue

            entry = grouped.setdefault(
                plate_number,
                {
                    "best_detection": stats.detection,
                    "best_score": (
                        self._score_plate_candidate(stats.detection),
                        stats.fresh_count,
                        stats.display_count,
                        stats.detection.confidence,
                    ),
                    "color_votes": {},
                    "fresh_count": 0,
                    "display_count": 0,
                    "vehicle_type_votes": {},
                },
            )
            entry["fresh_count"] = int(entry["fresh_count"]) + stats.fresh_count
            entry["display_count"] = int(entry["display_count"]) + stats.display_count

            color_votes = entry["color_votes"]
            color_key = self._normalize_plate_color_for_plate(stats.detection.plate_color, plate_number)
            vote_weight = max(stats.fresh_count * 1.4 + stats.display_count * 0.35 + stats.detection.confidence, 0.1)
            color_votes[color_key] = float(color_votes.get(color_key, 0.0)) + vote_weight
            self._merge_vehicle_type_vote_dict(entry["vehicle_type_votes"], stats.vehicle_type_votes)

            candidate_score = (
                self._score_plate_candidate(stats.detection),
                stats.fresh_count,
                stats.display_count,
                stats.detection.confidence,
            )
            if candidate_score > entry["best_score"]:
                entry["best_detection"] = stats.detection
                entry["best_score"] = candidate_score

        merged_stats: list[VideoDetectionStats] = []
        for plate_number, entry in grouped.items():
            best_detection = entry["best_detection"]
            stable_color = self._pick_stable_track_color(
                entry["color_votes"],
                plate_number,
                fallback_color=best_detection.plate_color,
            )
            stable_vehicle_type = self._pick_stable_track_vehicle_type(
                entry["vehicle_type_votes"],
                fallback_type=best_detection.vehicle_type,
            )
            merged_stats.append(
                VideoDetectionStats(
                    detection=best_detection.model_copy(
                        update={
                            "plate_color": stable_color,
                            "vehicle_type": stable_vehicle_type,
                        }
                    ),
                    fresh_count=int(entry["fresh_count"]),
                    display_count=int(entry["display_count"]),
                    vehicle_type_votes=dict(entry["vehicle_type_votes"]),
                )
            )
        return merged_stats

    def _finalize_video_detections(self, best_detections: dict[str, VideoDetectionStats]) -> list[PlateDetection]:
        accepted: list[PlateDetection] = []
        fallback: list[PlateDetection] = []
        ranked_stats = sorted(
            self._merge_video_detection_stats_by_number(best_detections),
            key=lambda item: (item.fresh_count, item.display_count, item.detection.confidence),
            reverse=True,
        )
        for stats in ranked_stats:
            confidence = stats.detection.confidence
            if stats.fresh_count >= 2 and stats.display_count >= 4 and confidence >= 0.34:
                accepted.append(stats.detection)
                continue
            if stats.fresh_count >= 3 and confidence >= 0.38:
                accepted.append(stats.detection)
                continue
            if stats.fresh_count >= 2 and stats.display_count >= 2 and confidence >= 0.46:
                accepted.append(stats.detection)
                continue
            if stats.fresh_count >= 1 and stats.display_count >= 5 and confidence >= 0.44:
                accepted.append(stats.detection)
                continue
            if stats.fresh_count >= 1 and confidence >= 0.66:
                fallback.append(stats.detection)

        if not accepted:
            accepted = fallback[:5]
        accepted = [self._stabilize_video_detection_color(item) for item in accepted]
        accepted = [
            item.model_copy(
                update={
                    "vehicle_type": self._fallback_vehicle_type_for_bbox(
                        item.vehicle_type,
                        item.bbox,
                        video_mode=True,
                    )
                }
            )
            for item in accepted
        ]
        accepted.sort(key=lambda item: item.confidence, reverse=True)
        return accepted

    def _stabilize_video_detection_color(self, detection: PlateDetection) -> PlateDetection:
        plate_number = detection.plate_number or ""
        if len(plate_number) != 7:
            return detection
        if detection.plate_color != PLATE_COLOR_YELLOW:
            return detection
        _, _, width, height = detection.bbox
        if width * height > 900:
            return detection
        return PlateDetection(
            plate_number=detection.plate_number,
            plate_color=PLATE_COLOR_BLUE,
            vehicle_type=detection.vehicle_type,
            confidence=detection.confidence,
            bbox=list(detection.bbox),
        )
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
                    vehicle_type=detection.vehicle_type,
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
        tag_background = (12, 12, 12)
        primary_text = (255, 254, 236)
        secondary_text = (214, 250, 207)
        use_pil_text = self._can_use_pil_annotation_text()
        text_overlays: list[tuple[tuple[int, int], str, object, tuple[int, int, int]]] = []
        plate_font = None
        meta_font = None
        pil_image = None
        pil_draw = None

        if use_pil_text:
            plate_font = self._load_annotation_font(22)
            meta_font = self._load_annotation_font(16)
            use_pil_text = plate_font is not None and meta_font is not None

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
            vehicle_type = self._normalize_vehicle_type_from_label(detection.vehicle_type)
            meta_text = confidence_text
            if vehicle_type != VEHICLE_TYPE_UNKNOWN:
                meta_text = f"{vehicle_type} | {confidence_text}"
            padding_x = 10
            padding_y = 8
            line_gap = 5

            if use_pil_text:
                plate_w, plate_h = self._measure_font_text(plate_text, plate_font)
                meta_w, meta_h = self._measure_font_text(meta_text, meta_font)
            else:
                font = cv2.FONT_HERSHEY_SIMPLEX
                plate_scale = 0.52
                meta_scale = 0.40
                plate_thickness = 2
                meta_thickness = 1
                (plate_w, plate_h), _ = cv2.getTextSize(plate_text, font, plate_scale, plate_thickness)
                (meta_w, meta_h), _ = cv2.getTextSize(meta_text, font, meta_scale, meta_thickness)
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
            cv2.rectangle(annotated, (tag_left, tag_top), (tag_right, tag_bottom), line_color, 2)

            if use_pil_text:
                text_overlays.append(
                    ((tag_left + padding_x, tag_top + padding_y - 1), plate_text, plate_font, primary_text)
                )
                text_overlays.append(
                    (
                        (tag_left + padding_x, tag_top + padding_y + plate_h + line_gap - 1),
                        meta_text,
                        meta_font,
                        secondary_text,
                    )
                )
            else:
                plate_origin = (tag_left + padding_x, tag_top + padding_y + plate_h)
                meta_origin = (tag_left + padding_x, plate_origin[1] + line_gap + meta_h)
                cv2.putText(annotated, plate_text, plate_origin, font, plate_scale, primary_text, plate_thickness, cv2.LINE_AA)
                cv2.putText(annotated, meta_text, meta_origin, font, meta_scale, secondary_text, meta_thickness, cv2.LINE_AA)

        if use_pil_text and text_overlays:
            pil_image, pil_draw = self._prepare_pil_annotation_context(annotated)
            if pil_image is not None and pil_draw is not None:
                for position, text, font, color in text_overlays:
                    self._draw_pil_text(pil_draw, position, text, font, color)
                annotated = self._convert_pil_image_to_bgr(pil_image)
        elif use_pil_text and pil_image is not None:
            annotated = self._convert_pil_image_to_bgr(pil_image)
        return annotated

    def _can_use_pil_annotation_text(self) -> bool:
        try:
            from PIL import Image, ImageDraw, ImageFont  # noqa: F401
        except ImportError:
            return False
        return True

    def _prepare_pil_annotation_context(self, frame):
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return None, None

        rgb_frame = frame[:, :, ::-1]
        image = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(image)
        return image, draw

    def _load_annotation_font(self, font_size: int):
        font_path = self._find_annotation_font_path()
        cache_key = (str(font_path), font_size)
        cached_font = self._annotation_font_cache.get(cache_key)
        if cached_font is not None:
            return cached_font

        try:
            from PIL import ImageFont
        except ImportError:
            return None

        try:
            font = ImageFont.truetype(str(font_path), font_size)
        except Exception:
            logger.warning("Failed to load annotation font: %s", font_path, exc_info=True)
            return None

        self._annotation_font_cache[cache_key] = font
        return font

    def _find_annotation_font_path(self) -> Path:
        candidates = [
            Path(r"C:\Windows\Fonts\msyh.ttc"),
            Path(r"C:\Windows\Fonts\msyhbd.ttc"),
            Path(r"C:\Windows\Fonts\NotoSansSC-Regular.ttf"),
            Path(r"C:\Windows\Fonts\simhei.ttf"),
            Path(r"C:\Windows\Fonts\simsun.ttc"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _measure_font_text(self, text: str, font) -> tuple[int, int]:
        left, top, right, bottom = font.getbbox(text)
        return max(int(right - left), 1), max(int(bottom - top), 1)

    def _draw_pil_text(self, draw, position: tuple[int, int], text: str, font, color_bgr: tuple[int, int, int]) -> None:
        draw.text(position, text, font=font, fill=(color_bgr[2], color_bgr[1], color_bgr[0]))

    def _convert_pil_image_to_bgr(self, image):
        import numpy as np

        rgb_array = np.array(image)
        return rgb_array[:, :, ::-1].copy()

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
        if patch.size == 0 or patch.shape[0] < 6 or patch.shape[1] < 12:
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

    def _save_history(self, detections: list[PlateDetection], source_path: str | None) -> None:
        records = [
            PlateRecord(
                plate_number=detection.plate_number,
                plate_color=detection.plate_color,
                confidence=detection.confidence,
                source_path=source_path,
            )
            for detection in detections
            if detection.plate_number
        ]
        if not records:
            return

        with SessionLocal() as session:
            session.add_all(records)
            session.commit()

    def _persist_upload(self, image_bytes: bytes, filename: str) -> str:
        suffix = Path(filename).suffix or ".jpg"
        target_dir = self.get_upload_root()
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{uuid4().hex}{suffix}"
        target_path.write_bytes(image_bytes)
        return str(target_path)

    def _require_cv2(self):
        try:
            import cv2
        except ImportError as exc:
            raise InferenceDependencyError("Missing opencv-python-headless. Install video-processing dependencies in the backend environment first.") from exc
        return cv2
