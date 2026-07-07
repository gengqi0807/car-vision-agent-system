from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from sqlalchemy import select

from app.core.config import settings
from app.core.database import SessionLocal
from app.core.logger import get_logger
from app.models.plate_record import PlateRecord
from app.models_infer.errors import InferenceConfigurationError, InferenceDependencyError
from app.models_infer.hyperlpr_recognizer import HyperLPRRecognizer
from app.schemas.plate import PlateDetection, PlateRecognitionResponse, PlateRecordSummary

logger = get_logger(__name__)


@dataclass
class PlateTrack:
    track_id: str
    plate_number: str
    plate_color: str
    confidence: float
    bbox: list[int]
    template: Any
    last_seen_frame: int
    last_recognized_frame: int
    misses: int = 0


class PlateService:
    def __init__(self) -> None:
        self.recognizer = HyperLPRRecognizer()
        self._backend_dir = Path(__file__).resolve().parents[2]

    async def recognize_image(self, filename: str, image_bytes: bytes | None = None) -> PlateRecognitionResponse:
        return self.recognize_image_bytes(image_bytes or b"", filename)

    def recognize_image_bytes(self, image_bytes: bytes, filename: str = "unknown.jpg") -> PlateRecognitionResponse:
        if not image_bytes:
            return PlateRecognitionResponse(frame_id=filename, detections=[])

        source_path = self._persist_upload(image_bytes, filename) if settings.plate_save_uploads else None
        detections = self._recognize_detections(image_bytes)

        if detections:
            self._save_history(detections, source_path)

        return PlateRecognitionResponse(frame_id=filename, detections=detections)

    def stream_rtsp(self, rtsp_url: str):
        capture, pending_frame = self._open_rtsp_capture(rtsp_url)

        frame_index = 0
        last_sent_at = 0.0
        last_history_saved_at = 0.0
        last_recognition_frame = -settings.plate_stream_process_every_n_frames
        tracks: list[PlateTrack] = []

        try:
            if pending_frame is not None:
                preview_frame = self._resize_stream_frame(pending_frame)
                yield self._build_stream_payload(preview_frame, [])

            while True:
                if pending_frame is not None:
                    source_frame = pending_frame
                    pending_frame = None
                    ok = True
                else:
                    ok, source_frame = capture.read()

                if not ok or source_frame is None:
                    break

                frame_index += 1
                display_frame = self._resize_stream_frame(source_frame)

                tracks = self._update_tracks(source_frame, tracks, frame_index)

                should_recognize = self._should_rerecognize(tracks, frame_index, last_recognition_frame)
                if should_recognize:
                    detections = self._recognize_detections(source_frame)
                    tracks = self._merge_recognized_tracks(source_frame, tracks, detections, frame_index)
                    last_recognition_frame = frame_index

                    if detections:
                        now = time.monotonic()
                        if now - last_history_saved_at >= settings.plate_stream_history_interval_seconds:
                            self._save_history(detections, None)
                            last_history_saved_at = now

                active_detections = self._tracks_to_detections(tracks)
                scaled_detections = self._scale_detections(
                    active_detections,
                    source_frame.shape[1],
                    source_frame.shape[0],
                    display_frame.shape[1],
                    display_frame.shape[0],
                )
                annotated = self._annotate_frame(display_frame, scaled_detections)

                if settings.plate_stream_max_fps > 0:
                    min_interval = 1.0 / settings.plate_stream_max_fps
                    elapsed = time.monotonic() - last_sent_at
                    if elapsed < min_interval:
                        time.sleep(min_interval - elapsed)
                last_sent_at = time.monotonic()

                yield self._build_stream_payload(annotated, scaled_detections)
        finally:
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
                    plate_number="沪A12345",
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
            "无法打开视频流。请确认当前电脑已接入老师提供的内网，RTSP 地址可用，且本机 OpenCV 支持 RTSP/FFMPEG。"
            f" 诊断信息: {detail}"
        )

    def _create_video_capture(self, rtsp_url: str, backend: int | None, ffmpeg_options: str | None):
        cv2 = self._require_cv2()
        if ffmpeg_options:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_options
        else:
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)

        capture = cv2.VideoCapture(rtsp_url, backend) if backend is not None else cv2.VideoCapture(rtsp_url)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _recognize_detections(self, image_source) -> list[PlateDetection]:
        max_side_override = settings.plate_stream_recognition_max_side if not isinstance(image_source, (bytes, bytearray)) else None
        return [
            PlateDetection(
                plate_number=item.plate_number,
                plate_color=item.plate_color,
                confidence=item.confidence,
                bbox=item.bbox,
            )
            for item in self.recognizer.recognize_all(image_source, max_side_override=max_side_override)
        ]

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

    def _should_rerecognize(self, tracks: list[PlateTrack], frame_index: int, last_recognition_frame: int) -> bool:
        if frame_index == 1:
            return True
        if not tracks:
            return True
        if any(track.misses > 0 for track in tracks):
            return True
        interval = max(settings.plate_stream_process_every_n_frames, 1)
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
                    PlateTrack(
                        track_id=uuid4().hex,
                        plate_number=detection.plate_number,
                        plate_color=detection.plate_color,
                        confidence=detection.confidence,
                        bbox=list(detection.bbox),
                        template=template,
                        last_seen_frame=frame_index,
                        last_recognized_frame=frame_index,
                        misses=0,
                    )
                )
                continue

            track = merged_tracks[match_index]
            track.plate_number = detection.plate_number
            track.plate_color = detection.plate_color
            track.confidence = detection.confidence
            track.bbox = list(detection.bbox)
            track.template = template
            track.last_seen_frame = frame_index
            track.last_recognized_frame = frame_index
            track.misses = 0
            matched_indices.add(match_index)

        return merged_tracks

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

    def _build_stream_payload(self, frame, detections: list[PlateDetection]):
        cv2 = self._require_cv2()
        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), settings.plate_stream_jpeg_quality],
        )
        if not success:
            raise InferenceConfigurationError("视频帧编码失败，无法回传到前端。")

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

    def _annotate_frame(self, frame, detections: list[PlateDetection]):
        cv2 = self._require_cv2()
        annotated = frame.copy()
        for index, detection in enumerate(detections, start=1):
            x, y, width, height = detection.bbox
            x2 = x + width
            y2 = y + height
            is_blue_plate = "蓝" in detection.plate_color
            color = (184, 196, 208) if is_blue_plate else (45, 212, 191)
            cv2.rectangle(annotated, (x, y), (x2, y2), color, 2)
            label = f"{index}. {detection.plate_number} {int(detection.confidence * 100)}%"
            label_y = y - 10 if y > 24 else y2 + 20
            cv2.putText(
                annotated,
                label,
                (max(x, 4), max(label_y, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return annotated

    def _resize_stream_frame(self, frame):
        cv2 = self._require_cv2()
        max_side_limit = settings.plate_stream_max_side
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
        target_dir = (self._backend_dir / settings.plate_upload_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{uuid4().hex}{suffix}"
        target_path.write_bytes(image_bytes)
        return str(target_path)

    def _require_cv2(self):
        try:
            import cv2
        except ImportError as exc:
            raise InferenceDependencyError(
                "缺少 opencv-python-headless，请先在后端环境中安装视频流处理依赖。"
            ) from exc
        return cv2
