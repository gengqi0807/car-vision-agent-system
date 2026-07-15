from __future__ import annotations

import asyncio
import logging
import subprocess
import threading
import time
from datetime import datetime, timezone

import cv2

from app.core.config import settings
from app.core.database import SessionLocal
from app.schemas.gesture import GestureFrameResult, Keypoint, StreamState
from app.services.mediamtx_runtime import MediaMTXRuntime
from app.services.camera_source import open_camera_source
from app.services.camera_lease import CameraLeaseManager
from app.services.monitor_service import MonitorService
from app.services.police_gesture_local_runtime import PoliceGestureVideoSession


logger = logging.getLogger(__name__)


class PoliceGestureStreamService:
    _instance: "PoliceGestureStreamService | None" = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None
        self._ffmpeg: subprocess.Popen[bytes] | None = None
        self._state = StreamState(running=False)
        self._latest: GestureFrameResult | None = None
        self._frame_condition = threading.Condition()
        self._latest_jpeg: bytes | None = None

    @classmethod
    def instance(cls) -> "PoliceGestureStreamService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start(self, source: str = "0", fps: int = 15) -> StreamState:
        self.stop()
        CameraLeaseManager.acquire("交警手势识别")
        try:
            MediaMTXRuntime.ensure_running()
        except Exception:
            CameraLeaseManager.release("交警手势识别")
            raise
        with self._lock:
            self._running = True
            self._state = StreamState(
                running=True,
                source=source,
                fps=fps,
                published=False,
                publish_rtsp_url=self._publish_url(),
                playback_url=self._playback_url(),
                started_at=datetime.now(timezone.utc),
            )
            self._latest_jpeg = None
            self._thread = threading.Thread(target=self._worker, args=(source, fps), daemon=True)
            self._thread.start()
            return self._state

    def stop(self) -> StreamState:
        with self._lock:
            self._running = False
            thread = self._thread
            capture = self._capture
            ffmpeg = self._ffmpeg
        with self._frame_condition:
            self._frame_condition.notify_all()
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
        if ffmpeg is not None:
            try:
                if ffmpeg.stdin:
                    ffmpeg.stdin.close()
            except Exception:
                pass
            try:
                if ffmpeg.poll() is None:
                    ffmpeg.terminate()
            except Exception:
                pass
        if thread and thread.is_alive():
            thread.join(timeout=4)
        with self._lock:
            self._thread = None
            self._capture = None
            self._ffmpeg = None
            self._latest = None
            self._latest_jpeg = None
            self._state = StreamState(running=False)
            CameraLeaseManager.release("交警手势识别")
            return self._state

    def status(self) -> StreamState:
        with self._lock:
            return self._state.model_copy(deep=True)

    def current(self) -> GestureFrameResult:
        with self._lock:
            if self._latest is not None:
                return self._latest.model_copy(deep=True)
        return GestureFrameResult(gesture="no_gesture", confidence=0.0, keypoints=[], updated_at=datetime.now(timezone.utc))

    def mjpeg_frames(self):
        last_frame: bytes | None = None
        while True:
            with self._frame_condition:
                self._frame_condition.wait_for(
                    lambda: (self._latest_jpeg is not None and self._latest_jpeg is not last_frame)
                    or not self._is_running(),
                    timeout=1.0,
                )
                frame = self._latest_jpeg
            if frame is not None and frame is not last_frame:
                last_frame = frame
                yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-store\r\n\r\n" + frame + b"\r\n"
            if not self._is_running() and frame is last_frame:
                break

    @staticmethod
    def _is_effective_gesture(gesture: str | None) -> bool:
        return gesture not in {
            None,
            "",
            "no_gesture",
            "unknown",
            "other",
            "no_pose",
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
        with SessionLocal() as session:
            asyncio.run(
                MonitorService(session).capture_event(
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
            )

    def _worker(self, source: str, fps: int) -> None:
        capture, resolved_source = open_camera_source(source)
        ffmpeg: subprocess.Popen[bytes] | None = None
        low_confidence_streak = 0
        last_low_confidence_at: float | None = None
        no_gesture_started_at = time.monotonic()
        try:
            with self._lock:
                self._capture = capture
            if not capture.isOpened():
                raise RuntimeError(f"无法打开视频源：{source}")
            with self._lock:
                self._state = self._state.model_copy(update={"source": resolved_source})
            with PoliceGestureVideoSession(realtime=True) as session:
                while self._is_running():
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        raise RuntimeError("摄像头读取失败")
                    result = session.process_frame(frame)
                    now = time.monotonic()
                    if self._is_effective_gesture(result.gesture):
                        no_gesture_started_at = now
                    elif now - no_gesture_started_at >= settings.police_gesture_no_gesture_alert_seconds:
                        self._capture_monitor_log_sync(
                            event_type="police_gesture_no_gesture_timeout",
                            title="交警手势长时间无动作",
                            summary=(
                                f"实时监控已开启，但连续 {settings.police_gesture_no_gesture_alert_seconds} 秒"
                                " 未识别到有效动作。"
                            ),
                            details={
                                "source": resolved_source,
                                "mode": "realtime",
                            },
                            trigger_alert=True,
                            level="warning",
                        )
                        no_gesture_started_at = now

                    completed_gesture = result.completed_gesture
                    completed_confidence = round(float(result.completed_confidence or 0.0), 4)
                    if self._is_effective_gesture(completed_gesture):
                        no_gesture_started_at = now
                        if self._is_success_confidence(completed_confidence):
                            low_confidence_streak = 0
                            last_low_confidence_at = None
                            self._capture_monitor_log_sync(
                                event_type="police_gesture_success",
                                title="交警手势识别成功",
                                summary=(
                                    f"实时监控成功识别动作 {completed_gesture}，"
                                    f"置信率 {completed_confidence:.2f}。"
                                ),
                                confidence=completed_confidence,
                                details={
                                    "source": resolved_source,
                                    "mode": "realtime",
                                    "gesture": completed_gesture,
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
                                    f"实时监控识别到动作 {completed_gesture}，"
                                    f"但置信率仅 {completed_confidence:.2f}，"
                                    f"连续低置信次数 {low_confidence_streak}。"
                                ),
                                confidence=completed_confidence,
                                details={
                                    "source": resolved_source,
                                    "mode": "realtime",
                                    "gesture": completed_gesture,
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
                    preview = result.annotated_frame
                    # Keep the browser preview as close as possible to the original camera frame.
                    ok, encoded = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    if ok:
                        with self._frame_condition:
                            self._latest_jpeg = encoded.tobytes()
                            self._frame_condition.notify_all()
                    height, width = result.annotated_frame.shape[:2]
                    if ffmpeg is None:
                        ffmpeg = self._start_ffmpeg(width, height, fps)
                        with self._lock:
                            self._ffmpeg = ffmpeg
                    if ffmpeg.poll() is not None or ffmpeg.stdin is None:
                        raise RuntimeError("FFmpeg 交警手势推流已退出")
                    ffmpeg.stdin.write(result.annotated_frame.tobytes())
                    with self._lock:
                        self._latest = GestureFrameResult(
                            gesture=result.gesture,
                            confidence=result.confidence,
                            keypoints=[Keypoint(**item) for item in result.keypoints],
                            updated_at=datetime.now(timezone.utc),
                        )
                        if not self._state.published:
                            self._state = self._state.model_copy(update={"published": True})
        except Exception as exc:
            with self._lock:
                self._state = self._state.model_copy(update={"running": False, "published": False, "last_error": str(exc)})
        finally:
            capture.release()
            if ffmpeg is not None:
                try:
                    if ffmpeg.stdin:
                        ffmpeg.stdin.close()
                    ffmpeg.wait(timeout=3)
                except Exception:
                    ffmpeg.kill()
            with self._lock:
                self._running = False
                self._thread = None
                self._capture = None
                self._ffmpeg = None
            with self._frame_condition:
                self._frame_condition.notify_all()
            CameraLeaseManager.release("交警手势识别")

    def _start_ffmpeg(self, width: int, height: int, fps: int) -> subprocess.Popen[bytes]:
        command = [
            settings.owner_gesture_ffmpeg_bin, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-pix_fmt", "yuv420p", "-g", str(fps), "-bf", "0", "-f", "rtsp", "-rtsp_transport", "udp",
            self._publish_url(),
        ]
        return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, bufsize=0)

    def _is_running(self) -> bool:
        with self._lock:
            return self._running

    @staticmethod
    def _publish_url() -> str:
        return f"{settings.owner_gesture_rtsp_base_url.rstrip('/')}/police-gesture-live"

    @staticmethod
    def _playback_url() -> str:
        return f"{settings.owner_gesture_playback_base_url.rstrip('/')}/police-gesture-live"
