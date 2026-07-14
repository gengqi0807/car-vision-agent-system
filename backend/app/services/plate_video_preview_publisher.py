from __future__ import annotations

import logging
import subprocess
import threading

import numpy as np

from app.core.config import settings
from app.services.mediamtx_runtime import MediaMTXRuntime

logger = logging.getLogger(__name__)


class PlateVideoPreviewPublisher:
    def __init__(self, job_id: str, on_ready=None) -> None:
        safe_job_id = "".join(ch for ch in job_id if ch.isalnum() or ch in "-_")[:64]
        self.stream_name = f"plate-video-{safe_job_id}"
        self.publish_url = f"{settings.plate_push_rtsp_base_url.rstrip('/')}/{self.stream_name}"
        self.playback_url = f"{settings.plate_push_playback_base_url.rstrip('/')}/{self.stream_name}"
        self._on_ready = on_ready
        self._condition = threading.Condition()
        self._latest_frame: np.ndarray | None = None
        self._stop = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        MediaMTXRuntime.ensure_running()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"preview-{self.stream_name}")
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        with self._condition:
            self._latest_frame = frame
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

    def _run(self) -> None:
        process: subprocess.Popen[bytes] | None = None
        ready_notified = False
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._latest_frame is not None or self._stop)
                    if self._stop:
                        break
                    frame = self._latest_frame
                    self._latest_frame = None
                if frame is None:
                    continue
                frame = self._resize(frame)
                if process is None:
                    height, width = frame.shape[:2]
                    process = self._start_ffmpeg(width, height)
                if process.poll() is not None or process.stdin is None:
                    raise RuntimeError("FFmpeg plate preview publisher exited unexpectedly")
                process.stdin.write(frame.tobytes())
                if not ready_notified:
                    ready_notified = True
                    if self._on_ready is not None:
                        self._on_ready(self.playback_url)
        except Exception:
            logger.exception("Plate video MediaMTX preview publisher failed: %s", self.publish_url)
        finally:
            if process is not None:
                try:
                    if process.stdin:
                        process.stdin.close()
                    process.wait(timeout=2)
                except Exception:
                    process.kill()

    def _start_ffmpeg(self, width: int, height: int) -> subprocess.Popen[bytes]:
        fps = max(1, settings.plate_video_preview_fps)
        command = [
            settings.plate_push_ffmpeg_bin, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-threads", "1", "-pix_fmt", "yuv420p", "-g", str(fps), "-bf", "0",
            "-f", "rtsp", "-rtsp_transport", "tcp", self.publish_url,
        ]
        creationflags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
        return subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            creationflags=creationflags,
        )

    @staticmethod
    def _resize(frame: np.ndarray) -> np.ndarray:
        import cv2

        limit = max(settings.plate_video_preview_max_side, 1)
        height, width = frame.shape[:2]
        longest = max(width, height)
        if longest <= limit:
            return frame
        scale = limit / longest
        return cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
