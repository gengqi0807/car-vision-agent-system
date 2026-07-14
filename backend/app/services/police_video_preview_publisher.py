from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections import deque

import numpy as np

from app.core.config import settings
from app.services.mediamtx_runtime import MediaMTXRuntime

logger = logging.getLogger(__name__)


class PoliceVideoPreviewPublisher:
    def __init__(self, task_id: str, fps: float) -> None:
        safe_task_id = "".join(ch for ch in task_id if ch.isalnum() or ch in "-_")[:64]
        self.stream_name = f"police-video-{safe_task_id}"
        self.source_fps = max(1.0, float(fps))
        self.fps = max(1.0, min(settings.police_video_preview_output_fps, 60.0))
        self.publish_url = f"{settings.owner_gesture_rtsp_base_url.rstrip('/')}/{self.stream_name}"
        self.playback_url = f"{settings.owner_gesture_playback_base_url.rstrip('/')}/{self.stream_name}"
        self._initial_buffer_frames = max(1, settings.police_video_preview_initial_buffer_frames)
        self._frames: deque[tuple[int, np.ndarray]] = deque(
            maxlen=max(self._initial_buffer_frames, settings.police_video_preview_buffer_max_frames)
        )
        self._rate_samples: deque[tuple[float, int]] = deque(maxlen=60)
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def start(self) -> None:
        MediaMTXRuntime.ensure_running()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"preview-{self.stream_name}")
        self._thread.start()

    def submit(self, frame_index: int, frame: np.ndarray) -> None:
        now = time.perf_counter()
        with self._condition:
            self._frames.append((frame_index, frame.copy()))
            self._rate_samples.append((now, frame_index))
            self._condition.notify_all()

    def close(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=4)

    def _run(self) -> None:
        process: subprocess.Popen[bytes] | None = None
        frame_interval = 1.0 / self.fps
        latest_frame: np.ndarray | None = None
        target_frame_index: float | None = None
        try:
            with self._condition:
                self._condition.wait_for(
                    lambda: len(self._frames) >= self._initial_buffer_frames or self._stop.is_set(),
                    timeout=10.0,
                )
                if self._frames:
                    first_index, latest_frame = self._frames.popleft()
                    target_frame_index = float(first_index)

            if latest_frame is None:
                return

            height, width = latest_frame.shape[:2]
            process = self._start_ffmpeg(width, height)
            next_frame_at = time.perf_counter()
            first_frame_written_at: float | None = None

            while not self._stop.is_set():
                with self._condition:
                    producer_fps = self._producer_fps_locked()
                    target_frame_index += max(1.0, producer_fps / self.fps)
                    if self._frames:
                        selected_position = min(
                            range(len(self._frames)),
                            key=lambda index: abs(self._frames[index][0] - target_frame_index),
                        )
                        selected_index, latest_frame = self._frames[selected_position]
                        target_frame_index = float(selected_index)
                        for _ in range(selected_position + 1):
                            self._frames.popleft()

                delay = next_frame_at - time.perf_counter()
                if delay > 0:
                    self._stop.wait(delay)
                if self._stop.is_set():
                    break
                next_frame_at += frame_interval
                if next_frame_at < time.perf_counter() - frame_interval:
                    next_frame_at = time.perf_counter()
                if process.poll() is not None or process.stdin is None:
                    error = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
                    raise RuntimeError(f"FFmpeg preview publisher exited: {error.strip()}")
                process.stdin.write(latest_frame.tobytes())
                if first_frame_written_at is None:
                    first_frame_written_at = time.perf_counter()
                elif time.perf_counter() - first_frame_written_at >= 0.8:
                    self._ready.set()
        except Exception:
            logger.exception("Police video MediaMTX preview publisher failed: %s", self.publish_url)
        finally:
            if process is not None:
                try:
                    if process.stdin:
                        process.stdin.close()
                    process.wait(timeout=3)
                except Exception:
                    process.kill()

    def _producer_fps_locked(self) -> float:
        if len(self._rate_samples) < 2:
            return self.fps
        first_time, first_index = self._rate_samples[0]
        last_time, last_index = self._rate_samples[-1]
        elapsed = last_time - first_time
        if elapsed <= 0 or last_index <= first_index:
            return self.fps
        return max(1.0, min((last_index - first_index) / elapsed, 240.0))

    def _start_ffmpeg(self, width: int, height: int) -> subprocess.Popen[bytes]:
        command = [
            settings.owner_gesture_ffmpeg_bin, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-r", str(self.fps), "-use_wallclock_as_timestamps", "1", "-fflags", "+nobuffer+discardcorrupt", "-i", "-",
            "-flags", "low_delay", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "ultrafast", "-tune", "zerolatency",
            "-x264-params", "bframes=0:sync-lookahead=0:rc-lookahead=0:sliced-threads=1:force-cfr=0", "-threads", "1",
            "-g", str(max(1, round(self.fps))), "-keyint_min", str(max(1, round(self.fps))), "-sc_threshold", "0",
            "-bf", "0", "-b:v", settings.owner_gesture_push_bitrate, "-fps_mode", "passthrough",
            "-flush_packets", "1", "-muxdelay", "0", "-muxpreload", "0",
            "-rtsp_transport", "tcp", "-f", "rtsp", self.publish_url,
        ]
        return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, bufsize=0)
