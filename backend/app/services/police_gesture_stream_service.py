from __future__ import annotations

import subprocess
import threading
from datetime import datetime, timezone

import cv2

from app.core.config import settings
from app.schemas.gesture import GestureFrameResult, Keypoint, StreamState
from app.services.mediamtx_runtime import MediaMTXRuntime
from app.services.camera_source import open_camera_source
from app.services.police_gesture_local_runtime import PoliceGestureVideoSession


class PoliceGestureStreamService:
    _instance: "PoliceGestureStreamService | None" = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._state = StreamState(running=False)
        self._latest: GestureFrameResult | None = None

    @classmethod
    def instance(cls) -> "PoliceGestureStreamService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start(self, source: str = "0", fps: int = 15) -> StreamState:
        self.stop()
        MediaMTXRuntime.ensure_running()
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
            self._thread = threading.Thread(target=self._worker, args=(source, fps), daemon=True)
            self._thread.start()
            return self._state

    def stop(self) -> StreamState:
        with self._lock:
            self._running = False
            thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=4)
        with self._lock:
            self._thread = None
            self._state = StreamState(running=False)
            return self._state

    def status(self) -> StreamState:
        with self._lock:
            return self._state.model_copy(deep=True)

    def current(self) -> GestureFrameResult:
        with self._lock:
            if self._latest is not None:
                return self._latest.model_copy(deep=True)
        return GestureFrameResult(gesture="no_gesture", confidence=0.0, keypoints=[], updated_at=datetime.now(timezone.utc))

    def _worker(self, source: str, fps: int) -> None:
        capture, resolved_source = open_camera_source(source)
        ffmpeg: subprocess.Popen[bytes] | None = None
        try:
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
                    height, width = result.annotated_frame.shape[:2]
                    if ffmpeg is None:
                        ffmpeg = self._start_ffmpeg(width, height, fps)
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
