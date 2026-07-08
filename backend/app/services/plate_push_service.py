from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import subprocess
import threading
import time

from app.core.config import settings
from app.core.logger import get_logger
from app.models_infer.errors import InferenceConfigurationError
from app.schemas.plate import PlateStreamControlResponse
from app.services.plate_service import PlateService

logger = get_logger(__name__)


@dataclass
class PlatePushState:
    running: bool = False
    published: bool = False
    rtsp_url: str | None = None
    stream_name: str | None = None
    publish_rtsp_url: str | None = None
    playback_url: str | None = None
    last_error: str | None = None
    started_at: datetime | None = None


class PlatePushService:
    def __init__(self) -> None:
        self._plate_service = PlateService()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._state = PlatePushState()

    def start(self, rtsp_url: str, stream_name: str | None = None) -> PlateStreamControlResponse:
        worker_to_stop: threading.Thread | None = None
        previous_rtsp_url: str | None = None

        with self._lock:
            stream_name = (stream_name or settings.plate_push_stream_name).strip()
            if not stream_name:
                raise InferenceConfigurationError("推流名称不能为空。")

            if self._state.running:
                if self._state.rtsp_url == rtsp_url and self._state.stream_name == stream_name:
                    return self._to_response()

                previous_rtsp_url = self._state.rtsp_url
                worker_to_stop = self._worker
                self._stop_event.set()

        if worker_to_stop is not None:
            logger.info("Switching plate push worker from %s to %s", previous_rtsp_url, rtsp_url)
            worker_to_stop.join(timeout=6)

        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise InferenceConfigurationError("上一条推流还未完全停止，请稍后重试。")

            self._worker = None
            self._stop_event = threading.Event()
            self._state = PlatePushState(
                running=True,
                published=False,
                rtsp_url=rtsp_url,
                stream_name=stream_name,
                publish_rtsp_url=self._build_publish_rtsp_url(stream_name),
                playback_url=self._build_playback_url(stream_name),
                last_error=None,
                started_at=datetime.utcnow(),
            )
            self._worker = threading.Thread(
                target=self._run_worker,
                args=(rtsp_url, stream_name, self._stop_event),
                daemon=True,
            )
            self._worker.start()
            logger.info("Started plate push worker for %s -> %s", rtsp_url, self._state.publish_rtsp_url)
            return self._to_response()

    def stop(self) -> PlateStreamControlResponse:
        worker: threading.Thread | None
        with self._lock:
            worker = self._worker
            self._stop_event.set()

        if worker is not None:
            worker.join(timeout=6)

        with self._lock:
            running = self._worker is not None and self._worker.is_alive()
            if not running:
                self._worker = None
                self._state.running = False
                self._state.published = False
            return self._to_response()

    def status(self) -> PlateStreamControlResponse:
        with self._lock:
            if self._worker is not None and not self._worker.is_alive():
                self._worker = None
                self._state.running = False
                self._state.published = False
            should_probe = self._state.running and not self._state.published and bool(self._state.publish_rtsp_url)
            publish_rtsp_url = self._state.publish_rtsp_url

        if should_probe and publish_rtsp_url:
            published = self._probe_publish_stream(publish_rtsp_url)
            if published:
                with self._lock:
                    if self._state.running and self._state.publish_rtsp_url == publish_rtsp_url:
                        self._state.published = True

        with self._lock:
            return self._to_response()

    def _run_worker(self, rtsp_url: str, stream_name: str, stop_event: threading.Event) -> None:
        ffmpeg_process: subprocess.Popen[bytes] | None = None
        first_frame_written = False
        try:
            publish_url = self._build_publish_rtsp_url(stream_name)
            for frame, _ in self._plate_service.iter_annotated_stream(rtsp_url, stop_event=stop_event):
                if stop_event.is_set():
                    break

                if ffmpeg_process is None:
                    height, width = frame.shape[:2]
                    ffmpeg_process = self._start_ffmpeg(width=width, height=height, publish_url=publish_url)

                if ffmpeg_process.poll() is not None:
                    stderr_output = self._read_process_error(ffmpeg_process)
                    raise InferenceConfigurationError(
                        f"ffmpeg 推流进程意外退出。{stderr_output or '请确认 mediamtx 已启动，并且推流地址可写。'}"
                    )

                try:
                    assert ffmpeg_process.stdin is not None
                    ffmpeg_process.stdin.write(frame.tobytes())
                    ffmpeg_process.stdin.flush()
                    if not first_frame_written:
                        logger.info("First annotated frame written to ffmpeg stdin for %s", publish_url)
                        first_frame_written = True
                except BrokenPipeError as exc:
                    raise InferenceConfigurationError("ffmpeg 管道已断开，推流中止。") from exc
        except Exception as exc:
            logger.exception("Plate push worker failed")
            with self._lock:
                self._state.last_error = str(exc)
        finally:
            self._shutdown_ffmpeg(ffmpeg_process)
            with self._lock:
                self._state.running = False
                self._state.published = False
                self._worker = None

    def _start_ffmpeg(self, width: int, height: int, publish_url: str) -> subprocess.Popen[bytes]:
        command = [
            settings.plate_push_ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(settings.plate_push_fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-g",
            str(settings.plate_push_fps),
            "-keyint_min",
            str(settings.plate_push_fps),
            "-sc_threshold",
            "0",
            "-bf",
            "0",
            "-b:v",
            settings.plate_push_bitrate,
            "-flush_packets",
            "1",
            "-muxdelay",
            "0.1",
            "-muxpreload",
            "0",
            "-rtsp_transport",
            "tcp",
            "-f",
            "rtsp",
            publish_url,
        ]
        try:
            logger.info("Starting ffmpeg publisher: %s -> %s", settings.plate_push_ffmpeg_bin, publish_url)
            return subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=None,
                bufsize=0,
            )
        except FileNotFoundError as exc:
            raise InferenceConfigurationError(
                f"找不到 ffmpeg，可在 .env 中配置 PLATE_PUSH_FFMPEG_BIN。当前值为 {settings.plate_push_ffmpeg_bin!r}。"
            ) from exc

    def _probe_publish_stream(self, publish_rtsp_url: str) -> bool:
        cv2 = self._plate_service._require_cv2()
        old_capture_options = None
        try:
            import os

            old_capture_options = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            capture = cv2.VideoCapture(publish_rtsp_url, cv2.CAP_FFMPEG)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not capture.isOpened():
                capture.release()
                return False

            deadline = time.monotonic() + 0.6
            while time.monotonic() < deadline:
                ok, frame = capture.read()
                if ok and frame is not None:
                    capture.release()
                    return True
                time.sleep(0.05)
            capture.release()
            return False
        except Exception:
            return False
        finally:
            try:
                import os

                if old_capture_options is None:
                    os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
                else:
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = old_capture_options
            except Exception:
                pass

    def _shutdown_ffmpeg(self, process: subprocess.Popen[bytes] | None) -> None:
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
            process.wait(timeout=5)
        except Exception:
            process.kill()

    def _read_process_error(self, process: subprocess.Popen[bytes]) -> str:
        try:
            if process.stderr is None:
                return ""
            raw = process.stderr.read()
            return raw.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

    def _build_publish_rtsp_url(self, stream_name: str) -> str:
        return f"{settings.plate_push_rtsp_base_url.rstrip('/')}/{stream_name}"

    def _build_playback_url(self, stream_name: str) -> str:
        return f"{settings.plate_push_playback_base_url.rstrip('/')}/{stream_name}"

    def _to_response(self) -> PlateStreamControlResponse:
        return PlateStreamControlResponse(
            running=self._state.running,
            published=self._state.published,
            rtsp_url=self._state.rtsp_url,
            stream_name=self._state.stream_name,
            publish_rtsp_url=self._state.publish_rtsp_url,
            playback_url=self._state.playback_url,
            last_error=self._state.last_error,
            started_at=self._state.started_at,
        )
