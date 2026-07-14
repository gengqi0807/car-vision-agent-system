from dataclasses import dataclass
from datetime import datetime
import re
import subprocess
import threading
import time
from urllib.parse import urlparse

from app.core.config import settings
from app.core.logger import get_logger
from app.models_infer.errors import InferenceConfigurationError
from app.schemas.plate import PlateStreamControlResponse
from app.services.plate_service import PlateService
from app.services.mediamtx_runtime import MediaMTXRuntime

logger = get_logger(__name__)


@dataclass
class PlatePushState:
    running: bool = False
    published: bool = False
    publisher_started: bool = False
    process_frames: bool = True
    rtsp_url: str | None = None
    stream_name: str | None = None
    publish_rtsp_url: str | None = None
    playback_url: str | None = None
    last_error: str | None = None
    started_at: datetime | None = None


class PlatePushService:
    def __init__(self, plate_service: PlateService | None = None) -> None:
        self._plate_service = plate_service or PlateService()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._state = PlatePushState()

    def start(
        self,
        rtsp_url: str,
        stream_name: str | None = None,
        process_frames: bool = True,
    ) -> PlateStreamControlResponse:
        worker_to_stop: threading.Thread | None = None
        previous_rtsp_url: str | None = None
        rtsp_url = rtsp_url.strip()
        MediaMTXRuntime.ensure_running()

        with self._lock:
            stream_name = self._resolve_stream_name(rtsp_url, stream_name)
            if not stream_name:
                raise InferenceConfigurationError("Stream name cannot be empty.")

            if self._state.running:
                if (
                    self._state.rtsp_url == rtsp_url
                    and self._state.stream_name == stream_name
                    and self._state.process_frames == process_frames
                ):
                    return self._to_response()

                previous_rtsp_url = self._state.rtsp_url
                worker_to_stop = self._worker
                self._stop_event.set()

        if worker_to_stop is not None:
            logger.info("Switching plate push worker from %s to %s", previous_rtsp_url, rtsp_url)
            worker_to_stop.join(timeout=6)

        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise InferenceConfigurationError("The previous stream is still shutting down. Please retry shortly.")

            self._worker = None
            self._stop_event = threading.Event()
            self._state = PlatePushState(
                running=True,
                published=False,
                publisher_started=False,
                process_frames=process_frames,
                rtsp_url=rtsp_url,
                stream_name=stream_name,
                publish_rtsp_url=self._build_publish_rtsp_url(stream_name),
                playback_url=self._build_playback_url(stream_name),
                last_error=None,
                started_at=datetime.utcnow(),
            )
            self._worker = threading.Thread(
                target=self._run_worker,
                args=(rtsp_url, stream_name, process_frames, self._stop_event),
                daemon=True,
            )
            self._worker.start()
            logger.info(
                "Started plate push worker for %s -> %s | process_frames=%s",
                rtsp_url,
                self._state.publish_rtsp_url,
                process_frames,
            )
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
                self._state.publisher_started = False
            return self._to_response()

    def status(self) -> PlateStreamControlResponse:
        with self._lock:
            if self._worker is not None and not self._worker.is_alive():
                self._worker = None
                self._state.running = False
                self._state.published = False
                self._state.publisher_started = False
            should_probe = (
                self._state.running
                and self._state.publisher_started
                and not self._state.published
                and bool(self._state.publish_rtsp_url)
            )
            publish_rtsp_url = self._state.publish_rtsp_url

        if should_probe and publish_rtsp_url:
            published = self._probe_publish_stream(publish_rtsp_url)
            if published:
                with self._lock:
                    if self._state.running and self._state.publish_rtsp_url == publish_rtsp_url:
                        self._state.published = True

        with self._lock:
            return self._to_response()

    def _run_worker(
        self,
        rtsp_url: str,
        stream_name: str,
        process_frames: bool,
        stop_event: threading.Event,
    ) -> None:
        if process_frames:
            self._run_processed_worker(rtsp_url, stream_name, stop_event)
        else:
            self._run_passthrough_worker(rtsp_url, stream_name, stop_event)

    def _run_processed_worker(self, rtsp_url: str, stream_name: str, stop_event: threading.Event) -> None:
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
                    self._mark_publisher_started(publish_url)

                if ffmpeg_process.poll() is not None:
                    stderr_output = self._read_process_error(ffmpeg_process)
                    raise InferenceConfigurationError(
                        f"ffmpeg publisher exited unexpectedly. {stderr_output or 'Check mediamtx and the publish URL.'}"
                    )

                try:
                    assert ffmpeg_process.stdin is not None
                    ffmpeg_process.stdin.write(frame.tobytes())
                    ffmpeg_process.stdin.flush()
                    if not first_frame_written:
                        logger.info("First annotated frame written to ffmpeg stdin for %s", publish_url)
                        first_frame_written = True
                except BrokenPipeError as exc:
                    raise InferenceConfigurationError("ffmpeg pipe closed while pushing the annotated stream.") from exc
        except Exception as exc:
            logger.exception("Plate push worker failed")
            with self._lock:
                self._state.last_error = str(exc)
        finally:
            self._shutdown_ffmpeg(ffmpeg_process)
            self._reset_state_after_worker_exit()

    def _run_passthrough_worker(self, rtsp_url: str, stream_name: str, stop_event: threading.Event) -> None:
        ffmpeg_process: subprocess.Popen[bytes] | None = None
        try:
            publish_url = self._build_publish_rtsp_url(stream_name)
            ffmpeg_process = self._start_ffmpeg_passthrough(rtsp_url=rtsp_url, publish_url=publish_url)
            self._mark_publisher_started(publish_url)
            logger.info("Started passthrough publisher for %s -> %s", rtsp_url, publish_url)

            while not stop_event.is_set():
                if ffmpeg_process.poll() is not None:
                    stderr_output = self._read_process_error(ffmpeg_process)
                    raise InferenceConfigurationError(
                        f"ffmpeg passthrough exited unexpectedly. {stderr_output or 'Check mediamtx, ffmpeg, and the source RTSP URL.'}"
                    )
                time.sleep(0.2)
        except Exception as exc:
            logger.exception("Plate passthrough worker failed")
            with self._lock:
                self._state.last_error = str(exc)
        finally:
            self._shutdown_ffmpeg(ffmpeg_process)
            self._reset_state_after_worker_exit()

    def _mark_publisher_started(self, publish_url: str) -> None:
        with self._lock:
            if self._state.running and self._state.publish_rtsp_url == publish_url:
                self._state.publisher_started = True

    def _reset_state_after_worker_exit(self) -> None:
        with self._lock:
            self._state.running = False
            self._state.published = False
            self._state.publisher_started = False
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
                f"ffmpeg was not found. Set PLATE_PUSH_FFMPEG_BIN in .env. Current value: {settings.plate_push_ffmpeg_bin!r}."
            ) from exc

    def _start_ffmpeg_passthrough(self, rtsp_url: str, publish_url: str) -> subprocess.Popen[bytes]:
        command = [
            settings.plate_push_ffmpeg_bin,
            "-rtsp_transport",
            "tcp",
            "-i",
            rtsp_url,
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "copy",
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",
            publish_url,
        ]
        try:
            logger.info("Starting ffmpeg passthrough publisher: %s -> %s", rtsp_url, publish_url)
            return subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError as exc:
            raise InferenceConfigurationError(
                f"ffmpeg was not found. Set PLATE_PUSH_FFMPEG_BIN in .env. Current value: {settings.plate_push_ffmpeg_bin!r}."
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

    def _resolve_stream_name(self, rtsp_url: str, stream_name: str | None) -> str:
        candidate = (stream_name or "").strip()
        if not candidate:
            candidate = self._build_default_stream_name(rtsp_url)

        sanitized = re.sub(r"[^A-Za-z0-9_-]+", "-", candidate).strip("-")
        if not sanitized:
            raise InferenceConfigurationError("Invalid stream name. Use letters, numbers, underscores, or hyphens.")
        return sanitized[:64]

    def _build_default_stream_name(self, rtsp_url: str) -> str:
        try:
            parsed = urlparse(rtsp_url)
            path_parts = [part for part in parsed.path.split("/") if part]
            if path_parts:
                tail = path_parts[-1]
                if tail:
                    return f"plate-{tail}"
        except Exception:
            pass
        return settings.plate_push_stream_name

    def _to_response(self) -> PlateStreamControlResponse:
        phase, status_message = self._build_phase_and_message()
        return PlateStreamControlResponse(
            running=self._state.running,
            published=self._state.published,
            publisher_started=self._state.publisher_started,
            phase=phase,
            status_message=status_message,
            process_frames=self._state.process_frames,
            rtsp_url=self._state.rtsp_url,
            stream_name=self._state.stream_name,
            publish_rtsp_url=self._state.publish_rtsp_url,
            playback_url=self._state.playback_url,
            last_error=self._state.last_error,
            started_at=self._state.started_at,
        )

    def _build_phase_and_message(self) -> tuple[str, str]:
        if self._state.running:
            if self._state.published:
                return "running", "识别推流中" if self._state.process_frames else "实时直推预览中"
            if self._state.publisher_started:
                return (
                    "waiting_publish",
                    "已连接源 RTSP，等待本地播放流发布"
                    if self._state.process_frames
                    else "源 RTSP 已接入，等待本地播放流发布",
                )
            return (
                "connecting_source",
                "正在连接源 RTSP，等待首帧"
                if self._state.process_frames
                else "正在连接源 RTSP，准备直接转发到前端",
            )

        if self._state.last_error:
            lower = self._state.last_error.lower()
            if "failed to open the rtsp stream" in lower or "timeout" in lower or "timed out" in lower:
                return "source_unavailable", "源 RTSP 未开启、不可达，或当前没有可读视频帧"
            return "interrupted", "推流中断"

        return "idle", "未启动推流"
