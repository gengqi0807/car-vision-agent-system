"""Video streaming helper — MediaMTX + FFmpeg.

Requires MediaMTX and FFmpeg installed.  Paths can be configured via
environment variables or the ``StreamConfig`` dataclass.

Environment variables
---------------------
MEDIAMTX_PATH : str
    Absolute path to ``mediamtx.exe`` (Windows) or ``mediamtx`` (Linux/macOS).
FFMPEG_PATH : str
    Absolute path to ``ffmpeg`` / ``ffmpeg.exe``.  If unset, ``ffmpeg`` is
    resolved from the system PATH.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _resolve_ffmpeg() -> str:
    """Try to locate ffmpeg: explicit env → PATH → raise."""
    env_path = os.environ.get("FFMPEG_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    # On Windows, ffmpeg.exe ; elsewhere, ffmpeg
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    try:
        result = subprocess.run(
            [exe, "-version"],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode == 0:
            return exe
    except FileNotFoundError:
        pass
    raise FileNotFoundError(
        "FFmpeg not found. Set FFMPEG_PATH env var or add ffmpeg to system PATH. "
        "Download: https://github.com/BtbN/FFmpeg-Builds/releases"
    )


@dataclass
class StreamConfig:
    """Configuration for MediaMTX + FFmpeg streaming."""

    mediamtx_path: str = field(
        default_factory=lambda: os.environ.get(
            "MEDIAMTX_PATH", ""
        )
    )
    ffmpeg_path: str = field(default_factory=_resolve_ffmpeg)
    rtsp_host: str = field(
        default_factory=lambda: os.environ.get("RTSP_HOST", "127.0.0.1")
    )
    rtsp_port: int = field(
        default_factory=lambda: int(os.environ.get("RTSP_PORT", "8554"))
    )

    def __post_init__(self) -> None:
        if not self.mediamtx_path or not Path(self.mediamtx_path).exists():
            raise FileNotFoundError(
                f"MediaMTX not found at '{self.mediamtx_path}'. "
                "Set MEDIAMTX_PATH env var or update StreamConfig. "
                "Download: https://github.com/bluenviron/mediamtx"
            )


# ---------------------------------------------------------------------------
# Stream manager
# ---------------------------------------------------------------------------


class StreamManager:
    """Manages MediaMTX server and FFmpeg push streams."""

    def __init__(self, config: StreamConfig | None = None) -> None:
        self._cfg = config or StreamConfig()
        self._mediamtx_proc: subprocess.Popen | None = None
        self._ffmpeg_proc: subprocess.Popen | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_mediamtx(self) -> None:
        """Launch the MediaMTX server process."""
        self._mediamtx_proc = subprocess.Popen(
            self._cfg.mediamtx_path,
            cwd=str(Path(self._cfg.mediamtx_path).parent),
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        time.sleep(1.5)
        if self._mediamtx_proc.poll() is not None:
            stderr_output = (
                self._mediamtx_proc.stderr.read().decode("utf-8", errors="replace")
                if self._mediamtx_proc.stderr
                else "(no output)"
            )
            raise RuntimeError(f"MediaMTX failed to start:\n{stderr_output}")
        print(f"MediaMTX started, PID={self._mediamtx_proc.pid}")

    def start_push(self, video_source: str, stream_name: str = "carview") -> None:
        """Push a video source to the RTSP server."""
        rtsp_url = f"rtsp://{self._cfg.rtsp_host}:{self._cfg.rtsp_port}/{stream_name}"
        ffmpeg_cmd = [
            self._cfg.ffmpeg_path,
            "-re",
            "-i",
            video_source,
            "-c",
            "copy",
            "-f",
            "rtsp",
            rtsp_url,
        ]
        self._ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        print(f"FFmpeg pushing → {rtsp_url}")

    def test_pull_stream(self, stream_name: str = "test") -> None:
        """Verify we can read a frame from the RTSP stream."""
        rtsp_url = f"rtsp://{self._cfg.rtsp_host}:{self._cfg.rtsp_port}/{stream_name}"
        cap = cv2.VideoCapture(rtsp_url)
        ret, frame = cap.read()
        if ret:
            print(f"Pull OK — frame shape={frame.shape}")
        else:
            print("Pull FAILED — is a video being pushed?")
        cap.release()

    def stop(self) -> None:
        """Stop FFmpeg push + MediaMTX."""
        for proc, label in (
            (self._ffmpeg_proc, "FFmpeg"),
            (self._mediamtx_proc, "MediaMTX"),
        ):
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                print(f"{label} stopped")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    manager = StreamManager()
    manager.start_mediamtx()
    time.sleep(2)
    manager.start_push(video_source="test.mp4", stream_name="test")
    time.sleep(3)
    manager.test_pull_stream()
    manager.stop()
