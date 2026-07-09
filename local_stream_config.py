from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StreamRuntimeConfig:
    mediamtx_home: Path
    ffmpeg_bin_home: Path
    stream_host: str = "127.0.0.1"
    stream_port: int = 8554
    default_stream_name: str = "test"

    @property
    def mediamtx_executable(self) -> Path:
        return _resolve_executable(self.mediamtx_home, "mediamtx.exe")

    @property
    def ffmpeg_executable(self) -> Path:
        return _resolve_executable(self.ffmpeg_bin_home, "ffmpeg.exe")

    @property
    def ffprobe_executable(self) -> Path:
        return _resolve_executable(self.ffmpeg_bin_home, "ffprobe.exe")

    def build_rtsp_url(self, stream_name: str | None = None) -> str:
        name = (stream_name or self.default_stream_name).strip() or self.default_stream_name
        return f"rtsp://{self.stream_host}:{self.stream_port}/{name}"


def _resolve_executable(base_path: Path, executable_name: str) -> Path:
    if base_path.is_file():
        return base_path
    direct_target = base_path / executable_name
    if direct_target.exists():
        return direct_target
    matches = list(base_path.rglob(executable_name))
    if matches:
        return matches[0]
    return direct_target


STREAM_RUNTIME = StreamRuntimeConfig(
    mediamtx_home=Path(r"D:\tool\mediamtx\mediamtx_v1.19.2_windows_amd64"),
    ffmpeg_bin_home=Path(r"D:\tool\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin"),
)
