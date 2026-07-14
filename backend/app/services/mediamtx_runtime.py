from __future__ import annotations

import socket
import subprocess
import threading
import time
from pathlib import Path

from app.core.config import settings


class MediaMTXRuntime:
    _lock = threading.Lock()
    _process: subprocess.Popen[bytes] | None = None

    @classmethod
    def ensure_running(cls) -> None:
        if cls._port_open(8554):
            return
        with cls._lock:
            if cls._port_open(8554):
                return
            executable = Path(settings.owner_gesture_mediamtx_bin)
            if not executable.exists():
                raise RuntimeError(f"找不到 MediaMTX：{executable}")
            config_path = executable.with_name("mediamtx.yml")
            command = [str(executable)]
            if config_path.exists():
                command.append(str(config_path))
            cls._process = subprocess.Popen(
                command,
                cwd=str(executable.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            deadline = time.monotonic() + 6.0
            while time.monotonic() < deadline:
                if cls._port_open(8554):
                    return
                if cls._process.poll() is not None:
                    break
                time.sleep(0.1)
            raise RuntimeError("MediaMTX 启动失败，RTSP 端口 8554 未就绪。")

    @staticmethod
    def _port_open(port: int) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return True
        except OSError:
            return False
