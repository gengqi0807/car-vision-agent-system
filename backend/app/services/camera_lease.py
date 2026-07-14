from __future__ import annotations

import threading


class CameraLeaseManager:
    _lock = threading.Lock()
    _owner: str | None = None

    @classmethod
    def acquire(cls, owner: str) -> None:
        with cls._lock:
            if cls._owner is not None and cls._owner != owner:
                raise RuntimeError(f"摄像头已被{cls._owner}占用，请先关闭该功能后再继续。")
            cls._owner = owner

    @classmethod
    def release(cls, owner: str) -> None:
        with cls._lock:
            if cls._owner == owner:
                cls._owner = None
