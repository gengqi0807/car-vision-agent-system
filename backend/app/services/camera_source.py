from __future__ import annotations

import cv2

from app.core.config import settings


def open_camera_source(source: str) -> tuple[cv2.VideoCapture, str]:
    normalized = str(source or "").strip()
    if normalized.lower() not in {"", "auto", "default"}:
        resolved: str | int = int(normalized) if normalized.isdigit() else normalized
        return cv2.VideoCapture(resolved), normalized

    tried: set[int] = set()
    for index in (settings.external_camera_index, settings.local_camera_index):
        if index in tried:
            continue
        tried.add(index)
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if capture.isOpened():
            ok, _ = capture.read()
            if ok:
                return capture, str(index)
        capture.release()

    return cv2.VideoCapture(settings.external_camera_index, cv2.CAP_DSHOW), str(settings.external_camera_index)
