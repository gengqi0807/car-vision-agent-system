from __future__ import annotations

import os

from app.core.config import settings
from app.core.logger import get_logger


logger = get_logger(__name__)
_configured = False


def configure_cpu_runtime() -> None:
    """Apply bounded CPU thread pools once per process."""
    global _configured
    if _configured:
        return

    torch_threads = max(int(settings.inference_torch_cpu_threads), 1)
    opencv_threads = max(int(settings.inference_opencv_cpu_threads), 1)
    os.environ.setdefault("OMP_NUM_THREADS", str(torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(torch_threads))

    try:
        import cv2

        cv2.setNumThreads(opencv_threads)
    except Exception:
        logger.warning("Failed to configure OpenCV CPU threads.", exc_info=True)

    try:
        import torch

        torch.set_num_threads(torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Inter-op threads can only be set before parallel work begins.
            pass
    except Exception:
        logger.warning("Failed to configure Torch CPU threads.", exc_info=True)

    _configured = True

