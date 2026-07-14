from __future__ import annotations

import ctypes
import os
from importlib import resources
from typing import Sequence

from mediapipe.tasks.python.core import mediapipe_c_bindings
from mediapipe.tasks.python.core.mediapipe_c_utils import CFunction


def patch_windows_mediapipe_free_symbol() -> None:
    """Patch MediaPipe's Windows ctypes loader when libmediapipe.dll lacks free().

    Some MediaPipe wheels for Windows expose the C task APIs from
    ``libmediapipe.dll`` without exporting ``free``. The Python bindings still
    expect that symbol during landmarker initialisation and crash before any
    inference can run. Falling back to the system CRT ``free`` keeps the Tasks
    API usable in this environment.
    """

    if os.name != "nt":
        return

    if getattr(mediapipe_c_bindings, "_car_vision_windows_free_patch", False):
        return

    crt = ctypes.CDLL("msvcrt")

    def load_raw_library(signatures: Sequence[CFunction] = ()) -> ctypes.CDLL:
        shared_lib = getattr(mediapipe_c_bindings, "_shared_lib", None)
        if shared_lib is None:
            absolute_lib_path = str(resources.files("mediapipe.tasks.c") / "libmediapipe.dll")
            shared_lib = ctypes.CDLL(absolute_lib_path)
            mediapipe_c_bindings._shared_lib = shared_lib

        for signature in signatures:
            c_func = getattr(shared_lib, signature.func_name)
            c_func.argtypes = signature.argtypes
            c_func.restype = signature.restype

        try:
            free_func = getattr(shared_lib, "free")
        except AttributeError:
            free_func = crt.free
            setattr(shared_lib, "free", free_func)

        free_func.argtypes = [ctypes.c_void_p]
        free_func.restype = None
        return shared_lib

    mediapipe_c_bindings.load_raw_library = load_raw_library
    mediapipe_c_bindings._car_vision_windows_free_patch = True
