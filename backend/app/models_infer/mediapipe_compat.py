from __future__ import annotations


def patch_windows_mediapipe_free_symbol() -> None:
    """No-op compatibility shim for MediaPipe 0.10.21.

    The Windows ``free`` symbol regression (google-ai-edge/mediapipe#6187)
    only affects newer MediaPipe wheels that ship
    ``mediapipe.tasks.python.core.mediapipe_c_bindings``. MediaPipe 0.10.21
    does not expose that module and does not suffer from the bug, so no
    ctypes patching is required here. This shim is kept so the call sites in
    ``mediapipe_hands`` / ``mediapipe_pose`` remain valid.
    """
    return
