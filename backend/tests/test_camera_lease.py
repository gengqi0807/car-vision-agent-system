import pytest

from app.services.camera_lease import CameraLeaseManager


def test_camera_lease_rejects_another_feature() -> None:
    CameraLeaseManager.release("车内手势识别")
    CameraLeaseManager.release("交警手势识别")
    CameraLeaseManager.acquire("车内手势识别")
    try:
        with pytest.raises(RuntimeError, match="车内手势识别"):
            CameraLeaseManager.acquire("交警手势识别")
    finally:
        CameraLeaseManager.release("车内手势识别")

    CameraLeaseManager.acquire("交警手势识别")
    CameraLeaseManager.release("交警手势识别")
