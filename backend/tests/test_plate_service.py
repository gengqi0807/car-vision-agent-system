from dataclasses import dataclass

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.database import Base
from app.services import plate_service as plate_service_module
from app.services.plate_service import PlateService

FAKE_PLATE = "TEST123"
FAKE_COLOR = "\u84dd\u724c"


@dataclass
class FakeDetection:
    plate_number: str
    plate_color: str
    confidence: float
    bbox: list[int]


class FakeRecognizer:
    def recognize_all(self, _image_source, **_kwargs):
        return [
            FakeDetection(
                plate_number=FAKE_PLATE,
                plate_color="BLUE",
                confidence=0.93,
                bbox=[10, 20, 120, 40],
            )
        ]


class FakeDetector:
    def detect(self, _source):
        return [
            {
                "label": "plate",
                "kind": "plate",
                "bbox": [30, 40, 140, 44],
                "confidence": 0.88,
            }
        ]


class FakeCropRecognizer:
    def is_available(self):
        return True

    def recognize(self, _image_source):
        return FakeDetection(
            plate_number="\u7ca4B12345",
            plate_color="blue",
            confidence=0.96,
            bbox=[0, 0, 0, 0],
        )


def test_recognize_image_bytes_persists_history(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'plate.db'}", future=True)
    testing_session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(plate_service_module, "SessionLocal", testing_session)

    service = PlateService()
    monkeypatch.setattr(service, "recognizer", FakeRecognizer())
    monkeypatch.setattr(service, "_should_use_detector", lambda: False)

    result = service.recognize_image_bytes(b"fake-image-bytes", "sample.jpg")

    assert result.frame_id == "sample.jpg"
    assert len(result.detections) == 1
    assert result.detections[0].plate_number == FAKE_PLATE
    assert result.detections[0].plate_color == FAKE_COLOR
    assert result.detections[0].confidence == 0.93

    history = service.list_history()
    assert history
    assert history[0].plate_number == FAKE_PLATE


def test_recognize_image_bytes_handles_empty_input():
    service = PlateService()

    result = service.recognize_image_bytes(b"", "sample.jpg")

    assert result.frame_id == "sample.jpg"
    assert result.detections == []


def test_recognize_image_bytes_uses_detector_then_ocr(monkeypatch):
    import cv2
    import numpy as np

    service = PlateService()
    monkeypatch.setattr(service, "recognizer", FakeRecognizer())
    monkeypatch.setattr(service, "detector", FakeDetector())
    monkeypatch.setattr(service, "_should_use_detector", lambda: True)
    monkeypatch.setattr(service, "_save_history", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(settings, "plate_save_uploads", False)

    image = np.zeros((120, 240, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    result = service.recognize_image_bytes(encoded.tobytes(), "detector.jpg")

    assert len(result.detections) == 1
    assert result.detections[0].plate_number == FAKE_PLATE
    assert result.detections[0].bbox == [30, 40, 140, 44]
    assert result.detections[0].confidence > 0.9


def test_recognize_image_bytes_prefers_open_traffic_flow_crop_ocr(monkeypatch):
    import cv2
    import numpy as np

    service = PlateService()
    monkeypatch.setattr(service, "recognizer", FakeRecognizer())
    monkeypatch.setattr(service, "crop_recognizer", FakeCropRecognizer())
    monkeypatch.setattr(service, "detector", FakeDetector())
    monkeypatch.setattr(service, "_should_use_detector", lambda: True)
    monkeypatch.setattr(service, "_save_history", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(settings, "plate_save_uploads", False)

    image = np.zeros((120, 240, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    result = service.recognize_image_bytes(encoded.tobytes(), "detector-lprnet.jpg")

    assert len(result.detections) == 1
    assert result.detections[0].plate_number == "\u7ca4B12345"
    assert result.detections[0].plate_color == "\u84dd\u724c"
    assert result.detections[0].bbox == [30, 40, 140, 44]
    assert result.detections[0].confidence > 0.9
