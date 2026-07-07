from dataclasses import dataclass

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.services import plate_service as plate_service_module
from app.services.plate_service import PlateService


@dataclass
class FakeDetection:
    plate_number: str
    plate_color: str
    confidence: float
    bbox: list[int]


class FakeRecognizer:
    def recognize_all(self, _image_source):
        return [
            FakeDetection(
                plate_number="沪A12345",
                plate_color="蓝牌",
                confidence=0.93,
                bbox=[10, 20, 120, 40],
            )
        ]


def test_recognize_image_bytes_persists_history(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'plate.db'}", future=True)
    testing_session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(plate_service_module, "SessionLocal", testing_session)

    service = PlateService()
    monkeypatch.setattr(service, "recognizer", FakeRecognizer())

    result = service.recognize_image_bytes(b"fake-image-bytes", "sample.jpg")

    assert result.frame_id == "sample.jpg"
    assert len(result.detections) == 1
    assert result.detections[0].plate_number == "沪A12345"
    assert result.detections[0].plate_color == "蓝牌"
    assert result.detections[0].confidence == 0.93

    history = service.list_history()
    assert history
    assert history[0].plate_number == "沪A12345"


def test_recognize_image_bytes_handles_empty_input():
    service = PlateService()

    result = service.recognize_image_bytes(b"", "sample.jpg")

    assert result.frame_id == "sample.jpg"
    assert result.detections == []
