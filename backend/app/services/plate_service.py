from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.plate_record import PlateRecord
from app.models_infer.hyperlpr_recognizer import HyperLPRRecognizer
from app.schemas.plate import PlateDetection, PlateRecognitionResponse, PlateRecordSummary


class PlateService:
    def __init__(self) -> None:
        self.recognizer = HyperLPRRecognizer()
        self._backend_dir = Path(__file__).resolve().parents[2]

    async def recognize_image(self, filename: str, image_bytes: bytes | None = None) -> PlateRecognitionResponse:
        return self.recognize_image_bytes(image_bytes or b"", filename)

    def recognize_image_bytes(self, image_bytes: bytes, filename: str = "unknown.jpg") -> PlateRecognitionResponse:
        if not image_bytes:
            return PlateRecognitionResponse(frame_id=filename, detections=[])

        source_path = self._persist_upload(image_bytes, filename) if settings.plate_save_uploads else None
        detections = [
            PlateDetection(
                plate_number=item.plate_number,
                plate_color=item.plate_color,
                confidence=item.confidence,
                bbox=item.bbox,
            )
            for item in self.recognizer.recognize_all(image_bytes)
        ]

        if detections:
            self._save_history(detections, source_path)

        return PlateRecognitionResponse(frame_id=filename, detections=detections)

    def list_history(self) -> list[PlateRecordSummary]:
        with SessionLocal() as session:
            statement = (
                select(PlateRecord)
                .order_by(PlateRecord.created_at.desc())
                .limit(settings.plate_history_limit)
            )
            records = session.scalars(statement).all()

        if not records:
            return [
                PlateRecordSummary(
                    id=1,
                    plate_number="沪A12345",
                    plate_color="蓝牌",
                    created_at=datetime.utcnow(),
                )
            ]

        return [
            PlateRecordSummary(
                id=record.id,
                plate_number=record.plate_number,
                plate_color=record.plate_color,
                created_at=record.created_at,
            )
            for record in records
        ]

    def _save_history(self, detections: list[PlateDetection], source_path: str | None) -> None:
        records = [
            PlateRecord(
                plate_number=detection.plate_number,
                plate_color=detection.plate_color,
                confidence=detection.confidence,
                source_path=source_path,
            )
            for detection in detections
            if detection.plate_number
        ]
        if not records:
            return

        with SessionLocal() as session:
            session.add_all(records)
            session.commit()

    def _persist_upload(self, image_bytes: bytes, filename: str) -> str:
        suffix = Path(filename).suffix or ".jpg"
        target_dir = (self._backend_dir / settings.plate_upload_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{uuid4().hex}{suffix}"
        target_path.write_bytes(image_bytes)
        return str(target_path)
