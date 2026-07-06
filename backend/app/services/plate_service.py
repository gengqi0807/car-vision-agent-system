from datetime import datetime

from app.schemas.plate import PlateDetection, PlateRecognitionResponse, PlateRecordSummary


class PlateService:
    async def recognize_image(self, filename: str) -> PlateRecognitionResponse:
        detection = PlateDetection(
            plate_number="浙A12345",
            plate_color="蓝牌",
            confidence=0.94,
            bbox=[120, 220, 280, 290],
        )
        return PlateRecognitionResponse(frame_id=filename, detections=[detection])

    def list_history(self) -> list[PlateRecordSummary]:
        return [
            PlateRecordSummary(
                id=1,
                plate_number="浙A12345",
                plate_color="蓝牌",
                created_at=datetime.utcnow(),
            )
        ]
