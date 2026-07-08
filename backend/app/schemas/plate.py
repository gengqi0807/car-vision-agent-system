from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PlateDetection(BaseModel):
    plate_number: str
    plate_color: str
    confidence: float
    bbox: list[int]


class PlateRecognitionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "frame_id": "frame-20260708-001",
                "detections": [
                    {
                        "plate_number": "粤B12345",
                        "plate_color": "蓝牌",
                        "confidence": 0.982,
                        "bbox": [128, 96, 256, 148],
                    }
                ],
            }
        }
    )

    frame_id: str
    detections: list[PlateDetection]


class PlateRecordSummary(BaseModel):
    id: int
    plate_number: str
    plate_color: str
    created_at: datetime
