from datetime import datetime

from pydantic import BaseModel


class PlateDetection(BaseModel):
    plate_number: str
    plate_color: str
    confidence: float
    bbox: list[int]


class PlateRecognitionResponse(BaseModel):
    frame_id: str
    detections: list[PlateDetection]


class PlateRecordSummary(BaseModel):
    id: int
    plate_number: str
    plate_color: str
    created_at: datetime
