from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


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


class PlateVideoRecognitionResponse(BaseModel):
    source_filename: str
    processed_video_url: str
    detections: list[PlateDetection]
    unread_samples: list[str] = Field(default_factory=list)
    processed_frame_count: int
    duration_seconds: float | None = None


class PlateRecordSummary(BaseModel):
    id: int
    plate_number: str
    plate_color: str
    created_at: datetime


class PlateStreamStartRequest(BaseModel):
    rtsp_url: str = Field(min_length=1)
    stream_name: str | None = Field(default=None, min_length=1)


class PlateStreamControlResponse(BaseModel):
    running: bool
    published: bool = False
    rtsp_url: str | None = None
    stream_name: str | None = None
    publish_rtsp_url: str | None = None
    playback_url: str | None = None
    last_error: str | None = None
    started_at: datetime | None = None
