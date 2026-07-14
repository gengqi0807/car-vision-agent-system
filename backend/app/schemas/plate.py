from datetime import datetime

from pydantic import BaseModel, Field


class PlateDetection(BaseModel):
    plate_number: str
    plate_color: str
    vehicle_type: str = "\u672a\u8bc6\u522b"
    confidence: float
    bbox: list[int]


class PlateRecognitionResponse(BaseModel):
    frame_id: str
    detections: list[PlateDetection]


class PlateVideoRecognitionResponse(BaseModel):
    source_filename: str
    processed_video_url: str
    detections: list[PlateDetection]
    unread_samples: list[str] = Field(default_factory=list)
    processed_frame_count: int
    duration_seconds: float | None = None


class PlateVideoJobCreateResponse(BaseModel):
    job_id: str
    status: str


class PlateVideoJobStatusResponse(BaseModel):
    job_id: str
    source_filename: str
    status: str
    progress: float = 0.0
    processed_frame_count: int = 0
    total_frames: int = 0
    detections: list[PlateDetection] = Field(default_factory=list)
    preview_image_url: str | None = None
    processed_video_url: str | None = None
    unread_samples: list[str] = Field(default_factory=list)
    duration_seconds: float | None = None
    error_message: str | None = None


class PlateRecordSummary(BaseModel):
    id: int
    plate_number: str
    plate_color: str
    vehicle_type: str = "未识别"
    created_at: datetime


class PlateStreamStartRequest(BaseModel):
    rtsp_url: str = Field(min_length=1)
    stream_name: str | None = Field(default=None, min_length=1)
    process_frames: bool = True


class PlateStreamControlResponse(BaseModel):
    running: bool
    published: bool = False
    publisher_started: bool = False
    phase: str = "idle"
    status_message: str | None = None
    process_frames: bool = True
    rtsp_url: str | None = None
    stream_name: str | None = None
    publish_rtsp_url: str | None = None
    playback_url: str | None = None
    last_error: str | None = None
    started_at: datetime | None = None
