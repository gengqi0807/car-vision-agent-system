from datetime import datetime

from pydantic import BaseModel, Field


class Keypoint(BaseModel):
    x: float
    y: float
    score: float


class ControlPanelState(BaseModel):
    system_awake: bool
    volume: int
    climate_temperature: int
    phone_call_active: bool
    current_mode: str
    media_playing: bool = True
    comfort_scene: str = "标准"
    vehicle_status: str = "就绪"
    focus_tile: str = "home"
    last_gesture: str | None = None
    last_command: str | None = None
    last_command_at: datetime | None = None
    last_feedback: str | None = None
    updated_at: datetime | None = None


class GestureFrameResult(BaseModel):
    gesture: str
    action: str | None = None
    confidence: float
    keypoints: list[Keypoint]
    annotated_image: str | None = None
    control_command: str | None = None
    triggered: bool = False
    panel_state: ControlPanelState | None = None
    updated_at: datetime


class GestureHistoryItem(BaseModel):
    gesture: str
    confidence: float
    source_path: str | None = None
    updated_at: datetime


class PoliceGestureVideoResult(BaseModel):
    source_filename: str
    gesture: str
    confidence: float
    keypoints: list[Keypoint]
    task_id: str | None = None
    processed_video_url: str
    processed_frame_count: int
    duration_seconds: float | None = None
    updated_at: datetime


class PoliceGestureVideoEvent(BaseModel):
    gesture: str
    confidence: float
    frame_index: int
    timestamp_seconds: float | None = None
    message: str = ""
    updated_at: datetime


class PoliceGestureVideoProgress(BaseModel):
    task_id: str
    source_filename: str = ""
    status: str
    progress: float = 0.0
    message: str = ""
    processed_frame_count: int = 0
    total_frames: int | None = None
    gesture: str | None = None
    confidence: float | None = None
    annotated_frame: str | None = None
    playback_url: str | None = None
    processed_video_url: str | None = None
    duration_seconds: float | None = None
    events: list[PoliceGestureVideoEvent] = Field(default_factory=list)
    updated_at: datetime


class PoliceGestureVideoJobCreateResponse(BaseModel):
    task_id: str
    status: str


class OwnerGestureResult(BaseModel):
    gesture: str
    action: str
    confidence: float
    keypoints: list[Keypoint]
    annotated_image: str | None = None
    hand_count: int = Field(default=0, description="检测到的手数量")
    control_command: str | None = None
    triggered: bool = False
    panel_state: ControlPanelState | None = None
    updated_at: datetime


class StreamControlRequest(BaseModel):
    command: str = Field(default="start")
    source: str = ""
    fps: int = Field(default=15, ge=1, le=60)


class StreamState(BaseModel):
    running: bool
    source: str = ""
    fps: int = 15
    published: bool = False
    publish_rtsp_url: str | None = None
    playback_url: str | None = None
    last_error: str | None = None
    started_at: datetime | None = None
