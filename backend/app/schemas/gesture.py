from datetime import datetime

from pydantic import BaseModel


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
    confidence: float
    keypoints: list[Keypoint]
    control_command: str | None = None
    triggered: bool = False
    panel_state: ControlPanelState | None = None
    updated_at: datetime


class GestureHistoryItem(BaseModel):
    gesture: str
    confidence: float
    updated_at: datetime
