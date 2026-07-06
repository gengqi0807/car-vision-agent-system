from datetime import datetime

from pydantic import BaseModel


class Keypoint(BaseModel):
    x: float
    y: float
    score: float


class GestureFrameResult(BaseModel):
    gesture: str
    confidence: float
    keypoints: list[Keypoint]
    updated_at: datetime


class GestureHistoryItem(BaseModel):
    gesture: str
    confidence: float
    updated_at: datetime


class ControlPanelState(BaseModel):
    volume: int
    climate_temperature: int
    phone_call_active: bool
    current_mode: str
