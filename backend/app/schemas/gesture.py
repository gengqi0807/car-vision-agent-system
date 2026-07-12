from datetime import datetime

from pydantic import BaseModel, Field


# ---- 基础复用 ----

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


# ---- 车主手势 ----

class OwnerGestureResult(BaseModel):
    """单帧推理结果，包含手势 + 映射后的车机动作。"""
    gesture: str        # 手势原始标签: palm, fist, circle_ccw, circle_cw, swipe_left, swipe_right, thumb_up, thumb_down, wave, pointing, unknown
    action: str         # 映射后的车机动作: wake, confirm, volume_down, volume_up, prev_func, next_func, call_answer, call_hangup, home, idle
    confidence: float
    keypoints: list[Keypoint]
    hand_count: int = Field(default=0, description="检测到的手数量")
    updated_at: datetime


class StreamControlRequest(BaseModel):
    """控制拉流生命周期的请求体。"""
    command: str            # "start" | "stop"
    source: str = ""        # RTSP URL / 摄像头索引 / 视频文件路径
    fps: int = Field(default=15, ge=1, le=60)


class StreamState(BaseModel):
    """当前拉流状态。"""
    running: bool
    source: str = ""
    fps: int = 15
    started_at: datetime | None = None


# ---- 车机控制面板（保留，留给车机控制模块） ----

class ControlPanelState(BaseModel):
    volume: int
    climate_temperature: int
    phone_call_active: bool
    current_mode: str
