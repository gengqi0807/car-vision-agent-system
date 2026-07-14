"""
自定义手势 Pydantic Schema。
"""

from datetime import datetime

from pydantic import BaseModel, Field


# ── 手势 ──────────────────────────────────────────────────────────

class CustomGestureCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64, description="手势英文标识（如 peace / ok）")
    display_name: str = Field(default="", description="手势显示名称")
    description: str = Field(default="", description="手势说明")


class CustomGestureUpdate(BaseModel):
    display_name: str | None = None
    description: str | None = None


class CustomGestureOut(BaseModel):
    id: int
    name: str
    display_name: str
    description: str
    sample_count: int
    is_trained: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class CustomGestureListOut(BaseModel):
    gestures: list[CustomGestureOut]
    total: int


# ── 样本 ──────────────────────────────────────────────────────────

class KeypointItem(BaseModel):
    x: float
    y: float
    z: float = 0.0


class CustomGestureSampleCreate(BaseModel):
    """上传单个样本时的手部关键点（21 个点）。"""
    keypoints: list[KeypointItem] = Field(min_length=21, max_length=21)
    source_type: str = Field(default="upload")
    filename: str = Field(default="")


class CustomGestureSampleOut(BaseModel):
    id: int
    gesture_id: int
    keypoints: list[KeypointItem]
    source_type: str
    filename: str
    created_at: datetime

    model_config = {"from_attributes": True}


class CustomGestureSampleListOut(BaseModel):
    samples: list[CustomGestureSampleOut]
    total: int


class CustomGestureSampleRejected(BaseModel):
    """被拒绝的图片及其原因。"""
    filename: str
    reason: str


class CustomGestureSampleBatchOut(BaseModel):
    """批量上传样本的响应。"""
    samples: list[CustomGestureSampleOut]
    rejected: list[CustomGestureSampleRejected]
    total_uploaded: int
    total_accepted: int
    total_rejected: int


# ── 训练 ──────────────────────────────────────────────────────────

class CustomGestureTrainRequest(BaseModel):
    """触发训练请求。gesture_names 为空则训练全部自定义手势。"""
    gesture_names: list[str] = Field(default_factory=list)


class CustomGestureTrainOut(BaseModel):
    status: str  # "success" | "no_data" | "single_class"
    message: str
    n_samples: int = 0
    n_classes: int = 0
    class_names: list[str] = Field(default_factory=list)
    model_path: str = ""
    evaluation: dict = Field(default_factory=dict)
