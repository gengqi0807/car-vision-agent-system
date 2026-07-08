from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AlertLevel = Literal["critical", "warning", "info"]


class AlertEventCreate(BaseModel):
    level: AlertLevel
    source: str = Field(min_length=2, max_length=64)
    title: str = Field(min_length=2, max_length=128)
    summary: str = Field(min_length=2, max_length=2000)


class AlertEvent(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    level: AlertLevel
    source: str
    title: str
    summary: str
    created_at: datetime


class AlertOverview(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total": 12,
                "critical": 2,
                "warning": 4,
                "info": 6,
                "latest": [
                    {
                        "id": 101,
                        "level": "critical",
                        "source": "plate-recognition",
                        "title": "连续识别失败",
                        "summary": "最近 5 分钟内车牌识别连续失败 8 次，请检查模型与输入源。",
                        "created_at": "2026-07-08T12:00:00",
                    }
                ],
            }
        }
    )
    total: int
    critical: int
    warning: int
    info: int
    latest: list[AlertEvent]


class AlertPushRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    channel: str
    target: str
    success: bool
    created_at: datetime


class BehaviorLogRecord(BaseModel):
    id: int
    source: str
    title: str
    summary: str
    created_at: datetime


class OperationLogRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    operation_type: str
    response_status: str | None = None
    created_at: datetime
