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
